#include "droid/simulation.hpp"

#include "droid/module_catalog.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace droid {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMotorCount = 4;
constexpr std::size_t kTouchSensorCount = 2;
constexpr std::size_t kPhysicsStepsPerBatch = 5;
constexpr double kTrackLength = 40.0;
constexpr double kBatteryCapacityJ = 72'000.0;
constexpr double kLightSourceXM = 16.0;
constexpr double kLightSourceYM = 0.0;
constexpr double kLightSearchMinimumDistanceM = 12.0;
constexpr double kLightSearchMaximumDistanceM = 20.0;
constexpr double kLightSearchDistanceStepM = 0x1p-21;
constexpr std::uint32_t kLightSearchDistanceBucketMask = 0x00ff'ffffU;
constexpr std::uint32_t kLightSearchPositiveSideKeyBit = 0x0100'0000U;
constexpr double kMaximumAgentControlDtS = 1.0;
constexpr std::string_view kAgentAssemblyId{"demo_rover_v0"};
constexpr std::string_view kFixedDemoEpisodeProfile{"fixed_demo_v0"};
constexpr std::string_view kLightSearch1dEpisodeProfile{
    "light_search_1d_v1"};
constexpr std::string_view kBodyDiscoveryEpisodeProfile{
    "body_discovery_1d_v1"};
constexpr std::array<std::string_view, 3> kBodyAssemblyIds{
    "transmission_normal_v1", "transmission_reversed_v1",
    "transmission_disconnected_v1"};
constexpr std::string_view kBodyShaftJoint{"body_discovery_shaft_joint"};
constexpr std::string_view kBodyTransmission{"body_discovery_transmission"};
constexpr double kBodyLightSamplePeriodS = 0.1;
constexpr double kBodyLightLatencyS = 0.04;
constexpr std::string_view kLightSensorSite{"light_sensor_site"};
constexpr std::uint64_t kLightSearchSeedXorTag{0x4c49474854563100ULL};

constexpr std::array<std::string_view, kMotorCount> kWheelJoints{
    "wheel_front_left_joint",
    "wheel_front_right_joint",
    "wheel_rear_left_joint",
    "wheel_rear_right_joint",
};

constexpr std::array<std::string_view, kMotorCount> kWheelActuators{
    "drive_front_left",
    "drive_front_right",
    "drive_rear_left",
    "drive_rear_right",
};

constexpr std::array<std::string_view, kMotorCount> kMotorInstanceIds{
    "motor-0001",
    "motor-0002",
    "motor-0003",
    "motor-0004",
};

constexpr std::array<std::string_view, kMotorCount> kEncoderPositionSensors{
    "encoder_front_left_position",
    "encoder_front_right_position",
    "encoder_rear_left_position",
    "encoder_rear_right_position",
};

constexpr std::array<std::string_view, kMotorCount> kEncoderVelocitySensors{
    "encoder_front_left_velocity",
    "encoder_front_right_velocity",
    "encoder_rear_left_velocity",
    "encoder_rear_right_velocity",
};

constexpr std::array<std::string_view, kTouchSensorCount> kTouchSensors{
    "touch_front_force",
    "touch_rear_force",
};

struct ModuleDefinition {
    std::string_view name;
    std::string_view kind;
};

constexpr std::array<ModuleDefinition, 7> kModules{{
    {"droid", "chassis"},
    {"module_power", "power"},
    {"module_sensor", "sensor"},
    {"module_front_left_wheel", "wheel"},
    {"module_front_right_wheel", "wheel"},
    {"module_rear_left_wheel", "wheel"},
    {"module_rear_right_wheel", "wheel"},
}};

struct ModelDeleter {
    void operator()(mjModel* model) const noexcept {
        if (model != nullptr) {
            mj_deleteModel(model);
        }
    }
};

struct DataDeleter {
    void operator()(mjData* data) const noexcept {
        if (data != nullptr) {
            mj_deleteData(data);
        }
    }
};

using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

[[nodiscard]] double round_to(double value, int digits = 6) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("simulation produced a non-finite value");
    }
    const double scale = std::pow(10.0, digits);
    const double rounded = std::round(value * scale) / scale;
    return rounded == 0.0 ? 0.0 : rounded;
}

template <typename Range>
    requires(!std::is_pointer_v<std::remove_reference_t<Range>>)
[[nodiscard]] Json rounded_values(const Range& values, int digits = 6) {
    Json result = Json::array();
    for (const auto value : values) {
        result.push_back(round_to(static_cast<double>(value), digits));
    }
    return result;
}

[[nodiscard]] Json rounded_values(
    const mjtNum* values,
    std::size_t count,
    int digits = 6) {
    Json result = Json::array();
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(round_to(static_cast<double>(values[index]), digits));
    }
    return result;
}

[[nodiscard]] double yaw_from_quaternion(const mjtNum* quaternion) {
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    return std::atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z));
}

[[nodiscard]] double tilt_from_quaternion(const mjtNum* quaternion) {
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double body_up_dot_world_up =
        std::clamp(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0);
    constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
    return std::acos(body_up_dot_world_up) * kRadiansToDegrees;
}

[[nodiscard]] double precise_sum(std::span<const double> values) {
    long double total = 0.0L;
    for (const double value : values) {
        total += static_cast<long double>(value);
    }
    return static_cast<double>(total);
}

[[nodiscard]] std::uint64_t splitmix64_finalizer(
    std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

[[nodiscard]] double light_source_x_for_episode(
    std::string_view episode_profile,
    std::uint64_t seed) {
    if (episode_profile == kFixedDemoEpisodeProfile) {
        return kLightSourceXM;
    }
    if (episode_profile == kLightSearch1dEpisodeProfile ||
        episode_profile == kBodyDiscoveryEpisodeProfile) {
        return light_search_1d_realization(seed).source_x_m;
    }
    throw std::invalid_argument(
        "unsupported episode_profile '" + std::string(episode_profile) +
        "'; expected fixed_demo_v0, light_search_1d_v1, or body_discovery_1d_v1");
}

[[nodiscard]] std::string runtime_version() {
    const char* version = mj_versionString();
    return version == nullptr ? "unknown" : std::string(version);
}

}  // namespace

LightSearch1dRealization light_search_1d_realization(
    std::uint64_t seed) noexcept {
    const std::uint64_t mixed =
        splitmix64_finalizer(seed ^ kLightSearchSeedXorTag);
    const bool positive_side = (mixed >> 63U) != 0U;
    const auto distance_bucket = static_cast<std::uint32_t>(mixed) &
        kLightSearchDistanceBucketMask;
    const double distance_m = kLightSearchMinimumDistanceM +
        static_cast<double>(distance_bucket) * kLightSearchDistanceStepM;
    return LightSearch1dRealization{
        .positive_side = positive_side,
        .distance_bucket = distance_bucket,
        .source_x_m = positive_side ? distance_m : -distance_m,
    };
}

std::uint32_t light_search_1d_realization_key(
    std::uint64_t seed) noexcept {
    const LightSearch1dRealization realization =
        light_search_1d_realization(seed);
    return realization.distance_bucket |
        (realization.positive_side ? kLightSearchPositiveSideKeyBit : 0U);
}

struct DroidSimulation::Impl {
    struct ModuleRef {
        int body_id{-1};
        std::string name;
        std::string kind;
    };

    explicit Impl(const std::filesystem::path& model_path, bool realtime)
        : model_path_(model_path), catalog_(catalog_json()),
          worker_enabled_(realtime), running_(realtime) {
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error(
                "MuJoCo model not found: " + model_path.string());
        }

        std::array<char, 2048> error{};
        model_.reset(mj_loadXML(
            model_path.string().c_str(), nullptr, error.data(), error.size()));
        if (!model_) {
            throw std::runtime_error(
                "unable to load MuJoCo model " + model_path.string() + ": " +
                std::string(error.data()));
        }
        data_.reset(mj_makeData(model_.get()));
        if (!data_) {
            throw std::runtime_error("unable to allocate MuJoCo simulation data");
        }
        physics_timestep_s_ = model_->opt.timestep;

        const Json& physics = motor_physics();
        reference_ambient_c_ = physics.at("reference_ambient_c").get<double>();
        nominal_bus_voltage_v_ =
            physics.at("nominal_bus_voltage_v").get<double>();
        bus_voltage_v_ = nominal_bus_voltage_v_;
        reset_motor_states();

        const double control_period_s = catalog_
            .at("motor_skus")
            .at(std::string(kReferenceMotorSku))
            .at("action")
            .at("control_period_s")
            .get<double>();
        motor_control_interval_steps_ = std::max<std::uint64_t>(
            1,
            static_cast<std::uint64_t>(
                std::llround(control_period_s / model_->opt.timestep)));

        resolve_model_references();

        mj_forward(model_.get(), data_.get());
        previous_xy_ = {
            static_cast<double>(body_position(robot_body_id_)[0]),
            static_cast<double>(body_position(robot_body_id_)[1]),
        };

        if (worker_enabled_) {
            worker_ = std::thread([this] { simulation_loop(); });
        }
    }

    void resolve_model_references() {
        robot_body_id_ = object_id(mjOBJ_BODY, "droid", "body");
        light_sensor_site_id_ =
            object_id(mjOBJ_SITE, kLightSensorSite, "site");
        modules_.clear();
        modules_.reserve(kModules.size());
        for (const ModuleDefinition& module : kModules) {
            modules_.push_back(ModuleRef{
                object_id(mjOBJ_BODY, module.name, "body"),
                std::string(module.name),
                std::string(module.kind),
            });
        }

        for (std::size_t index = 0; index < kMotorCount; ++index) {
            const int joint_id =
                object_id(mjOBJ_JOINT, kWheelJoints[index], "joint");
            wheel_qpos_addresses_[index] = model_->jnt_qposadr[joint_id];
            wheel_qvel_addresses_[index] = model_->jnt_dofadr[joint_id];
            shaft_qvel_addresses_[index] = wheel_qvel_addresses_[index];
            actuator_ids_[index] = object_id(
                mjOBJ_ACTUATOR, kWheelActuators[index], "actuator");
            encoder_position_ids_[index] = object_id(
                mjOBJ_SENSOR, kEncoderPositionSensors[index], "sensor");
            encoder_velocity_ids_[index] = object_id(
                mjOBJ_SENSOR, kEncoderVelocitySensors[index], "sensor");
        }
        for (std::size_t index = 0; index < kTouchSensorCount; ++index) {
            touch_sensor_ids_[index] =
                object_id(mjOBJ_SENSOR, kTouchSensors[index], "sensor");
        }

        if (body_model_active_) {
            const int shaft_joint = object_id(
                mjOBJ_JOINT, kBodyShaftJoint, "joint");
            shaft_qvel_addresses_[0] = model_->jnt_dofadr[shaft_joint];
            body_shaft_qpos_address_ = model_->jnt_qposadr[shaft_joint];
            body_transmission_id_ = object_id(
                mjOBJ_EQUALITY, kBodyTransmission, "equality");
        }
    }

    ~Impl() {
        stop_requested_.store(true, std::memory_order_release);
        wake_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    [[nodiscard]] const Json& motor_physics() const {
        return catalog_
            .at("motor_skus")
            .at(std::string(kReferenceMotorSku))
            .at("physics");
    }

    // This generated successor has a real output-shaft degree of freedom.
    // The actuator and encoders always remain on that shaft. A signed joint
    // equality transmits torque to the wheel; releasing it leaves the shaft
    // and wheel mechanically independent (apart from their shared chassis).
    [[nodiscard]] ModelPtr make_body_model() const {
        for (const double interval : {kBodyLightSamplePeriodS, kBodyLightLatencyS}) {
            const double steps = std::round(interval / model_->opt.timestep);
            if (steps < 1.0 ||
                std::abs(steps * model_->opt.timestep - interval) > 1e-12) {
                throw std::invalid_argument(
                    "body_discovery_1d_v1 requires light period and latency to "
                    "be integral physics steps");
            }
        }
        std::array<char, 2048> error{};
        const auto delete_spec = [](mjSpec* spec) { mj_deleteSpec(spec); };
        std::unique_ptr<mjSpec, decltype(delete_spec)> spec(
            mj_parseXML(model_path_.string().c_str(), nullptr,
                        error.data(), error.size()), delete_spec);
        if (!spec) {
            throw std::runtime_error("body model parse failed: " +
                                     std::string(error.data()));
        }
        mjsBody* parent = mjs_findBody(spec.get(), "droid");
        mjsActuator* actuator = mjs_asActuator(mjs_findElement(
            spec.get(), mjOBJ_ACTUATOR, "drive_front_left"));
        mjsSensor* position = mjs_asSensor(mjs_findElement(
            spec.get(), mjOBJ_SENSOR, "encoder_front_left_position"));
        mjsSensor* velocity = mjs_asSensor(mjs_findElement(
            spec.get(), mjOBJ_SENSOR, "encoder_front_left_velocity"));
        if (!parent || !actuator || !position || !velocity) {
            throw std::runtime_error("body model lacks required reference parts");
        }
        mjsBody* shaft = mjs_addBody(parent, nullptr);
        mjs_setName(shaft->element, "body_discovery_shaft");
        shaft->pos[0] = 0.22;
        shaft->pos[1] = 0.225;
        shaft->pos[2] = -0.11;
        // Small casing-supported shaft mass; geared rotor inertia is reflected
        // separately at the output joint using the catalog's ratio squared.
        shaft->mass = 0.02;
        shaft->inertia[0] = shaft->inertia[1] = shaft->inertia[2] = 0.00001;
        shaft->explicitinertial = true;
        mjsJoint* joint = mjs_addJoint(shaft, nullptr);
        mjs_setName(joint->element, kBodyShaftJoint.data());
        joint->type = mjJNT_HINGE;
        joint->axis[0] = 0.0;
        joint->axis[1] = 1.0;
        joint->axis[2] = 0.0;
        joint->limited = mjLIMITED_FALSE;
        std::fill(std::begin(joint->damping), std::end(joint->damping), 0.0);
        const double ratio = motor_physics().at("gear_ratio_motor_to_output");
        joint->armature = motor_physics().at("rotor_inertia_kg_m2").get<double>() *
            ratio * ratio;
        mjs_setString(actuator->target, kBodyShaftJoint.data());
        mjs_setString(position->objname, kBodyShaftJoint.data());
        mjs_setString(velocity->objname, kBodyShaftJoint.data());
        mjsEquality* transmission = mjs_addEquality(spec.get(), nullptr);
        mjs_setName(transmission->element, kBodyTransmission.data());
        transmission->type = mjEQ_JOINT;
        transmission->objtype = mjOBJ_JOINT;
        mjs_setString(transmission->name1, kBodyShaftJoint.data());
        mjs_setString(transmission->name2, kWheelJoints[0].data());
        std::fill(std::begin(transmission->data), std::end(transmission->data), 0.0);
        transmission->data[1] = 1.0;
        transmission->active = true;
        transmission->solref[0] = 0.004;
        transmission->solref[1] = 1.0;
        transmission->solimp[0] = 0.999;
        transmission->solimp[1] = 0.999;
        ModelPtr result(mj_compile(spec.get(), nullptr));
        if (!result) {
            throw std::runtime_error("body model compile failed: " +
                                     std::string(mjs_getError(spec.get())));
        }
        if (result->opt.timestep != model_->opt.timestep) {
            throw std::runtime_error("body model must retain the physics timestep");
        }
        return result;
    }

    void select_model_locked(bool body) {
        if (body == body_model_active_) {
            return;
        }
        if (body) {
            ModelPtr successor = make_body_model();
            DataPtr successor_data(mj_makeData(successor.get()));
            if (!successor_data) {
                throw std::runtime_error("unable to allocate body simulation data");
            }
            legacy_model_ = std::move(model_);
            model_ = std::move(successor);
            data_ = std::move(successor_data);
        } else {
            DataPtr legacy_data(mj_makeData(legacy_model_.get()));
            if (!legacy_data) {
                throw std::runtime_error("unable to restore legacy simulation data");
            }
            model_ = std::move(legacy_model_);
            data_ = std::move(legacy_data);
        }
        body_model_active_ = body;
        resolve_model_references();
    }

    [[nodiscard]] Json body_experiment_spec() const {
        std::scoped_lock lock(mutex_);
        return Json{
            {"schema_version", "droid-blocks.body-experiment.v1"},
            {"episode_profile", kBodyDiscoveryEpisodeProfile},
            {"supported_assembly_ids", kBodyAssemblyIds},
            {"base_assembly_id", kAgentAssemblyId},
            {"generator_id", "mujoco_output_shaft_transmission_v1"},
            {"changed_motor_id", kMotorInstanceIds[0]},
            {"mechanics", "independent output shaft; signed compliant joint equality to wheel"},
            {"shaft_mass_kg", 0.02},
            {"shaft_body_diagonal_inertia_kg_m2", Json::array({0.00001, 0.00001, 0.00001})},
            {"reflected_rotor_inertia_kg_m2", 0.00375},
            {"transmission_constraint_solref", Json::array({0.004, 1.0})},
            {"transmission_constraint_solimp_bounds", Json::array({0.999, 0.999})},
            {"light_timing", Json{
                {"sample_period_s", kBodyLightSamplePeriodS},
                {"latency_s", kBodyLightLatencyS},
                {"first_sample_time_s", 0.0},
                {"valid_before_first_delivery", false},
                {"field_model", "legacy analytic radial light; no noise or occlusion"},
                {"reward_uses_delivered_sample", true},
                {"override_scope", "ambient light only; latency is experimental, not catalog nominal"}}},
            {"policy_visible_topology", false},
            {"exposure", "development_only"},
        };
    }

    [[nodiscard]] int object_id(
        mjtObj object_type,
        std::string_view name,
        std::string_view type_name) const {
        const std::string owned_name(name);
        const int id = mj_name2id(model_.get(), object_type, owned_name.c_str());
        if (id < 0) {
            throw std::runtime_error(
                "MuJoCo model is missing " + std::string(type_name) + " '" +
                owned_name + "'");
        }
        return id;
    }

    [[nodiscard]] const mjtNum* body_position(int body_id) const {
        return data_->xpos + 3 * body_id;
    }

    [[nodiscard]] const mjtNum* body_quaternion(int body_id) const {
        return data_->xquat + 4 * body_id;
    }

    [[nodiscard]] const mjtNum* site_position(int site_id) const {
        return data_->site_xpos + 3 * site_id;
    }

    [[nodiscard]] double sensor_scalar(int sensor_id) const {
        const int address = model_->sensor_adr[sensor_id];
        if (model_->sensor_dim[sensor_id] < 1) {
            throw std::runtime_error("MuJoCo scalar sensor has no data");
        }
        return round_to(static_cast<double>(data_->sensordata[address]));
    }

    void reset_motor_states() {
        for (MotorState& state : motor_states_) {
            state.current_a = 0.0;
            state.temperature_c = reference_ambient_c_;
        }
    }

    void update_demo_targets() {
        const mjtNum* position = body_position(robot_body_id_);
        const double x = position[0];
        const double y = position[1];
        const double yaw = yaw_from_quaternion(body_quaternion(robot_body_id_));
        const double half_track = kTrackLength * 0.5;
        if (x > half_track - 2.0) {
            direction_ = -1.0;
        } else if (x < -half_track + 2.0) {
            direction_ = 1.0;
        }

        const double cruise =
            direction_ * (6.0 + 0.35 * std::sin(data_->time * 0.7));
        const double desired_yaw =
            direction_ > 0.0 ? 0.0 : 3.14159265358979323846;
        const double yaw_error = std::atan2(
            std::sin(desired_yaw - yaw), std::cos(desired_yaw - yaw));
        const double steering =
            std::clamp(0.8 * yaw_error - 0.25 * y, -1.1, 1.1);
        const double left = cruise - steering;
        const double right = cruise + steering;
        targets_ = {left, right, left, right};
    }

    void apply_actuation() {
        const bool update_commands = !external_agent_mode_ &&
            (demo_command_refresh_ ||
             tick_ % motor_control_interval_steps_ == 0);
        if (update_commands) {
            update_demo_targets();
            demo_command_refresh_ = false;
        }

        const double dt_s = model_->opt.timestep;
        const double energy_fraction = energy_j_ / kBatteryCapacityJ;
        bus_voltage_v_ =
            nominal_bus_voltage_v_ * (0.8 + 0.2 * energy_fraction);

        motor_outputs_.clear();
        motor_feedback_.clear();
        motor_outputs_.reserve(kMotorCount);
        motor_feedback_.reserve(kMotorCount);

        for (std::size_t index = 0; index < kMotorCount; ++index) {
            const double wheel_speed =
                data_->qvel[shaft_qvel_addresses_[index]];
            if (!external_agent_mode_ && update_commands) {
                motor_efforts_[index] = clamp(
                    (targets_[index] - wheel_speed) / 3.0, -1.0, 1.0);
            }

            double effort = motor_efforts_[index];
            if (safety_veto_) {
                effort = 0.0;
            }
            MotorStepResult output = step_motor_current_thermal(
                motor_states_[index],
                effort,
                wheel_speed,
                bus_voltage_v_,
                dt_s,
                std::nullopt,
                Json::object(),
                catalog_);
            // Keep the raw external request stable for the whole control
            // transition. The motor model owns clamping and safety vetoes.
            if (!external_agent_mode_) {
                motor_efforts_[index] = effort;
            }
            motor_states_[index] = output.state;

            ActuatorFeedbackInput feedback_input;
            feedback_input.position_rad =
                sensor_scalar(encoder_position_ids_[index]);
            feedback_input.velocity_rad_s = wheel_speed;
            feedback_input.current_a = output.state.current_a;
            feedback_input.bus_voltage_v = bus_voltage_v_;
            feedback_input.temperature_c = output.state.temperature_c;
            feedback_input.output_torque_nm = output.output_torque_nm;
            feedback_input.commanded_effort = effort;
            feedback_input.previous_stuck_score = stuck_scores_[index];
            feedback_input.dt_s = dt_s;
            feedback_input.fault_flags = output.safety.flags;
            ActuatorFeedback feedback = actuator_feedback_sample(
                feedback_input, Json::object(), catalog_);
            stuck_scores_[index] = feedback.stuck_score;

            data_->ctrl[actuator_ids_[index]] =
                clamp(output.output_torque_nm, -4.0, 4.0);
            motor_outputs_.push_back(std::move(output));
            motor_feedback_.push_back(std::move(feedback));
        }

        std::array<double, kMotorCount> powers{};
        for (std::size_t index = 0; index < kMotorCount; ++index) {
            powers[index] = motor_outputs_[index].electrical_power_w;
        }
        const double total_electrical_power_w = precise_sum(powers);
        bus_current_a_ = bus_voltage_v_ > 0.0
            ? total_electrical_power_w / bus_voltage_v_
            : 0.0;
        energy_j_ = clamp(
            energy_j_ - total_electrical_power_w * dt_s,
            0.0,
            kBatteryCapacityJ);
    }

    [[nodiscard]] Json reward_component(
        std::string_view module_id,
        std::string_view family_id,
        const Json& observations,
        std::vector<std::string> safety_flags = {}) const {
        const double rate = round_to(
            sensor_reward_rate(
                family_id, observations, Json::object(), catalog_),
            6);
        return Json{
            {"module_id", module_id},
            {"family_id", family_id},
            {"valid", true},
            {"observations", observations},
            {"reward_rate", rate},
            {"safety_flags", std::move(safety_flags)},
        };
    }

    [[nodiscard]] double ambient_light_lux() const {
        const mjtNum* sensor_position = site_position(light_sensor_site_id_);
        const double x_distance = sensor_position[0] - light_source_x_m_;
        const double y_distance = sensor_position[1] - kLightSourceYM;
        const double radial_distance_squared =
            x_distance * x_distance + y_distance * y_distance;
        return 1000.0 / (1.0 + 0.04 * radial_distance_squared);
    }

    struct LightSample {
        std::uint64_t sequence{0};
        std::uint64_t sampled_tick{0};
        std::uint64_t delivery_tick{0};
        double lux{0.0};
    };

    void initialize_body_light() {
        pending_light_.clear();
        delivered_light_ = {};
        next_light_sequence_ = 1;
        if (body_model_active_) {
            light_sample_steps_ = static_cast<std::uint64_t>(
                std::llround(kBodyLightSamplePeriodS / model_->opt.timestep));
            light_latency_steps_ = static_cast<std::uint64_t>(
                std::llround(kBodyLightLatencyS / model_->opt.timestep));
            pending_light_.push_back(LightSample{
                next_light_sequence_++, 0, light_latency_steps_, ambient_light_lux()});
        }
    }

    void update_body_light(std::uint64_t current_tick) {
        if (!body_model_active_) {
            return;
        }
        if (current_tick % light_sample_steps_ == 0) {
            pending_light_.push_back(LightSample{
                next_light_sequence_++, current_tick,
                current_tick + light_latency_steps_, ambient_light_lux()});
        }
        while (!pending_light_.empty() &&
               pending_light_.front().delivery_tick <= current_tick) {
            delivered_light_ = pending_light_.front();
            pending_light_.pop_front();
        }
    }

    [[nodiscard]] double observed_light_lux() const {
        return body_model_active_ ? delivered_light_.lux : ambient_light_lux();
    }

    [[nodiscard]] Json light_timing(std::uint64_t current_tick) const {
        const bool valid = delivered_light_.sequence != 0;
        const auto time = [this](std::uint64_t tick) {
            return round_to(static_cast<double>(tick) * model_->opt.timestep);
        };
        return Json{
            {"sequence", delivered_light_.sequence},
            {"sample_time_s", valid ? Json(time(delivered_light_.sampled_tick)) : Json(nullptr)},
            {"delivered_time_s", valid ? Json(time(delivered_light_.delivery_tick)) : Json(nullptr)},
            {"age_s", valid ? Json(time(current_tick - delivered_light_.sampled_tick)) : Json(nullptr)},
            {"sample_period_s", kBodyLightSamplePeriodS},
            {"latency_s", kBodyLightLatencyS},
        };
    }

    void update_rewards(double dt_s) {
        std::vector<Json> components;
        components.reserve(3);
        std::vector<std::string> hard_flags;

        for (std::size_t index = 0; index < kMotorCount; ++index) {
            const MotorStepResult& output = motor_outputs_.at(index);
            if (!output.safety.veto) {
                continue;
            }
            for (const std::string& flag : output.safety.flags) {
                hard_flags.push_back(
                    std::string(kMotorInstanceIds[index]) + ":" + flag);
            }
        }

        const double light_lux = observed_light_lux();
        components.push_back(reward_component(
            "sensor-0001",
            "ambient_light_v0",
            Json{
                {"illuminance_lux", light_lux},
                {"saturated", light_lux >= 1000.0},
            }));
        if (body_model_active_) {
            components.back().update(light_timing(tick_ + 1));
            components.back()["valid"] = delivered_light_.sequence != 0;
            if (delivered_light_.sequence == 0) {
                components.back()["reward_rate"] = catalog_.at("sensor_families")
                    .at("ambient_light_v0").at("missing_reward");
            }
        }

        const double touch_hard_n = catalog_
            .at("sensor_families")
            .at("touch_force_v0")
            .at("reward")
            .at("parameters")
            .at("force_hard_n")
            .get<double>();
        for (std::size_t index = 0; index < kTouchSensorCount; ++index) {
            const double force_n =
                std::max(0.0, sensor_scalar(touch_sensor_ids_[index]));
            std::vector<std::string> touch_flags;
            if (force_n >= touch_hard_n) {
                touch_flags.emplace_back("force_limit");
                hard_flags.push_back(
                    "sensor-000" + std::to_string(index + 2) +
                    ":force_limit");
            }
            components.push_back(reward_component(
                "sensor-000" + std::to_string(index + 2),
                "touch_force_v0",
                Json{
                    {"contact", force_n > 0.0},
                    {"normal_force_n", force_n},
                },
                std::move(touch_flags)));
        }

        std::vector<RegisteredSensorSample> samples;
        samples.reserve(components.size());
        std::vector<double> rates;
        rates.reserve(components.size());
        for (const Json& component : components) {
            const double rate = component.at("reward_rate").get<double>();
            samples.push_back(RegisteredSensorSample{
                component.at("module_id").get<std::string>(),
                component.at("family_id").get<std::string>(),
                rate,
                component.at("valid").get<bool>(),
                false,
            });
            rates.push_back(rate);
        }

        reward_components_ = std::move(components);
        reward_rate_ = precise_sum(rates);
        transition_reward_ =
            transition_reward(dt_s, samples, catalog_);
        cumulative_reward_ += transition_reward_;

        std::sort(hard_flags.begin(), hard_flags.end());
        hard_flags.erase(
            std::unique(hard_flags.begin(), hard_flags.end()),
            hard_flags.end());
        safety_flags_ = std::move(hard_flags);
        safety_veto_ = !safety_flags_.empty();
    }

    void step_once() {
        const mjtNum* before = body_position(robot_body_id_);
        const std::array<double, 2> before_xy{before[0], before[1]};
        apply_actuation();
        mj_step(model_.get(), data_.get());
        if (body_model_active_) {
            // Sample the actual post-transition configuration, including the
            // mounted site. The legacy pipeline intentionally stays untouched.
            mj_forward(model_.get(), data_.get());
            update_body_light(tick_ + 1);
        }
        update_rewards(model_->opt.timestep);
        const mjtNum* after = body_position(robot_body_id_);
        const std::array<double, 2> after_xy{after[0], after[1]};
        distance_ += std::hypot(
            after_xy[0] - before_xy[0], after_xy[1] - before_xy[1]);
        previous_xy_ = after_xy;
        ++tick_;
    }

    void simulation_loop() noexcept {
        try {
            const auto batch_period = std::chrono::duration<double>(
                physics_timestep_s_ *
                static_cast<double>(kPhysicsStepsPerBatch));
            Clock::time_point deadline = Clock::now();

            while (!stop_requested_.load(std::memory_order_acquire)) {
                std::unique_lock lock(mutex_);
                const bool active = running_;
                if (active) {
                    for (std::size_t step = 0;
                         step < kPhysicsStepsPerBatch;
                         ++step) {
                        step_once();
                    }
                }

                if (!active) {
                    wake_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                        return stop_requested_.load(std::memory_order_acquire) ||
                            running_;
                    });
                    deadline = Clock::now();
                    continue;
                }

                deadline += std::chrono::duration_cast<Clock::duration>(
                    batch_period);
                wake_.wait_until(lock, deadline, [this] {
                    return stop_requested_.load(std::memory_order_acquire) ||
                        !running_;
                });
                const auto lag = Clock::now() - deadline;
                if (lag > std::chrono::duration_cast<Clock::duration>(
                              4.0 * batch_period)) {
                    deadline = Clock::now();
                }
            }
        } catch (const std::exception& error) {
            std::scoped_lock lock(mutex_);
            running_ = false;
            last_error_ = error.what();
        } catch (...) {
            std::scoped_lock lock(mutex_);
            running_ = false;
            last_error_ = "unknown simulation failure";
        }
    }

    [[nodiscard]] bool running() const {
        std::scoped_lock lock(mutex_);
        return running_;
    }

    void set_running(bool running) {
        std::scoped_lock lock(mutex_);
        if (!worker_enabled_ && running) {
            throw std::logic_error(
                "a deterministic simulation has no realtime worker");
        }
        if (running) {
            if (body_model_active_) {
                select_model_locked(false);
                reset_state_locked();
                agent_assembly_id_ = kAgentAssemblyId;
            }
            external_agent_mode_ = false;
            agent_terminated_ = false;
            demo_command_refresh_ = true;
            episode_profile_ = kFixedDemoEpisodeProfile;
            light_source_x_m_ = kLightSourceXM;
        }
        running_ = running;
        wake_.notify_all();
    }

    void reset_state_locked() {
        mj_resetData(model_.get(), data_.get());
        mj_forward(model_.get(), data_.get());
        tick_ = 0;
        distance_ = 0.0;
        direction_ = 1.0;
        targets_.fill(0.0);
        motor_efforts_.fill(0.0);
        reset_motor_states();
        motor_outputs_.clear();
        motor_feedback_.clear();
        stuck_scores_.fill(0.0);
        energy_j_ = kBatteryCapacityJ;
        bus_voltage_v_ = nominal_bus_voltage_v_;
        bus_current_a_ = 0.0;
        reward_rate_ = 0.0;
        transition_reward_ = 0.0;
        cumulative_reward_ = 0.0;
        reward_components_.clear();
        safety_veto_ = false;
        safety_flags_.clear();
        agent_terminated_ = false;
        episode_profile_ = kFixedDemoEpisodeProfile;
        light_source_x_m_ = kLightSourceXM;
        demo_command_refresh_ = !external_agent_mode_;
        last_error_.clear();
        pending_light_.clear();
        delivered_light_ = {};
        std::fill(data_->ctrl, data_->ctrl + model_->nu, 0.0);
        const mjtNum* position = body_position(robot_body_id_);
        previous_xy_ = {position[0], position[1]};
    }

    void reset() {
        std::scoped_lock lock(mutex_);
        const bool was_running = running_;
        if (body_model_active_) {
            select_model_locked(false);
            agent_assembly_id_ = kAgentAssemblyId;
        }
        reset_state_locked();
        running_ = was_running;
        wake_.notify_all();
    }

    void advance_steps(std::size_t count) {
        std::scoped_lock lock(mutex_);
        if (worker_enabled_ && running_) {
            throw std::logic_error(
                "synchronous stepping requires a paused realtime worker");
        }
        for (std::size_t step = 0; step < count; ++step) {
            step_once();
        }
    }

    [[nodiscard]] Json agent_spec() const {
        std::scoped_lock lock(mutex_);
        const Json& motor = catalog_
            .at("motor_skus")
            .at(std::string(kReferenceMotorSku));
        const Json& action = motor.at("action");

        Json sensor_topology = Json::array();
        const auto append_sensor = [this, &sensor_topology](
                                       std::string_view module_id,
                                       std::string_view family_id) {
            sensor_topology.push_back(Json{
                {"module_id", std::string(module_id)},
                {"family_id", std::string(family_id)},
                {"channels",
                 catalog_.at("sensor_families")
                     .at(std::string(family_id))
                     .at("channels")},
            });
        };
        append_sensor("sensor-0001", "ambient_light_v0");
        append_sensor("sensor-0002", "touch_force_v0");
        append_sensor("sensor-0003", "touch_force_v0");

        Json feedback_topology = Json::array();
        for (const std::string_view module_id : kMotorInstanceIds) {
            feedback_topology.push_back(Json{
                {"module_id", std::string(module_id)},
                {"sku_id", std::string(kReferenceMotorSku)},
                {"fields", motor.at("actuator_feedback").at("fields")},
            });
        }

        return Json{
            {"api_version", "agent_environment_v0"},
            {"engine",
             Json{
                 {"id", "mujoco"},
                 {"version", runtime_version()},
             }},
            {"assembly_id", std::string(kAgentAssemblyId)},
            {"supported_assembly_ids",
             Json::array({std::string(kAgentAssemblyId)})},
            {"default_episode_profile",
             std::string(kFixedDemoEpisodeProfile)},
            {"episode_profiles",
             Json::array({
                 Json{
                     {"id", std::string(kFixedDemoEpisodeProfile)},
                     {"description",
                      "canonical fixed world used by the visual demo and "
                      "legacy agent resets"},
                     {"seed_affects_realization", false},
                     {"realization_policy_visible", false},
                     {"light_source_x_m", kLightSourceXM},
                 },
                 Json{
                     {"id", std::string(kLightSearch1dEpisodeProfile)},
                      {"description",
                       "seeded one-dimensional search task with a hidden, "
                       "balanced source side and exact 24-bit distance"},
                     {"seed_affects_realization", true},
                     {"realization_policy_visible", false},
                      {"distribution",
                       Json{
                           {"parameter", "light_source_x_m"},
                           {"kind", "balanced_sign_uniform_24bit_distance_grid"},
                           {"side_support",
                            Json::array({"negative_x", "positive_x"})},
                           {"absolute_distance_m",
                            Json{
                                {"minimum_inclusive_m",
                                 kLightSearchMinimumDistanceM},
                                {"maximum_exclusive_m",
                                 kLightSearchMaximumDistanceM},
                                {"step_m", kLightSearchDistanceStepM},
                                {"bucket_count", 1U << 24U},
                            }},
                       }},
                     {"sensor_field",
                      Json{
                          {"sample_position", "mounted_sensor_site_world_xy"},
                          {"source_y_m", kLightSourceYM},
                          {"illuminance_lux",
                           "1000 / (1 + 0.04 * radial_distance_m^2)"},
                      }},
                     {"seed_mapping",
                      Json{
                           {"integer_mixer", "splitmix64_finalizer"},
                           {"input", "seed XOR 0x4c49474854563100"},
                           {"side_bit", 63},
                           {"side_bit_0", "negative_x"},
                           {"side_bit_1", "positive_x"},
                           {"distance_bucket_expression",
                            "mixed & 0x00ffffff"},
                           {"distance_formula_m", "12 + bucket * 2^-21"},
                           {"source_x_formula_m",
                            "(side_bit ? +1 : -1) * distance"},
                           {"realization_key_expression",
                            "(side_bit << 24) | bucket"},
                           {"realization_key_bits", 25},
                           {"integer_only", true},
                       }},
                 },
             })},
            {"physics_timestep_s", model_->opt.timestep},
            {"nominal_control_dt_s", action.at("control_period_s")},
            {"maximum_control_dt_s", kMaximumAgentControlDtS},
            {"action",
             Json{
                 {"name", action.at("name")},
                 {"motor_ids", kMotorInstanceIds},
                 {"ordering", kMotorInstanceIds},
                 {"minimum", action.at("minimum")},
                 {"maximum", action.at("maximum")},
                 {"unit", action.at("unit")},
                 {"validation",
                  "exactly one finite value per motor; unknown and missing "
                  "IDs are rejected; out-of-range values are clamped by "
                  "the motor driver"},
                 {"semantics",
                  "direct normalized winding-current effort; the privileged "
                  "demo velocity controller is bypassed"},
             }},
            {"timing",
             Json{
                 {"physics_timestep_s", model_->opt.timestep},
                 {"nominal_control_dt_s", action.at("control_period_s")},
                 {"maximum_control_dt_s", kMaximumAgentControlDtS},
                 {"control_dt_rule",
                  "positive integer multiple of physics_timestep_s"},
             }},
            {"observation",
             Json{
                 {"reward_sensors", std::move(sensor_topology)},
                 {"actuator_feedback", std::move(feedback_topology)},
                 {"collection_semantics",
                  "variable-length entries keyed by opaque module IDs; array "
                  "order has no physical meaning"},
                 {"excluded_privileged_data",
                  Json::array({
                      "world position and orientation",
                      "chassis velocity and travelled distance",
                      "global contact count",
                      "light-source position and distance",
                      "authored front/rear/left/right and part/task roles",
                      "demo-controller target and direction",
                  })},
             }},
            {"reward",
             Json{
                 {"aggregation",
                  "sum over physics transitions of "
                  "physics_timestep_s * sum(mounted sensor reward_rate)"},
                 {"voters",
                  Json::array({"sensor-0001", "sensor-0002", "sensor-0003"})},
                 {"actuator_feedback_votes", false},
             }},
            {"termination",
             Json{
                 {"terminated", "true after any hard safety veto"},
                 {"truncated", "always false; episode horizons are external"},
             }},
        };
    }

    [[nodiscard]] Json policy_observation_locked() const {
        Json sensor_samples = Json::array();
        const double light_lux = observed_light_lux();
        sensor_samples.push_back(Json{
            {"module_id", "sensor-0001"},
            {"family_id", "ambient_light_v0"},
            {"valid", true},
            {"observations",
             Json{
                 {"illuminance_lux", round_to(light_lux)},
                 {"saturated", light_lux >= 1000.0},
             }},
        });
        if (body_model_active_) {
            sensor_samples.back().update(light_timing(tick_));
            sensor_samples.back()["valid"] = delivered_light_.sequence != 0;
        }
        for (std::size_t index = 0; index < kTouchSensorCount; ++index) {
            const double force_n =
                std::max(0.0, sensor_scalar(touch_sensor_ids_[index]));
            sensor_samples.push_back(Json{
                {"module_id", "sensor-000" + std::to_string(index + 2)},
                {"family_id", "touch_force_v0"},
                {"valid", true},
                {"observations",
                 Json{
                     {"contact", force_n > 0.0},
                     {"normal_force_n", round_to(force_n)},
                 }},
            });
        }

        Json actuator_samples = Json::array();
        for (std::size_t index = 0; index < kMotorCount; ++index) {
            Json feedback;
            if (motor_feedback_.size() == kMotorCount) {
                feedback = motor_feedback_[index];
            } else {
                feedback = ActuatorFeedback{
                    .position_rad =
                        sensor_scalar(encoder_position_ids_[index]),
                    .velocity_rad_s =
                        sensor_scalar(encoder_velocity_ids_[index]),
                    .current_a = motor_states_[index].current_a,
                    .bus_voltage_v = bus_voltage_v_,
                    .temperature_c = motor_states_[index].temperature_c,
                    .output_torque_nm = 0.0,
                    .load_impedance_nm_s_per_rad = 0.0,
                    .stuck_score = stuck_scores_[index],
                    .stuck = false,
                    .fault_flags = {},
                };
            }
            actuator_samples.push_back(Json{
                {"module_id", std::string(kMotorInstanceIds[index])},
                {"sku_id", std::string(kReferenceMotorSku)},
                {"valid", true},
                {"feedback", std::move(feedback)},
            });
        }

        return Json{
            {"reward_sensors", std::move(sensor_samples)},
            {"actuator_feedback", std::move(actuator_samples)},
        };
    }

    [[nodiscard]] Json policy_observation() const {
        std::scoped_lock lock(mutex_);
        return policy_observation_locked();
    }

    [[nodiscard]] Json reset_agent(
        std::string_view assembly_id,
        std::uint64_t seed,
        std::string_view episode_profile) {
        const bool body = episode_profile == kBodyDiscoveryEpisodeProfile;
        const auto assembly = std::find(kBodyAssemblyIds.begin(),
                                        kBodyAssemblyIds.end(), assembly_id);
        if ((body && assembly == kBodyAssemblyIds.end()) ||
            (!body && assembly_id != kAgentAssemblyId)) {
            throw std::invalid_argument(
                "unsupported assembly_id '" + std::string(assembly_id) +
                "' for episode_profile '" + std::string(episode_profile) + "'");
        }
        const double episode_light_source_x_m =
            light_source_x_for_episode(episode_profile, seed);

        std::scoped_lock lock(mutex_);
        select_model_locked(body);
        running_ = false;
        external_agent_mode_ = true;
        agent_assembly_id_ = assembly_id;
        agent_seed_ = seed;
        reset_state_locked();
        episode_profile_ = episode_profile;
        light_source_x_m_ = episode_light_source_x_m;
        if (body) {
            const bool connected = assembly_id != kBodyAssemblyIds[2];
            const double sign = assembly_id == kBodyAssemblyIds[1] ? -1.0 : 1.0;
            model_->eq_data[body_transmission_id_ * mjNEQDATA + 1] = sign;
            data_->eq_active[body_transmission_id_] = connected;
            mj_forward(model_.get(), data_.get());
        }
        initialize_body_light();
        wake_.notify_all();
        return Json{
            {"observation", policy_observation_locked()},
            {"info",
             Json{
                 {"assembly_id", agent_assembly_id_},
                 {"seed", agent_seed_},
                 {"episode_profile", episode_profile_},
                 {"mode", "external_effort_v0"},
                 {"deterministic", true},
             }},
        };
    }

    [[nodiscard]] Json step_agent(
        const std::map<std::string, double>& actions,
        double control_dt_s) {
        if (!std::isfinite(control_dt_s) || control_dt_s <= 0.0) {
            throw std::invalid_argument(
                "control_dt_s must be finite and greater than zero");
        }
        if (control_dt_s > kMaximumAgentControlDtS) {
            throw std::invalid_argument(
                "control_dt_s exceeds maximum_control_dt_s");
        }

        std::array<double, kMotorCount> ordered_actions{};
        std::vector<std::string> clamped_action_ids;
        for (const auto& [module_id, value] : actions) {
            const auto known = std::find(
                kMotorInstanceIds.begin(),
                kMotorInstanceIds.end(),
                std::string_view(module_id));
            if (known == kMotorInstanceIds.end()) {
                throw std::invalid_argument(
                    "unknown motor action ID: " + module_id);
            }
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "motor action must be finite: " + module_id);
            }
        }
        for (std::size_t index = 0; index < kMotorCount; ++index) {
            const std::string module_id(kMotorInstanceIds[index]);
            const auto action = actions.find(module_id);
            if (action == actions.end()) {
                throw std::invalid_argument(
                    "missing motor action ID: " + module_id);
            }
            ordered_actions[index] = action->second;
            if (action->second < -1.0 || action->second > 1.0) {
                clamped_action_ids.push_back(module_id);
            }
        }
        if (actions.size() != kMotorCount) {
            throw std::invalid_argument(
                "action map must contain exactly four motor IDs");
        }

        std::scoped_lock lock(mutex_);
        if (!external_agent_mode_) {
            throw std::logic_error(
                "reset_agent must be called before synchronous agent stepping");
        }
        if (running_) {
            throw std::logic_error(
                "synchronous agent stepping requires a paused realtime worker");
        }
        if (agent_terminated_) {
            throw std::logic_error(
                "episode is terminated; call reset_agent before stepping again");
        }

        const double physics_dt_s = model_->opt.timestep;
        const double requested_steps = control_dt_s / physics_dt_s;
        const auto physics_steps =
            static_cast<std::int64_t>(std::llround(requested_steps));
        const double represented_dt_s =
            static_cast<double>(physics_steps) * physics_dt_s;
        const double timing_tolerance = std::max(
            1e-12, std::abs(control_dt_s) * 1e-10);
        if (physics_steps < 1 ||
            std::abs(control_dt_s - represented_dt_s) > timing_tolerance) {
            throw std::invalid_argument(
                "control_dt_s must be an integer multiple of the physics "
                "timestep");
        }

        motor_efforts_ = ordered_actions;
        const double cumulative_before = cumulative_reward_;
        std::map<std::string, double> component_rewards;
        std::map<std::string, std::vector<std::string>> actuator_flags;
        std::vector<std::string> hard_flags;
        std::size_t executed_steps = 0;
        bool terminated = false;

        for (std::int64_t step = 0; step < physics_steps; ++step) {
            step_once();
            ++executed_steps;

            for (const Json& component : reward_components_) {
                component_rewards[component.at("module_id").get<std::string>()] +=
                    physics_dt_s * component.at("reward_rate").get<double>();
            }
            for (std::size_t index = 0;
                 index < motor_outputs_.size();
                 ++index) {
                auto& observed = actuator_flags[
                    std::string(kMotorInstanceIds[index])];
                observed.insert(
                    observed.end(),
                    motor_outputs_[index].safety.flags.begin(),
                    motor_outputs_[index].safety.flags.end());
            }
            hard_flags.insert(
                hard_flags.end(), safety_flags_.begin(), safety_flags_.end());
            if (safety_veto_) {
                terminated = true;
                break;
            }
        }

        std::sort(hard_flags.begin(), hard_flags.end());
        hard_flags.erase(
            std::unique(hard_flags.begin(), hard_flags.end()), hard_flags.end());
        for (auto& [unused, flags] : actuator_flags) {
            (void)unused;
            std::sort(flags.begin(), flags.end());
            flags.erase(std::unique(flags.begin(), flags.end()), flags.end());
        }

        const double reward = cumulative_reward_ - cumulative_before;
        Json integrated_components = reward_components_;
        std::vector<double> component_values;
        component_values.reserve(integrated_components.size());
        for (Json& component : integrated_components) {
            const std::string module_id =
                component.at("module_id").get<std::string>();
            const double value = component_rewards[module_id];
            component["transition_reward"] = value;
            component_values.push_back(value);
        }
        if (!integrated_components.empty()) {
            const double correction = reward - precise_sum(component_values);
            Json& final_component = integrated_components.back();
            const double corrected =
                final_component.at("transition_reward").get<double>() +
                correction;
            final_component["transition_reward"] = corrected;
        }

        Json actuator_safety = Json::array();
        for (const std::string_view module_id : kMotorInstanceIds) {
            actuator_safety.push_back(Json{
                {"module_id", std::string(module_id)},
                {"flags", actuator_flags[std::string(module_id)]},
            });
        }

        agent_terminated_ = terminated;
        return Json{
            {"observation", policy_observation_locked()},
            {"reward", reward},
            {"reward_components", std::move(integrated_components)},
            {"safety",
             Json{
                 {"veto", terminated},
                 {"flags", std::move(hard_flags)},
                 {"actuator_flags", std::move(actuator_safety)},
             }},
            {"terminated", terminated},
            {"truncated", false},
            {"info",
             Json{
                 {"assembly_id", agent_assembly_id_},
                 {"seed", agent_seed_},
                 {"episode_profile", episode_profile_},
                 {"mode", "external_effort_v0"},
                 {"requested_control_dt_s", control_dt_s},
                 {"elapsed_control_dt_s",
                  static_cast<double>(executed_steps) * physics_dt_s},
                 {"physics_steps_requested", physics_steps},
                 {"physics_steps_executed", executed_steps},
                 {"clamped_action_ids", std::move(clamped_action_ids)},
             }},
        };
    }

    [[nodiscard]] Json state() const {
        std::scoped_lock lock(mutex_);
        const mjtNum* position = body_position(robot_body_id_);
        const mjtNum* quaternion = body_quaternion(robot_body_id_);
        const double horizontal_speed =
            std::hypot(data_->qvel[0], data_->qvel[1]);

        std::array<double, kMotorCount> wheel_angles{};
        std::array<double, kMotorCount> motor_torques{};
        std::array<double, kMotorCount> motor_powers{};
        std::array<double, kMotorCount> motor_currents{};
        std::array<double, kMotorCount> motor_temperatures{};
        std::array<double, kMotorCount> encoder_positions{};
        std::array<double, kMotorCount> encoder_velocities{};
        for (std::size_t index = 0; index < kMotorCount; ++index) {
            const double angle = data_->qpos[wheel_qpos_addresses_[index]];
            wheel_angles[index] = std::atan2(std::sin(angle), std::cos(angle));
            motor_currents[index] = motor_states_[index].current_a;
            motor_temperatures[index] = motor_states_[index].temperature_c;
            encoder_positions[index] =
                sensor_scalar(encoder_position_ids_[index]);
            encoder_velocities[index] =
                sensor_scalar(encoder_velocity_ids_[index]);
            if (motor_outputs_.size() == kMotorCount) {
                motor_torques[index] = motor_outputs_[index].output_torque_nm;
                motor_powers[index] = motor_outputs_[index].electrical_power_w;
            }
        }

        std::array<double, kTouchSensorCount> touch_forces{};
        for (std::size_t index = 0; index < kTouchSensorCount; ++index) {
            touch_forces[index] =
                std::max(0.0, sensor_scalar(touch_sensor_ids_[index]));
        }

        Json modules = Json::array();
        for (const ModuleRef& module : modules_) {
            modules.push_back(Json{
                {"name", module.name},
                {"type", module.kind},
                {"position", rounded_values(body_position(module.body_id), 3)},
                {"quaternion", rounded_values(body_quaternion(module.body_id), 4)},
            });
        }

        Json feedback_json = Json::array();
        Json impedance = Json::array();
        Json stuck_score = Json::array();
        Json stuck = Json::array();
        for (std::size_t index = 0; index < motor_feedback_.size(); ++index) {
            Json feedback = motor_feedback_[index];
            feedback["module_id"] = kMotorInstanceIds[index];
            feedback["reward_vote"] = false;
            feedback_json.push_back(std::move(feedback));
            impedance.push_back(round_to(
                motor_feedback_[index].load_impedance_nm_s_per_rad));
            stuck_score.push_back(round_to(motor_feedback_[index].stuck_score));
            stuck.push_back(motor_feedback_[index].stuck);
        }

        const double tilt_deg = tilt_from_quaternion(quaternion);
        const double energy_fraction = energy_j_ / kBatteryCapacityJ;
        const double light_lux = observed_light_lux();

        Json result{
            {"engine", "MuJoCo " + runtime_version()},
            {"runtime", "native-cpp20"},
            {"module_schema", catalog_.at("schema_version")},
            {"running", running_},
            {"sim_time", round_to(data_->time)},
            {"tick", tick_},
            {"controller",
             external_agent_mode_
                 ? Json{
                       {"id", "external_effort_v0"},
                       {"kind", "policy_evaluation"},
                   }
                 : Json{
                       {"id", "demo_cruise_v0"},
                       {"kind", "privileged_visualization_only"},
                   }},
            {"robot",
             Json{
                 {"position", rounded_values(position, 3)},
                 {"quaternion", rounded_values(quaternion, 4)},
                 {"yaw", round_to(yaw_from_quaternion(quaternion))},
                 {"speed", round_to(horizontal_speed)},
                 {"distance", round_to(distance_)},
                 {"wheel_angles", rounded_values(wheel_angles)},
                 {"modules", std::move(modules)},
             }},
            {"sensors",
             Json{
                 {"touch_force_n", rounded_values(touch_forces)},
                 {"ambient_light_lux", round_to(light_lux)},
                 {"instances", reward_components_},
                 {"privileged_diagnostics",
                  Json{
                      {"tilt_deg", round_to(tilt_deg)},
                      {"world_contact_count", data_->ncon},
                  }},
                 {"tilt_deg", round_to(tilt_deg)},
                 {"contact_count", data_->ncon},
             }},
            {"power_system",
             Json{
                 {"policy_visible", false},
                 {"reward_vote", false},
                 {"bus_voltage_v", round_to(bus_voltage_v_)},
                 {"bus_current_a", round_to(bus_current_a_)},
                 {"energy_fraction", round_to(energy_fraction, 9)},
             }},
            {"actuators",
             Json{
                 {"model_id", kReferenceMotorSku},
                 {"module_ids", kMotorInstanceIds},
                 {"commands", rounded_values(motor_efforts_)},
                 {"targets", rounded_values(targets_)},
                 {"target_velocity_rad_s", rounded_values(targets_)},
                 {"effort", rounded_values(motor_torques)},
                 {"torque_nm", rounded_values(motor_torques)},
                 {"shaft_position_rad", rounded_values(encoder_positions)},
                 {"shaft_velocity_rad_s", rounded_values(encoder_velocities)},
                 {"current_a", rounded_values(motor_currents)},
                 {"temperature_c", rounded_values(motor_temperatures)},
                 {"electrical_power_w", rounded_values(motor_powers)},
                 {"feedback", std::move(feedback_json)},
                 {"feedback_role", "observation_and_safety_only"},
                 {"load_impedance_nm_s_per_rad", std::move(impedance)},
                 {"stuck_score", std::move(stuck_score)},
                 {"stuck", std::move(stuck)},
             }},
            {"reward",
             Json{
                 {"total_rate", round_to(reward_rate_)},
                 {"transition", round_to(transition_reward_, 9)},
                 {"cumulative", round_to(cumulative_reward_)},
                 {"components", reward_components_},
                 {"aggregation", "dt_s * sum(sensor.reward_rate)"},
             }},
            {"safety",
             Json{
                 {"veto", safety_veto_},
                 {"flags", safety_flags_},
             }},
            {"world",
             Json{
                 {"track_length", kTrackLength},
                 {"episode_profile", episode_profile_},
                 {"light_source_position",
                  Json::array({light_source_x_m_, kLightSourceYM, 0.0})},
             }},
        };
        if (body_model_active_) {
            const bool connected = data_->eq_active[body_transmission_id_];
            const double ratio = model_->eq_data[
                body_transmission_id_ * mjNEQDATA + 1];
            const double shaft_angle = data_->qpos[body_shaft_qpos_address_];
            const double wheel_angle = data_->qpos[wheel_qpos_addresses_[0]];
            result["body_experiment_physics"] = Json{
                {"schema_version", "droid-blocks.body-experiment.v1"},
                {"assembly_id", agent_assembly_id_},
                {"episode_profile", episode_profile_},
                {"changed_motor_id", kMotorInstanceIds[0]},
                {"connected", connected},
                {"shaft_to_wheel_ratio", connected ? Json(ratio) : Json(nullptr)},
                {"shaft_position_rad", round_to(shaft_angle)},
                {"shaft_velocity_rad_s", round_to(data_->qvel[shaft_qvel_addresses_[0]])},
                {"wheel_position_rad", round_to(wheel_angle)},
                {"wheel_velocity_rad_s", round_to(data_->qvel[wheel_qvel_addresses_[0]])},
                {"constraint_position_error_rad", connected ?
                    Json(round_to(shaft_angle - ratio * wheel_angle, 9)) : Json(nullptr)},
                {"instantaneous_illuminance_lux", round_to(ambient_light_lux())},
                {"light_timing", light_timing(tick_)},
                {"light_valid", delivered_light_.sequence != 0},
                {"policy_visible", false},
            };
        }
        return result;
    }

    [[nodiscard]] Json catalog() const {
        Json result = catalog_;
        Json sensor_instances = Json::array({
            Json{{"module_id", "sensor-0001"},
                 {"family_id", "ambient_light_v0"}},
            Json{{"module_id", "sensor-0002"},
                 {"family_id", "touch_force_v0"}},
            Json{{"module_id", "sensor-0003"},
                 {"family_id", "touch_force_v0"}},
        });

        Json motor_instances = Json::array();
        Json actuator_feedback_instances = Json::array();
        for (const std::string_view module_id : kMotorInstanceIds) {
            const Json instance{
                {"module_id", module_id},
                {"sku_id", kReferenceMotorSku},
            };
            motor_instances.push_back(instance);
            actuator_feedback_instances.push_back(instance);
        }

        std::vector<std::string> unmounted;
        for (const auto& [family_id, unused] :
             result.at("sensor_families").items()) {
            (void)unused;
            if (family_id != "ambient_light_v0" &&
                family_id != "touch_force_v0") {
                unmounted.push_back(family_id);
            }
        }
        std::sort(unmounted.begin(), unmounted.end());

        result["runtime_demo"] = Json{
            {"controller",
             Json{
                 {"id", "demo_cruise_v0"},
                 {"privileged", true},
                 {"purpose",
                  "keeps the visual showcase moving; not an RL policy"},
             }},
            {"motor_instances", std::move(motor_instances)},
            {"actuator_feedback_instances",
             std::move(actuator_feedback_instances)},
            {"sensor_instances", std::move(sensor_instances)},
            {"unmounted_sensor_families", std::move(unmounted)},
        };
        return result;
    }

    [[nodiscard]] Json health() const {
        std::scoped_lock lock(mutex_);
        Json result{
            {"status", last_error_.empty() ? "ok" : "error"},
            {"engine", "MuJoCo"},
            {"version", runtime_version()},
            {"runtime", "native-cpp20"},
            {"module_schema", catalog_.at("schema_version")},
            {"running", running_},
        };
        if (!last_error_.empty()) {
            result["error"] = last_error_;
        }
        return result;
    }

    std::filesystem::path model_path_;
    const Json& catalog_;
    ModelPtr model_;
    ModelPtr legacy_model_;
    DataPtr data_;
    double physics_timestep_s_{0.002};
    bool body_model_active_{false};
    int body_shaft_qpos_address_{-1};
    int body_transmission_id_{-1};
    std::deque<LightSample> pending_light_;
    LightSample delivered_light_;
    std::uint64_t next_light_sequence_{1};
    std::uint64_t light_sample_steps_{50};
    std::uint64_t light_latency_steps_{20};
    bool worker_enabled_{false};
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::atomic_bool stop_requested_{false};
    std::thread worker_;
    bool running_{false};
    std::string last_error_;
    bool external_agent_mode_{false};
    bool agent_terminated_{false};
    bool demo_command_refresh_{false};
    std::string agent_assembly_id_{kAgentAssemblyId};
    std::uint64_t agent_seed_{0};
    std::string episode_profile_{kFixedDemoEpisodeProfile};
    double light_source_x_m_{kLightSourceXM};

    int robot_body_id_{-1};
    int light_sensor_site_id_{-1};
    std::vector<ModuleRef> modules_;
    std::array<int, kMotorCount> wheel_qpos_addresses_{};
    std::array<int, kMotorCount> wheel_qvel_addresses_{};
    std::array<int, kMotorCount> shaft_qvel_addresses_{};
    std::array<int, kMotorCount> actuator_ids_{};
    std::array<int, kMotorCount> encoder_position_ids_{};
    std::array<int, kMotorCount> encoder_velocity_ids_{};
    std::array<int, kTouchSensorCount> touch_sensor_ids_{};

    std::uint64_t tick_{0};
    double distance_{0.0};
    double direction_{1.0};
    std::array<double, kMotorCount> targets_{};
    std::array<double, kMotorCount> motor_efforts_{};
    std::array<MotorState, kMotorCount> motor_states_{};
    std::vector<MotorStepResult> motor_outputs_;
    std::vector<ActuatorFeedback> motor_feedback_;
    std::array<double, kMotorCount> stuck_scores_{};
    double reference_ambient_c_{25.0};
    double nominal_bus_voltage_v_{7.4};
    double energy_j_{kBatteryCapacityJ};
    double bus_voltage_v_{7.4};
    double bus_current_a_{0.0};
    double reward_rate_{0.0};
    double transition_reward_{0.0};
    double cumulative_reward_{0.0};
    std::vector<Json> reward_components_;
    bool safety_veto_{false};
    std::vector<std::string> safety_flags_;
    std::uint64_t motor_control_interval_steps_{10};
    std::array<double, 2> previous_xy_{};
};

DroidSimulation::DroidSimulation(
    const std::filesystem::path& model_path,
    bool realtime)
    : impl_(std::make_unique<Impl>(model_path, realtime)) {}

DroidSimulation::~DroidSimulation() = default;

bool DroidSimulation::running() const {
    return impl_->running();
}

void DroidSimulation::set_running(bool running) {
    impl_->set_running(running);
}

void DroidSimulation::reset() {
    impl_->reset();
}

Json DroidSimulation::agent_spec() const {
    return impl_->agent_spec();
}

Json DroidSimulation::body_experiment_spec() const {
    return impl_->body_experiment_spec();
}

Json DroidSimulation::reset_agent(
    std::string_view assembly_id,
    std::uint64_t seed,
    std::string_view episode_profile) {
    return impl_->reset_agent(assembly_id, seed, episode_profile);
}

Json DroidSimulation::step_agent(
    const std::map<std::string, double>& actions,
    double control_dt_s) {
    return impl_->step_agent(actions, control_dt_s);
}

Json DroidSimulation::policy_observation() const {
    return impl_->policy_observation();
}

void DroidSimulation::advance_steps(std::size_t count) {
    impl_->advance_steps(count);
}

Json DroidSimulation::state() const {
    return impl_->state();
}

Json DroidSimulation::catalog() const {
    return impl_->catalog();
}

Json DroidSimulation::health() const {
    return impl_->health();
}

}  // namespace droid
