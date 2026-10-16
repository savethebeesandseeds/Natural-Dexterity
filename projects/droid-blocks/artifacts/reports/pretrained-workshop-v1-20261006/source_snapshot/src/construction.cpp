#include "droid/construction.hpp"
#include "droid/module_catalog.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace droid {
namespace {
constexpr double kStudM = 0.04;
constexpr double kPhysicsDtS = 0.002;
constexpr int kControlSteps = 10;
constexpr std::uint64_t kImuPeriodTicks = 5;
constexpr std::uint64_t kImuLatencyTicks = 3;
constexpr std::uint64_t kLightPeriodTicks = 50;
constexpr std::uint64_t kLightLatencyTicks = 10;
constexpr double kLightFalloffDistanceM = 0.5;
constexpr double kCellMassKg = 0.015;
constexpr double kBlockMassKg = 0.012;
constexpr double kSensorMassKg = 0.02;
constexpr double kSensorHalfM = 0.018;
constexpr double kPassiveDamping = 0.006;
constexpr double kPassiveArmature = 0.00002;

struct ModelDelete { void operator()(mjModel* p) const { mj_deleteModel(p); } };
struct DataDelete { void operator()(mjData* p) const { mj_deleteData(p); } };
struct SpecDelete { void operator()(mjSpec* p) const { mj_deleteSpec(p); } };
using ModelPtr = std::unique_ptr<mjModel, ModelDelete>;
using DataPtr = std::unique_ptr<mjData, DataDelete>;

double finite_round(double value, double scale = 1e9) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite construction physics");
    const double result = std::round(value * scale) / scale;
    return result == 0.0 ? 0.0 : result;
}

Json values(const mjtNum* data, std::size_t count = 3) {
    Json result = Json::array();
    for (std::size_t i = 0; i < count; ++i) result.push_back(finite_round(data[i]));
    return result;
}

double seconds(std::uint64_t tick) { return finite_round(tick * kPhysicsDtS); }

void keys(const Json& object, std::initializer_list<std::string_view> expected,
          std::string_view context) {
    if (!object.is_object() || object.size() != expected.size()) {
        throw std::invalid_argument(std::string(context) + " has missing or unknown fields");
    }
    for (const auto key : expected) {
        if (!object.contains(key)) {
            throw std::invalid_argument(std::string(context) + " is missing " + std::string(key));
        }
    }
}

int integer(const Json& value, int low, int high, std::string_view context) {
    if (!value.is_number_integer() || value < low || value > high) {
        throw std::invalid_argument(std::string(context) + " must be an integer in [" +
            std::to_string(low) + ", " + std::to_string(high) + "]");
    }
    return value.get<int>();
}

std::size_t text_length(const std::string& text) {
    return static_cast<std::size_t>(std::count_if(text.begin(), text.end(),
        [](unsigned char c) { return (c & 0xc0U) != 0x80U; }));
}

bool reserved_part_id(const std::string& id) {
    if (id == "imu-0001" || id == "motor-0001") return true;
    for (int segment = 0; segment < 2; ++segment) {
        for (int slot = 0; slot < 6; ++slot) {
            if (id == "segment-" + std::to_string(segment) + "-cell-" +
                      std::to_string(slot)) return true;
        }
    }
    return false;
}

Json validate_assembly(const Json& input) {
    // Keep omitted legacy fields omitted in canonical assembly/state output.
    // The optional motor location extends both kits without weakening their
    // exact-field validation or adding assembly information to observations.
    Json required_fields = input;
    if (input.is_object() && input.contains("powered_hinge")) {
        (void)integer(input.at("powered_hinge"), 0, 1, "assembly.powered_hinge");
        required_fields.erase("powered_hinge");
    }
    const bool with_light = input.is_object() && input.contains("schema") &&
        input.at("schema") == "construction_kit_v2";
    if (with_light) {
        keys(required_fields, {"schema", "name", "segments", "blocks", "sensor", "light"}, "assembly");
    } else {
        keys(required_fields, {"schema", "name", "segments", "blocks", "sensor"}, "assembly");
    }
    if (!with_light && input.at("schema") != "construction_kit_v1") {
        throw std::invalid_argument("unsupported construction schema");
    }
    if (!input.at("name").is_string() ||
        text_length(input.at("name").get_ref<const std::string&>()) > 60) {
        throw std::invalid_argument("assembly.name must contain at most 60 characters");
    }
    const Json& segments = input.at("segments");
    if (!segments.is_array() || segments.size() != 2) {
        throw std::invalid_argument("assembly.segments must contain exactly two lengths");
    }
    const std::array<int, 2> lengths{
        integer(segments.at(0), 2, 6, "segment length"),
        integer(segments.at(1), 2, 6, "segment length")};
    const auto socket = [&lengths](const Json& part) {
        const int segment = integer(part.at("segment"), 0, 1, "part.segment");
        const int slot = integer(part.at("slot"), 0, lengths[segment] - 1, "part.slot");
        const int side = integer(part.at("side"), -1, 1, "part.side");
        if (side == 0) throw std::invalid_argument("part.side must be -1 or 1");
        return std::tuple(segment, slot, side);
    };
    keys(input.at("sensor"), {"segment", "slot", "side"}, "assembly.sensor");
    std::set<std::tuple<int, int, int>> occupied{socket(input.at("sensor"))};
    if (with_light) {
        keys(input.at("light"), {"segment", "slot", "side", "face"}, "assembly.light");
        if (!occupied.insert(socket(input.at("light"))).second) {
            throw std::invalid_argument("light and IMU cannot occupy the same construction socket");
        }
        const int face = integer(input.at("light").at("face"), -1, 1, "light.face");
        if (face == 0) throw std::invalid_argument("light.face must be -1 or 1");
    }
    const Json& blocks = input.at("blocks");
    if (!blocks.is_array() || blocks.size() > 12) {
        throw std::invalid_argument("assembly.blocks must contain at most 12 blocks");
    }
    std::set<std::string> identifiers;
    for (const Json& block : blocks) {
        keys(block, {"id", "segment", "slot", "side"}, "block");
        if (!block.at("id").is_string()) throw std::invalid_argument("block.id must be a string");
        const std::string& id = block.at("id").get_ref<const std::string&>();
        if (id.empty() || text_length(id) > 60 || reserved_part_id(id) ||
            (with_light && id == "light-0001") ||
            !identifiers.insert(id).second) {
            throw std::invalid_argument(
                "block.id must be unique, non-reserved and contain 1 to 60 characters");
        }
        if (!occupied.insert(socket(block)).second) {
            throw std::invalid_argument("two parts cannot occupy the same construction socket");
        }
    }
    // Exercise the JSON library's UTF-8 validation before any compilation or
    // active-state replacement. Labels are never interpolated into model XML.
    (void)input.dump();
    return input;
}

Json default_sun() {
    return Json{{"position_m", Json::array({0.30, 0.0, 0.20})}, {"intensity_lux", 1000.0}};
}

Json validate_sun(const Json& input) {
    keys(input, {"position_m", "intensity_lux"}, "sun");
    const Json& position = input.at("position_m");
    if (!position.is_array() || position.size() != 3) {
        throw std::invalid_argument("sun.position_m must contain exactly three coordinates");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!position.at(axis).is_number()) throw std::invalid_argument("sun coordinates must be numbers");
        const double coordinate = position.at(axis).get<double>();
        if (!std::isfinite(coordinate) || std::abs(coordinate) > 2.0 ||
            (axis == 1 && coordinate != 0.0)) {
            throw std::invalid_argument("sun requires finite X/Z within [-2,2]m and Y=0");
        }
    }
    if (!input.at("intensity_lux").is_number()) {
        throw std::invalid_argument("sun intensity must be a number");
    }
    const double intensity = input.at("intensity_lux").get<double>();
    if (!std::isfinite(intensity) || intensity < 0.0 || intensity > 1000.0) {
        throw std::invalid_argument("sun intensity must be finite and in [0,1000]lux");
    }
    return input;
}

const Json& motor_physics() {
    return catalog_json().at("motor_skus").at(std::string(kReferenceMotorSku)).at("physics");
}

double reflected_rotor_inertia() {
    const double ratio = motor_physics().at("gear_ratio_motor_to_output");
    return motor_physics().at("rotor_inertia_kg_m2").get<double>() * ratio * ratio;
}

struct Geometry {
    std::string model_name;
    std::string id;
    std::string kind;
    int segment{};
    int slot{};
    int side{};
    int geom_id{-1};
    std::array<double, 4> color{};
};

std::string generate_xml(const Json& assembly, std::vector<Geometry>& geometry) {
    const int powered_hinge = assembly.value("powered_hinge", 0);
    std::ostringstream xml;
    xml.precision(17);
    xml << "<mujoco model=\"" << assembly.at("schema").get<std::string>() << "\">"
        << "<compiler angle=\"radian\" inertiafromgeom=\"auto\"/>"
        << "<option timestep=\"0.002\" gravity=\"0 0 -9.81\" integrator=\"implicitfast\"/>"
        << "<default><geom type=\"box\" contype=\"0\" conaffinity=\"0\"/></default>"
        << "<worldbody><site name=\"anchor\" pos=\"0 0 0\" size=\"0.008\"/>";
    const auto add_geom = [&xml, &geometry](Geometry part, double x, double z,
                                          double sx, double sy, double sz, double mass) {
        part.model_name = "part_" + std::to_string(geometry.size());
        xml << "<geom name=\"" << part.model_name << "\" pos=\"" << x << " 0 " << z
            << "\" size=\"" << sx << ' ' << sy << ' ' << sz << "\" mass=\"" << mass
            << "\" rgba=\"" << part.color[0] << ' ' << part.color[1] << ' '
            << part.color[2] << ' ' << part.color[3] << "\"/>";
        geometry.push_back(std::move(part));
    };
    for (int segment = 0; segment < 2; ++segment) {
        const bool powered = segment == powered_hinge;
        const int length = assembly.at("segments").at(segment);
        const double offset = segment == 0 ? 0.0 :
            -kStudM * assembly.at("segments").at(0).get<int>();
        xml << "<body name=\"segment_" << segment << "\" pos=\"0 0 " << offset << "\">"
            << "<joint name=\"" << (powered ? "motor_joint" : "passive_joint")
            << "\" type=\"hinge\" axis=\"0 1 0\" limited=\"false\" damping=\""
            << (powered ? 0.0 : kPassiveDamping) << "\" armature=\""
            << (powered ? reflected_rotor_inertia() : kPassiveArmature) << "\"/>";
        const std::array<double, 4> beam_color = segment == 0
            ? std::array<double, 4>{0.15, 0.55, 0.75, 1.0}
            : std::array<double, 4>{0.9, 0.55, 0.17, 1.0};
        for (int slot = 0; slot < length; ++slot) {
            add_geom(Geometry{"", "segment-" + std::to_string(segment) + "-cell-" +
                std::to_string(slot), "beam", segment, slot, 0, -1, beam_color},
                0.0, -(slot + 0.5) * kStudM, 0.02, 0.011, 0.02, kCellMassKg);
        }
        for (const Json& block : assembly.at("blocks")) {
            if (block.at("segment") != segment) continue;
            const int slot = block.at("slot");
            const int side = block.at("side");
            add_geom(Geometry{"", block.at("id"), "block", segment, slot, side, -1,
                {0.7, 0.35, 0.63, 1.0}}, side * kStudM,
                -(slot + 0.5) * kStudM, 0.02, 0.02, 0.02, kBlockMassKg);
        }
        const Json& sensor = assembly.at("sensor");
        if (sensor.at("segment") == segment) {
            const int slot = sensor.at("slot");
            const int side = sensor.at("side");
            const double x = side * (0.02 + kSensorHalfM);
            const double z = -(slot + 0.5) * kStudM;
            add_geom(Geometry{"", "imu-0001", "sensor", segment, slot, side, -1,
                {0.22, 0.78, 0.52, 1.0}}, x, z,
                kSensorHalfM, kSensorHalfM, kSensorHalfM, kSensorMassKg);
            xml << "<site name=\"imu_site\" pos=\"" << x << " 0 " << z
                << "\" size=\"0.003\"/>";
        }
        if (assembly.contains("light") && assembly.at("light").at("segment") == segment) {
            const Json& light = assembly.at("light");
            const int slot = light.at("slot");
            const int side = light.at("side");
            const int face = light.at("face");
            const double x = side * (0.02 + kSensorHalfM);
            const double z = -(slot + 0.5) * kStudM;
            add_geom(Geometry{"", "light-0001", "light", segment, slot, side, -1,
                {0.95, 0.79, 0.18, 1.0}}, x, z,
                kSensorHalfM, kSensorHalfM, kSensorHalfM, kSensorMassKg);
            xml << "<site name=\"light_site\" pos=\"" << x << " 0 "
                << z + face * kSensorHalfM << "\" quat=\""
                << (face == 1 ? "1 0 0 0" : "0 1 0 0") << "\" size=\"0.003\"/>";
        }
        xml << "<site name=\"segment_" << segment << "_end\" pos=\"0 0 "
            << -length * kStudM << "\" size=\"0.003\"/>";
    }
    xml << "</body></body></worldbody>"
        << "<actuator><motor name=\"drive\" joint=\"motor_joint\" gear=\"1\" "
        << "ctrllimited=\"true\" ctrlrange=\"-4 4\"/></actuator>"
        << "<sensor><accelerometer name=\"imu_accel\" site=\"imu_site\"/>"
        << "<gyro name=\"imu_gyro\" site=\"imu_site\"/></sensor></mujoco>";
    return xml.str();
}

double norm(const std::array<double, 3>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}
}  // namespace

struct ConstructionWorld::Impl {
    struct Sample {
        std::uint64_t sequence{};
        std::uint64_t acquired_tick{};
        std::uint64_t delivery_tick{};
        std::array<double, 3> acceleration{};
        std::array<double, 3> gyro{};
    };

    struct LightSample {
        std::uint64_t sequence{};
        std::uint64_t acquired_tick{};
        std::uint64_t delivery_tick{};
        double illuminance_lux{};
    };

    explicit Impl(const Json& input, const Json& initial_sun = default_sun())
        : assembly(validate_assembly(input)),
          has_light(assembly.at("schema") == "construction_kit_v2"),
          powered_hinge(assembly.value("powered_hinge", 0)),
          sun(validate_sun(initial_sun)) {
        const std::string xml = generate_xml(assembly, geometry);
        std::array<char, 2048> error{};
        std::unique_ptr<mjSpec, SpecDelete> spec(
            mj_parseXMLString(xml.c_str(), nullptr, error.data(), error.size()));
        if (!spec) throw std::runtime_error("construction parse failed: " + std::string(error.data()));
        model.reset(mj_compile(spec.get(), nullptr));
        if (!model) throw std::runtime_error("construction compile failed: " + std::string(mjs_getError(spec.get())));
        data.reset(mj_makeData(model.get()));
        if (!data) throw std::runtime_error("construction data allocation failed");
        for (Geometry& geom : geometry) geom.geom_id = id(mjOBJ_GEOM, geom.model_name);
        for (int segment = 0; segment < 2; ++segment) {
            body_ids[segment] = id(mjOBJ_BODY, "segment_" + std::to_string(segment));
            end_sites[segment] = id(mjOBJ_SITE, "segment_" + std::to_string(segment) + "_end");
        }
        imu_site = id(mjOBJ_SITE, "imu_site");
        accel_sensor = id(mjOBJ_SENSOR, "imu_accel");
        gyro_sensor = id(mjOBJ_SENSOR, "imu_gyro");
        if (has_light) light_site = id(mjOBJ_SITE, "light_site");
        const int motor_joint = id(mjOBJ_JOINT, "motor_joint");
        const int passive_joint = id(mjOBJ_JOINT, "passive_joint");
        motor_qpos = model->jnt_qposadr[motor_joint];
        motor_qvel = model->jnt_dofadr[motor_joint];
        passive_qpos = model->jnt_qposadr[passive_joint];
        passive_qvel = model->jnt_dofadr[passive_joint];
        if (model->nu != 1 || model->nv != 2 || model->nq != 2 ||
            model->actuator_trnid[0] != motor_joint || motor_qpos == passive_qpos ||
            motor_qvel == passive_qvel || model->jnt_type[motor_joint] != mjJNT_HINGE ||
            model->jnt_type[passive_joint] != mjJNT_HINGE) {
            throw std::runtime_error("construction requires one drive bound to the selected motor hinge and one passive hinge");
        }
        reset();
    }

    int id(mjtObj type, const std::string& name) const {
        const int result = mj_name2id(model.get(), type, name.c_str());
        if (result < 0) throw std::runtime_error("construction model missing " + name);
        return result;
    }

    Sample acquire() {
        Sample result;
        result.sequence = next_sequence++;
        result.acquired_tick = tick;
        result.delivery_tick = tick + kImuLatencyTicks;
        for (int i = 0; i < 3; ++i) {
            result.acceleration[i] = data->sensordata[model->sensor_adr[accel_sensor] + i];
            result.gyro[i] = data->sensordata[model->sensor_adr[gyro_sensor] + i];
            if (!std::isfinite(result.acceleration[i]) || !std::isfinite(result.gyro[i])) {
                throw std::runtime_error("construction IMU produced a non-finite sample");
            }
        }
        return result;
    }

    Json imu_channels(const Sample& sample) const {
        Json acceleration = Json::array();
        Json gyro = Json::array();
        for (int i = 0; i < 3; ++i) {
            acceleration.push_back(finite_round(sample.acceleration[i]));
            gyro.push_back(finite_round(sample.gyro[i]));
        }
        return Json{{"specific_force_m_s2", acceleration}, {"angular_velocity_rad_s", gyro}};
    }

    Json imu_sample() const {
        const bool valid = delivered.sequence != 0;
        return Json{{"module_id", "imu-0001"}, {"family_id", "imu_6axis_v0"},
            {"valid", valid}, {"sequence", delivered.sequence},
            {"sample_time_s", valid ? Json(seconds(delivered.acquired_tick)) : Json(nullptr)},
            {"delivered_time_s", valid ? Json(seconds(delivered.delivery_tick)) : Json(nullptr)},
            {"age_s", valid ? Json(seconds(tick - delivered.acquired_tick)) : Json(nullptr)},
            {"sample_period_s", seconds(kImuPeriodTicks)},
            {"latency_s", seconds(kImuLatencyTicks)},
            {"observations", imu_channels(delivered)}};
    }

    std::array<double, 3> light_direction() const {
        const mjtNum* rotation = data->site_xmat + 9 * light_site;
        return {rotation[2], rotation[5], rotation[8]};
    }

    double instantaneous_light() const {
        const mjtNum* position = data->site_xpos + 3 * light_site;
        const auto direction = light_direction();
        std::array<double, 3> toward_sun{};
        double dot = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            toward_sun[axis] = sun.at("position_m").at(axis).get<double>() - position[axis];
            dot += direction[axis] * toward_sun[axis];
        }
        const double distance = norm(toward_sun);
        if (distance <= 1e-9) return 0.0;
        const double cosine = clamp(dot / distance, 0.0, 1.0);
        const double scaled_distance = distance / kLightFalloffDistanceM;
        return clamp(sun.at("intensity_lux").get<double>() * cosine /
            (1.0 + scaled_distance * scaled_distance), 0.0, 1000.0);
    }

    LightSample acquire_light() {
        return LightSample{next_light_sequence++, tick, tick + kLightLatencyTicks,
                           instantaneous_light()};
    }

    Json light_channels() const {
        return Json{{"illuminance_lux", finite_round(delivered_light.illuminance_lux)},
                    {"saturated", delivered_light.illuminance_lux >= 1000.0}};
    }

    Json light_sample() const {
        const bool valid = delivered_light.sequence != 0;
        return Json{{"module_id", "light-0001"}, {"family_id", "ambient_light_v0"},
            {"valid", valid}, {"sequence", delivered_light.sequence},
            {"sample_time_s", valid ? Json(seconds(delivered_light.acquired_tick)) : Json(nullptr)},
            {"delivered_time_s", valid ? Json(seconds(delivered_light.delivery_tick)) : Json(nullptr)},
            {"age_s", valid ? Json(seconds(tick - delivered_light.acquired_tick)) : Json(nullptr)},
            {"sample_period_s", seconds(kLightPeriodTicks)},
            {"latency_s", seconds(kLightLatencyTicks)}, {"observations", light_channels()}};
    }

    void set_sun(const Json& input) {
        if (!has_light) throw std::invalid_argument("sun control requires construction_kit_v2");
        Json candidate = validate_sun(input);
        sun = std::move(candidate);
    }

    void reset() {
        mj_resetData(model.get(), data.get());
        mj_forward(model.get(), data.get());
        tick = 0;
        next_sequence = 1;
        delivered = {};
        pending.clear();
        pending.push_back(acquire());
        next_light_sequence = 1;
        delivered_light = {};
        pending_light.clear();
        if (has_light) pending_light.push_back(acquire_light());
        motor = MotorState{0.0, motor_physics().at("reference_ambient_c")};
        output = {};
        feedback = ActuatorFeedback{};
        feedback.bus_voltage_v = motor_physics().at("nominal_bus_voltage_v");
        feedback.temperature_c = motor.temperature_c;
        command = 0.0;
        energy_used_j = 0.0;
        cumulative_reward = 0.0;
        reward_rate = catalog_json().at("sensor_families").at("imu_6axis_v0").at("missing_reward");
        imu_reward_rate = reward_rate;
        light_reward_rate = 0.0;
        if (has_light) {
            light_reward_rate = catalog_json().at("sensor_families")
                .at("ambient_light_v0").at("missing_reward");
            reward_rate += light_reward_rate;
        }
        veto = false;
        flags.clear();
    }

    Json observation() const {
        Json result{{"schema_version", "construction_observation_v1"},
            {"reward_sensors", Json::array({imu_sample()})},
            {"actuator_feedback", Json::array({Json{{"module_id", "motor-0001"},
                {"sku_id", kReferenceMotorSku}, {"valid", true}, {"feedback", feedback}}})}};
        if (has_light) {
            result["schema_version"] = "construction_observation_v2";
            result["reward_sensors"].push_back(light_sample());
        }
        return result;
    }

    void sensor_safety() {
        if (delivered.sequence == 0) return;
        const Json& parameters = catalog_json().at("sensor_families")
            .at("imu_6axis_v0").at("reward").at("parameters");
        const double deviation = std::abs(norm(delivered.acceleration) -
            parameters.at("gravity_m_s2").get<double>());
        if (deviation >= parameters.at("acceleration_deviation_hard_m_s2").get<double>()) {
            veto = true;
            flags.insert("imu-0001:acceleration_limit");
        }
        if (norm(delivered.gyro) >= parameters.at("angular_speed_hard_rad_s").get<double>()) {
            veto = true;
            flags.insert("imu-0001:angular_rate_limit");
        }
    }

    Json step(double effort) {
        if (!std::isfinite(effort) || effort < -1.0 || effort > 1.0) {
            throw std::invalid_argument("construction effort must be finite and in [-1,1]");
        }
        if (veto) throw std::logic_error("construction safety stop requires reset");
        command = effort;
        flags.clear();
        std::set<std::string> motor_flags;
        double transition_value = 0.0;
        double imu_transition_value = 0.0;
        double light_transition_value = 0.0;
        for (int substep = 0; substep < kControlSteps; ++substep) {
            const double bus = motor_physics().at("nominal_bus_voltage_v");
            output = step_motor_current_thermal(motor, veto ? 0.0 : effort,
                data->qvel[motor_qvel], bus, kPhysicsDtS);
            motor = output.state;
            for (const std::string& flag : output.safety.flags) {
                motor_flags.insert(flag);
                flags.insert("motor-0001:" + flag);
            }
            veto = veto || output.safety.veto;
            data->ctrl[0] = veto ? 0.0 : clamp(output.output_torque_nm, -4.0, 4.0);
            mj_step(model.get(), data.get());
            mj_forward(model.get(), data.get());
            // MuJoCo can reset unstable state after issuing a warning. Detect
            // both its warning and a time rollback; never display that reset
            // as a successful, repeatable construction transition.
            const double expected_time = seconds(tick + 1);
            const auto finite_values = [](const mjtNum* first, std::size_t count) {
                return std::all_of(first, first + count,
                    [](mjtNum value) { return std::isfinite(value); });
            };
            bool numerical_warning = false;
            for (const int warning : {mjWARN_INERTIA, mjWARN_BADQPOS, mjWARN_BADQVEL,
                                      mjWARN_BADQACC, mjWARN_BADCTRL}) {
                numerical_warning = numerical_warning || data->warning[warning].number != 0;
            }
            if (numerical_warning || !finite_values(data->qpos, model->nq) ||
                !finite_values(data->qvel, model->nv) || !finite_values(data->qacc, model->nv) ||
                !std::isfinite(data->time) ||
                std::abs(data->time - expected_time) > std::max(1e-9, expected_time * 1e-9)) {
                veto = true;
                flags.insert("construction:numerical_instability");
                data->ctrl[0] = 0.0;
                throw std::runtime_error("MuJoCo construction numerical instability; reset required");
            }
            ++tick;
            if (tick % kImuPeriodTicks == 0) pending.push_back(acquire());
            while (!pending.empty() && pending.front().delivery_tick <= tick) {
                delivered = pending.front();
                pending.pop_front();
            }
            if (has_light) {
                if (tick % kLightPeriodTicks == 0) pending_light.push_back(acquire_light());
                while (!pending_light.empty() && pending_light.front().delivery_tick <= tick) {
                    delivered_light = pending_light.front();
                    pending_light.pop_front();
                }
            }
            sensor_safety();
            const bool valid = delivered.sequence != 0;
            reward_rate = valid ? sensor_reward_rate("imu_6axis_v0", imu_channels(delivered)) :
                catalog_json().at("sensor_families").at("imu_6axis_v0").at("missing_reward").get<double>();
            imu_reward_rate = reward_rate;
            const std::array<RegisteredSensorSample, 1> voters{{
                {"imu-0001", "imu_6axis_v0", reward_rate, valid, false}}};
            if (has_light) {
                const bool light_valid = delivered_light.sequence != 0;
                light_reward_rate = light_valid ? sensor_reward_rate("ambient_light_v0", light_channels()) :
                    catalog_json().at("sensor_families").at("ambient_light_v0").at("missing_reward").get<double>();
                const std::array<RegisteredSensorSample, 1> light_voters{{
                    {"light-0001", "ambient_light_v0", light_reward_rate, light_valid, false}}};
                const std::array<RegisteredSensorSample, 2> combined{{voters[0], light_voters[0]}};
                transition_value += transition_reward(kPhysicsDtS, combined);
                imu_transition_value += transition_reward(kPhysicsDtS, voters);
                light_transition_value += transition_reward(kPhysicsDtS, light_voters);
                reward_rate += light_reward_rate;
            } else {
                transition_value += transition_reward(kPhysicsDtS, voters);
            }
            energy_used_j += output.electrical_power_w * kPhysicsDtS;
            ActuatorFeedbackInput input;
            input.position_rad = data->qpos[motor_qpos];
            input.velocity_rad_s = data->qvel[motor_qvel];
            input.current_a = motor.current_a;
            input.bus_voltage_v = bus;
            input.temperature_c = motor.temperature_c;
            input.output_torque_nm = veto ? 0.0 : output.output_torque_nm;
            input.commanded_effort = effort;
            input.previous_stuck_score = feedback.stuck_score;
            input.dt_s = kPhysicsDtS;
            input.fault_flags = std::vector<std::string>(motor_flags.begin(), motor_flags.end());
            feedback = actuator_feedback_sample(input);
        }
        if (veto) data->ctrl[0] = 0.0;
        cumulative_reward += transition_value;
        Json result{{"observation", observation()}, {"reward", transition_value},
            {"reward_components", Json::array({Json{{"module_id", "imu-0001"},
                {"family_id", "imu_6axis_v0"}, {"transition_reward", transition_value},
                {"reward_rate", reward_rate}, {"valid", delivered.sequence != 0}}})},
            {"safety", safety()}, {"terminated", veto}, {"truncated", false},
            {"info", Json{{"elapsed_control_dt_s", 0.02}, {"physics_steps_executed", kControlSteps}}}};
        if (has_light) {
            Json& imu_component = result["reward_components"].at(0);
            imu_component["transition_reward"] = imu_transition_value;
            imu_component["reward_rate"] = imu_reward_rate;
            // Match the integrated global sum after floating-point accumulation;
            // the correction is only its arithmetic rounding residual.
            light_transition_value += transition_value -
                (imu_transition_value + light_transition_value);
            result["reward_components"].push_back(Json{{"module_id", "light-0001"},
                {"family_id", "ambient_light_v0"}, {"transition_reward", light_transition_value},
                {"reward_rate", light_reward_rate}, {"valid", delivered_light.sequence != 0}});
        }
        return result;
    }

    Json safety() const {
        return Json{{"veto", veto}, {"flags", std::vector<std::string>(flags.begin(), flags.end())}};
    }

    Json state() const {
        Json displayed = Json::array();
        for (const Geometry& geom : geometry) {
            mjtNum quaternion[4];
            mju_mat2Quat(quaternion, data->geom_xmat + 9 * geom.geom_id);
            const mjtNum* half_size = model->geom_size + 3 * geom.geom_id;
            displayed.push_back(Json{{"id", geom.id}, {"kind", geom.kind},
                {"segment", geom.segment}, {"slot", geom.slot}, {"side", geom.side},
                {"position_m", values(data->geom_xpos + 3 * geom.geom_id)},
                {"size_m", Json::array({2 * half_size[0], 2 * half_size[1], 2 * half_size[2]})},
                {"quaternion", values(quaternion, 4)}, {"rgba", geom.color}});
        }
        Json segments = Json::array();
        Json masses = Json::array();
        Json inertias = Json::array();
        for (int segment = 0; segment < 2; ++segment) {
            segments.push_back(Json{{"index", segment},
                {"length_studs", assembly.at("segments").at(segment)},
                {"start_m", values(data->xpos + 3 * body_ids[segment])},
                {"end_m", values(data->site_xpos + 3 * end_sites[segment])}});
            masses.push_back(finite_round(model->body_mass[body_ids[segment]]));
            inertias.push_back(Json{{"diagonal_kg_m2", values(model->body_inertia + 3 * body_ids[segment])},
                {"inertial_frame_quaternion", values(model->body_iquat + 4 * body_ids[segment], 4)}});
        }
        mjtNum sensor_quaternion[4];
        mju_mat2Quat(sensor_quaternion, data->site_xmat + 9 * imu_site);
        Json sensor = imu_sample();
        sensor["position_m"] = values(data->site_xpos + 3 * imu_site);
        sensor["quaternion"] = values(sensor_quaternion, 4);
        Json result{{"schema", "construction_state_v1"}, {"assembly", assembly},
            {"elapsed_s", seconds(tick)}, {"sim_time", seconds(tick)}, {"tick", tick},
            {"geometry", displayed}, {"segments", segments},
            {"joints", Json{{"motor_m", powered_hinge == 0 ? Json::array({0.0, 0.0, 0.0}) :
                    values(data->xpos + 3 * body_ids[1])},
                {"passive_m", powered_hinge == 0 ? values(data->xpos + 3 * body_ids[1]) :
                    Json::array({0.0, 0.0, 0.0})}}},
            {"tip_position_m", values(data->site_xpos + 3 * end_sites[1])},
            {"sensor", sensor}, {"actuator", Json(feedback)}, {"command", command},
            {"reward", Json{{"rate", reward_rate}, {"cumulative", cumulative_reward}}},
            {"safety", safety()},
            {"diagnostics", Json{{"policy_visible", false}, {"actuator_count", model->nu},
                {"degrees_of_freedom", model->nv},
                {"passive_angle_rad", finite_round(data->qpos[passive_qpos])},
                {"passive_velocity_rad_s", finite_round(data->qvel[passive_qvel])},
                {"passive_actuator_force_nm", finite_round(data->qfrc_actuator[passive_qvel])},
                {"motor_angle_rad", finite_round(data->qpos[motor_qpos])},
                {"motor_velocity_rad_s", finite_round(data->qvel[motor_qvel])},
                {"total_mass_kg", finite_round(mj_getTotalmass(model.get()))},
                {"segment_masses_kg", masses}, {"segment_inertias_kg_m2", inertias},
                {"electrical_energy_j", finite_round(energy_used_j)},
                {"instantaneous_imu", Json{
                    {"specific_force_m_s2", values(data->sensordata + model->sensor_adr[accel_sensor])},
                    {"angular_velocity_rad_s", values(data->sensordata + model->sensor_adr[gyro_sensor])}}}}}};
        if (has_light) {
            result["schema"] = "construction_state_v2";
            Json light = light_sample();
            light["position_m"] = values(data->site_xpos + 3 * light_site);
            const auto direction = light_direction();
            light["direction_m"] = Json::array({finite_round(direction[0]),
                finite_round(direction[1]), finite_round(direction[2])});
            result["light_sensor"] = std::move(light);
            result["sun"] = sun;
            result["diagnostics"]["instantaneous_light_lux"] = finite_round(instantaneous_light());
        }
        return result;
    }

    Json assembly;
    bool has_light{};
    int powered_hinge{};
    Json sun;
    std::vector<Geometry> geometry;
    ModelPtr model;
    DataPtr data;
    std::array<int, 2> body_ids{};
    std::array<int, 2> end_sites{};
    int imu_site{}, accel_sensor{}, gyro_sensor{};
    int light_site{-1};
    int motor_qpos{}, motor_qvel{}, passive_qpos{}, passive_qvel{};
    std::uint64_t tick{}, next_sequence{1};
    std::deque<Sample> pending;
    Sample delivered;
    std::uint64_t next_light_sequence{1};
    std::deque<LightSample> pending_light;
    LightSample delivered_light;
    MotorState motor;
    MotorStepResult output;
    ActuatorFeedback feedback;
    double command{}, energy_used_j{}, cumulative_reward{}, reward_rate{};
    double imu_reward_rate{}, light_reward_rate{};
    bool veto{};
    std::set<std::string> flags;
};

ConstructionWorld::ConstructionWorld() : impl_(std::make_unique<Impl>(default_assembly())) {}
ConstructionWorld::~ConstructionWorld() = default;
void ConstructionWorld::rebuild(const Json& assembly) {
    auto candidate = std::make_unique<Impl>(assembly,
        impl_->has_light ? impl_->sun : default_sun());
    impl_.swap(candidate);
}
void ConstructionWorld::reset() { impl_->reset(); }
void ConstructionWorld::set_sun(const Json& sun) { impl_->set_sun(sun); }
Json ConstructionWorld::assembly() const { return impl_->assembly; }
Json ConstructionWorld::observation() const { return impl_->observation(); }
Json ConstructionWorld::state() const { return impl_->state(); }
Json ConstructionWorld::step(double effort) { return impl_->step(effort); }

Json ConstructionWorld::default_assembly() {
    return Json{{"schema", "construction_kit_v1"}, {"name", "My first creature"},
        {"segments", Json::array({3, 3})}, {"blocks", Json::array()},
        {"sensor", Json{{"segment", 1}, {"slot", 2}, {"side", -1}}}};
}

Json ConstructionWorld::light_assembly() {
    Json result = default_assembly();
    result["schema"] = "construction_kit_v2";
    result["name"] = "My first sun-seeker";
    result["light"] = Json{{"segment", 1}, {"slot", 2}, {"side", 1}, {"face", 1}};
    return result;
}

Json ConstructionWorld::specification() {
    return Json{{"schema", "construction_kit_v1"}, {"observation_schema", "construction_observation_v1"},
        {"generator", "anchored_two_link_cells_v1"}, {"name_max_characters", 60},
        {"segment_count", 2}, {"segment_length_studs", Json::array({2, 6})},
        {"maximum_added_blocks", 12}, {"stud_m", kStudM},
        {"beam_cell_size_m", Json::array({0.04, 0.022, 0.04})}, {"beam_cell_mass_kg", kCellMassKg},
        {"block_cube_m", 0.04}, {"block_mass_kg", kBlockMassKg},
        {"sensor_cube_m", 0.036}, {"sensor_mass_kg", kSensorMassKg},
        {"physics_dt_s", kPhysicsDtS}, {"control_dt_s", 0.02},
        {"motor_id", "motor-0001"}, {"sensor_id", "imu-0001"},
        {"motor_model", kReferenceMotorSku}, {"motor_reflected_inertia_kg_m2", reflected_rotor_inertia()},
        {"actuation_layout", Json{{"schema", "optional_powered_hinge_v1"},
            {"assembly_field", "powered_hinge"}, {"optional", true}, {"default", 0},
            {"values", Json::array({0, 1})},
            {"hinge_order", "0: anchored base; 1: elbow relative to the first segment"},
            {"motor_binding", "selected hinge gets catalog rotor armature, torque, friction and local encoder/current feedback; other hinge gets passive armature and damping"},
            {"observation", "unchanged local motor/IMU/light contract; no hinge index or geometry"}}},
        {"passive_joint", Json{{"powered", false}, {"damping_nm_s_per_rad", kPassiveDamping},
            {"armature_kg_m2", kPassiveArmature}}},
        {"motion_plane", "XZ; anchored at origin; hinges about +Y; initially hanging toward -Z"},
        {"contacts", "disabled in this isolated planar fixture; no self-collision or ground"},
        {"imu", Json{{"family_id", "imu_6axis_v0"}, {"sample_period_s", 0.01},
            {"nominal_catalog_latency_s", 0.005}, {"latency_s", 0.006},
            {"latency_rule", "round nominal delivery up to the next .002s physics tick"},
            {"initial_valid", false}, {"noise", "none"},
            {"axes", "mounted segment local frame; no world orientation observation"}}},
        {"motor_feedback_timing", "synchronous local servo feedback at the step endpoint"},
        {"safety", Json{{"motor", "existing current, voltage, thermal and speed model; all flags retained"},
            {"imu", "new construction_v1 supplemental veto: delivered sample reaches the catalog reward's hard acceleration-deviation or angular-speed threshold"},
            {"imu_threshold_scope", "construction fixture only; no old world safety rule is changed"},
            {"numerical", "MuJoCo numerical warnings, non-finite state or a simulator time reset fail closed"},
            {"veto_behavior", "physical veto: zero powered torque, finish current .02s interval, require reset; numerical failure: abort immediately and require reset"}}},
        {"reward", "only the mounted IMU votes using unchanged imu_6axis_v0 reward and missing_reward"},
        {"power_supply", "fixed nominal catalog bus voltage; energy recorded, battery depletion not modeled"},
        {"exposure", "development-only construction; no old world, policy artifact, or heldout set"}};
}

Json ConstructionWorld::light_specification() {
    Json result = specification();
    result["schema"] = "construction_kit_v2";
    result["observation_schema"] = "construction_observation_v2";
    result["generator"] = "anchored_two_link_cells_light_v2";
    result["light"] = Json{{"module_id", "light-0001"}, {"family_id", "ambient_light_v0"},
        {"cube_m", 0.036}, {"mass_kg", kSensorMassKg}, {"face_values", Json::array({-1, 1})},
        {"face_axis", "mounted segment local +/-Z; optical site is on the corresponding cube face"},
        {"sample_period_s", 0.1}, {"latency_s", 0.02}, {"initial_valid", false},
        {"falloff_distance_m", kLightFalloffDistanceM},
        {"formula", "I * max(0, dot(unit(sun-site), world_face_normal)) / (1 + (distance_m / 0.5)^2)"},
        {"coincident_source", "zero illuminance when distance <= 1e-9m because incident direction is undefined"},
        {"noise_and_occlusion", "none; the mounted hemisphere and distance are modeled, shadows are not"}};
    result["sun"] = Json{{"default", default_sun()}, {"position_xz_bounds_m", Json::array({-2.0, 2.0})},
        {"position_y_m", 0.0}, {"intensity_bounds_lux", Json::array({0.0, 1000.0})},
        {"move_semantics", "no reset; already acquired samples retain their original value"},
        {"reset_semantics", "keep selected sun; reset physical state and acquisition queues"}};
    result["reward"] = "mounted IMU plus light each vote once using unchanged catalog rewards and missing_reward; sum integrated every .002s";
    return result;
}
}  // namespace droid
