"""Prospective, source-only curriculum for a bounded native PPO pilot.

This module uses only the Python standard library. Creating or validating its
manifest never starts a simulator. Reserved assemblies are declarations only;
the sampler copies only source cases and has no reserved-case sampling mode.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random


SCHEMA = "generalization_curriculum_v1"
SAMPLER_SCHEMA = "shuffled_source_cycles_v1"


def _case(lengths, weighted, hinge, side):
    return {
        "assembly": {
            "schema": "construction_kit_v2",
            "name": "Pooled sun-seeker development fixture",
            "segments": list(lengths),
            "powered_hinge": hinge,
            "blocks": ([{"id": "block-1", "segment": 1, "slot": 0, "side": -1}]
                       if weighted else []),
            "sensor": {"segment": 1, "slot": 2, "side": -1},
            "light": {"segment": 1, "slot": 2, "side": 1, "face": 1},
        },
        "sun": {"position_m": [side * .3, 0., -.3], "intensity_lux": 1000.},
    }


def _layout_cases(prefix, layouts):
    cases, ids = [], []
    for label, lengths, weighted in layouts:
        for hinge in (0, 1):
            for side in (-1, 1):
                cases.append(_case(lengths, weighted, hinge, side))
                ids.append(f"{prefix}-{label}-{'base' if hinge == 0 else 'elbow'}-"
                           f"{'left' if side < 0 else 'right'}")
    return cases, ids


def manifest():
    """Return an independent copy of the declared pilot, without native exposure."""
    source_layouts = [(f"L{a}-{b}-{'distal-block' if weighted else 'plain'}", (a, b), weighted)
                      for a, b in ((3, 3), (4, 3), (3, 4), (4, 4))
                      for weighted in (False, True)]
    source, source_ids = _layout_cases("source", source_layouts)
    development, development_ids = _layout_cases("development", [
        ("long-upper", (4, 3), False),
        ("long-lower", (3, 4), False),
        ("weighted", (4, 4), True),
    ])
    reserved, reserved_ids = _layout_cases("reserved", [
        ("long-upper", (5, 3), False),
        ("long-lower", (3, 5), False),
        ("weighted", (5, 5), True),
    ])
    return {
        "schema": SCHEMA,
        "status": "declared_before_native_pooled_learning",
        "question": "Can one pooled local-history policy learn useful behavior across the known native assembly family?",
        "scope": "Development-only anchored planar two-link fixture, one motor, one IMU and one light. This pilot tests pooled sensor-history PPO, not topology generalization or a first child-built creature.",
        "source_cases": source,
        "development_cases": development,
        "reserved_cases": reserved,
        "case_ids": {"source": source_ids, "development": development_ids, "reserved": reserved_ids},
        "pilot": {"seed": 404, "source_decisions": 16384, "n_envs": 8, "n_steps": 256,
                  "horizon_decisions": 640, "development_decisions_per_arm": 640,
                  "development_arms": ["frozen", "untrained", "quiet"]},
        "development_gate": {"required_cases": 12, "complete_safe_arms": ["frozen", "quiet"],
                             "native_reward_gain_strictly_greater_than": 3.2,
                             "interpretation": "Source competence diagnostic only: every frozen policy opportunity must safely complete and beat its paired quiet native return by more than 3.2. Pilot completion alone is not a research pass and never authorizes reserved-case exposure."},
        "full_study": {"source_seeds": [101, 202, 303], "source_decisions_per_seed": 98304,
                       "status": "not_authorized_by_pilot"},
        "decision_dt_s": .1,
        "native_dt_s": .02,
        "sampling": {"schema": SAMPLER_SCHEMA, "curriculum_seed_offset": 500000,
                     "rule": "Independent random.Random(source_seed+500000); shuffle all source indices once per cycle; draw each once; never rewind when policy/action RNG resets."},
        "ppo": {"implementation": "stable-baselines3", "version": "2.9.0",
                "network": "Separate two-layer 64x64 tanh actor/value networks; categorical 3-action output",
                "learning_rate": .0003, "n_epochs": 4, "batch_size": 64,
                "gamma": .92 ** (.1 / .4), "gae_lambda": .95, "clip_range": .2,
                "ent_coef": .01, "vf_coef": .5, "max_grad_norm": .5,
                "optimizer": "Adam", "optimizer_eps": 1e-5, "ortho_init": True,
                "clip_range_vf": None, "target_kl": None,
                "source_n_steps": 256, "source_n_envs": 8,
                "target_n_steps": 64, "target_n_envs": 1,
                "normalization": "Advantage normalization only; fixed sensor scaling; no VecNormalize.",
                "transfer": "Pilot development evaluation uses final actor+critic weights frozen; no development optimizer updates."},
        "policy_input": {"frames": 12, "frame_size": 17, "dimension": 204,
                         "order": "oldest to newest", "fixed_clip": [-5., 5.],
                         "frame": ["executed_effort/.15", "sin(motor_position)", "cos(motor_position)",
                                   "motor_velocity/10", "motor_current/.5", "imu_force_xyz/50 (3)",
                                   "imu_gyro_xyz/10 (3)", "light_lux/1000", "previous_request/.15",
                                   "previous_macro_reward/(2*.1)", "imu_valid", "light_valid", "frame_present"],
                         "initialization": "11 zero absent frames plus current local frame; invalid sensor values zero under validity masks.",
                         "excluded": ["assembly", "powered_hinge", "geometry", "passive_joint", "sun_position", "absolute_time", "case_id"],
                         "critic": "Same local stack as actor; no privileged critic inputs."},
        "native_reward": "Unmodified mounted light plus IMU rewards, integrated by the native world every .002s and summed over five .02s controls per decision; no shaping or reward normalization.",
        "reset": "New deterministic hanging native world with q=v=0, fresh motor state and sensor queues; curriculum case draws use their own recorded seed and schedule.",
        "safety": "The existing native governor runs every .02s. Any native veto or motor feedback flag ends a development opportunity; preserve the raw prefix and do not reset-rescue it. Source episodes may reset after safety termination and every such stop is counted.",
        "checkpoint_rule": "Final pilot checkpoint only, sealed before all development evaluations; no outcome-based checkpoint, seed, hyperparameter or case selection.",
        "untrained_baseline": "Matched architecture and source initialization seed 404, frozen without learning; reset the same per-case action RNG after construction as for the trained frozen branch. Report trained-minus-untrained native returns separately; this comparison adds no pass threshold.",
        "reserve_rule": "The twelve reserved cases are manifest data only in this pilot: no native construction, probes, policy rollout, adaptation or reward inspection. A later prospectively declared study is required before their exposure.",
    }


def _keys(value, expected, where):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{where} fields must be exactly {sorted(expected)}")


def _integer(value, low, high, where):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{where} must be an integer in [{low}, {high}]")
    return value


def _label(value, where, *, empty=False):
    if type(value) is not str or not (0 if empty else 1) <= len(value) <= 60:
        raise ValueError(f"{where} must contain {'0' if empty else '1'} to 60 characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where} must be valid UTF-8 text") from exc


def validate_case(value):
    """Independently check the relevant native v2 assembly and lamp contract."""
    _keys(value, {"assembly", "sun"}, "case")
    assembly = value["assembly"]
    required = {"schema", "name", "segments", "blocks", "sensor", "light"}
    if type(assembly) is not dict:
        raise ValueError("assembly must be an object")
    _keys(assembly, required | ({"powered_hinge"} if "powered_hinge" in assembly else set()), "assembly")
    if assembly["schema"] != "construction_kit_v2":
        raise ValueError("this curriculum requires construction_kit_v2")
    _label(assembly["name"], "assembly.name", empty=True)
    lengths = assembly["segments"]
    if type(lengths) is not list or len(lengths) != 2:
        raise ValueError("assembly.segments must contain exactly two lengths")
    for length in lengths:
        _integer(length, 2, 6, "segment length")
    _integer(assembly.get("powered_hinge", 0), 0, 1, "powered_hinge")

    def socket(part):
        segment = _integer(part["segment"], 0, 1, "part.segment")
        slot = _integer(part["slot"], 0, lengths[segment] - 1, "part.slot")
        side = _integer(part["side"], -1, 1, "part.side")
        if side == 0:
            raise ValueError("part.side must be -1 or 1")
        return segment, slot, side

    _keys(assembly["sensor"], {"segment", "slot", "side"}, "sensor")
    _keys(assembly["light"], {"segment", "slot", "side", "face"}, "light")
    occupied = {socket(assembly["sensor"])}
    light_socket = socket(assembly["light"])
    if light_socket in occupied:
        raise ValueError("light and IMU sockets overlap")
    occupied.add(light_socket)
    if _integer(assembly["light"]["face"], -1, 1, "light.face") == 0:
        raise ValueError("light.face must be -1 or 1")
    blocks = assembly["blocks"]
    if type(blocks) is not list or len(blocks) > 12:
        raise ValueError("blocks must be an array of at most 12 entries")
    reserved_ids = {"motor-0001", "imu-0001", "light-0001"}
    reserved_ids.update(f"segment-{segment}-cell-{slot}" for segment in (0, 1) for slot in range(6))
    identifiers = set()
    for block in blocks:
        _keys(block, {"id", "segment", "slot", "side"}, "block")
        identifier = block["id"]
        _label(identifier, "block.id")
        if identifier in reserved_ids or identifier in identifiers:
            raise ValueError("block.id must be unique and non-reserved")
        identifiers.add(identifier)
        location = socket(block)
        if location in occupied:
            raise ValueError("construction sockets overlap")
        occupied.add(location)
    sun = value["sun"]
    _keys(sun, {"position_m", "intensity_lux"}, "sun")
    position = sun["position_m"]
    if type(position) is not list or len(position) != 3:
        raise ValueError("sun.position_m must have three coordinates")
    for coordinate in position:
        if type(coordinate) not in (int, float) or not -2 <= coordinate <= 2 or not math.isfinite(coordinate):
            raise ValueError("sun coordinates must be finite and in [-2,2]")
    if position[1] != 0:
        raise ValueError("sun y must equal 0 in the planar fixture")
    intensity = sun["intensity_lux"]
    if type(intensity) not in (int, float) or not 0 <= intensity <= 1000 or not math.isfinite(intensity):
        raise ValueError("sun intensity must be finite and in [0,1000]")


def mechanics_identity(assembly):
    """Canonical physical assembly identity, ignoring labels and block IDs.

    Block IDs only address displayed parts; block socket placements determine
    mechanics. The default powered hinge is normalized to 0. The lamp is not part
    of an assembly identity, so changing it cannot disguise source/reserve overlap.
    """
    validate_case({"assembly": assembly,
                   "sun": {"position_m": [0., 0., 0.], "intensity_lux": 0.}})
    mechanics = {"schema": assembly["schema"], "segments": assembly["segments"],
                 "powered_hinge": assembly.get("powered_hinge", 0),
                 "blocks": sorted([block["segment"], block["slot"], block["side"]]
                                  for block in assembly["blocks"]),
                 "sensor": assembly["sensor"], "light": assembly["light"]}
    return json.dumps(mechanics, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_manifest(value):
    """Reject malformed, overlapping, reordered or prospectively altered cases/budgets."""
    expected = manifest()
    _keys(value, expected, "manifest")
    case_keys = ("source_cases", "development_cases", "reserved_cases")
    identities = {}
    for key in case_keys:
        cases = value[key]
        if type(cases) is not list or len(cases) != len(expected[key]):
            raise ValueError(f"{key} must contain exactly {len(expected[key])} conditions")
        identities[key] = set()
        conditions = set()
        for case in cases:
            validate_case(case)
            identity = mechanics_identity(case["assembly"])
            identities[key].add(identity)
            condition = (identity, tuple(case["sun"]["position_m"]), case["sun"]["intensity_lux"])
            if condition in conditions:
                raise ValueError(f"{key} contains duplicate physical conditions")
            conditions.add(condition)
    if identities["source_cases"] & identities["reserved_cases"]:
        raise ValueError("source and reserved whole-assembly mechanics overlap")
    if not identities["development_cases"] <= identities["source_cases"]:
        raise ValueError("pilot development assemblies must be known source mechanics")
    for key in case_keys:
        for actual, declared in zip(value[key], expected[key]):
            if (mechanics_identity(actual["assembly"]) != mechanics_identity(declared["assembly"])
                    or actual["sun"] != declared["sun"]):
                raise ValueError(f"{key} differs from the declared ordered physical conditions")
    metadata = {key: item for key, item in value.items() if key not in case_keys}
    declared_metadata = {key: item for key, item in expected.items() if key not in case_keys}
    try:
        actual_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest metadata must be finite JSON values") from exc
    if actual_bytes != json.dumps(declared_metadata, sort_keys=True, separators=(",", ":"), allow_nan=False):
        raise ValueError("manifest metadata, identities or budgets differ from the declaration")


class DeterministicSourceSampler:
    """Independent seeded cycles with complete source coverage and exposure counts."""

    def __init__(self, protocol, seed):
        validate_manifest(protocol)
        self.source_seed = _integer(seed, 0, 2 ** 32 - 1, "source seed")
        self.curriculum_seed = seed + protocol["sampling"]["curriculum_seed_offset"]
        self._random = random.Random(self.curriculum_seed)
        self._sources = copy.deepcopy(protocol["source_cases"])
        self._order = []
        self._position = 0
        self._counts = [0] * len(self._sources)
        self._digest = hashlib.sha256()
        self.draws = 0
        self.latest_index = None

    def __call__(self):
        if self._position == len(self._order):
            self._order = list(range(len(self._sources)))
            self._random.shuffle(self._order)
            self._position = 0
        index = self._order[self._position]
        self._position += 1
        self.draws += 1
        self.latest_index = index
        self._counts[index] += 1
        self._digest.update(f"{index}\n".encode("ascii"))
        return copy.deepcopy(self._sources[index])

    def exposure_report(self):
        return {"schema": SAMPLER_SCHEMA, "source_seed": self.source_seed,
                "curriculum_seed": self.curriculum_seed, "draws": self.draws,
                "completed_cycles": self.draws // len(self._sources),
                "current_cycle_position": self.draws % len(self._sources),
                "source_count": len(self._sources), "draw_counts": self._counts.copy(),
                "draw_index_sha256": self._digest.hexdigest()}
