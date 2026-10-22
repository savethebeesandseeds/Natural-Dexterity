"""Parametric, seam-walked Scout robe toile geometry. All coordinates are mm.

This is an original loose drop-shoulder drafting experiment, not a graded or
fit-tested commercial block. Illustrative defaults MUST be replaced with the
wearer's measurements and a toile checked before cutting the final cloth.

The closed ``seam`` polygon is a stitching line on joined edges, a final fold
line on hemmed edges, and the RAW/final boundary on bias-bound edges. Read each
piece's ``edge_finishes``; the ``outline`` already includes those allowances.
"""

from __future__ import annotations

import math
from typing import Any


# (illustrative default, minimum, maximum, human-readable measurement)
_MEASUREMENTS = {
    "chest": (680, 450, 1400, "Chest circumference over the clothes worn beneath."),
    "hip": (720, 450, 1500, "Fullest hip/seat circumference over underclothes."),
    "waist": (630, 400, 1500, "Waist circumference at the chosen sash level."),
    "shoulder_width": (310, 200, 600, "Back width between the anatomical shoulder points."),
    "neck_circumference": (300, 230, 550, "Comfortably measured circumference at the neck base."),
    "head_circumference": (540, 440, 700, "Around the fullest part of the head, above eyebrows."),
    "head_height": (285, 200, 450, "Vertical height from high shoulder/neck level to crown."),
    "shoulder_to_waist": (340, 220, 650, "High shoulder beside neck down to the chosen sash level."),
    "upper_arm": (230, 150, 650, "Fullest upper-arm circumference with arm relaxed."),
    "hand_circumference": (170, 120, 320, "Around knuckles with thumb tucked, for passage through cuff."),
    "wrist": (140, 100, 280, "Wrist circumference; recorded for possible future long-sleeve adjustment."),
    "sleeve_length": (180, 80, 850, "Outer sleeve: anatomical shoulder point to finished edge along arm."),
    "tunic_sleeve_length": (230, 80, 850, "Inner sleeve: anatomical shoulder point to finished edge along arm."),
    "robe_length": (750, 400, 1500, "High shoulder beside neck to finished outer hem. Default is a knee-length idea."),
    "tunic_length": (510, 350, 1100, "High shoulder beside neck to finished inner hem."),
    "chest_ease": (160, 100, 400, "Added outer chest circumference. Inner uses 40 mm less."),
    "hip_ease": (160, 100, 400, "Added outer hip circumference. Inner uses 40 mm less."),
    "sleeve_ease": (120, 60, 250, "Added outer upper-arm circumference at armhole. Inner uses 20 mm less."),
    "cuff_ease": (70, 40, 200, "Added to max(upper arm, hand) for these loose cuffs."),
    "neck_ease": (40, 20, 100, "Drafting addition to neck measurement for the open V-neck proportions."),
    "hood_ease": (40, 20, 100, "Added independently to head/3 hood depth and shoulder-to-crown height."),
    "wrap_overlap": (95, 40, 220, "Each front extends this far beyond centre front at sash/hem; total overlap is twice this."),
    "hem_flare": (35, 0, 100, "Outward flare added at each front/back side hem."),
    "seam_allowance": (10, 6, 20, "Joining seam allowance. Already included in CUT outlines."),
    "hem_allowance": (25, 20, 50, "Total double-turn hem allowance; first turn 10 mm, second remainder."),
    "binding_width": (10, 8, 15, "Finished double-fold bias binding width; cut strips four times this width."),
    "sash_width": (50, 30, 100, "Finished width of the short fastening sash."),
    "sash_overlap": (80, 50, 150, "End overlap for a low-profile hook-and-loop fastening."),
    "sash_ease": (40, 20, 100, "Added waist circumference for sash over garment layers."),
    "tabard_width": (85, 50, 160, "Finished width of each shoulder tabard."),
    "tabard_drop": (110, 40, 300, "Distance tabard extends below sash level, front and back."),
}

DEFAULT_PARAMS: dict[str, Any] = {key: values[0] for key, values in _MEASUREMENTS.items()}
DEFAULT_PARAMS.update(include_hood=True, include_tunic=True, include_tabards=True, include_sash=True)
PARAM_SPECS = {
    key: {"default": value[0], "min": value[1], "max": value[2], "unit": "mm", "description": value[3]}
    for key, value in _MEASUREMENTS.items()
}
for _key in ("include_hood", "include_tunic", "include_tabards", "include_sash"):
    PARAM_SPECS[_key] = {"default": True, "type": "boolean", "description": "Generate " + _key.removeprefix("include_") + " pieces."}


def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def polygon_area(points):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1])) / 2


def polyline_length(points):
    return sum(distance(a, b) for a, b in zip(points, points[1:]))


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _segments_cross(a, b, c, d):
    """Detect non-neighbour edge crossings, including touching/collinear edges."""
    eps = 1e-7
    r, s = _sub(b, a), _sub(d, c)
    den = _cross(r, s)
    if abs(den) < eps:
        if abs(_cross(_sub(c, a), r)) > eps:
            return False
        axis = 0 if abs(r[0]) >= abs(r[1]) else 1
        return min(max(a[axis], b[axis]), max(c[axis], d[axis])) >= max(min(a[axis], b[axis]), min(c[axis], d[axis])) - eps
    t, u = _cross(_sub(c, a), s) / den, _cross(_sub(c, a), r) / den
    return -eps <= t <= 1 + eps and -eps <= u <= 1 + eps


def assert_simple_polygon(points, name="polygon"):
    if len(points) < 3 or polygon_area(points) <= 0:
        raise ValueError(f"{name}: expected a nonempty clockwise-on-paper polygon")
    for i in range(len(points)):
        if distance(points[i], points[(i + 1) % len(points)]) < 1e-6:
            raise ValueError(f"{name}: zero length edge")
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            if _segments_cross(points[i], points[(i + 1) % len(points)], points[j], points[(j + 1) % len(points)]):
                raise ValueError(f"{name}: intersecting edges {i} and {j}; revise measurements/allowances")


def offset_polygon(points, allowances):
    """Offset each oriented edge outwards; mitre normal corners, bevel acute tips.

    Concave vertices use the actual offset-line intersection. An extreme
    concavity is rejected instead of filling it with a misleading bevel.
    Result is checked for self intersections; this is not a general clipper.
    """
    assert_simple_polygon(points, "stitch/fold boundary")
    if len(points) != len(allowances):
        raise ValueError("Each polygon edge must have exactly one allowance")
    shifted = []
    directions = []
    for a, b, allowance in zip(points, points[1:] + points[:1], allowances):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        directions.append((dx / length, dy / length))
        normal = (dy / length * allowance, -dx / length * allowance)
        shifted.append(((a[0] + normal[0], a[1] + normal[1]), (b[0] + normal[0], b[1] + normal[1])))
    result = []
    for i, vertex in enumerate(points):
        previous_end = shifted[i - 1][1]
        current_start = shifted[i][0]
        r, s = directions[i - 1], directions[i]
        den = _cross(r, s)
        if ((allowances[i - 1] == 0) != (allowances[i] == 0)) and abs(den) < 0.5:
            # Squarely terminate a joining/hem allowance where a raw bound
            # edge begins. A mitre between zero and positive allowances can
            # fold back over a short curved segment or gouge the neckline.
            candidates = [previous_end, current_start]
        elif abs(den) < 1e-10:
            candidates = [previous_end] if distance(previous_end, current_start) < 1e-7 else [previous_end, current_start]
        else:
            t = _cross(_sub(current_start, previous_end), s) / den
            meet = (previous_end[0] + t * r[0], previous_end[1] + t * r[1])
            cap = max(allowances[i - 1], allowances[i], 1) * 5
            if distance(meet, vertex) > cap:
                if min(allowances[i - 1], allowances[i]) == 0:
                    # At a zero-allowance bound edge, stop the joining
                    # allowance squarely at its sewing endpoint. Extending
                    # near-collinear offset lines would create a false long
                    # spike or gouge at the neck-to-wrap transition.
                    candidates = [previous_end, current_start]
                elif den < 0:
                    raise ValueError("Seam allowance overwhelms a concave corner; reduce allowance or revise draft")
                else:
                    candidates = [previous_end, current_start]
            else:
                candidates = [meet]
        for point in candidates:
            if not result or distance(point, result[-1]) > 1e-7:
                result.append(point)
    if distance(result[0], result[-1]) < 1e-7:
        result.pop()
    assert_simple_polygon(result, "cut outline")
    return result


def _check(name, a, b, tolerance=0.05):
    return {"name": name, "a_mm": round(a, 4), "b_mm": round(b, 4), "tolerance_mm": tolerance,
            "difference_mm": round(abs(a - b), 4), "passed": abs(a - b) <= tolerance}


def _line(points, kind, label=None):
    result = {"points": points, "kind": kind}
    if label:
        result["label"] = label
    return result


def _shaped_double_turn_hem(points, allowances, hem_edge, hem, first_turn=10.0):
    """Reflect adjacent side seams at BOTH folds of a double-turn hem.

    Simply offsetting a tapered cuff makes its allowance too narrow to turn
    back. A flared body hem has the converse problem. This envelope walks the
    side seams through the final fold, then reflects them at the first turn.
    The returned envelope is offset only on joining edges, never on the raw
    edge of the already included hem extension.
    """
    a, b = points[hem_edge], points[(hem_edge + 1) % len(points)]
    edge = _sub(b, a)
    edge_length = distance(a, b)
    normal = (edge[1] / edge_length, -edge[0] / edge_length)
    prev = points[(hem_edge - 1) % len(points)]
    following = points[(hem_edge + 2) % len(points)]
    in_direction, out_direction = _sub(a, prev), _sub(following, b)

    def extension(vertex, direction, depth, lateral_depth):
        projection = direction[0] * normal[0] + direction[1] * normal[1]
        if abs(projection) < 1e-8:
            raise ValueError("Cannot shape a hem whose adjacent edge is parallel to the hem")
        drift = (normal[0] - direction[0] / projection, normal[1] - direction[1] / projection)
        return (vertex[0] + depth * normal[0] + lateral_depth * drift[0],
                vertex[1] + depth * normal[1] + lateral_depth * drift[1])

    second_turn = hem - first_turn
    a1 = extension(a, in_direction, second_turn, second_turn)
    a2 = extension(a, in_direction, hem, hem - 2 * first_turn)
    b1 = extension(b, out_direction, second_turn, second_turn)
    b2 = extension(b, out_direction, hem, hem - 2 * first_turn)
    extra = [a1, a2, b2, b1]
    envelope = points[:hem_edge + 1] + extra + points[hem_edge + 1:]
    previous_sa = allowances[(hem_edge - 1) % len(points)]
    next_sa = allowances[(hem_edge + 1) % len(points)]
    envelope_sa = allowances[:hem_edge] + [previous_sa, previous_sa, 0, next_sa, next_sa] + allowances[hem_edge + 1:]
    guide_lines = [_line([a1, b1], "fold", "FIRST TURN 10 MM"),
                   _line([a, a1, a2], "seam_extension"),
                   _line([b, b1, b2], "seam_extension")]
    return offset_polygon(envelope, envelope_sa), guide_lines


def _piece(piece_id, name, cut_count, cut, material, group, seam, finishes, grain, *, lines=None, labels=None, notes=None):
    allowances = [item[1] for item in finishes]
    hem_edges = [i for i, item in enumerate(finishes) if item[0] == "hem"]
    additional_lines = []
    if len(hem_edges) == 1:
        edge = hem_edges[0]
        outline, additional_lines = _shaped_double_turn_hem(seam, allowances, edge, allowances[edge])
    else:
        outline = offset_polygon(seam, allowances)
    # Normalization includes the cut allowance; every cut coordinate is positive.
    shift_x = 1 - min(p[0] for p in outline)
    shift_y = 1 - min(p[1] for p in outline)
    def point(p):
        return [round(p[0] + shift_x, 5), round(p[1] + shift_y, 5)]
    transformed_lines = [dict(item, points=[point(p) for p in item["points"]]) for item in (lines or []) + additional_lines]
    transformed_labels = []
    for label in labels or []:
        x, y = point((label["x"], label["y"]))
        transformed_labels.append(dict(label, x=x, y=y))
    result = {"id": piece_id, "name": name, "cut": cut, "cut_count": cut_count,
              "material": material, "group": group, "outline": [point(p) for p in outline],
              "seam": [point(p) for p in seam], "lines": transformed_lines,
              "labels": transformed_labels, "grain": [point(p) for p in grain],
              "notes": notes or [], "edge_allowances": allowances,
              "edge_finishes": [{"edge": i, "type": finish[0], "allowance_mm": finish[1], "label": finish[2]} for i, finish in enumerate(finishes)]}
    result["width_mm"] = round(max(p[0] for p in result["outline"]) + 1, 4)
    result["height_mm"] = round(max(p[1] for p in result["outline"]) + 1, 4)
    result["cut_area_mm2"] = round(polygon_area(outline), 4)
    return result


def _validate_params(given):
    unknown = sorted(set(given) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError("Unknown parameter(s): " + ", ".join(unknown))
    params = dict(DEFAULT_PARAMS, **given)
    for key, spec in PARAM_SPECS.items():
        value = params[key]
        if spec.get("type") == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{key} must be a finite number between {spec['min']} and {spec['max']} mm")
    if params["robe_length"] < params["shoulder_to_waist"] + 100:
        raise ValueError("robe_length must be at least shoulder_to_waist + 100 mm")
    if params["include_tunic"] and params["tunic_length"] < params["shoulder_to_waist"] + 80:
        raise ValueError("tunic_length must be at least shoulder_to_waist + 80 mm")
    if params["tabard_drop"] > (params["tunic_length"] if params["include_tunic"] else params["robe_length"]) - params["shoulder_to_waist"] - 20:
        if params["include_tabards"]:
            raise ValueError("tabard_drop must finish at least 20 mm above the inner/outer hem")
    return params


def generate(params=None):
    """Return JSON-serializable millimetre geometry, derivations and seam checks."""
    p = _validate_params(params or {})
    pieces, checks, warnings = [], [], []
    sa, hem = p["seam_allowance"], p["hem_allowance"]
    JOIN = lambda label: ("sew", sa, label)
    HEM = lambda label: ("hem", hem, label)
    BIND = lambda label: ("bind", 0, label)
    derived: dict[str, Any] = {"illustrative_defaults_only": p == DEFAULT_PARAMS, "binding": {}}
    body_records = {}

    for group in ["robe"] + (["tunic"] if p["include_tunic"] else []):
        outer = group == "robe"
        title = "Outer robe" if outer else "Inner tunic"
        material = "Charcoal outer cloth" if outer else "Slate-blue inner cloth"
        ease_reduction = 0 if outer else 40
        circumference = max(p["chest"] + p["chest_ease"] - ease_reduction,
                            p["hip"] + p["hip_ease"] - ease_reduction)
        q = circumference / 4
        length = p["robe_length"] if outer else p["tunic_length"]
        neck_scale = p["neck_circumference"] + p["neck_ease"]
        nw, bd, fd = neck_scale / 6, neck_scale * 0.065, neck_scale * 0.42
        slope = p["shoulder_width"] * 0.055
        opening = p["upper_arm"] + p["sleeve_ease"] - (0 if outer else 20)
        depth = opening / 2
        arm_y = slope + depth
        flare = p["hem_flare"] if outer else p["hem_flare"] / 2
        overlap = p["wrap_overlap"] if outer else p["wrap_overlap"] * 0.8
        waist_y = p["shoulder_to_waist"]
        shoulder_drop = q - p["shoulder_width"] / 2
        anatomical_sleeve = p["sleeve_length"] if outer else p["tunic_sleeve_length"]
        sleeve_length = anatomical_sleeve - shoulder_drop
        cuff = max(p["upper_arm"], p["hand_circumference"]) + p["cuff_ease"]
        if q <= nw + 35:
            raise ValueError(f"{group}: neckline leaves less than 35 mm shoulder seam")
        if shoulder_drop < 0:
            raise ValueError(f"{group}: body width is narrower than shoulders; increase circumference ease")
        if sleeve_length < 60:
            raise ValueError(f"{group}: sleeve length after subtracting shoulder drop is under 60 mm; increase sleeve length or reduce body ease")
        if waist_y <= max(fd, arm_y) + 25:
            raise ValueError(f"{group}: sash level must be more than 25 mm below neckline and underarm")
        if cuff > opening:
            warnings.append(f"{title}: cuff flares wider than upper sleeve; check sleeve clearance and desired silhouette in toile.")
        if anatomical_sleeve > 350:
            warnings.append(f"{title}: longer sleeve still uses a loose upper-arm-based cuff; adjust only after a toile.")
        if slope + depth > 0.75 * waist_y:
            warnings.append(f"{title}: low underarm; check overhead reach before final fabric.")
        hood_join = outer and p["include_hood"]
        neck_finish = JOIN("hood attachment") if hood_join else BIND("neckline bias binding")

        # Symmetrical full back, deliberately no CUT ON FOLD ambiguity.
        back = [(0, slope), (q - nw, 0), (q, bd), (q + nw, 0),
                (2 * q, slope), (2 * q, arm_y), (2 * q + flare, length),
                (-flare, length), (0, arm_y)]
        back_finishes = [JOIN("left shoulder"), neck_finish, neck_finish,
                         JOIN("right shoulder"), JOIN("right arm opening"),
                         JOIN("right side"), HEM("bottom hem"), JOIN("left side"), JOIN("left arm opening")]
        back_lines = [_line([(q, bd + 20), (q, length - 30)], "centre", "CENTRE BACK"),
                      _line([(20, waist_y), (2 * q - 20, waist_y)], "placement", "SASH LEVEL"),
                      _line([(0, arm_y), (8, arm_y)], "notch", "UNDERARM"),
                      _line([(2 * q - 8, arm_y), (2 * q, arm_y)], "notch")]
        pieces.append(_piece(group + "_back", title + " - full back", 1, "CUT 1 FULL PIECE; no fold", material, group,
                             back, back_finishes, [(q * 0.6, length * 0.40), (q * 0.6, length * 0.70)],
                             lines=back_lines, labels=[{"x": q * 0.25, "y": length * 0.28, "text": "FULL BACK / CUT 1"}],
                             notes=["Trace the full piece once, not on a fold.",
                                    "Join shoulders and side seams at seam allowance; stop at underarm marks.",
                                    "Staystitch the angular back neckline. Clip inside the V allowance only as needed, stopping short of stitching." if hood_join else "Bind neckline raw/final edge; zero extra allowance on those two edges.",
                                    f"Bottom hem: turn 10 mm, then {hem - 10:g} mm to the marked finished line."]))

        # Front centre neckline endpoint is distinct from the wrap extension.
        front = [(nw, 0), (q, slope), (q, arm_y), (q + flare, length),
                 (-overlap, length), (-overlap, waist_y), (0, fd)]
        front_finishes = [JOIN("shoulder"), JOIN("arm opening"), JOIN("side"), HEM("bottom hem"),
                          BIND("vertical front opening"), BIND("diagonal wrap opening"), neck_finish]
        front_lines = [_line([(0, fd + 20), (0, length - 30)], "centre", "CENTRE FRONT"),
                       _line([(-overlap + 15, waist_y), (q - 15, waist_y)], "placement", "SASH LEVEL"),
                       _line([(q - 8, arm_y), (q, arm_y)], "notch", "UNDERARM"),
                       _line([(0, fd), (8, fd + 2)], "notch", "HOOD ENDS HERE" if hood_join else "NECK/WRAP CORNER")]
        pieces.append(_piece(group + "_front", title + " - wrap front", 2, "CUT 2 MIRRORED (one left, one right)", material, group,
                             front, front_finishes, [(q * 0.55, length * 0.40), (q * 0.55, length * 0.72)],
                             lines=front_lines, labels=[{"x": q * 0.13, "y": length * 0.28, "text": "FRONT / MIRRORED PAIR"}],
                             notes=["Cut one left and one right; do not cut two identical handed fronts.",
                                    "Front openings have zero allowance and receive double-fold bias binding.",
                                    "The diagonal below the marked neck endpoint is the wrap edge; do not sew it to the hood." if hood_join else "Bind neckline and front opening as one continuous edge after joining shoulders.",
                                    f"Bottom hem: turn 10 mm, then {hem - 10:g} mm to the marked finished line.",
                                    "Position short internal hook-and-loop tabs during fitting; no long internal ties are required."]))

        cuff_top, cuff_bottom = (opening - cuff) / 2, (opening + cuff) / 2
        sleeve = [(0, 0), (sleeve_length, cuff_top), (sleeve_length, cuff_bottom), (0, opening)]
        sleeve_finishes = [JOIN("first underarm edge"), HEM("cuff hem"), JOIN("second underarm edge"), JOIN("armhole attachment")]
        sleeve_lines = [_line([(0, depth), (10, depth)], "notch", "SHOULDER SEAM"),
                        _line([(sleeve_length * 0.25, depth), (sleeve_length * 0.75, depth)], "centre", "TOP OF SLEEVE")]
        pieces.append(_piece(group + "_sleeve", title + " - flat sleeve", 2, "CUT 2 FULL PIECES; no fold", material, group,
                             sleeve, sleeve_finishes, [(sleeve_length * 0.2, depth * 0.80), (sleeve_length * 0.8, depth * 0.80)],
                             lines=sleeve_lines, labels=[{"x": sleeve_length * 0.22, "y": depth * 0.65, "text": "SLEEVE / CUT 2"}],
                             notes=["The long body edge joins the front and back arm openings without gathering.",
                                    "Match centre notch on attachment edge to shoulder seam; remaining ends are underarms.",
                                    "Join the two equal sloping underarm edges to close each sleeve.",
                                    f"Cuff hem: turn 10 mm, then {hem - 10:g} mm. The cut extension is reflected at both folds to match the sleeve taper.",
                                    "Follow the marked underarm seam extensions through the shaped hem allowance; do not trim them to a straight continuation."]))

        back_neck_half = distance(back[1], back[2])
        front_neck = distance(front[6], front[0])
        binding_body = 2 * (distance(front[4], front[5]) + distance(front[5], front[6]))
        if not hood_join:
            binding_body += 2 * (back_neck_half + front_neck)
        body_records[group] = {"back_neck_half": back_neck_half, "front_neck": front_neck,
                               "neckline_half": back_neck_half + front_neck, "binding_length": binding_body}
        checks.extend([
            _check(title + " shoulders: front / back", distance(front[0], front[1]), distance(back[3], back[4])),
            _check(title + " side seams: front / back", distance(front[2], front[3]), distance(back[5], back[6])),
            _check(title + " sleeve attachment / front+back arm opening", distance(sleeve[3], sleeve[0]), distance(front[1], front[2]) + distance(back[4], back[5])),
            _check(title + " sleeve underside pair", distance(sleeve[0], sleeve[1]), distance(sleeve[2], sleeve[3])),
            _check(title + " sleeve centre / front arm opening", opening / 2, distance(front[1], front[2])),
        ])
        derived[group] = {"finished_base_circumference_mm": circumference,
                          "finished_overlap_at_sash_mm": 2 * overlap, "hem_circumference_excluding_overlap_mm": circumference + 4 * flare,
                          "shoulder_drop_each_side_mm": shoulder_drop, "sleeve_seam_length_along_centre_mm": sleeve_length,
                          "upper_sleeve_circumference_mm": opening, "cuff_circumference_mm": cuff,
                          "neckline_seam_length_mm": 2 * (back_neck_half + front_neck),
                          "hood_base_half_mm": back_neck_half + front_neck if hood_join else None,
                          "front_neck_depth_mm": fd, "back_neck_depth_mm": bd, "neck_half_width_mm": nw,
                          "armhole_depth_below_shoulder_tip_mm": depth, "shoulder_slope_mm": slope,
                          "quarter_body_mm": q, "length_mm": length}

    if p["include_hood"]:
        record = body_records["robe"]
        base = record["neckline_half"]
        height = p["head_height"] + p["hood_ease"]
        hood_depth = p["head_circumference"] / 3 + p["hood_ease"]
        radius = min(height, hood_depth) * 0.18
        # Start at back neck, travel up/over crown to forehead, then face/base.
        hood = [(0, height), (0, radius)]
        for step in range(1, 5):
            angle = math.pi + step * math.pi / 8
            hood.append((radius + radius * math.cos(angle), radius + radius * math.sin(angle)))
        hood.append((hood_depth - radius, 0))
        for step in range(1, 5):
            angle = -math.pi / 2 + step * math.pi / 8
            hood.append((hood_depth - radius + radius * math.cos(angle), radius + radius * math.sin(angle)))
        hood.append((base, height))
        face_index = len(hood) - 2
        finishes = [JOIN("crown / centre-back panel seam") for _ in hood]
        finishes[face_index] = BIND("hood face opening")
        finishes[-1] = JOIN("neckline attachment")
        shoulder_x = record["back_neck_half"]
        hood_lines = [_line([(shoulder_x, height), (shoulder_x, height - 9)], "notch", "SHOULDER"),
                      _line([(0, height), (8, height - 8)], "notch", "CENTRE BACK"),
                      _line([(base, height), (base - 9, height - 8)], "notch", "FRONT NECK END")]
        pieces.append(_piece("robe_hood", "Outer robe - hood side", 2, "CUT 2 MIRRORED (one left, one right)", "Charcoal outer cloth", "robe",
                             hood, finishes, [(hood_depth * 0.5, height * 0.25), (hood_depth * 0.5, height * 0.72)],
                             lines=hood_lines, labels=[{"x": hood_depth * 0.20, "y": height * 0.43, "text": "HOOD / MIRRORED PAIR"}],
                             notes=["Join the two panels around the back and crown; face edge remains open.",
                                    "Bottom attachment edge runs from centre-back to the marked front neckline endpoint.",
                                    "Match shoulder notch to body shoulder seam; hood face and garment wrap edges meet at the front neck endpoint.",
                                    "Bias-bind the face opening with zero extra edge allowance. No drawcord channel.",
                                    "Crown is drafted as short straight segments; sew along the supplied polyline for the first toile."]))
        face_length = distance(hood[face_index], hood[face_index + 1])
        body_records["robe"]["binding_length"] += 2 * face_length
        checks.extend([_check("Hood base / complete garment neckline", 2 * distance(hood[-1], hood[0]), 2 * record["neckline_half"]),
                       _check("Hood CB-to-shoulder notch / back half-neck", shoulder_x, record["back_neck_half"]),
                       _check("Hood shoulder-to-front / front neck", base - shoulder_x, record["front_neck"])])
        derived["hood"] = {"panel_base_mm": base, "panel_height_mm": height, "panel_upper_depth_mm": hood_depth,
                           "panel_face_edge_mm": face_length, "panel_crown_seam_mm": polyline_length(hood[:face_index + 1]),
                           "shoulder_notch_from_cb_mm": shoulder_x}

    for group, record in body_records.items():
        # Joined square ends consume 2*SA per splice. There is a 100mm handling
        # reserve. Joining instructions deliberately use straight seam joins.
        strip_length = 1000.0
        needed = record["binding_length"]
        count = max(1, math.ceil((needed + 100 - 2 * sa) / (strip_length - 2 * sa)))
        usable = strip_length * count - 2 * sa * (count - 1)
        width = p["binding_width"] * 4
        strip = [(0, 0), (strip_length, 0), (strip_length, width), (0, width)]
        binding_lines = [_line([(5, width * f), (strip_length - 5, width * f)], "fold", "FOLD" if f == 0.5 else None) for f in (0.25, 0.5, 0.75)]
        pieces.append(_piece(group + "_bias_strip", ("Outer" if group == "robe" else "Inner") + " - bias binding strip", count,
                             f"CUT {count} STRIPS ON TRUE BIAS; may piece shorter strips", "Charcoal outer cloth" if group == "robe" else "Slate-blue inner cloth", group,
                             strip, [("raw", 0, "raw strip edge") for _ in strip],
                             [(strip_length * 0.55, width * 0.15), (strip_length * 0.55 + width * 0.7, width * 0.85)],
                             lines=binding_lines, labels=[], notes=[
                                 f"Cut width {width:g} mm on TRUE BIAS (45 degrees to straight grain), no added allowance.",
                                 f"Supply at least {needed + 100:.1f} mm usable binding after joins; shown strips supply {usable:.1f} mm.",
                                 f"Join short ends with {sa:g} mm straight seams; press open. Shorter strips need extra length for each additional join.",
                                 f"Fold outer long raw edges to centre; then fold in half, yielding {p['binding_width']:g} mm finished binding.",
                                 "Enclose the zero-allowance garment edges; do not trim or move the marked boundary inward.",
                                 "This rectangle is the RAW CUT strip; dashed quarter lines are folds, not joining seams."]))
        derived["binding"][group] = {"finished_edge_length_mm": needed, "handling_reserve_mm": 100,
                                      "cut_width_mm": width, "strip_length_mm": strip_length, "strip_count": count,
                                      "usable_length_after_joins_mm": usable}

    if p["include_tabards"]:
        length = 2 * (p["shoulder_to_waist"] + p["tabard_drop"])
        width = p["tabard_width"]
        rectangle = [(0, 0), (width, 0), (width, length), (0, length)]
        tabard_lines = [_line([(0, length / 2), (width, length / 2)], "placement", "SHOULDER MIDPOINT"),
                        _line([(0, p["tabard_drop"]), (width, p["tabard_drop"])], "placement", "SASH LEVEL"),
                        _line([(0, length - p["tabard_drop"]), (width, length - p["tabard_drop"])], "placement", "SASH LEVEL")]
        pieces.append(_piece("tabard", "Shoulder tabard", 4, "CUT 4 = two tabards, each with two fabric layers", "Charcoal outer cloth", "accessories",
                             rectangle, [JOIN("bagged tabard perimeter") for _ in rectangle], [(width / 2, length * 0.25), (width / 2, length * 0.70)],
                             lines=tabard_lines, labels=[], notes=["Sew in pairs right sides together; leave a turning gap, turn and close gap.",
                                 "One finished tabard drapes over each shoulder with equal front/back drop.",
                                 "Mark actual shoulder and sash levels during fitting; secure tabards with a few removable tacks."]))
        derived["tabard"] = {"finished_width_mm": width, "finished_length_mm": length, "finished_quantity": 2}

    if p["include_sash"]:
        length = p["waist"] + p["sash_ease"] + p["sash_overlap"]
        width = p["sash_width"]
        rectangle = [(0, 0), (length, 0), (length, width), (0, width)]
        closure = p["sash_overlap"]
        sash_lines = [_line([(closure, 0), (closure, width)], "placement", "OVERLAP"),
                      _line([(length - closure, 0), (length - closure, width)], "placement", "OVERLAP")]
        pieces.append(_piece("sash", "Short fastening sash", 2, "CUT 2 = one sash with two fabric layers", "Charcoal outer cloth", "accessories",
                             rectangle, [JOIN("bagged sash perimeter") for _ in rectangle], [(length * 0.30, width / 2), (length * 0.70, width / 2)],
                             lines=sash_lines, labels=[], notes=["Sew right sides together around the perimeter; leave a turning gap, turn and close gap.",
                                 f"Finished overlap is {closure:g} mm; position low-profile hook-and-loop on opposite faces after checking fit over all layers.",
                                 "Keep fastening ends short. This clothing sash is not a load-bearing device harness."]))
        derived["sash"] = {"finished_length_mm": length, "finished_width_mm": width,
                           "closed_circumference_mm": length - closure, "overlap_mm": closure}

    for piece in pieces:
        assert_simple_polygon(piece["seam"], piece["id"] + " stitch/fold")
        assert_simple_polygon(piece["outline"], piece["id"] + " cut")
        if polygon_area(piece["outline"]) + 1e-3 < polygon_area(piece["seam"]):
            raise ValueError(piece["id"] + ": allowance outline lost area")
    if any(not item["passed"] for item in checks):
        raise ValueError("Seam walk failed")
    derived["cut_piece_count"] = sum(piece["cut_count"] for piece in pieces)
    derived["net_cut_area_m2"] = sum(piece["cut_area_mm2"] * piece["cut_count"] for piece in pieces) / 1_000_000
    return {"schema_version": 1, "units": "mm", "params": p, "pieces": pieces,
            "derived": derived, "seam_checks": checks, "warnings": warnings,
            "fit_status": "Unfitted experimental toile draft; seam agreement is a geometry check, not proof of fit.",
            "legend": {"outline": "CUT line; allowances included", "seam": "SEW on joined edges / FOLD on hemmed edges / RAW FINAL edge where bias bound",
                       "fold": "First hem turn or bias strip fold", "seam_extension": "Continue joining seam through shaped hem allowance", "centre": "Reference centre line", "placement": "Placement guide; adjust during fitting", "notch": "Transfer as a mark; do not cut into seam line"},
            "formula_notes": ["Body base circumference = max(chest + chest_ease, hip + hip_ease); inner subtracts 40 mm ease.",
                              "Shoulder drop = quarter body circumference - half anatomical shoulder width; subtract drop from sleeve measurement.",
                              "Arm opening on each body panel = half upper-sleeve circumference. Flat sleeve attachment = sum of both openings.",
                              "Neck half-width = (neck + neck_ease)/6; back V depth = 0.065*(neck+ease); front V depth = 0.42*(neck+ease). These are design proportions, not an anthropometric sizing standard.",
                              "Each hood base = measured back half-neck seam + measured front-neck seam; shoulder notch marks their junction.",
                              "Hood upper depth = head circumference/3 + hood_ease; height = shoulder-to-crown height + hood_ease. Check hood vision and turning in a toile.",
                              "Wrap overlap below sash = twice each front extension. Tabard length = 2*(shoulder_to_waist + tabard_drop).",
                              "No allowance is added to bias-bound edges. Binding width is 4 times finished width, and supply includes joins plus 100 mm handling reserve.",
                              "Tapered cuff and flared body hem extensions reflect adjacent seams at the final fold and again at the first 10 mm turn.",
                              "Net cut area is piece area only; it is NOT fabric yardage and excludes layout waste, bias placement and shrinkage."]}


if __name__ == "__main__":
    import json
    import sys
    supplied = json.loads(open(sys.argv[1], encoding="utf-8").read()) if len(sys.argv) > 1 else {}
    print(json.dumps(generate(supplied), indent=2))
