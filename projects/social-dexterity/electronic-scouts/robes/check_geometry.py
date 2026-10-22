"""Geometric regression checks; no claim about garment fit or wear safety.

Run: python check_geometry.py
Checks representative small/large drafts, optional components, edge allowances,
100 deterministic varied drafts, and actual generated double-fold seam marks.
"""

import itertools
import math
import random

import pattern_geometry as g


SMALL = dict(chest=590, hip=610, waist=560, shoulder_width=270,
             neck_circumference=270, head_circumference=500, head_height=250,
             shoulder_to_waist=290, upper_arm=185, hand_circumference=150,
             sleeve_length=150, tunic_sleeve_length=190, robe_length=650,
             tunic_length=430, wrap_overlap=80, tabard_drop=100)
LARGE = dict(chest=820, hip=860, waist=780, shoulder_width=360,
             neck_circumference=340, head_circumference=570, head_height=320,
             shoulder_to_waist=430, upper_arm=280, hand_circumference=195,
             sleeve_length=220, tunic_sleeve_length=280, robe_length=900,
             tunic_length=640)


def reflect(point, line):
    """Reflect a point across a line using its perpendicular projection."""
    a, b = line
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)
    foot = (a[0] + t * dx, a[1] + t * dy)
    return (2 * foot[0] - point[0], 2 * foot[1] - point[1])


def distance_to_line(point, line):
    a, b = line
    return abs((b[0] - a[0]) * (a[1] - point[1]) - (a[0] - point[0]) * (b[1] - a[1])) / g.distance(a, b)


def check_foldback(piece):
    """Physically fold the generated raw seam endpoints twice.

    They must land on the actual adjacent body/sleeve stitching line, proving
    the turn-up has neither a width deficit nor surplus at the side seam.
    This inspects the public generated geometry, independent of the drafting
    helper that constructed the extension.
    """
    hem_edges = [item["edge"] for item in piece["edge_finishes"] if item["type"] == "hem"]
    if not hem_edges:
        return
    assert len(hem_edges) == 1
    edge = hem_edges[0]
    seam = piece["seam"]
    final_fold = [seam[edge], seam[(edge + 1) % len(seam)]]
    first_fold = next(item["points"] for item in piece["lines"] if item.get("label") == "FIRST TURN 10 MM")
    extensions = [item["points"] for item in piece["lines"] if item["kind"] == "seam_extension"]
    assert len(extensions) == 2
    adjacent = [[seam[(edge - 1) % len(seam)], seam[edge]],
                [seam[(edge + 1) % len(seam)], seam[(edge + 2) % len(seam)]]]
    for extension, stitching in zip(extensions, adjacent):
        folded = reflect(reflect(extension[-1], first_fold), final_fold)
        assert distance_to_line(folded, stitching) < 0.001, (piece["id"], folded, stitching)


def run():
    cases = [{}, SMALL, LARGE]
    flags = ["include_hood", "include_tunic", "include_tabards", "include_sash"]
    cases.extend(dict(zip(flags, mask)) for mask in itertools.product([False, True], repeat=4))
    for key, values in {"seam_allowance": [6, 20], "hem_allowance": [20, 50],
                        "hem_flare": [0, 100], "neck_ease": [20, 100],
                        "hood_ease": [20, 100], "binding_width": [8, 15]}.items():
        cases.extend({key: value} for value in values)
    rng = random.Random(17)
    for _ in range(100):
        t = rng.random()
        varied = {key: SMALL[key] * (1 - t) + LARGE[key] * t for key in sorted(set(SMALL) & set(LARGE))}
        varied.update(seam_allowance=rng.uniform(6, 20), hem_allowance=rng.uniform(20, 50),
                      hem_flare=rng.uniform(0, 80), wrap_overlap=rng.uniform(50, 140))
        cases.append(varied)
    checks = 0
    for index, params in enumerate(cases):
        data = g.generate(params)
        assert all(check["passed"] for check in data["seam_checks"]), index
        for piece in data["pieces"]:
            g.assert_simple_polygon(piece["seam"], piece["id"])
            g.assert_simple_polygon(piece["outline"], piece["id"])
            assert all(math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0 for x, y in piece["outline"])
            check_foldback(piece)
            checks += 1
        for binding in data["derived"]["binding"].values():
            assert binding["usable_length_after_joins_mm"] >= binding["finished_edge_length_mm"] + 100
    invalid = [dict(chest=float("nan")), dict(chest=True), dict(unknown=1),
               dict(sleeve_length=80), dict(robe_length=400), dict(hem_allowance=15),
               dict(include_hood="yes")]
    for params in invalid:
        try:
            g.generate(params)
        except ValueError:
            pass
        else:
            raise AssertionError(("Accepted invalid inputs", params))
    print(f"PASS: {len(cases)} valid parameter combinations, {checks} piece geometries, {len(invalid)} invalid inputs rejected.")
    print("Seam walks, nonintersecting positive outlines, binding reserve and actual two-fold hem seam agreement passed.")
    print("These checks do not validate fit. Make and assess a toile.")


if __name__ == "__main__":
    run()
