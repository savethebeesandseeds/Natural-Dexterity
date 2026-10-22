"""Numerical guarantees of the single shared, deterministic audition PCA."""
import math

import pytest

from web_radio.semantic_projection import FIT_EXTENT, pca


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def test_repeated_fit_is_identical_and_positions_use_one_shared_basis():
    vectors = [[1, 2, 0, 1], [2, 1, 3, 0], [-1, 0, 2, 1], [0, 2, -2, 4], [1, 2, 0, 1]]
    fit = pca(vectors)
    assert fit == pca(vectors)
    assert fit["positions"][0] == fit["positions"][-1]
    for raw, position in zip(vectors, fit["positions"]):
        norm = math.hypot(*raw)
        centered = [value / norm - mean for value, mean in zip(raw, fit["mean"])]
        expected = [dot(centered, axis) * fit["scale"] for axis in fit["basis"]]
        assert position == pytest.approx(expected, abs=1e-12)
    assert max(abs(v) for row in fit["positions"] for v in row) == pytest.approx(FIT_EXTENT)
    for axis in zip(*fit["positions"]):
        assert math.fsum(axis) == pytest.approx(0, abs=1e-12)


def test_identical_vectors_and_single_vector_have_zero_rank_and_coordinates():
    for vectors in ([[1, 2, 3]] * 5, [[1, 2, 3]], [[0, 0, 0]] * 3):
        fit = pca(vectors)
        assert fit["positions"] == [[0.0, 0.0, 0.0]] * len(vectors)
        assert fit["basis"] == [[0.0, 0.0, 0.0]] * 3
        assert fit["diagnostics"]["rank"] == 0


def test_rank_one_has_only_one_axis_and_preserves_identical_endpoints():
    fit = pca([[1, 0], [0, 1], [1, 0], [0, 1]])
    assert fit["diagnostics"]["rank"] == 1
    assert fit["positions"][0] == fit["positions"][2]
    assert fit["positions"][1] == fit["positions"][3]
    assert all(row[1:] == [0.0, 0.0] for row in fit["positions"])
    assert fit["positions"][0][0] == pytest.approx(-fit["positions"][1][0])
    assert abs(fit["positions"][0][0]) == pytest.approx(FIT_EXTENT)


def test_retained_basis_is_orthonormal_and_sign_rule_is_canonical():
    vectors = [[1, 2, 0, 1], [2, 1, 3, 0], [-1, 0, 2, 1],
               [0, 2, -2, 4], [3, -2, 1, 1], [0, -1, -2, 1]]
    fit = pca(vectors)
    assert fit["diagnostics"]["rank"] >= 3
    for i, axis in enumerate(fit["basis"]):
        assert dot(axis, axis) == pytest.approx(1.0, abs=1e-12)
        pivot = max(range(len(axis)), key=lambda feature: abs(axis[feature]))
        assert axis[pivot] > 0
        for previous in fit["basis"][:i]:
            assert dot(axis, previous) == pytest.approx(0.0, abs=1e-12)
    assert fit["diagnostics"]["remaining_off_diagonal_max"] <= 1e-12


def test_positive_rescaling_of_individual_inputs_does_not_change_mapping():
    vectors = [[1, 2, 3], [2, -1, 4], [-3, 2, 1], [4, 1, -2]]
    scaled = [[value * factor for value in row] for row, factor in zip(vectors, [2, 8, 0.5, 16])]
    for original, rescaled in zip(pca(vectors)["positions"], pca(scaled)["positions"]):
        assert original == pytest.approx(rescaled, abs=1e-12)


@pytest.mark.parametrize("vectors", [
    [], [[]], [[1], [1, 2]], [[math.nan]], [[math.inf]], [[True]],
    [[0.0] * 3073], [[1.0]] * 513,
])
def test_invalid_or_unbounded_inputs_are_rejected(vectors):
    with pytest.raises(ValueError):
        pca(vectors)


def test_four_components_preserve_xyz_and_share_the_same_scale():
    vectors = [[float(i == j) * sign for j in range(4)] for i in range(4) for sign in (-1, 1)]
    three = pca(vectors)
    four = pca(vectors, dimensions=4)
    assert [row[:3] for row in four["positions"]] == three["positions"]
    assert four["basis"][:3] == three["basis"]
    assert four["mean"] == three["mean"]
    assert four["scale"] == three["scale"]
    assert four["diagnostics"]["rank"] == 4
    assert four["diagnostics"]["retained_eigenvalues"][3] > 0
    assert max(abs(row[3]) for row in four["positions"]) > 0.1
    assert max(abs(row[3]) for row in four["positions"]) <= 1
    for i, axis in enumerate(four["basis"]):
        assert dot(axis, axis) == pytest.approx(1, abs=1e-12)
        for previous in four["basis"][:i]:
            assert dot(axis, previous) == pytest.approx(0, abs=1e-12)
    assert four == pca(vectors, dimensions=4)


def test_four_components_keep_rank_deficient_axes_zero():
    fit = pca([[1, 0], [0, 1], [1, 0]], dimensions=4)
    assert all(row[1:] == [0.0, 0.0, 0.0] for row in fit["positions"])
    assert len(fit["basis"]) == 4
    identical = pca([[1, 2, 3]] * 4, dimensions=4)
    assert identical["positions"] == [[0.0] * 4] * 4


@pytest.mark.parametrize("dimensions", [0, 2, 5, True, 3.0])
def test_pca_dimension_must_be_exactly_three_or_four(dimensions):
    with pytest.raises(ValueError):
        pca([[1, 2], [2, 1]], dimensions=dimensions)
