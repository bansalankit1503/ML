"""Tests for the beginner-facing vector implementation."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ml_from_scratch.linear_algebra.vectors import (  # noqa: E402
    add,
    as_vector,
    cosine_similarity,
    dot,
    magnitude,
    normalize,
    plot_vectors,
    project_onto,
    scale,
    subtract,
)


@pytest.mark.parametrize("values", ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3])))
def test_as_vector_converts_numeric_sequences(values: object) -> None:
    result = as_vector(values)

    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "values",
    [1, [], [[1, 2]], [1, np.nan], [np.inf]],
)
def test_as_vector_rejects_invalid_vectors(values: object) -> None:
    with pytest.raises(ValueError):
        as_vector(values)


def test_add_subtract_and_scale_are_component_wise() -> None:
    np.testing.assert_array_equal(add([1, 2], [3, -1]), np.array([4.0, 1.0]))
    np.testing.assert_array_equal(subtract([1, 2], [3, -1]), np.array([-2.0, 3.0]))
    np.testing.assert_array_equal(scale([1, -2], -0.5), np.array([-0.5, 1.0]))


@pytest.mark.parametrize("function", [add, subtract, dot, cosine_similarity, project_onto])
def test_pair_operations_reject_mismatched_dimensions(function: object) -> None:
    with pytest.raises(ValueError, match="same dimension"):
        function([1, 2], [1, 2, 3])  # type: ignore[operator]


def test_scale_rejects_non_finite_scalar() -> None:
    with pytest.raises(ValueError, match="finite"):
        scale([1, 2], float("inf"))


def test_dot_and_magnitude_match_hand_calculated_values() -> None:
    assert dot([1, 2, 3], [4, -5, 6]) == pytest.approx(12.0)
    assert magnitude([3, 4]) == pytest.approx(5.0)


def test_normalize_returns_a_unit_vector() -> None:
    result = normalize([3, 4])

    np.testing.assert_allclose(result, np.array([0.6, 0.8]))
    assert magnitude(result) == pytest.approx(1.0)


def test_normalize_rejects_the_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        normalize([0, 0])


def test_cosine_similarity_handles_parallel_and_orthogonal_vectors() -> None:
    assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 7]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([1, 0], [0, 0])


def test_projection_is_parallel_to_the_basis() -> None:
    result = project_onto([3, 4], [1, 0])

    np.testing.assert_allclose(result, np.array([3.0, 0.0]))
    assert dot(result, [0, 1]) == pytest.approx(0.0)


def test_projection_rejects_the_zero_basis() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        project_onto([3, 4], [0, 0])


def test_plot_vectors_returns_a_figure_and_axes() -> None:
    figure, axes = plot_vectors([1, 0], [0, 1], labels=("x", "y"))

    assert figure is axes.figure
    assert axes.get_xlabel() == "x"
    assert axes.get_ylabel() == "y"
    assert len(axes.collections) == 2
    figure.canvas.draw()


def test_plot_vectors_validates_dimensions_and_labels() -> None:
    with pytest.raises(ValueError, match="two components"):
        plot_vectors([1, 2, 3])
    with pytest.raises(ValueError, match="one label"):
        plot_vectors([1, 2], labels=("first", "second"))
