"""Small, well-tested vector operations for the first linear algebra chapter.

The functions in this module deliberately validate their inputs. Catching a shape
mistake early makes numerical code much easier to reason about, especially when
the same ideas later appear as feature vectors, embeddings, and gradients.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]

__all__ = [
    "add",
    "as_vector",
    "cosine_similarity",
    "dot",
    "magnitude",
    "normalize",
    "plot_vectors",
    "project_onto",
    "scale",
    "subtract",
]


def as_vector(values: ArrayLike) -> Vector:
    """Convert a numeric one-dimensional sequence to a finite float vector.

    Args:
        values: A non-empty, one-dimensional sequence or NumPy array.

    Returns:
        A copy of ``values`` as a one-dimensional ``float64`` NumPy array.

    Raises:
        ValueError: If the input is not a finite, non-empty one-dimensional vector.

    Examples:
        >>> as_vector([1, 2, 3])
        array([1., 2., 3.])
    """
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Vector values must be numeric.") from error

    if vector.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional vector, received {vector.ndim} dimensions."
        )
    if vector.size == 0:
        raise ValueError("A vector must contain at least one component.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Vector components must all be finite numbers.")

    return vector.copy()


def add(left: ArrayLike, right: ArrayLike) -> Vector:
    """Add two vectors component by component.

    Args:
        left: The first vector.
        right: The second vector with the same number of components.

    Returns:
        The component-wise sum of ``left`` and ``right``.

    Raises:
        ValueError: If the vectors have different dimensions.
    """
    left_vector, right_vector = _validated_pair(left, right)
    return left_vector + right_vector


def subtract(left: ArrayLike, right: ArrayLike) -> Vector:
    """Subtract ``right`` from ``left`` component by component.

    Args:
        left: The vector to subtract from.
        right: The vector to subtract.

    Returns:
        The component-wise difference ``left - right``.

    Raises:
        ValueError: If the vectors have different dimensions.
    """
    left_vector, right_vector = _validated_pair(left, right)
    return left_vector - right_vector


def scale(vector: ArrayLike, scalar: float) -> Vector:
    """Multiply every component of a vector by one finite scalar.

    Args:
        vector: The vector to scale.
        scalar: The finite number that stretches, shrinks, or reverses the vector.

    Returns:
        The scaled vector.

    Raises:
        ValueError: If ``scalar`` is not a finite real number.
    """
    vector_array = as_vector(vector)
    try:
        scalar_value = float(scalar)
    except (TypeError, ValueError) as error:
        raise ValueError("A scalar must be a finite real number.") from error

    if not np.isfinite(scalar_value):
        raise ValueError("A scalar must be a finite real number.")
    return vector_array * scalar_value


def dot(left: ArrayLike, right: ArrayLike) -> float:
    """Calculate the dot product of two equal-length vectors.

    The result is a single number rather than another vector.

    Args:
        left: The first vector.
        right: The second vector.

    Returns:
        ``sum(left[i] * right[i])`` as a Python float.

    Raises:
        ValueError: If the vectors have different dimensions.
    """
    left_vector, right_vector = _validated_pair(left, right)
    return float(np.dot(left_vector, right_vector))


def magnitude(vector: ArrayLike) -> float:
    """Return the Euclidean length (L2 norm) of a vector.

    Args:
        vector: The vector whose length is required.

    Returns:
        The non-negative Euclidean length of ``vector``.
    """
    return float(np.linalg.vector_norm(as_vector(vector)))


def normalize(vector: ArrayLike) -> Vector:
    """Return a unit vector pointing in the same direction as ``vector``.

    Args:
        vector: A non-zero vector.

    Returns:
        A vector with magnitude one and the same direction as ``vector``.

    Raises:
        ValueError: If ``vector`` is the zero vector.
    """
    vector_array = as_vector(vector)
    vector_magnitude = magnitude(vector_array)
    if vector_magnitude == 0.0:
        raise ValueError("The zero vector has no direction and cannot be normalized.")
    return vector_array / vector_magnitude


def cosine_similarity(left: ArrayLike, right: ArrayLike) -> float:
    """Measure directional similarity between two non-zero vectors.

    A result of ``1`` means the vectors point in the same direction, ``0`` means
    they are perpendicular, and ``-1`` means they point in opposite directions.

    Args:
        left: The first non-zero vector.
        right: The second non-zero vector.

    Returns:
        The cosine of the angle between the vectors.

    Raises:
        ValueError: If either vector is the zero vector or dimensions differ.
    """
    left_vector, right_vector = _validated_pair(left, right)
    denominator = magnitude(left_vector) * magnitude(right_vector)
    if denominator == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    # Numerical round-off can produce values a tiny amount outside [-1, 1].
    return float(np.clip(np.dot(left_vector, right_vector) / denominator, -1.0, 1.0))


def project_onto(vector: ArrayLike, basis: ArrayLike) -> Vector:
    """Project one vector onto a non-zero basis vector.

    Args:
        vector: The vector to project.
        basis: The non-zero direction receiving the projection.

    Returns:
        The component of ``vector`` that points along ``basis``.

    Raises:
        ValueError: If dimensions differ or ``basis`` is the zero vector.
    """
    vector_array, basis_array = _validated_pair(vector, basis)
    basis_length_squared = dot(basis_array, basis_array)
    if basis_length_squared == 0.0:
        raise ValueError("Cannot project onto the zero vector.")
    return (dot(vector_array, basis_array) / basis_length_squared) * basis_array


def plot_vectors(
    *vectors: ArrayLike,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one or more two-dimensional vectors from the origin.

    Args:
        *vectors: One or more vectors with exactly two components.
        labels: Optional labels, one for each vector.
        ax: An existing Matplotlib axes; a new figure is created when omitted.

    Returns:
        The Matplotlib figure and axes containing the plot.

    Raises:
        ValueError: If no vectors are given, a vector is not two-dimensional, or
            the labels do not match the number of vectors.
    """
    if not vectors:
        raise ValueError("Provide at least one vector to plot.")

    parsed_vectors = [as_vector(vector) for vector in vectors]
    if any(vector.size != 2 for vector in parsed_vectors):
        raise ValueError("plot_vectors only supports vectors with exactly two components.")

    if labels is not None and len(labels) != len(parsed_vectors):
        raise ValueError("Provide exactly one label for each vector.")
    resolved_labels = list(labels) if labels is not None else [
        f"v{i + 1}" for i in range(len(parsed_vectors))
    ]

    if ax is None:
        figure, axes = plt.subplots(figsize=(7, 6))
    else:
        axes = ax
        figure = axes.figure

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (vector, label) in enumerate(zip(parsed_vectors, resolved_labels)):
        color = palette[index % len(palette)]
        axes.quiver(
            0,
            0,
            vector[0],
            vector[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.009,
            label=label,
        )
        axes.annotate(
            label,
            xy=(vector[0], vector[1]),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

    endpoints = np.vstack(parsed_vectors)
    limit = max(1.0, float(np.max(np.abs(endpoints))) + 1.0)
    axes.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x", ylabel="y")
    axes.axhline(0, color="0.55", linewidth=0.8)
    axes.axvline(0, color="0.55", linewidth=0.8)
    axes.grid(True, alpha=0.25)
    axes.set_aspect("equal", adjustable="box")
    axes.legend(loc="upper left")
    figure.tight_layout()
    return figure, axes


def _validated_pair(left: ArrayLike, right: ArrayLike) -> tuple[Vector, Vector]:
    """Return two valid vectors after checking that their dimensions agree."""
    left_vector = as_vector(left)
    right_vector = as_vector(right)
    if left_vector.shape != right_vector.shape:
        raise ValueError(
            "Vectors must have the same dimension; "
            f"received {left_vector.size} and {right_vector.size} components."
        )
    return left_vector, right_vector
