"""Matplotlib assets used by the linear algebra documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .vectors import add, plot_vectors


def save_vector_operations_figure(output_path: str | Path) -> Path:
    """Create an SVG showing two vectors, their sum, and the addition rule.

    Args:
        output_path: Destination for the image. Parent directories are created.

    Returns:
        The resolved output path.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    vector_u = np.array([3.0, 1.0])
    vector_v = np.array([1.0, 2.0])
    total = add(vector_u, vector_v)

    figure, axes = plot_vectors(
        vector_u,
        vector_v,
        total,
        labels=("u = [3, 1]", "v = [1, 2]", "u + v = [4, 3]"),
    )
    axes.plot(
        [vector_u[0], total[0]],
        [vector_u[1], total[1]],
        color="C1",
        linestyle="--",
        linewidth=1.8,
        label="translated v",
    )
    axes.set_title("Vector addition: place the second arrow at the first arrow's tip")
    axes.legend(loc="upper left")
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output.resolve()


def main() -> None:
    """Render the Chapter 1 documentation image from the command line."""
    parser = argparse.ArgumentParser(description="Render the vector addition diagram.")
    parser.add_argument(
        "--output",
        default="docs/assets/images/vector-operations.svg",
        help="Path for the generated SVG (default: %(default)s).",
    )
    arguments = parser.parse_args()
    output = save_vector_operations_figure(arguments.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
