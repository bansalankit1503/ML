# ML From Scratch

> A public, beginner-first journey through the mathematics and code behind modern machine learning.

[![Documentation](https://img.shields.io/badge/docs-MkDocs%20Material-526CFE?logo=materialformkdocs&logoColor=white)](https://bansalankit1503.github.io/ML/)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![GitHub Pages](https://img.shields.io/badge/deploy-GitHub%20Pages-222222?logo=github)](https://github.com/bansalankit1503/ML/actions)
[![Visitors](https://img.shields.io/badge/visitors-coming%20soon-lightgrey)](#visitor-counter-placeholder)

This repository is becoming a documentation-first Machine Learning portfolio. Each lesson explains the idea from first principles, derives the relevant mathematics, and pairs it with typed, tested NumPy code.

## Start learning

When the site is deployed, begin at the [documentation home](https://bansalankit1503.github.io/ML/) or read [Chapter 1: Vectors](docs/module-01-linear-algebra/chapter-01-vectors.md) directly.

## Progress

**Published chapters:** `1`  
`[██░░░░░░░░░░░░░░░░░░]` First chapter complete

| Status | Module | Current focus |
| --- | --- | --- |
| :white_check_mark: | 01 · Linear Algebra | [Chapter 1 — Vectors](docs/module-01-linear-algebra/chapter-01-vectors.md) |
| :hourglass_flowing_sand: | 02 · Calculus | Planned |
| :hourglass_flowing_sand: | 03 · Probability & Statistics | Planned |
| :hourglass_flowing_sand: | 04 · Deep Learning | Planned |
| :hourglass_flowing_sand: | 05 · Transformers | Planned |
| :hourglass_flowing_sand: | 06 · Vision Transformers | Planned |
| :hourglass_flowing_sand: | 07 · Segment Anything Model | Planned |

### Completed

- [x] Module 01, Chapter 01 — Vectors

### Up next

- [ ] Module 01, Chapter 02 — Matrices

## Learning philosophy

The goal is understanding rather than memorisation. Every chapter follows the same loop:

1. Build intuition with a concrete example.
2. Express that idea with precise mathematics.
3. Implement it with small, readable Python functions.
4. Test the implementation and connect it to an ML use case.
5. Practice with exercises and interview questions.

## Local development

This project targets Python 3.12. Create a virtual environment, install the project tooling, and serve the documentation site:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdocs serve
```

Run the numerical tests with:

```powershell
pytest
```

## GitHub Pages deployment

The workflow in `.github/workflows/deploy.yml` builds the site on every push to `main` using Python 3.12. After pushing:

1. Open the repository **Settings → Pages**.
2. Choose **GitHub Actions** as the build and deployment source.
3. Wait for the workflow to finish; the live site URL is shown in the deployment.

## Screenshot placeholder

Add a screenshot of the deployed site here after the first GitHub Pages deployment.

<a id="visitor-counter-placeholder"></a>

## Repository layout

```text
docs/         Documentation source for MkDocs
src/          Reusable, typed NumPy implementations
notebooks/    Interactive learning notebooks
tests/        Unit tests for the implementations
INITIAL/      Earlier, preserved learning notes
```
