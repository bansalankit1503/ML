# About this project

**ML From Scratch** is a public documentation website for learning machine learning from first principles. It is intentionally structured like a technical documentation project rather than a loose collection of notes.

## What "from scratch" means here

It does not mean avoiding useful libraries forever. It means understanding the smallest correct version of an idea before hiding it behind a framework:

- derive the mathematics;
- implement the core operation with NumPy;
- test the implementation;
- connect it to a real machine-learning application.

## Scope

The learning path starts with linear algebra and calculus, then moves through probability, deep learning, transformers, Vision Transformers, and Segment Anything.

The repository also contains an `INITIAL/` directory of earlier, preserved notes. Those files are not part of the new documentation navigation; the MkDocs site is the curated learning path going forward.

## Status

The project is actively being built chapter by chapter. The first published lesson is [Vectors](module-01-linear-algebra/chapter-01-vectors.md).

## How the site is published

The site uses MkDocs Material and deploys through GitHub Actions to GitHub Pages. See the setup instructions in the repository [README](../README.md).
