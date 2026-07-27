Excellent. This chapter answers one of the most common questions about Vision Transformers:

> Why do we split the image into patches at all? Why not feed the entire image directly into the Transformer?



The answer lies in the computational complexity of self-attention and the need to convert a 2D image into a sequence of tokens.

By the end of this chapter, you'll understand one of the most important design choices in Vision Transformers.


---

Part IV – Vision Transformers (ViT)

Chapter 12 — Patch Embedding: Turning Images into Transformer Tokens


---

Learning Objectives

By the end of this lesson, you will understand:

Why images are split into patches

Why flattening is necessary

How linear projection creates patch embeddings

Why ViT uses a patch size such as 16×16

The trade-offs of different patch sizes

How patch embedding differs from CNN feature extraction



---

12.1 The Challenge

Transformers were originally designed for text.

A sentence looks like:

"I love AI"

↓

Word1
Word2
Word3

A transformer expects a sequence of tokens.

An image, however, is very different.

Example:

224 × 224 × 3

This is a 3D tensor, not a sequence.

So the first challenge is:

> How do we convert an image into something a Transformer understands?




---

12.2 Why Not Use Every Pixel as a Token?

Suppose we decide:

> One pixel = One token.



For a 224×224 image:

Total pixels:

\[
224 \times 224 = 50,176
\]

The transformer would receive:

50,176 tokens

Remember:

Self-attention compares every token with every other token.

That means:

\[
50,176^2
\approx
2.5 \text{ billion}
\]

attention comparisons for a single attention head in one layer.

This is far too expensive for practical models.


---

12.3 The Computational Problem

Suppose we have:

N tokens

Self-attention complexity is:

\[
O(N^2)
\]

Let's compare.

Tokens	Attention Comparisons (≈ N²)

196	38,416
784	614,656
50,176	≈ 2.5 billion


Notice how quickly the cost grows.

This is the main reason ViT does not use one token per pixel.


---

12.4 The Patch Idea

Instead of treating every pixel as a token,

we group neighboring pixels.

Example:

16 × 16

One patch now contains:

256 pixels

Instead of:

256 tokens

we create:

1 token

This dramatically reduces the sequence length.


---

12.5 Dividing the Image

Suppose the image is:

224 × 224

Patch size:

16 × 16

Along each side:

\[
224/16 = 14
\]

Therefore:

14 patches horizontally

14 patches vertically

Total:

\[
14 \times 14 = 196
\]

Instead of 50,176 tokens,

we now have only:

196 tokens

This makes self-attention practical.


---

12.6 Visualizing the Patches

Original image:

+----------------------+
|                      |
|      🐶 on grass      |
|                      |
+----------------------+

After splitting:

+----+----+----+----+
| P1 | P2 | P3 | P4 |
+----+----+----+----+
| P5 | P6 | P7 | P8 |
+----+----+----+----+
| P9 |... |... |... |
+----+----+----+----+

Each patch becomes one transformer token.


---

12.7 What's Inside a Patch?

Suppose:

Patch size:

16 × 16

RGB image:

3 channels

Every pixel has:

Red

Green

Blue

Total numbers:

\[
16 \times16\times3
=
768
\]

Each patch is therefore a small cube of data.


---

12.8 Why Flatten the Patch?

A transformer expects vectors.

But each patch is:

16 ×16 ×3

This is a three-dimensional tensor.

We flatten it into:

768 numbers

Example:

Before:

16

×

16

×

3

After:

[0.21, 0.73, 0.15, ...]

A long vector.

Flattening doesn't change the values—it simply changes how they are arranged in memory.


---

12.9 Why Isn't Flattening Enough?

A flattened patch still contains raw pixel values.

Example:

[124, 126, 121, 220, ...]

Raw pixels are not ideal features for a transformer.

Just as words are converted into embeddings,

patches are converted into patch embeddings.


---

12.10 Linear Projection

Suppose one flattened patch contains:

768 numbers

The model learns a matrix:

\[
W
\]

to compute:

\[
\text{Patch Embedding}
=
\text{Flattened Patch}
\times
W
\]

For example:

768

↓

Linear Layer

↓

768

or

768

↓

Linear Layer

↓

1024

depending on the model architecture.

This learned projection maps raw pixel values into a feature space that is more useful for attention.


---

12.11 Is This Similar to Word Embeddings?

Yes, conceptually.

Language:

Word

↓

Embedding Vector

Vision:

Patch

↓

Embedding Vector

The difference is:

Words are discrete symbols looked up in an embedding table.

Image patches are continuous pixel values transformed by a learned linear layer.



---

12.12 Why 16×16?

This is one of the most important design decisions.

Let's compare.


---

Very Small Patches (8×8)

Image:

224×224

Number of patches:

\[
28 \times28
=
784
\]

Advantages:

Fine details preserved.

Better localization.


Disadvantages:

Much longer token sequence.

Higher memory usage.

Slower attention.



---

Medium Patches (16×16)

Number of patches:

196

Advantages:

Good balance between detail and efficiency.

Widely used in the original ViT.



---

Large Patches (32×32)

Number of patches:

\[
7\times7=49
\]

Advantages:

Very fast.

Low memory usage.


Disadvantages:

Lose fine details.

Small objects may disappear within a patch.



---

12.13 Trade-Off Table

Patch Size	Tokens (224×224 Image)	Advantages	Disadvantages

8×8	784	Captures fine details	High computational cost
16×16	196	Good balance	May miss some very small structures
32×32	49	Fast and memory efficient	Coarse representation; small details may be lost


Choosing the patch size is a trade-off between accuracy and efficiency.


---

12.14 Does ViT Really Use Flattening?

Yes, but there is an interesting implementation detail.

The original ViT paper describes patch extraction followed by a linear projection.

Many practical implementations achieve the same result using a convolution layer with:

kernel size = patch size

stride = patch size


For example:

Conv2D

Kernel = 16×16

Stride = 16

Each convolution output corresponds to one non-overlapping patch embedding.

Mathematically, this is equivalent to applying the same learned linear projection independently to every flattened patch, while often being more efficient on modern hardware.


---

12.15 Patch Embedding in Vision Transformers

The complete process is:

Input Image
      │
      ▼
Split into 16×16 Patches
      │
      ▼
Flatten Each Patch
      │
      ▼
Linear Projection
      │
      ▼
Patch Embeddings
      │
      ▼
Add Position Embeddings
      │
      ▼
Transformer Encoder

This is the bridge between raw pixels and transformer tokens.


---

12.16 Connection to SAM

The Segment Anything Model also begins by converting the input image into patch embeddings.

However, unlike a classification-focused ViT, SAM keeps rich spatial feature representations because later components need to predict pixel-level segmentation masks, not just a single class label.

This means preserving useful spatial information throughout the encoder is especially important.


---

Common Misconceptions

❌ "Flattening extracts image features."

No.

Flattening only rearranges the pixel values into a vector. The learned linear projection is what begins transforming raw pixels into useful features.


---

❌ "Patch size doesn't matter."

It matters a great deal. Smaller patches usually preserve more detail but require more computation. Larger patches are more efficient but may lose fine-grained information.


---

❌ "Patch embedding is the same as a CNN feature extractor."

Not exactly.

Patch embedding is primarily an input projection step. A CNN feature extractor repeatedly applies convolutional filters across many layers to build hierarchical representations.


---

Key Takeaways

Images are split into patches because self-attention scales quadratically with the number of tokens.

Flattening converts each 3D image patch into a 1D vector.

A learned linear projection transforms raw pixel vectors into patch embeddings.

Patch size controls the balance between computational efficiency and visual detail.

Many implementations use a convolution with kernel size = stride = patch size as an efficient way to implement patch embedding.



---

Practice Questions

Conceptual

1. Why can't a ViT use one token per pixel for a typical image?


2. Why is flattening alone insufficient?


3. How does the linear projection differ from flattening?


4. What are the advantages and disadvantages of using 8×8 versus 32×32 patches?


5. Why can a convolution layer implement patch embedding efficiently?



Numerical

1. A 256 × 256 RGB image is divided into 16 × 16 patches.

How many patches are created?

How many values are in each flattened patch?



2. A 384 × 384 RGB image is divided into 32 × 32 patches.

How many transformer tokens are produced before adding the [CLS] token?





---

Chapter Summary

Patch embedding is the crucial first step that enables a transformer to process images. Instead of treating every pixel as a separate token, Vision Transformers divide the image into fixed-size patches, flatten each patch into a vector, and apply a learned linear projection to produce patch embeddings. This dramatically reduces the sequence length, making self-attention computationally feasible while preserving enough visual information for effective learning. The choice of patch size is a fundamental trade-off between computational cost and the ability to capture fine image details.


---

Next Lesson

Now that we understand how images become tokens, we'll examine what the attention maps inside a Vision Transformer actually learn.

We'll answer questions such as:

Do different attention heads focus on different parts of an image?

How does a ViT learn object boundaries without convolutions?

What do early, middle, and late transformer layers attend to?

How do attention maps evolve during processing?

How do these learned attention patterns help models like SAM perform accurate segmentation?


This chapter will connect the mathematics of attention with the visual behavior observed inside real Vision Transformer models.
