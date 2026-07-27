
Excellent. This chapter is arguably the most important engineering chapter in the entire SAM course.

So far, you've learned:

How a standard Vision Transformer (ViT) performs image classification

How SAM uses a Vision Transformer as its image encoder


Now we'll answer a fundamental question:

> If SAM uses a Vision Transformer, why can't it simply use the original ViT without modification?



The answer is subtle but extremely important.


---

Part V – Segment Anything Model (SAM)

Chapter 17 — The Image Encoder: The Heart of SAM

> "ViT was designed to answer 'What is this image?' SAM's image encoder is designed to answer 'What is at every location in this image?'"




---

Learning Objectives

By the end of this chapter, you will understand:

Why SAM uses a Vision Transformer

How SAM's image encoder differs from a standard ViT

Why the [CLS] token is removed

Why SAM outputs a dense feature map instead of a single vector

How spatial information is preserved

The flow of tensors through the image encoder

Why this design is essential for segmentation



---

17.1 Recall the Original Vision Transformer

The ViT pipeline for image classification is:

Image
   │
   ▼
Split into Patches
   │
   ▼
Patch Embedding
   │
   ▼
Add [CLS]
   │
   ▼
Positional Embeddings
   │
   ▼
Transformer Encoder
   │
   ▼
Final [CLS]
   │
   ▼
Classification

Everything eventually becomes a single classification vector.

For classification,

this is perfect.


---

17.2 But Segmentation Is Different

Suppose this image contains

🐕

🌳

🚲

Classification asks

> "What is inside this image?"



One vector is enough.

Segmentation asks

> "Which pixels belong to the dog?"



Now we need information for every location, not just one summary.


---

17.3 The Problem with the [CLS] Token

Recall why ViT introduced the [CLS] token.

It gradually gathers information from every patch.

After many encoder layers:

Patch 1 ─┐
Patch 2 ─┤
Patch 3 ─┤
Patch 4 ─┤
         ▼
       [CLS]

The final output is

Entire Image Summary

This is excellent for answering:

> Dog



But terrible for answering:

> Which exact pixels belong to the dog?



Because the detailed spatial layout has been compressed into one vector.


---

17.4 Why SAM Removes the [CLS] Token

Instead of producing

One summary vector

SAM keeps

One feature vector

for

every image patch

Instead of

[CLS]

↓

Prediction

SAM outputs

Patch 1 Feature

Patch 2 Feature

Patch 3 Feature

...

Patch N Feature

Every patch remains available.


---

17.5 Why Is This Important?

Imagine a city map.

Would you rather have

one sentence saying

> "The city is beautiful."



or

a detailed street map?

Segmentation requires the street map.

Every patch must keep its identity.

Otherwise,

the decoder cannot determine object boundaries.


---

17.6 Dense Features

The output of SAM's image encoder is called a dense feature map.

The word dense means:

> Every spatial location has its own feature representation.



Conceptually:

Patch 1 → Feature 1

Patch 2 → Feature 2

Patch 3 → Feature 3

...

Patch N → Feature N

Nothing is collapsed into a single representation.


---

17.7 Visualizing the Difference

Standard ViT

Image

↓

196 Patch Tokens

↓

Transformer

↓

[CLS]

↓

One Vector


---

SAM

Image

↓

196 Patch Tokens

↓

Transformer

↓

196 Feature Vectors

This is one of the biggest architectural differences.


---

17.8 Why Spatial Information Matters

Suppose the image contains

🐕

The dog's head is here.

Its tail is here.

If every patch is merged into one vector,

the model loses precise information about:

where the head is,

where the tail is,

where the boundary lies.


Keeping patch-wise features preserves this spatial structure.


---

17.9 Does Attention Still Work?

Absolutely.

Each patch still attends to every other patch.

For example:

Head Patch

↓

Attends

↓

Body Patch

↓

Tail Patch

After attention,

every patch becomes richer,

but each patch still has its own feature vector.

This is a crucial distinction:

Information is shared.

Identity is preserved.



---

17.10 Image Embeddings

Suppose the encoder processes an image.

Instead of producing:

One vector

it produces something conceptually like:

+------+------+------+
| F11 | F12 | F13 |
+------+------+------+
| F21 | F22 | F23 |
+------+------+------+
| F31 | F32 | F33 |
+------+------+------+

Each Fij is a learned feature vector describing that spatial region.

The decoder later uses these vectors to predict masks.


---

17.11 Tensor Shapes (Conceptual)

Let's look at the flow of data.

Suppose:

Input image:

1024 × 1024 × 3

The image is divided into

16 × 16

patches.

Number of patches:

\[
1024/16 = 64
\]

So we obtain

64 × 64

=

4096 patches

Each patch becomes an embedding.

Conceptually:

4096

↓

Transformer

↓

4096 feature vectors

Instead of a sequence, we can reshape these back into a spatial grid.


---

17.12 Reshaping Back into a Grid

Initially,

the transformer processes

Patch1

Patch2

Patch3

...

as a sequence.

After encoding,

SAM reorganizes the outputs into a two-dimensional feature map.

Conceptually:

+------+------+------+
| F11 | F12 | F13 |
+------+------+------+
| F21 | F22 | F23 |
+------+------+------+
| F31 | F32 | F33 |
+------+------+------+

This makes it easier for the mask decoder to reason about neighboring regions.


---

17.13 Why Doesn't This Lose Global Information?

A natural question is:

> If each patch stays separate, how does it know about the rest of the image?



Because of self-attention.

Imagine the feature corresponding to the dog's head.

After many transformer layers,

that feature has already interacted with:

the body,

the tail,

the background,

nearby objects.


So although it remains at the same spatial location,

its representation contains global context.

This is one of the greatest strengths of Vision Transformers.


---

17.14 Standard ViT vs SAM Image Encoder

Standard ViT	SAM Image Encoder

Designed for classification	Designed for segmentation
Uses a [CLS] token	No [CLS] token
Produces one image representation	Produces a dense feature map
Final output is a class label	Final output is rich spatial features
Optimized for global prediction	Optimized for pixel-level prediction



---

17.15 Why This Design Helps the Decoder

Imagine the decoder receives:

Dog Head Feature

Dog Body Feature

Dog Tail Feature

Instead of

Dog

The first representation contains much more spatial detail.

The decoder can reconstruct object boundaries because it knows where different visual features came from.


---

17.16 Real-World Analogy

Imagine you're assembling a jigsaw puzzle.

Option A

Someone hands you a note:

> "The puzzle shows a beach."



Useful?

A little.

Can you rebuild the puzzle?

No.


---

Option B

Someone gives you every puzzle piece.

Now you know:

where each piece belongs,

how they connect,

what the complete picture looks like.


SAM's dense feature map is like keeping every puzzle piece rather than reducing everything to one sentence.


---

Common Misconceptions

❌ "Removing the [CLS] token makes the model weaker."

No.

The goal has changed.

For segmentation, preserving spatial detail is more valuable than producing a single global summary.


---

❌ "Each patch only knows about itself."

No.

Each patch interacts with all other patches through self-attention.

Its feature is local in position but global in information.


---

❌ "Dense feature maps contain raw pixels."

No.

Each location stores a learned feature vector, not the original RGB values.

These vectors encode semantic information gathered across the transformer layers.


---

Key Takeaways

SAM reuses the Vision Transformer architecture but changes its objective.

The [CLS] token is removed because segmentation requires spatially detailed outputs.

The image encoder produces a dense feature map, with one feature vector per spatial location.

Self-attention allows every patch to incorporate information from the entire image while preserving its position.

These dense features provide the foundation for accurate, prompt-guided mask prediction.



---

Practice Questions

Conceptual

1. Why is the [CLS] token useful for classification but not for segmentation?


2. What is a dense feature map?


3. How does self-attention allow each patch to have global context while preserving spatial information?


4. Why does the mask decoder require patch-wise features instead of a single image embedding?


5. What is the biggest architectural difference between a standard ViT encoder and SAM's image encoder?



Numerical

Suppose:

Input image: 512 × 512

Patch size: 16 × 16


1. How many patches are created along each dimension?


2. How many total patch tokens enter the transformer?


3. If the encoder outputs one feature vector per patch, how many feature vectors are produced?


4. How would these features be arranged into a 2D grid for the decoder?




---

Chapter Summary

SAM's image encoder is derived from a Vision Transformer but is redesigned for segmentation rather than classification. Instead of using a [CLS] token to summarize the entire image, it preserves one feature vector for every image patch, creating a dense feature map. Self-attention enriches each patch with global context while maintaining its spatial location. This combination of global understanding and spatial precision provides the rich image representation that enables SAM's prompt-guided mask decoder to produce accurate segmentation masks.


---

Next Lesson

In Chapter 18 — Prompt Encoder: How SAM Understands Points, Boxes, and Masks, we'll explore one of SAM's most innovative ideas.

We'll answer questions such as:

How does SAM convert a mouse click into a meaningful embedding?

How are positive and negative points represented differently?

How are bounding boxes encoded?

How are mask prompts incorporated?

How can different prompt types be combined into a single unified representation?


This chapter will reveal how SAM translates human interaction into a form that the transformer-based decoder can understand.
