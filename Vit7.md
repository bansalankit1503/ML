Excellent. This chapter introduces one of the most unique innovations of the Segment Anything Model (SAM).

Traditional segmentation models only accept an image as input.

SAM, however, accepts instructions from the user.

The natural question is:

> How can a neural network understand something as simple as a mouse click?



A click is just two numbers:

(x, y)

How does that become something meaningful for a transformer?

Let's find out.


---

Part V – Segment Anything Model (SAM)

Chapter 18 — Prompt Encoder: How SAM Understands Points, Boxes, and Masks

> "The Prompt Encoder is the translator between human intentions and neural network representations."




---

Learning Objectives

By the end of this chapter, you will understand:

What a prompt is in SAM

Why prompts must be encoded

The different types of prompts SAM supports

How point prompts are represented

How positive and negative points differ

How box prompts are encoded

How mask prompts are encoded

How all prompt types are converted into a common embedding space



---

18.1 What Is a Prompt?

A prompt is simply a hint that tells SAM what object you want to segment.

Unlike traditional segmentation models, SAM does not always guess your target.

Instead, you guide it.

Examples:

Click on an object

Draw a box around it

Provide a rough mask


The prompt tells SAM:

> "This is what I'm interested in."




---

18.2 Everyday Analogy

Imagine you're in a library.

You ask the librarian:

> "Please give me the blue book."



The library contains thousands of books.

Without your instruction, the librarian doesn't know which one you want.

Your instruction is the prompt.

The librarian's understanding of your request is analogous to the Prompt Encoder.


---

18.3 Why Can't SAM Use Raw Coordinates?

Suppose you click here:

+----------------------+
|                      |
|        •             |
|                      |
|        🐕            |
|                      |
+----------------------+

The computer receives:

(420, 315)

These are only numbers.

A transformer cannot automatically infer that:

the point is on a dog's head,

the click indicates the desired object,

nearby pixels are likely part of the same object.


Therefore, the coordinates must be converted into a learned representation.


---

18.4 The Main Job of the Prompt Encoder

The Prompt Encoder converts human inputs into vectors.

Conceptually:

Mouse Click

↓

Coordinates

↓

Embedding Vector

Similarly:

Bounding Box

↓

Embedding Vector

Mask

↓

Embedding

Everything eventually becomes embeddings because transformers operate on vectors.


---

18.5 Types of Prompts in SAM

SAM supports three major prompt types.

Point

Bounding Box

Mask

Multiple prompts can also be used together.

For example:

a positive point,

a negative point,

a bounding box,


all at the same time.


---

18.6 Point Prompts

Suppose we have:

+----------------------+
|                      |
|       🐕             |
|      •               |
|                      |
+----------------------+

The user clicks on the dog.

This is called a positive point.

Meaning:

> "This location belongs to the object I want."




---

18.7 Positive vs Negative Points

SAM supports two kinds of point prompts.

Positive Point

•

Meaning:

> Include this region.




---

Negative Point

✕

Meaning:

> Exclude this region.



Example:

+--------------------------+
|        🌳               |
|                          |
|      🐕     ✕ Grass      |
|      •                   |
+--------------------------+

Interpretation:

Include the dog.

Exclude the grass.


Negative points are especially useful when nearby objects overlap or touch each other.


---

18.8 Why Are Negative Points Useful?

Imagine two dogs sitting close together.

🐕🐕

You click:

•

on the left dog.

SAM might accidentally include both.

Now add:

✕

on the right dog.

Now SAM understands:

> Segment the first dog, not the second.



This simple interaction greatly improves precision.


---

18.9 How Are Point Prompts Represented?

Conceptually, each point contains:

(x, y)

+

Point Type

where the point type indicates whether it is:

positive,

negative.


The Prompt Encoder learns different embeddings for these two cases.

Conceptually:

Positive Point

↓

Coordinate Encoding

+

Positive Label Embedding

↓

Final Point Embedding

Negative points follow the same process but use a different learned label embedding.


---

18.10 Position Matters

Consider two clicks.

Dog

•

and

Tree

•

The coordinate encoding tells the model where the point is.

The label embedding tells the model how to interpret the point.

Both pieces of information are required.


---

18.11 Bounding Box Prompts

Suppose the user draws:

+-------------+
|     🐕      |
+-------------+

The box provides a stronger hint than a single point.

Instead of saying:

> Somewhere here...



the box says:

> The object is probably inside these boundaries.




---

18.12 How Is a Box Encoded?

A box is defined by two corners.

(x₁, y₁)

(x₂, y₂)

Instead of treating the box as one object,

SAM encodes the two corner points.

Conceptually:

Top-Left Corner

↓

Embedding

Bottom-Right Corner

↓

Embedding

The decoder learns that these two embeddings together describe a rectangle.


---

18.13 Mask Prompts

Sometimes a user already has a rough mask.

Example:

██████░░
████░░░░
██░░░░░░

Perhaps it came from:

a previous prediction,

another segmentation model,

user editing.


Instead of starting over,

SAM uses this mask as an additional prompt.


---

18.14 Why Mask Prompts Matter

Suppose the first prediction is:

████░░
████░░
██░░░░

The ears are missing.

The user edits the mask slightly.

SAM uses the edited mask as a prompt and predicts a more accurate segmentation.

This enables iterative refinement.


---

18.15 Dense and Sparse Prompts

SAM groups prompts into two categories.

Sparse Prompts

Small pieces of information:

positive points,

negative points,

box corners.


These are represented as a small set of embeddings.


---

Dense Prompts

Large spatial inputs:

masks.


These contain information across many image locations.


---

18.16 Combining Multiple Prompts

Suppose the user provides:

Positive Point

+

Negative Point

+

Bounding Box

The Prompt Encoder converts each one into embeddings.

Conceptually:

Positive Point

↓

Embedding A

Negative Point

↓

Embedding B

Bounding Box

↓

Embedding C

↓

Combined Prompt Representation

The Mask Decoder receives all of these together.


---

18.17 Why Use One Shared Embedding Space?

The image encoder produces image embeddings.

The prompt encoder produces prompt embeddings.

Both are designed to exist in compatible feature spaces so the Mask Decoder can reason about them jointly.

You can think of it like two people speaking the same language.

If the image features "speak English" but the prompt features "speak Japanese," communication would fail.

The Prompt Encoder translates human interactions into the same "language" used by the rest of the model.


---

18.18 Complete Prompt Flow

Mouse Click
      │
      ▼
Coordinates
      │
      ▼
Coordinate Encoding
      │
      ▼
Add Point-Type Embedding
      │
      ▼
Point Embedding

For a box:

Bounding Box
      │
      ▼
Two Corner Coordinates
      │
      ▼
Coordinate Encodings
      │
      ▼
Corner Embeddings

For a mask:

Mask
      │
      ▼
Mask Encoder
      │
      ▼
Dense Prompt Embedding


---

18.19 Real-World Analogy

Imagine giving directions to a taxi driver.

You might say:

"Pick me up here." (Point)

"I'm somewhere inside this shopping mall." (Box)

"I'm already walking along this route." (Mask)


Although these instructions look different, they all communicate the same thing:

> Where the driver should focus.



The Prompt Encoder performs a similar translation for SAM.


---

Common Misconceptions

❌ "The Prompt Encoder performs segmentation."

No.

It only converts prompts into embeddings.

The Mask Decoder generates the segmentation mask.


---

❌ "Coordinates alone are enough."

No.

Coordinates specify location, but the model also needs to know whether the point is positive, negative, or part of another prompt type.


---

❌ "SAM only accepts one prompt."

No.

SAM can combine multiple prompts—such as several points, a box, and a mask—to produce a more accurate result.


---

Key Takeaways

A prompt tells SAM what object to segment.

The Prompt Encoder converts user inputs into learned embeddings.

Point prompts include both coordinates and a point-type label (positive or negative).

Box prompts are represented through their corner coordinates.

Mask prompts provide dense spatial guidance for refinement.

All prompt types are transformed into a common embedding space so the Mask Decoder can combine them with image features.



---

Practice Questions

Conceptual

1. Why can't SAM use raw mouse coordinates directly?


2. What is the difference between a positive point and a negative point?


3. Why is a bounding box generally a stronger prompt than a single point?


4. What is the difference between sparse and dense prompts?


5. Why do image embeddings and prompt embeddings need to be compatible?



Thought Exercise

Suppose an image contains:

🐕   🌳   🚲

You provide:

One positive point on the dog.

One negative point on the tree.

A bounding box around the dog.

A rough mask covering part of the dog's body.


Describe:

1. Which prompts are sparse?


2. Which prompt is dense?


3. How might each prompt help the Mask Decoder produce a better segmentation?




---

Chapter Summary

The Prompt Encoder is the bridge between human interaction and the neural network. It transforms points, boxes, and masks into learned embeddings that the Mask Decoder can understand. Positive and negative points allow users to include or exclude regions, bounding boxes provide coarse localization, and mask prompts enable iterative refinement. By expressing all prompt types in a shared embedding space, SAM can flexibly combine different forms of guidance with rich image features to generate accurate segmentation masks.


---

Next Lesson

In Chapter 19 — Mask Decoder: How SAM Converts Image Features and Prompts into Segmentation Masks, we'll study the final stage of the pipeline.

We'll explore:

Why the Mask Decoder is comparatively lightweight

How it combines image and prompt embeddings

The role of transformer-based interactions inside the decoder

How mask tokens work

Why SAM predicts multiple candidate masks

How the final segmentation mask is generated from dense image features


This chapter will complete your understanding of SAM's core architecture.
