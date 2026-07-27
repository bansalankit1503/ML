Excellent. This chapter is the culmination of everything we've learned so far. Instead of studying SAM component by component, we'll now follow one image and one user click through the entire model, from raw pixels to the final segmentation mask.

Think of this chapter as a "flight recorder" for SAM—we'll observe exactly what happens at every stage.


---

Part V – Segment Anything Model (SAM)

Chapter 21 — From a Mouse Click to a Segmentation Mask: An End-to-End Walkthrough

> "Understanding a model is easiest when you follow one example from beginning to end."




---

Learning Objectives

By the end of this chapter, you will understand:

The complete inference pipeline of SAM

How data flows through every component

What happens internally after a user clicks

How image features and prompts interact

How the final segmentation mask is produced

Why SAM is fast enough for interactive applications



---

21.1 The Scenario

Suppose we have the following image:

+--------------------------------------+
|                                      |
|        🐕             🌳             |
|                                      |
|                 🚲                  |
|                                      |
+--------------------------------------+

The user wants to segment the dog.

Instead of drawing a detailed outline, they simply click on the dog's head.

+--------------------------------------+
|                                      |
|       •🐕            🌳             |
|                                      |
|                 🚲                  |
|                                      |
+--------------------------------------+

The entire journey begins with one click.


---

21.2 Step 1 — Image Input

The original RGB image enters SAM.

Conceptually:

Image

↓

1024 × 1024 × 3

This is simply a grid of pixels.

At this stage, the computer has no understanding of:

dogs

bicycles

trees


Only numbers representing color values.


---

21.3 Step 2 — Patch Embedding

The image is divided into patches.

Conceptually:

+----+----+----+----+
| P1 | P2 | P3 | P4 |
+----+----+----+----+
| P5 | P6 | P7 | P8 |
+----+----+----+----+

Each patch is flattened and projected into an embedding vector.

Instead of pixels, the transformer now processes tokens.


---

21.4 Step 3 — Add Positional Information

Without positional information:

P12

P89

P3

P41

would be meaningless.

Positional embeddings tell the model where each patch originated.

Now the model knows:

top-left

bottom-right

center

edge


This preserves spatial structure.


---

21.5 Step 4 — Vision Transformer Encoding

Now the image enters the Vision Transformer.

Each patch communicates with every other patch.

Example:

Dog Head

↓

Dog Body

↓

Dog Tail

↓

Grass

↓

Tree

After many encoder layers,

every patch contains:

local information

global context


This produces a dense feature map.


---

21.6 Image Embedding Is Created

The output is conceptually:

+------+------+------+
| F11 | F12 | F13 |
+------+------+------+
| F21 | F22 | F23 |
+------+------+------+
| F31 | F32 | F33 |
+------+------+------+

Each feature vector describes one image location.

Notice:

No segmentation has happened yet.

SAM has only understood the image.


---

21.7 Step 5 — User Click

Now the user clicks.

•

The application records:

(x,y)

Suppose:

(412,278)

To the computer,

these are simply coordinates.


---

21.8 Step 6 — Prompt Encoding

The Prompt Encoder receives:

Coordinate

+

Positive Label

It converts these into a learned embedding.

Conceptually:

(412,278)

↓

Coordinate Encoding

↓

Positive Point Embedding

Now the click becomes something meaningful to the neural network.


---

21.9 Step 7 — Decoder Receives Two Inputs

The decoder now has:

Image Features

and

Prompt Embedding

These are combined through attention.


---

21.10 Step 8 — Cross-Attention

Imagine the click landed on the dog's ear.

Cross-attention begins asking:

> Which image features are related to this prompt?



High attention:

Ear

↓

Head

↓

Body

↓

Tail

Low attention:

Tree

Grass

Sky

The prompt guides the decoder toward the desired object.


---

21.11 Step 9 — Mask Tokens Begin Reasoning

The mask tokens now collect information.

Conceptually:

Mask Token

↓

Looks at

↓

Dog Features

Each mask token forms its own hypothesis.

For example:

Token A:

Entire dog.

Token B:

Only the head.

Token C:

Dog plus shadow.


---

21.12 Step 10 — Candidate Masks

The decoder generates several possible masks.

Example:

Candidate 1

████████

████████

Entire dog.


---

Candidate 2

████░░░░

████░░░░

Only upper body.


---

Candidate 3

████████

██████░░

Dog plus nearby shadow.


---

21.13 Step 11 — IoU Prediction

Each candidate receives a quality estimate.

Candidate	Estimated IoU

1	0.94
2	0.72
3	0.83


Candidate 1 is predicted to be the highest-quality segmentation.


---

21.14 Step 12 — Final Mask

The application displays:

██████████

██████████

██████████

Only the dog is highlighted.

The user experiences this as an almost instantaneous response.


---

21.15 What Happens If the User Clicks Again?

Suppose the user now clicks the bicycle.

Does SAM repeat everything?

No.

Only these steps repeat:

New Prompt

↓

Prompt Encoder

↓

Mask Decoder

The Vision Transformer does not reprocess the image.

The cached image embedding is reused.

This is why interaction feels smooth.


---

21.16 Complete Pipeline

Input Image
      │
      ▼
Patch Embedding
      │
      ▼
Positional Encoding
      │
      ▼
Vision Transformer
      │
      ▼
Dense Image Features
      │
      ├───────────────┐
      │               │
      ▼               ▼
Mouse Click     Prompt Encoder
      │               │
      └──────┬────────┘
             ▼
      Cross-Attention
             │
             ▼
        Mask Tokens
             │
             ▼
     Candidate Masks
             │
             ▼
      IoU Prediction
             │
             ▼
      Best Segmentation


---

21.17 Time Perspective

Although the pipeline appears long, the work is divided intelligently.

Stage	Runs Once Per Image?	Runs Again for Each Prompt?

Image Encoder	✅ Yes	❌ No
Prompt Encoder	❌ No	✅ Yes
Mask Decoder	❌ No	✅ Yes


This separation is the key to interactive performance.


---

21.18 What Information Exists at Each Stage?

Stage	Information Available

Raw Image	Pixels only
Patch Embedding	Visual tokens
ViT Output	Rich semantic features with spatial information
Prompt Encoder	User intent encoded as vectors
Decoder	Combines image understanding with user intent
Final Output	Pixel-level segmentation mask


Notice how the representation becomes increasingly meaningful.


---

21.19 A Real-World Analogy

Imagine visiting a museum with a guide.

First Visit

The guide carefully walks through every room and memorizes:

paintings

sculptures

exhibits


This is like the Image Encoder.


---

Then you ask:

> "Show me the dinosaur."



The guide immediately takes you there.

This is like the Prompt Encoder and Mask Decoder working together.

If you later ask:

> "Now show me the Egyptian artifacts."



The guide doesn't relearn the museum.

They reuse their existing knowledge.

SAM behaves the same way.


---

21.20 Why This Architecture Is Elegant

Most traditional segmentation models repeatedly analyze the image whenever the task changes.

SAM separates:

understanding the image,

understanding the request.


This makes the system:

modular,

reusable,

efficient,

interactive.


It is one of the reasons SAM became so influential.


---

Common Misconceptions

❌ "The user click directly creates the mask."

No.

The click only expresses the user's intent. The Image Encoder, Prompt Encoder, and Mask Decoder must still work together to generate the final mask.


---

❌ "The Image Encoder is rerun after every click."

No.

The image embedding is computed once and reused for all subsequent prompts on the same image.


---

❌ "Cross-attention searches raw pixels."

No.

Cross-attention operates on learned feature embeddings, not on the original RGB pixels.


---

Key Takeaways

SAM processes the image once with the Vision Transformer.

The resulting dense image embedding is cached.

Every new prompt is encoded into a compatible embedding space.

The Mask Decoder combines image features and prompt embeddings through attention.

Multiple candidate masks are generated, scored, and the best one is selected.

Reusing image features makes SAM fast enough for interactive editing.



---

Practice Questions

Conceptual

1. Why is the image encoded before the user prompt is processed?


2. Why can the image embedding be reused for multiple clicks?


3. At which stage does the model first incorporate the user's intent?


4. Why does the decoder generate multiple candidate masks?


5. What is the purpose of the IoU prediction head?



End-to-End Exercise

Imagine an image containing:

🚗   🧍   🐕   🌳

The user performs the following sequence:

1. Clicks on the dog.


2. Adds a negative point on the grass.


3. Draws a bounding box around the dog.


4. Refines the result using a rough mask.



For each interaction, explain:

Which components execute?

Which data are reused?

How each new prompt changes the decoder's reasoning?



---

Chapter Summary

This chapter traced a complete inference through the Segment Anything Model. Starting with a raw image, we followed its transformation into patch embeddings, dense Vision Transformer features, prompt embeddings, and finally pixel-level segmentation masks. We saw how the Prompt Encoder communicates user intent, how the Mask Decoder uses attention and mask tokens to generate candidate masks, and how the IoU prediction head selects the most reliable prediction. Most importantly, we learned why SAM is efficient: the expensive image encoding is performed only once, while prompt encoding and decoding are lightweight enough to support interactive segmentation.


---

Next Lesson

In Chapter 22 — Inside the Mathematics of SAM: How Image Features, Prompt Embeddings, and Mask Tokens Interact, we'll move beyond the conceptual pipeline and study the actual mathematical operations inside the Mask Decoder.

We'll derive and explain:

Query, Key, and Value formation inside the decoder

Cross-attention equations

How prompt embeddings modify image features

How mask tokens produce segmentation masks

Why transformer attention is the perfect mechanism for prompt-guided segmentation


This chapter bridges the gap between intuition and the underlying mathematics, completing your understanding of how SAM works internally.
