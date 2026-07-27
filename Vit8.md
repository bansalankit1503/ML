Excellent. You've now understood:

✅ Vision Transformer Image Encoder

✅ Prompt Encoder


Now we arrive at the brain of SAM—the component that actually creates the segmentation mask.

This chapter is one of the most technically important parts of SAM because it introduces several ideas that are unique to SAM, including:

Mask Tokens

Cross-Attention

IoU Prediction Head

Multiple Mask Prediction


By the end of this lesson, you'll understand how SAM converts image features and prompts into an accurate segmentation mask.


---

Part V – Segment Anything Model (SAM)

Chapter 19 — Mask Decoder: How SAM Creates Segmentation Masks

> "The Image Encoder understands the scene. The Prompt Encoder understands the user's intent. The Mask Decoder combines both to decide exactly which pixels belong to the object."




---

Learning Objectives

By the end of this chapter, you will understand:

Why SAM needs a Mask Decoder

Why the decoder is lightweight

What mask tokens are

How cross-attention combines prompts with image features

How SAM predicts multiple masks

What the IoU prediction head does

The complete decoding pipeline



---

19.1 Where Are We?

Let's recall the architecture.

Image
                   │
                   ▼
          Image Encoder (ViT)
                   │
          Dense Image Features
                   │
                   │
      Prompt ──────┘
         │
         ▼
   Prompt Encoder
         │
         ▼
  Prompt Embeddings
         │
         ▼
     Mask Decoder
         │
         ▼
 Segmentation Mask

Everything we've learned so far prepares the inputs for the Mask Decoder.


---

19.2 Why Do We Need Another Network?

A natural question is:

> Why can't the Vision Transformer directly output the segmentation mask?



Because the Image Encoder doesn't know what object the user wants.

Consider this image:

🐕      🚲      🌳

The image encoder understands:

dog

bicycle

tree


But it doesn't know whether you want:

the dog,

the bicycle,

or the tree.


That information comes from the prompt.

The decoder combines:

Image Understanding

+

User Intention

to generate the final mask.


---

19.3 Why Is the Decoder Small?

The Image Encoder is computationally expensive because it processes every image patch.

The decoder works on features that have already been extracted.

Think of it like this.

Suppose you read an entire book.

That takes hours.

Answering questions about the book afterwards is much faster.

The decoder is answering questions about an image that has already been understood.


---

19.4 Inputs to the Decoder

The decoder receives two major inputs.

Image Features

F₁

F₂

F₃

...

Fₙ

Each feature corresponds to a spatial location.


---

Prompt Embeddings

For example:

Positive Point

↓

Embedding

or

Bounding Box

↓

Embedding

Both become vectors.


---

19.5 The Decoder's Goal

The decoder answers one question:

> Which image features belong to the object indicated by the prompt?



Notice that it is not classifying.

It is grouping relevant image features together.


---

19.6 Introducing Mask Tokens

This is one of SAM's most important innovations.

Earlier, you learned about the [CLS] token in Vision Transformers.

SAM introduces something different:

Mask Tokens

Instead of summarizing the image,

each mask token represents:

> "One possible segmentation mask."



Conceptually:

Mask Token 1

Mask Token 2

Mask Token 3

Each token competes to explain the requested object.


---

19.7 Why Use Mask Tokens?

Imagine asking three artists to trace the same object.

Artist A:

██████
██████

Artist B:

█████░
██████

Artist C:

██████
█████░

All are similar,

but one tracing may better match the true object.

Mask tokens work similarly by proposing different candidate masks.


---

19.8 Cross-Attention

Now comes the key operation.

The prompt embedding interacts with the image features.

Conceptually:

Prompt

↓

Cross-Attention

↓

Relevant Image Features

Suppose the prompt is:

•

on the dog's head.

Cross-attention helps identify image features corresponding to:

ears

nose

body

tail


while reducing attention to:

grass

tree

sky



---

19.9 Self-Attention vs Cross-Attention

This distinction is very important.

Self-Attention

Image Features

↓

Image Features

Image patches communicate with each other.


---

Cross-Attention

Prompt

↓

Image Features

The prompt guides which image features are emphasized.

In SAM, both mechanisms contribute to effective decoding.


---

19.10 Decoder Interaction

Conceptually:

Prompt Embeddings
        │
        ▼
Cross-Attention
        ▲
        │
Image Features

The decoder repeatedly exchanges information until the prompt and image features are aligned.


---

19.11 From Features to a Mask

After several decoding steps,

each mask token contains information describing a candidate object.

Conceptually:

Mask Token

↓

Object Representation

↓

Pixel Scores

↓

Binary Mask

Instead of directly predicting pixels,

the decoder first predicts scores indicating how likely each spatial location belongs to the object.


---

19.12 Heatmap Before Thresholding

Before producing a binary mask,

the decoder generates something like:

0.98 0.97 0.95

0.94 0.92 0.20

0.05 0.03 0.01

These values represent confidence.

High values indicate strong confidence that the location belongs to the object.


---

19.13 Thresholding

Suppose we use a threshold of 0.5.

Values greater than 0.5 become:

1

Others become:

0

Result:

██████

██████

░░░░░░

This produces the segmentation mask.


---

19.14 Why Does SAM Predict Multiple Masks?

Suppose you click here:

•

🧍

The point might refer to:

the person's shirt,

the entire person,

or a backpack.


One click can be ambiguous.

Instead of forcing one answer,

SAM predicts multiple candidate masks.

For example:

Mask A

Entire Person

Mask B

Only Shirt

Mask C

Person + Backpack

The system can then choose the most appropriate result.


---

19.15 IoU Prediction Head

Now another clever idea.

SAM predicts how good each candidate mask is likely to be.

This is done by the IoU Prediction Head.

It estimates the quality of each mask without needing the ground-truth mask at inference time.


---

19.16 What Is IoU?

IoU stands for Intersection over Union.

It measures overlap between:

predicted mask

ground-truth mask


Formula:

\[
IoU = \frac{\text{Intersection}}{\text{Union}}
\]

Perfect overlap:

IoU = 1.0

No overlap:

IoU = 0

The prediction head estimates this score for each candidate mask.


---

19.17 Choosing the Best Mask

Suppose the decoder predicts:

Mask	Estimated IoU

A	0.91
B	0.76
C	0.64


The highest-quality candidate is selected by default.

Applications can also expose multiple candidates to the user for interactive selection.


---

19.18 Complete Decoder Pipeline

Image Features
        │
        ▼
Prompt Embeddings
        │
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
Best Mask


---

19.19 Real-World Analogy

Imagine an architect receives:

a detailed map of a city (image features),

and a request:


> "Highlight every park."



The architect doesn't redraw the city.

Instead, they examine the existing map and mark only the requested regions.

The Image Encoder creates the map.

The Prompt Encoder explains the request.

The Mask Decoder highlights the correct region.


---

Common Misconceptions

❌ "The decoder looks at raw pixels."

No.

It operates on image embeddings generated by the Image Encoder, not on the original RGB image.


---

❌ "Cross-attention replaces self-attention."

No.

They serve different purposes.

Self-attention models relationships within a set of tokens.

Cross-attention lets one set of tokens (prompts or mask tokens) interact with another set (image features).



---

❌ "Predicting multiple masks means SAM is uncertain."

Not necessarily.

Multiple masks allow SAM to handle ambiguous prompts gracefully, giving the system flexibility when a single prompt could correspond to several valid objects or regions.


---

Key Takeaways

The Mask Decoder combines image features with prompt embeddings.

It is lightweight because the expensive image encoding has already been completed.

Mask tokens represent candidate segmentation masks.

Cross-attention allows prompts to guide which image features are relevant.

The decoder predicts multiple candidate masks for ambiguous prompts.

The IoU Prediction Head estimates the quality of each candidate mask, helping select the best one.



---

Practice Questions

Conceptual

1. Why can't the Image Encoder produce the final segmentation mask on its own?


2. What is the purpose of a mask token?


3. How does cross-attention differ from self-attention?


4. Why does SAM predict multiple candidate masks?


5. What role does the IoU Prediction Head play during inference?



Thought Exercise

Suppose an image contains:

👩 holding 🐕

You click on the dog's head.

Think through the pipeline:

1. Which image features are likely to receive higher attention?


2. Why might SAM generate multiple masks?


3. How would the IoU Prediction Head help choose the best mask?


4. How might adding a negative point on the person change the final prediction?




---

Chapter Summary

The Mask Decoder is the decision-making component of SAM. It receives dense image features from the Vision Transformer and prompt embeddings from the Prompt Encoder, then combines them using attention-based interactions. Through mask tokens, it generates multiple candidate segmentation masks, while the IoU Prediction Head estimates the quality of each candidate. This architecture enables SAM to respond quickly to user prompts, resolve ambiguous interactions, and produce accurate, pixel-level segmentation without reprocessing the entire image.


---

Next Lesson

In Chapter 20 — How SAM Is Trained: Data, SA-1B, Loss Functions, and the Three-Stage Data Engine, we'll explore how SAM became capable of segmenting such a wide variety of objects.

We'll cover:

Why training SAM required a fundamentally new dataset

The SA-1B dataset and why it's unprecedented in scale

Meta's three-stage data engine

How human annotators and SAM improved each other in a feedback loop

The loss functions used during training

Why data quality was just as important as model architecture


This chapter will reveal that SAM's success comes not only from its architecture, but also from the innovative way it was trained.
