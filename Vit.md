Excellent. This is the chapter where everything you've learned comes together.

Until now, we've studied the Transformer independently of any specific application. Now we'll see how researchers asked a simple but revolutionary question:

> "If transformers work so well for language, can we apply the same idea directly to images?"



The answer was Vision Transformer (ViT), introduced in 2020. It showed that an image could be treated like a sentence and image patches could be treated like words.

This idea changed computer vision.


---

Part IV – Vision Transformers (ViT)

Chapter 11 — Vision Transformer: From Image to Classification


---

Learning Objectives

By the end of this chapter, you will understand:

Why CNNs were dominant before ViT

The key insight behind Vision Transformers

The complete ViT pipeline

How images become sequences of tokens

The role of the [CLS] token

How ViT predicts an image class



---

11.1 A Brief History

For almost a decade, image recognition was dominated by Convolutional Neural Networks (CNNs).

Popular architectures included:

AlexNet

VGG

ResNet

EfficientNet


CNNs worked by applying convolution filters to local regions of an image.

For example:

Image

↓

3×3 Filter

↓

Feature Map

↓

More Convolutions

↓

Classification

CNNs achieved remarkable performance, but they had an important characteristic:

> They naturally focus on local neighborhoods.



To understand relationships between distant regions, CNNs often require many layers.


---

11.2 The Big Idea of ViT

Researchers asked:

> Instead of treating an image as pixels, why not treat it like a sentence?



Language model:

Sentence

↓

Words

↓

Tokens

↓

Transformer

Vision Transformer:

Image

↓

Patches

↓

Tokens

↓

Transformer

This is the entire philosophy of ViT.


---

11.3 A Simple Example

Suppose we have a small image:

┌────────────┐
│            │
│   🐶       │
│            │
└────────────┘

Instead of processing the whole image at once, ViT divides it into equal-sized patches.

Example:

┌────┬────┐
│ P1 │ P2 │
├────┼────┤
│ P3 │ P4 │
└────┴────┘

Each patch becomes a token.

Just like:

Sentence

↓

Word 1

Word 2

Word 3

we now have:

Image

↓

Patch 1

Patch 2

Patch 3

Patch 4


---

11.4 Real Image Sizes

The original ViT commonly uses:

Image size:

224 × 224

Patch size:

16 × 16

How many patches?

Along width:

\[
224/16=14
\]

Along height:

\[
224/16=14
\]

Total:

\[
14 \times 14 = 196
\]

Therefore,

A single image becomes:

196 image patches

This is directly analogous to a sentence containing 196 tokens.


---

11.5 Flattening Each Patch

Each patch is still a small image.

Suppose:

Patch size:

16 × 16

RGB image:

3 channels

One patch contains:

\[
16 \times16\times3 =768
\]

numbers.

Instead of treating it as a square,

we flatten it into one long vector.

16×16×3

↓

768 numbers

Exactly like flattening a matrix into a vector.


---

11.6 Patch Embedding

A vector of raw pixel values is not yet suitable for the Transformer.

Just as words are converted into embeddings,

each image patch is converted into a patch embedding.

Pipeline:

Patch

↓

Flatten

↓

Linear Layer

↓

Patch Embedding

If there are 196 patches,

we now obtain:

196 embedding vectors

These become the transformer's input tokens.


---

11.7 The [CLS] Token

Language models such as BERT introduced a special token called:

[CLS]

Its purpose is to collect information from the whole sequence.

ViT adopts the same idea.

Before the image patches,

we insert one additional learnable token.

[CLS]

P1

P2

P3

...

P196

Now the Transformer receives:

197 tokens

The [CLS] token participates in self-attention like every other token.

As information flows through the encoder blocks, it gathers information from all image patches.

By the end of the network, the [CLS] token becomes a compact summary of the entire image.


---

11.8 Adding Positional Embeddings

Imagine we shuffle the image patches.

P18

P91

P4

P72

The Transformer would no longer know where each patch originally came from.

So we add positional embeddings.

Patch Embedding

+

Position Embedding

↓

Final Input Token

Now every token knows:

what visual information it contains, and

where it came from in the image.



---

11.9 Passing Through Transformer Encoder Blocks

Now the input sequence is:

[CLS]

P1

P2

...

P196

This sequence passes through multiple encoder blocks.

For example:

Input Tokens

↓

Encoder Block 1

↓

Encoder Block 2

↓

Encoder Block 3

↓

...

↓

Encoder Block 12

During every block:

every patch attends to every other patch,

every patch updates its representation,

the [CLS] token also attends to all patches.


As the layers deepen, the representations become increasingly informative.


---

11.10 What Happens to the [CLS] Token?

Initially,

the [CLS] token contains only its learned embedding.

[CLS]

↓

Random Learnable Vector

After one encoder block:

[CLS]

↓

Information from many patches

After twelve encoder blocks:

[CLS]

↓

Information from the entire image

It effectively becomes an image summary.


---

11.11 Final Classification Head

After the last encoder block,

we keep only the final [CLS] representation.

Final [CLS]

↓

MLP Head

↓

Class Scores

Suppose the model predicts:

Class	Score

Dog	0.95
Cat	0.03
Horse	0.02


The highest score corresponds to the predicted class.


---

11.12 Complete ViT Pipeline

Putting everything together:

Input Image
      │
      ▼
Split into Patches
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
Add [CLS] Token
      │
      ▼
Add Positional Embeddings
      │
      ▼
Transformer Encoder × N
      │
      ▼
Final [CLS] Representation
      │
      ▼
Classification Head
      │
      ▼
Predicted Class

This is the complete Vision Transformer architecture for image classification.


---

11.13 Why Is This Different from CNNs?

CNN	Vision Transformer

Uses convolution filters	Uses self-attention
Strong local inductive bias	Learns global relationships directly
Processes local neighborhoods first	Every patch can attend to every other patch from the first layer
Translation invariance comes naturally	Positional information must be explicitly added


Neither approach is universally better; each has strengths depending on the task, dataset size, and computational resources.


---

11.14 Why Was ViT Revolutionary?

Before ViT, many researchers believed convolution was essential for vision tasks.

ViT demonstrated that, given sufficient training data and compute, a transformer architecture without convolutions could achieve state-of-the-art image classification performance.

This inspired a wave of transformer-based vision models, including architectures for detection, segmentation, and multimodal learning.


---

11.15 Connection to SAM

Understanding ViT is essential because the Segment Anything Model (SAM) uses a transformer-based image encoder.

However, SAM does not stop at image classification.

Instead of predicting:

Dog

SAM produces rich feature representations that can later be used to generate segmentation masks based on user prompts.

In other words:

ViT: Image → Class

SAM: Image → Rich Feature Map → Segmentation


The encoder pipeline is very similar, but the downstream task is different.


---

Common Misconceptions

❌ "A patch is one pixel."

No. A patch contains many pixels, such as a 16×16 RGB region.


---

❌ "The [CLS] token is copied from the image."

No. It is a learnable vector initialized randomly and optimized during training.


---

❌ "The Transformer sees the entire image directly."

No. It only sees the sequence of patch embeddings plus the [CLS] token.


---

Key Takeaways

ViT treats an image as a sequence of image patches.

Each patch is flattened and projected into an embedding.

A learnable [CLS] token is added to summarize the image.

Positional embeddings preserve spatial layout.

The sequence is processed by stacked Transformer encoder blocks.

The final [CLS] representation is used for image classification.



---

Practice Questions

Conceptual

1. Why does ViT divide an image into patches instead of processing the whole image directly?


2. Why is a linear projection applied after flattening each patch?


3. What is the purpose of the [CLS] token?


4. Why are positional embeddings necessary in ViT?


5. How does ViT differ fundamentally from a CNN?



Numerical

An image has size 384 × 384 and uses 16 × 16 patches.

1. How many patches are created?


2. If each patch is RGB, how many raw pixel values are in one patch before flattening?


3. After adding one [CLS] token, how many tokens enter the Transformer?




---

Chapter Summary

The Vision Transformer adapts the transformer architecture from language to vision by treating an image as a sequence of fixed-size patches. Each patch is flattened, projected into an embedding, combined with positional information, and processed alongside a learnable [CLS] token through stacked Transformer encoder blocks. The final [CLS] representation summarizes the image and is used for classification. This elegant reformulation of images as token sequences laid the foundation for many modern vision models, including the encoder used in SAM.


---

Next Lesson

In the next chapter, we'll go even deeper into the patch embedding process.

We'll answer questions that are often glossed over:

Why flatten patches?

Is flattening the only option?

How does the linear projection actually work?

Why does ViT commonly choose a 16×16 patch size?

What happens if we use 8×8, 32×32, or variable-sized patches?

How does patch size affect accuracy, computation, and memory?


By the end of that chapter, you'll understand one of the most important design trade-offs in Vision Transformers.
