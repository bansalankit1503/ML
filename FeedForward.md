Excellent. This is one of the most important chapters in the entire course.

So far, we've studied the individual building blocks:

✅ Embeddings

✅ Queries, Keys, Values

✅ Dot Product

✅ Self-Attention

✅ Multi-Head Attention

✅ Positional Encoding


Now it's time to assemble these pieces into a complete Transformer Encoder Block.

This block is the core component used in:

Vision Transformers (ViT)

DINO

MAE

CLIP Vision Encoder

SAM Image Encoder

Many other modern vision models


Once you understand this block, you understand the heart of these architectures.


---

Part III – Transformers: From First Principles

Chapter 9 — The Transformer Encoder Block: Putting Everything Together


---

Learning Objectives

By the end of this chapter, you will understand:

The complete flow of data through a Transformer encoder block

The purpose of every component

Why residual (skip) connections are necessary

Why layer normalization is used

Why a Feed-Forward Network (MLP) follows attention

How stacking encoder blocks creates powerful models

How this architecture is used in Vision Transformers and SAM



---

9.1 The Big Picture

A Transformer encoder block looks like this:

Input Embeddings
                        │
                        ▼
             + Positional Encoding
                        │
                        ▼
          Multi-Head Self-Attention
                        │
                 + Residual Connection
                        │
                        ▼
               Layer Normalization
                        │
                        ▼
          Feed-Forward Network (MLP)
                        │
                 + Residual Connection
                        │
                        ▼
               Layer Normalization
                        │
                        ▼
               Output Representations

At first glance, this may look complicated.

In reality, it is just a sequence of simple operations.


---

9.2 Step 1 – Input Embeddings

Suppose our sentence is:

The

cat

slept

Each word becomes an embedding.

Example:

Token	Embedding

The	e₁
cat	e₂
slept	e₃


These embeddings capture semantic meaning, but they do not yet include order.


---

9.3 Step 2 – Add Positional Encoding

We now add positional information.

Embedding
      +
Position
      =
Input Representation

For example:

Token	Embedding	Position	Final Input

The	e₁	p₁	e₁+p₁
cat	e₂	p₂	e₂+p₂
slept	e₃	p₃	e₃+p₃


Now each token knows both:

what it means, and

where it appears.



---

9.4 Step 3 – Multi-Head Self-Attention

The enriched token representations are fed into Multi-Head Attention.

Each head computes:

Q

K

V

Each head independently computes:

Attention(Q,K,V)

Example with four heads:

Input

     │

 ┌───┼───┬───┐

 ▼   ▼   ▼   ▼

H1  H2  H3  H4

 └───┼───┴───┘

     ▼

Concatenate

     ▼

Linear Layer

The output now contains contextual information gathered from all tokens.


---

9.5 Why Isn't Attention Enough?

Many beginners think:

> "Attention already mixes information. Why do we need anything else?"



Imagine a classroom discussion.

Students exchange ideas.

At the end of the discussion:

Everyone knows what everyone else thinks.

But nobody has yet processed the new information deeply.

Humans don't just listen.

They think.

The Feed-Forward Network plays this role.

Before that, however, we need to understand two important techniques that make deep transformers train reliably.


---

9.6 Residual (Skip) Connections

Suppose attention produces:

Attention Output

Should we discard the original input?

No.

Instead, we add it back.

Input
      │
      ▼
Attention
      │
      ▼
Output
      +
Input
      │
      ▼
Result

Mathematically:

\[
Y = X + \text{Attention}(X)
\]

This is called a Residual Connection or Skip Connection.


---

9.7 Why Do We Need Residual Connections?

Imagine you are editing an important document.

Would you rather:

overwrite the original completely, or

make edits while keeping the original available?


Most people prefer the second option.

Residual connections let the model preserve useful information while adding new information learned through attention.


---

9.8 Another Analogy

Think about climbing a mountain.

Without shortcuts:

A

↓

B

↓

C

↓

D

↓

E

Every step depends on the previous one.

Now imagine there are bridges connecting:

A ─────────► D

B ───────► E

Information can travel through multiple paths.

These shortcuts make optimization easier and help very deep networks learn effectively.


---

9.9 Why Deep Networks Become Difficult to Train

Imagine stacking 48 transformer layers.

Without skip connections:

Layer1

↓

Layer2

↓

Layer3

↓

...

↓

Layer48

Small numerical changes can accumulate, making it difficult for gradients to flow backward during training.

Residual connections provide a direct path for both information and gradients, greatly improving training stability.


---

9.10 Layer Normalization

After the residual connection, we apply Layer Normalization.

Why?

Neural networks work best when the values flowing through them stay within a reasonable range.

Imagine exam scores:

Student A = 15

Student B = 18

Student C = 95

The scale is inconsistent.

Normalization rescales the values while preserving their relative relationships.

This leads to more stable and efficient training.


---

9.11 What Does LayerNorm Actually Do?

Suppose a token representation is:

[4, 8, 10]

LayerNorm computes:

1. The mean of the features.


2. The variance (or standard deviation).


3. Normalizes the values.


4. Applies two learned parameters (scale and shift) so the model can adjust the normalized values if needed.



Conceptually:

Before:

[4 8 10]

↓

LayerNorm

↓

[-1.1 0.2 0.9]

The exact numbers are less important than the idea: each token's feature vector is normalized independently.


---

9.12 Feed-Forward Network (MLP)

Now every token has gathered information from the rest of the sequence.

Next, each token is processed independently by a small neural network called the Feed-Forward Network (FFN) or MLP.

Typical structure:

Input

↓

Linear

↓

Activation (e.g., GELU)

↓

Linear

↓

Output

Notice:

There is no communication between tokens here.

The FFN processes each token separately.


---

9.13 Why Do We Need the MLP?

Attention answers:

> "Which other tokens are important?"



The MLP answers:

> "Now that I've gathered information, how should I transform my own representation?"



Think of attention as collecting evidence and the MLP as analyzing that evidence.

Both are necessary.


---

9.14 Second Residual Connection

After the FFN, we again preserve the previous representation.

FFN Output

+

Previous Representation

↓

Result

Mathematically:

\[
Z = Y + \text{FFN}(Y)
\]

Again, the goal is to preserve useful information while allowing refinement.


---

9.15 Second Layer Normalization

Finally, another LayerNorm is applied.

The encoder block output is now ready.

This output becomes the input to the next encoder block.


---

9.16 Stacking Encoder Blocks

A single encoder block is useful.

Multiple encoder blocks are powerful.

Input

↓

Encoder Block 1

↓

Encoder Block 2

↓

Encoder Block 3

↓

...

↓

Encoder Block N

↓

Final Representation

Each block refines the token representations further.

Early layers often capture simpler relationships, while deeper layers can model more abstract concepts.


---

9.17 Transformer Encoder in Vision Transformers

In a Vision Transformer:

1. Split the image into patches.


2. Convert each patch into an embedding.


3. Add positional embeddings.


4. Pass the patch embeddings through many encoder blocks.



Image

↓

Patch Embeddings

↓

+ Position

↓

Encoder Block 1

↓

Encoder Block 2

↓

...

↓

Encoder Block 12

↓

Image Representation

Each block allows every patch to exchange information with every other patch and then refine that information.


---

9.18 Transformer Encoder in SAM

SAM's image encoder is built on the same principle.

The image is divided into patches, embedded, and processed through many transformer encoder blocks.

The resulting feature representations are then used by later components of SAM to predict segmentation masks based on prompts.

The encoder's job is to produce rich, context-aware visual features that capture objects, boundaries, textures, and long-range relationships across the image.


---

Common Misconceptions

❌ "Attention is the entire transformer."

No. Attention is only one component. Residual connections, normalization, and the FFN are equally important for a practical transformer.


---

❌ "The FFN mixes information between tokens."

No. Token-to-token communication happens in the attention layer. The FFN transforms each token independently.


---

❌ "Residual connections are optional."

For modern deep transformers, residual connections are a fundamental design element that enables effective training.


---

Key Takeaways

A Transformer encoder block consists of:

1. Multi-Head Self-Attention


2. Residual Connection


3. Layer Normalization


4. Feed-Forward Network (MLP)


5. Second Residual Connection


6. Second Layer Normalization



Residual connections preserve information and improve gradient flow.

Layer Normalization stabilizes training.

The FFN transforms each token after it has gathered contextual information.

Stacking many encoder blocks produces the deep representations used in ViTs and SAM.



---

Practice Questions

Conceptual

1. Why is positional encoding added before the attention layer?


2. Why are residual connections placed after both attention and the FFN?


3. What problem does Layer Normalization solve?


4. Why does the FFN process each token independently?


5. Why are multiple encoder blocks stacked instead of using just one?



Thought Exercise

Suppose you remove one component from the encoder block:

Residual Connection

Layer Normalization

Feed-Forward Network


For each removal, predict how the transformer's behavior or training might change. Which removal do you think would have the greatest impact, and why?


---

Chapter Summary

The Transformer encoder block is the fundamental building block of modern transformer architectures. It combines Multi-Head Self-Attention, residual connections, Layer Normalization, and a Feed-Forward Network into a sequence that allows tokens to exchange information, refine their representations, and train efficiently even in very deep models. Vision Transformers and the image encoder of SAM are built by stacking many such encoder blocks, enabling them to learn powerful global representations of images.


---

Next Lesson

The next chapter will dive deeply into the Feed-Forward Network (MLP) itself.

Although it often receives less attention than self-attention, the MLP accounts for a significant portion of the parameters and computation in many transformer models. We'll explore:

Why two linear layers are used

Why the hidden dimension is typically expanded (e.g., 768 → 3072 → 768)

Why activation functions like GELU are preferred

How the MLP complements attention

The role of the MLP in Vision Transformers and SAM


By the end of that chapter, you'll understand not just what the MLP does, but why transformer designers chose this architecture.
