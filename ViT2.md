Absolutely. The previous chapter introduced attention maps conceptually. Here is the complete textbook-quality version with deeper intuition, mathematics, visualization, examples, and practical insights.


---

Part IV – Vision Transformers (ViT)

Chapter 13 — Attention Maps: What Does a Vision Transformer Actually Look At?

> "Attention tells the model where to look before deciding what something is."




---

Learning Objectives

After completing this chapter, you will understand:

What an attention map actually is

How an attention map is mathematically computed

How attention maps are visualized

Why different attention heads learn different things

How attention evolves across transformer layers

How ViT recognizes entire objects without convolutions

Why attention maps are useful—but not perfect explanations

How these ideas are used inside the Segment Anything Model (SAM)



---

13.1 Motivation

Imagine someone shows you this image for only two seconds.

+--------------------------------+
|                                |
|        🐕                      |
|                                |
|             🌳                 |
|                                |
+--------------------------------+

Now they ask,

> "What did you see?"



Did your eyes examine every pixel?

No.

Your brain naturally focused on:

the dog's face

its body

maybe the tree


It ignored:

empty sky

plain grass

background noise


Humans naturally allocate attention.

Transformers learn to do something very similar.


---

13.2 From Self-Attention to Attention Maps

Recall the self-attention equation:

\[
\text{Attention}(Q,K,V)
=
\text{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Notice something important.

There are two outputs involved conceptually:

1. Attention weights



\[
A=
\text{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
\]

2. Final contextual embeddings



\[
AV
\]

The attention map visualizes A, not AV.

This distinction is very important.


---

13.3 What Exactly Is an Attention Map?

Suppose an image is divided into four patches.

+-------+-------+
|  P1   |  P2   |
+-------+-------+
|  P3   |  P4   |
+-------+-------+

Suppose we compute attention.

We obtain

Query	P1	P2	P3	P4

P1	0.70	0.10	0.15	0.05
P2	0.20	0.60	0.10	0.10
P3	0.10	0.15	0.65	0.10
P4	0.05	0.10	0.20	0.65


Every row sums to one.

For example,

Row 1 means

Patch P1 is looking:

70% at itself

10% at P2

15% at P3

5% at P4


That row is an attention map for Patch P1.


---

13.4 Visualizing Attention

Numbers are difficult to interpret.

Instead,

we convert them into brightness.

Suppose

0.8 → Very Bright

0.5 → Bright

0.2 → Medium

0.05 → Dark

Then

+--------+--------+
|████████|░░░░░░░░|
+--------+--------+
|██░░░░░░|░░░░░░░░|
+--------+--------+

Immediately we understand

> "This patch mostly attends to the top-left region."



That is exactly how attention visualizations in research papers are generated.


---

13.5 Example: A Dog Image

Suppose the image is

+-----------------------+
|                       |
|      🐕               |
|                       |
|          🌳           |
|                       |
+-----------------------+

Suppose we inspect the patch containing the dog's eye.

Attention map:

█████████████

██████████░░

██████░░░░░░

░░░░░░░░░░░░

The model mostly looks at

eye

ears

nose


rather than

sky

empty grass


The attention weights reveal which image regions contribute most strongly to the representation of that query patch.


---

13.6 Why Doesn't Every Patch Receive Equal Attention?

Imagine you're reading a book.

While reading this sentence

> The little dog chased the ball.



You naturally connect

dog ↔ chased

chased ↔ ball


You don't spend equal effort thinking about every word.

Similarly,

a Vision Transformer learns

> Some patches matter much more than others.



The attention mechanism automatically assigns higher weights to informative regions.


---

13.7 Attention Matrix as a Communication Network

Another way to think about attention is as a communication graph.

Suppose we have four patches.

P1

P2

P3

P4

Attention creates connections.

P1 ------> P2

 |          |

 |          |

 ▼          ▼

P3 ------> P4

The thicker the connection,

the larger the attention weight.

Therefore,

attention maps are actually visualizations of information flow inside the transformer.


---

13.8 Multi-Head Attention: Many Specialists

Recall Multi-Head Attention.

Instead of one attention matrix,

we have several.

Example:

Input

 │

 ├── Head 1

 ├── Head 2

 ├── Head 3

 └── Head 4

Each head has

different

WQ

WK

WV


Therefore,

each head learns different attention patterns.


---

13.9 What Might Different Heads Learn?

Imagine this street scene.

🚗

🚶

🚦

🌳

🏢

Head 1 might focus on

Vehicles

████████

████░░░░

░░░░░░░░


---

Head 2

Road boundaries

░░██████

████████

░░██████


---

Head 3

Traffic lights

░░░░████

░░░░████

░░░░████


---

Head 4

Overall scene layout

████████

████████

████████

No one tells the model to learn these patterns.

They emerge during training because they help solve the task.


---

13.10 Layer-by-Layer Evolution

One of the most fascinating discoveries about Vision Transformers is that attention changes with depth.


---

Layer 1

Mostly learns

edges

colors

brightness


████░░░░

░░████░░

░░░░████

Very local.


---

Layer 3

Begins grouping nearby patches.

████████

████████

░░████░░

Simple shapes emerge.


---

Layer 6

Now attention spans much larger regions.

The model begins recognizing

eyes

wheels

leaves



---

Layer 9

Object parts become connected.

For example

Dog

↓

Head

↓

Legs

↓

Tail

The model realizes

these separate patches belong to one object.


---

Layer 12

Global understanding.

Attention often covers

the entire object.

Instead of

"ear"

the model now thinks

"dog."


---

13.11 Why This Is Different from CNNs

Consider two patches.

Cat                       Ball

Very far apart.

A CNN usually requires many convolution layers before information from the cat reaches the ball.

A Vision Transformer can connect them immediately.

Cat Patch

↓

Attention

↓

Ball Patch

One attention layer is enough.

This ability to model long-range relationships directly is a key advantage of transformers.


---

13.12 The Journey of the [CLS] Token

Initially,

[CLS]

contains no knowledge about the image.

After Layer 1,

it gathers a little information.

[CLS]

↓

Sky

↓

Grass

After Layer 6,

[CLS]

↓

Dog

↓

Tree

↓

Ground

After Layer 12,

Entire Image Summary

This final representation is passed to the classification head.


---

13.13 Can We Trust Attention Maps Completely?

This is an important research question.

Many people believe

> Bright attention = Explanation



Not necessarily.

Remember,

after attention,

the transformer still applies

Value vectors

Feed-Forward Networks

Residual connections

LayerNorm

Multiple encoder layers


A region receiving high attention does not automatically mean it is the sole reason for the prediction.

Attention maps are informative, but they are not a complete explanation of the model's reasoning.


---

13.14 Attention Maps in Real Research

Researchers use attention maps to:

understand what the model has learned

debug training problems

compare different architectures

study how information flows through the network

identify failure cases

analyze head specialization


Attention visualization has become a valuable tool for interpreting Vision Transformers, although it should be combined with other analysis methods when explaining model behavior.


---

13.15 Connection to SAM

SAM uses a Vision Transformer as its image encoder.

Suppose the image contains

🐕

The image encoder has already learned relationships such as:

head ↔ body

body ↔ legs

legs ↔ tail


Later,

when the user clicks on the dog's head,

SAM can use these rich feature representations to infer that the head belongs to the same object as the body and tail.

This is one reason SAM can often segment an entire object even when the prompt is only a single point.


---

Real-World Example

Suppose you ask a friend:

> "Where is Rahul?"



Your friend first looks around the room.

Not every person receives equal attention.

Their eyes move toward:

Rahul's chair

Rahul's friends

Rahul's laptop


Only after gathering this information do they answer.

A Vision Transformer behaves similarly.

It first decides where to gather information before forming its final representation.


---

Common Misconceptions

❌ "Attention maps show exactly why the model made a prediction."

Not always. They show interaction strengths between tokens, but the final prediction also depends on the Value vectors, MLPs, residual paths, normalization, and many stacked layers.


---

❌ "All attention heads learn identical patterns."

No.

Different heads often specialize in different types of visual relationships, though some overlap can occur.


---

❌ "Higher attention always means higher importance."

Not necessarily.

Attention weights indicate how information is aggregated, but they do not directly measure feature importance or causal influence.


---

Chapter Summary

In this chapter, we transformed the abstract mathematics of self-attention into an intuitive visual concept. We learned that an attention map is a visualization of the attention weight matrix, showing how strongly one image patch attends to others. Different attention heads develop different specializations, and attention patterns typically evolve from low-level visual cues in early layers to whole-object understanding in deeper layers. These maps provide valuable insight into Vision Transformers and help explain how models like SAM build rich, globally informed image representations.


---

Key Takeaways

Attention maps visualize the Softmax(QKᵀ/√dₖ) matrix.

Every row represents how one query patch distributes its attention across all patches.

Different attention heads often learn complementary visual relationships.

Early layers tend to focus on simple patterns; deeper layers often capture object-level structure.

Vision Transformers can model long-range relationships from the very first attention layer.

Attention maps are powerful interpretability tools but should not be treated as complete explanations of model decisions.



---

Practice Questions

Conceptual

1. Why is an attention map a visualization of the attention weights rather than the final output \(AV\)?


2. What does one row of the attention matrix represent?


3. Why do different attention heads learn different patterns?


4. How do attention patterns typically change from early to deeper transformer layers?


5. Why can a Vision Transformer relate distant image regions more easily than a CNN?



Numerical

Suppose a query patch has attention weights:

\[
[0.55,\ 0.25,\ 0.10,\ 0.10]
\]

1. Which patch contributes the most to the query's updated representation?


2. Do the attention weights satisfy the Softmax property?


3. If the Value vector of the first patch changes significantly, how would you expect the query's output representation to be affected?




---

Next Chapter

Chapter 14 — Why Vision Transformers Work: Inductive Bias, Data Requirements, and Scaling

In this chapter, we'll answer some of the deepest questions about ViTs:

Why do CNNs perform well even with relatively small datasets?

Why did early Vision Transformers require enormous datasets?

What is inductive bias, and why do CNNs have more of it than ViTs?

Why do Vision Transformers improve dramatically as data and model size increase?

How did techniques such as stronger augmentation, distillation, and self-supervised learning help ViTs become practical?


This chapter will bridge the gap between understanding how ViTs work and why they succeed in practice.
