Excellent. This chapter is one of the most underestimated parts of the Transformer.

Most beginners think:

> "Attention is the brain of the Transformer. The MLP is just a couple of linear layers."



This is not true.

In many Transformer models:

The MLP contains more parameters than the attention layer.

The MLP performs a large fraction of the computation.

Research has shown that many factual concepts and feature transformations are stored in the MLP layers.


A useful way to think about it is:

Attention decides where to look.

The MLP decides how to transform what was found.


Let's understand why.


---

Part III – Transformers: From First Principles

Chapter 10 — The Feed-Forward Network (MLP): The Thinking Layer of the Transformer


---

Learning Objectives

By the end of this lesson, you will understand:

Why the Transformer needs an MLP after attention

Why one linear layer is not enough

Why the hidden dimension is expanded

What activation functions do

Why GELU is commonly used

How the MLP works in Vision Transformers and SAM



---

10.1 Where Are We in the Encoder?

Recall the encoder block:

Input
   │
   ▼
Multi-Head Attention
   │
Residual + LayerNorm
   │
   ▼
Feed-Forward Network (MLP)
   │
Residual + LayerNorm
   │
   ▼
Output

The attention layer has already allowed every token to gather information from every other token.

Now comes an important question:

> What should we do with all this new information?



The answer is:

Transform it.

That is exactly the purpose of the MLP.


---

10.2 A Human Analogy

Imagine you're studying for an exam.

Step 1:

You collect information from:

Books

Teachers

Friends

Videos


This is similar to Attention.

Step 2:

You sit alone and think.

You:

connect ideas,

remove incorrect assumptions,

form new understanding,

build intuition.


This is the MLP.

Notice:

During this thinking process,

you are not talking to anyone else.

Likewise,

the MLP processes each token independently.


---

10.3 Why Isn't Attention Enough?

Suppose attention gives the word:

Dog

the following contextual representation:

[0.6, 1.1, -0.3, 2.0]

This vector contains information gathered from other words.

But it is still just a numerical representation.

The Transformer now wants to build a better representation.

Examples:

combine features,

suppress irrelevant signals,

strengthen useful signals,

create higher-level concepts.


Attention cannot do all of this alone.

The MLP performs these transformations.


---

10.4 The Simplest MLP

The simplest possible network is:

Input

↓

Linear Layer

↓

Output

Mathematically,

\[
y = Wx + b
\]

where:

\(x\) is the input vector,

\(W\) is the weight matrix,

\(b\) is the bias.



---

10.5 Why One Linear Layer Isn't Enough

Suppose we have two linear layers:

\[
y=W_2(W_1x)
\]

Because matrix multiplication is associative,

\[
W_2W_1=W
\]

Therefore,

\[
y=Wx
\]

This means:

> Two linear layers without a non-linear activation are mathematically equivalent to one larger linear layer.



So stacking linear layers alone does not increase the expressive power of the model.


---

10.6 The Importance of Non-Linearity

Real-world relationships are rarely linear.

Imagine house prices.

Doubling the size of a house does not always double its price.

Similarly,

understanding language or images requires modelling complex, non-linear relationships.

To do this, we insert a non-linear activation function.

Input

↓

Linear

↓

Activation

↓

Linear

↓

Output

Now the model can learn much richer transformations.


---

10.7 Why Expand the Hidden Dimension?

One of the most surprising design choices in Transformers is this:

Suppose the embedding size is:

768

The MLP expands it to:

3072

Then compresses it back:

768

Diagram:

768

↓

3072

↓

768

Why make the vector larger only to shrink it again?


---

10.8 A Workshop Analogy

Imagine repairing a watch.

If your workbench is tiny, you cannot spread out the gears and tools.

Everything gets crowded.

Now imagine a much larger workbench.

You can:

separate the parts,

inspect them carefully,

rearrange them,

assemble them properly.


Finally, you put the repaired watch back together.

The larger workbench is like the expanded hidden dimension.

It provides the model with more "working space" to perform complex transformations.


---

10.9 Mathematical View

Suppose:

Embedding size:

\[
d = 768
\]

The first linear layer computes:

\[
768 \rightarrow 3072
\]

The activation is applied.

Then the second linear layer computes:

\[
3072 \rightarrow 768
\]

This expanded representation allows the model to combine and transform features in ways that would be difficult in the smaller space alone.


---

10.10 Why GELU?

Many neural networks use the ReLU activation:

\[
\text{ReLU}(x)=\max(0,x)
\]

Graphically:

/
     /
----/

Negative values become exactly zero.

Transformers often use GELU (Gaussian Error Linear Unit) instead.

GELU behaves more smoothly.

Instead of abruptly discarding all negative values, it gradually reduces them.

A simplified comparison:

ReLU

      /
     /
----/

GELU

     /
   /
 _/

The smooth transition often improves optimization and model performance, which is why GELU is widely used in architectures such as Vision Transformers.


---

10.11 Step-by-Step Flow Through the MLP

Suppose one token enters the MLP.

Embedding

↓

[768 numbers]

↓

Linear

↓

3072 numbers

↓

GELU

↓

Linear

↓

768 numbers

Notice:

Only one token is being processed.

The neighboring tokens are not involved.

Every token goes through the same MLP independently.


---

10.12 Attention vs MLP

This distinction is extremely important.

Attention	MLP

Communicates between tokens	Processes each token independently
Learns relationships	Learns feature transformations
Answers: "Who should I look at?"	Answers: "How should I update myself?"
Mixes information	Refines information


Both are essential.

Without attention, tokens cannot exchange information.

Without the MLP, tokens cannot deeply transform the information they have gathered.


---

10.13 Vision Transformer Example

Suppose an image contains:

🐦

One image patch attends to:

sky,

tree,

branch.


Attention gathers this context.

The MLP then transforms the patch representation into something richer, perhaps making the "bird" concept more distinct by emphasizing the combination of features that consistently appear together during training.

The exact features are learned by the model rather than being manually defined.


---

10.14 MLP in SAM

In the Segment Anything Model:

Attention gathers information from image patches across the scene.

The MLP then refines each patch's representation independently.

By repeating this process across many encoder blocks, SAM develops highly informative feature representations that help distinguish objects, boundaries, and regions before the segmentation decoder predicts the final mask.


---

10.15 Why the Same MLP for Every Token?

A common question is:

> Why doesn't every token have its own MLP?



Imagine a teacher grading 100 answer sheets.

The teacher uses the same grading rules for every student.

If each student had a different grading system, the evaluation would be inconsistent.

Similarly, the Transformer applies the same MLP parameters to every token.

This keeps the model consistent and greatly reduces the number of trainable parameters.


---

Common Misconceptions

❌ "The MLP mixes information between tokens."

No.

Each token passes through the MLP independently.

Only the attention layer mixes information across tokens.


---

❌ "The MLP is less important than attention."

No.

The MLP is a core part of the Transformer and often contains a large share of its parameters.


---

❌ "Expanding the hidden dimension is wasteful."

No.

The larger intermediate representation gives the network more capacity to learn rich feature transformations before projecting back to the original embedding size.


---

Key Takeaways

The MLP transforms each token after attention has gathered contextual information.

It consists of:

1. Linear layer (expand dimensions)


2. Non-linear activation (often GELU)


3. Linear layer (project back)



The non-linear activation is essential; without it, multiple linear layers collapse into a single linear transformation.

The expanded hidden dimension provides additional capacity for learning complex feature interactions.

Attention and the MLP play complementary roles in the Transformer.



---

Practice Questions

Conceptual

1. Why is a non-linear activation required between the two linear layers?


2. Why does the MLP expand the hidden dimension before shrinking it again?


3. How do the roles of attention and the MLP differ?


4. Why is the same MLP applied to every token?


5. Why is GELU often preferred over ReLU in Transformers?



Thought Exercise

Imagine you remove the MLP from every encoder block but keep the attention layers.

What capabilities would the model retain?

What kinds of transformations might become difficult?

How do you think this would affect performance on language understanding or image recognition tasks?



---

Chapter Summary

The Feed-Forward Network (MLP) is the transformer's "thinking" stage. After self-attention gathers information from across the sequence or image, the MLP independently refines each token's representation through two linear layers separated by a non-linear activation, typically GELU. Expanding the hidden dimension gives the model additional capacity to learn complex feature interactions before projecting back to the original embedding size. Together, attention and the MLP form the two complementary computational engines of every Transformer encoder block.


---

Next Lesson

In the next chapter, we'll move from individual components to the complete Vision Transformer (ViT) pipeline.

We'll build ViT from the ground up:

1. Splitting an image into patches


2. Patch embedding


3. Adding the [CLS] token


4. Adding positional embeddings


5. Passing tokens through stacked Transformer encoder blocks


6. Using the final [CLS] representation for image classification



This marks the transition from learning the Transformer itself to understanding how it becomes a powerful computer vision model.
