Excellent. This is one of the deepest and most important chapters in the entire Vision Transformer journey.

Many people learn how a Vision Transformer works but never understand why it succeeded—or why, at first, it actually performed worse than CNNs on many datasets.

This chapter explains the missing piece:

> Inductive Bias



Understanding this concept will also help you understand why models like SAM, DINOv2, CLIP, and modern multimodal models use transformer-based vision encoders.


---

Part IV – Vision Transformers (ViT)

Chapter 14 — Why Vision Transformers Work: Inductive Bias, Data Requirements, and Scaling


---

Learning Objectives

By the end of this chapter, you will understand:

What inductive bias means

Why CNNs have strong inductive biases

Why Vision Transformers have much weaker inductive biases

Why early ViTs required enormous datasets

How scaling changed the landscape of computer vision

Why self-supervised learning made ViTs practical

Why modern models such as SAM and CLIP rely on Vision Transformers



---

14.1 The Big Question

When the original Vision Transformer paper was published in 2020, many researchers asked:

> If Transformers are so powerful, why weren't they used for images years earlier?



The answer is surprisingly simple:

> Transformers know almost nothing about images before training.



CNNs, on the other hand, already "expect" certain properties of images.

This expectation is called an inductive bias.


---

14.2 What Is Inductive Bias?

An inductive bias is a built-in assumption that a learning algorithm makes before seeing any training data.

Think of it as the model's starting belief about the world.

For example, imagine two children.

Child A has never seen a dog.

Child B has been told:

> "Dogs usually have four legs, a tail, and a face."



When both children see a new dog, Child B learns faster because of that prior knowledge.

That prior knowledge is analogous to an inductive bias.


---

14.3 Everyday Analogy

Imagine learning chess.

Student 1

Starts with:

no rules

no examples

no guidance


They must discover everything themselves.

Student 2

Starts knowing:

how each piece moves

the goal of checkmate

basic opening principles


Who learns faster?

Obviously, Student 2.

CNNs are like Student 2.

Vision Transformers are closer to Student 1.


---

14.4 Why CNNs Have Strong Inductive Bias

Consider a 3×3 convolution filter.

+---+---+---+
|   |███|   |
+---+---+---+
|███|███|███|
+---+---+---+
|   |███|   |
+---+---+---+

This filter only looks at nearby pixels.

From the very beginning, CNNs assume:

nearby pixels are related,

edges are useful,

local textures matter.


These assumptions are built into the architecture itself.


---

14.5 Translation Equivariance

Suppose a cat appears in the top-left corner.

🐈

Now move the cat to the bottom-right.

🐈

The same convolution filter scans the entire image.

The filter doesn't care where the cat appears.

This property is called translation equivariance.

It allows CNNs to detect the same pattern at different locations using the same learned weights.


---

14.6 Locality

CNNs naturally assume:

> Nearby pixels are more closely related than distant pixels.



Example:

👁️ 👃 👄

The eye is more related to the nose than to a tree on the other side of the image.

This assumption is true for many natural images.

As a result, CNNs don't need to learn this relationship—it is already built into the architecture.


---

14.7 What About Vision Transformers?

A Vision Transformer starts with image patches.

P1  P2  P3

P4  P5  P6

P7  P8  P9

Initially, it has no built-in preference for nearby patches.

Patch P1 can attend to:

P2

P5

P9


All are treated as possible interactions.

The model must learn from data which relationships are useful.


---

14.8 The Cost of Fewer Assumptions

Because ViTs make fewer assumptions, they require more experience.

Imagine teaching two people to recognize birds.

Person A

Already knows:

wings

feathers

beaks


Person B

Knows nothing about animals.

Who needs more examples?

Clearly, Person B.

Similarly, ViTs generally need more data than CNNs to learn good visual representations from scratch.


---

14.9 Why the Original ViT Used Huge Datasets

The original Vision Transformer paper trained on very large datasets such as JFT-300M (an internal dataset at Google with hundreds of millions of labeled images).

Why?

Because with enough diverse examples, the model can learn useful visual patterns without relying on strong architectural assumptions.

The key lesson was:

> With sufficient data and compute, weaker inductive bias is not necessarily a disadvantage.




---

14.10 Scaling Changes Everything

A remarkable observation in deep learning is the scaling law:

As we increase:

model size,

dataset size,

training compute,


performance often improves in a predictable way.

A simple illustration:

Performance
 ^
 |                           ●
 |                      ●
 |                 ●
 |            ●
 |       ●
 |  ●
 +---------------------------->
      Data / Compute / Model Size

While the exact relationship depends on the model and task, many modern transformer models exhibit this general trend.


---

14.11 CNNs vs ViTs as Data Grows

Imagine evaluating both models on increasingly large datasets.

Accuracy
 ^
 |                         ViT
 |                      /
 |                   /
 |                /
 |             /
 | CNN  ______/
 |
 +---------------------------->
        Training Data

With limited data:

CNNs often outperform ViTs because of their strong inductive bias.


With abundant data:

ViTs frequently catch up and may surpass CNNs on many benchmarks.



---

14.12 Self-Supervised Learning Changed the Game

Collecting labeled images is expensive.

Instead of relying only on labels, researchers developed self-supervised learning.

The idea:

> Learn useful visual representations from the images themselves.



Examples include:

predicting masked image patches,

matching different views of the same image,

learning from image-text pairs.


This allows models to leverage enormous amounts of unlabeled data.


---

14.13 Why This Helped Vision Transformers

Self-supervised learning gives Vision Transformers extensive visual experience before they are fine-tuned for a specific task.

Instead of starting from random weights, the model already understands many visual concepts such as:

edges,

textures,

shapes,

object parts,

semantic relationships.


This dramatically reduces the amount of labeled data needed for downstream tasks.


---

14.14 Connection to CLIP, DINO, and MAE

Modern vision foundation models take advantage of large-scale pretraining.

Examples:

CLIP learns from image–text pairs.

DINO/DINOv2 learns visual representations through self-supervised objectives.

MAE (Masked Autoencoders) learns by reconstructing masked image patches.


Although the training objectives differ, they all benefit from the flexibility and scalability of Vision Transformers.


---

14.15 Why SAM Uses a Vision Transformer

SAM is expected to segment:

animals,

people,

roads,

furniture,

medical-like structures,

everyday objects,


and much more.

Building separate CNNs for every possible scenario would not be practical.

Instead, SAM uses a large Vision Transformer encoder that has learned broad, general-purpose visual representations from extensive training.

These rich representations are then combined with prompts to produce segmentation masks.


---

14.16 Strengths and Weaknesses

CNN	Vision Transformer

Strong inductive bias	Weak inductive bias
Learns efficiently from smaller datasets	Typically benefits from larger datasets
Naturally captures local patterns	Naturally models global relationships
Often more computationally efficient for smaller models	Scales well with larger models and datasets


Neither approach is universally superior; the best choice depends on the task, available data, and computational resources.


---

Common Misconceptions

❌ "Weak inductive bias means Vision Transformers are worse."

No.

It means they start with fewer built-in assumptions. With sufficient data and training, this flexibility can become a major advantage.


---

❌ "CNNs are obsolete."

Not at all.

CNNs remain highly effective in many applications, especially when data or compute is limited, or when efficient deployment is important.


---

❌ "Scaling guarantees perfect performance."

No.

Scaling often improves performance, but gains eventually diminish, and practical limitations such as compute, memory, data quality, and optimization still matter.


---

Key Takeaways

Inductive bias refers to the assumptions a model makes before training.

CNNs have strong inductive biases such as locality and translation equivariance.

Vision Transformers have weaker inductive biases and therefore often require more data when trained from scratch.

Large-scale pretraining and self-supervised learning made Vision Transformers practical and highly successful.

Modern vision foundation models—including CLIP, DINO, MAE, and SAM—build on these ideas.



---

Practice Questions

Conceptual

1. What is inductive bias, and why is it useful?


2. Why do CNNs often perform well on smaller datasets?


3. Why did the original Vision Transformer rely on extremely large datasets?


4. How does self-supervised learning reduce the need for labeled data?


5. Why is a weaker inductive bias not necessarily a disadvantage?



Thought Exercise

Imagine you need to build an image classifier in two scenarios:

1. You have 10,000 labeled images.


2. You have 100 million unlabeled images and 50,000 labeled images.



For each scenario:

Would you choose a CNN or a Vision Transformer?

Would you train from scratch or use a pretrained model?

Explain your reasoning based on inductive bias and data availability.



---

Chapter Summary

The success of Vision Transformers is not just due to the attention mechanism—it is also a story about data, compute, and inductive bias. CNNs embed strong assumptions about images, enabling efficient learning from limited data. Vision Transformers begin with far fewer assumptions, making them more flexible but also more data-hungry when trained from scratch. As large-scale pretraining, self-supervised learning, and increased computational resources became available, Vision Transformers demonstrated exceptional scalability and became the foundation for many state-of-the-art vision systems, including the image encoder used in SAM.


---

What's Next?

With this chapter, you now understand:

✔️ How Transformers work

✔️ How images become tokens

✔️ How Vision Transformers process images

✔️ Why Vision Transformers succeeded


You are now ready to begin the Segment Anything Model (SAM) itself.

Part V — Segment Anything Model (SAM)

The first chapter will be:

Chapter 15 — Why Segment Anything? The Evolution of Image Segmentation

We'll explore:

What image segmentation is

Semantic vs Instance vs Panoptic segmentation

Why traditional segmentation models had important limitations

The motivation behind SAM

The key idea that made "Segment Anything" possible


This marks the transition from understanding Vision Transformers to understanding one of the most influential foundation models in computer vision.
