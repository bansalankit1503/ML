Excellent. This is one of the **most important chapters in the entire book**.

Until now, we've learned **what images are**.

Now we answer a much bigger question:

> **What kinds of problems can Computer Vision solve?**

This chapter is important because almost every modern Computer Vision paper begins by stating **which vision task it addresses**.

For example:

* **ResNet** → Image Classification
* **YOLO** → Object Detection
* **Mask R-CNN** → Instance Segmentation
* **DETR** → Object Detection
* **SAM** → Image Segmentation
* **CLIP** → Vision-Language Understanding
* **Florence-2** → Multi-task Vision Model

Understanding these tasks will help you immediately understand what any new research paper is trying to solve.

---

# Module 1 – Foundations of Computer Vision

## Part 1 – Computer Vision Foundations

# Chapter 10 — Fundamental Computer Vision Tasks

---

# Learning Objectives

By the end of this chapter, you will understand:

* What a Computer Vision task is
* The hierarchy of vision tasks
* Image Classification
* Localization
* Object Detection
* Semantic Segmentation
* Instance Segmentation
* Panoptic Segmentation
* Keypoint Detection
* Pose Estimation
* Object Tracking
* Optical Flow
* OCR
* Image Captioning
* Visual Question Answering (VQA)
* Which deep learning models solve each task

---

# 10.1 What is a Computer Vision Task?

Imagine you're shown the following photograph.

```text
          🐶     🚗

      🌳         🚲
```

Different people may ask different questions.

One person asks:

> "What is in this image?"

Another asks:

> "Where is the dog?"

Another asks:

> "How many objects are there?"

Another asks:

> "Which pixels belong to the car?"

Notice something important.

The **image is the same**.

Only the **question changes**.

That question defines the Computer Vision task.

> **A Computer Vision task specifies what information we want the computer to extract from an image or video.**

---

# 10.2 The Evolution of Vision Tasks

Historically, Computer Vision became progressively more detailed.

Think of a detective.

### Stage 1

"I know there is a dog."

↓

### Stage 2

"I know where the dog is."

↓

### Stage 3

"I know every pixel belonging to the dog."

↓

### Stage 4

"I know this is Dog #1, not Dog #2."

Each stage requires more understanding.

---

# Vision Task Hierarchy

```text
Image

│

├── Classification

│

├── Localization

│

├── Object Detection

│

├── Semantic Segmentation

│

├── Instance Segmentation

│

├── Panoptic Segmentation

│

├── Pose Estimation

│

├── Tracking

│

├── OCR

│

├── Captioning

│

└── Visual Question Answering
```

---

# 10.3 Image Classification

The simplest vision task.

Question:

> **What is present in this image?**

Input:

```text
🐶
```

Output:

```text
Dog
```

The model predicts one or more labels.

Examples:

* Dog
* Cat
* Airplane
* Tree
* Building

No location information is provided.

---

## Real Example

Chest X-ray

↓

Prediction

```text
Pneumonia
```

The model says what exists.

It does **not** indicate where.

---

## Popular Models

* AlexNet
* VGG
* ResNet
* EfficientNet
* Vision Transformer (ViT)

---

# 10.4 Image Localization

Now we ask:

> **Where is the object?**

Instead of only predicting:

```text
Dog
```

The model predicts

```text
Dog

Bounding Box
```

Example

```text
+-------------+

|             |

|    🐶       |

|             |

+-------------+
```

Output

```text
Dog

(x,y,w,h)
```

Localization predicts one object and its location.

---

# 10.5 Object Detection

Real images contain many objects.

Example

```text
🐶 🚗 🚲 🌳
```

Output

```text
Dog

Car

Bicycle

Tree
```

Each object receives

* label
* bounding box

Example

```text
Dog

(40,70,120,180)

Car

(300,120,420,250)
```

Popular models:

* YOLO
* SSD
* Faster R-CNN
* RetinaNet
* DETR

---

# Why Detection is Harder

Classification:

One prediction.

Detection:

Many predictions.

The model must answer:

* What?
* Where?
* How many?

simultaneously.

---

# 10.6 Semantic Segmentation

Suppose we have three dogs.

```text
🐶 🐶 🐶
```

Semantic segmentation colors pixels according to their class.

Example

```text
Blue

Blue

Blue
```

All dogs receive the same label.

The model knows

> "These pixels belong to dogs."

But it cannot distinguish Dog 1 from Dog 2.

---

## Output

Instead of rectangles,

the output becomes

```text
Pixel Labels
```

Every pixel belongs to one semantic class.

Applications:

* Road segmentation
* Medical imaging
* Satellite mapping

---

# 10.7 Instance Segmentation

Now suppose there are three dogs.

Instead of

```text
Dog

Dog

Dog
```

the model predicts

```text
Dog 1

Dog 2

Dog 3
```

Each object receives:

* class
* individual mask

This is much more informative.

---

## Example

```text
Red Mask

↓

Dog 1

Blue Mask

↓

Dog 2

Green Mask

↓

Dog 3
```

Popular models:

* Mask R-CNN
* YOLACT
* SOLO
* SAM (when prompted)

---

# 10.8 Panoptic Segmentation

Panoptic segmentation combines

Semantic

*

Instance

segmentation.

Example

Road

↓

Semantic

Every road pixel has the same label.

Cars

↓

Instance

Each car receives its own identity.

Output

```text
Road

Road

Road

Car 1

Car 2

Person 1
```

Everything is labeled.

---

# 10.9 Pose Estimation

Instead of predicting objects,

we predict body joints.

Example

```text
🙂

● Head

● Shoulder

● Elbow

● Wrist

● Hip

● Knee

● Ankle
```

Applications:

* Fitness apps
* Sports analytics
* Animation
* Human-computer interaction

Popular models:

* OpenPose
* HRNet
* BlazePose

---

# 10.10 Object Tracking

Imagine watching a video.

Frame 1

```text
🚗
```

Frame 2

```text
🚗
```

Frame 3

```text
🚗
```

Tracking assigns the same identity across frames.

Instead of

```text
Car

Car

Car
```

the model predicts

```text
Car #7
```

throughout the video.

Applications:

* Surveillance
* Autonomous driving
* Sports analysis

---

# 10.11 Optical Flow

Tracking follows objects.

Optical Flow estimates **pixel motion**.

Suppose a ball moves.

Frame A

```text
⚽
```

Frame B

```text
      ⚽
```

Optical flow predicts

```text
→→→→→
```

for each moving pixel.

Applications:

* Video stabilization
* Motion estimation
* Autonomous vehicles

---

# 10.12 OCR (Optical Character Recognition)

Input

```text
STOP
```

Output

```text
"S T O P"
```

The task converts images into text.

Applications:

* Document scanning
* Passport reading
* License plate recognition
* Invoice processing

Popular engines:

* Tesseract OCR
* EasyOCR
* PaddleOCR

---

# 10.13 Image Captioning

Instead of one label,

the model generates a sentence.

Input

```text
👧🐶⚽
```

Output

> "A young girl is playing with a dog in a park."

The model combines:

Computer Vision

*

Natural Language Processing

---

# 10.14 Visual Question Answering (VQA)

Now the model receives

Image

*

Question

Example

Image

```text
🐶⚽
```

Question

> What is the dog playing with?

Output

```text
Ball
```

The model must understand both:

* vision
* language

Popular models:

* BLIP
* Flamingo
* Florence-2
* GPT-4V-style multimodal systems

---

# 10.15 Multi-Task Models

Modern models solve many tasks at once.

Examples:

| Model          | Tasks                                    |
| -------------- | ---------------------------------------- |
| YOLOv11        | Detection, segmentation, pose            |
| SAM            | Segmentation                             |
| Florence-2     | Captioning, OCR, detection, segmentation |
| CLIP           | Classification, retrieval                |
| Grounding DINO | Detection from text prompts              |

The trend in Computer Vision is moving toward **general-purpose vision models** rather than one model per task.

---

# Task Comparison

| Task                  | Output                      |
| --------------------- | --------------------------- |
| Classification        | Class label                 |
| Localization          | Label + one bounding box    |
| Detection             | Multiple labels + boxes     |
| Semantic Segmentation | Pixel class map             |
| Instance Segmentation | Individual object masks     |
| Panoptic Segmentation | Complete scene labeling     |
| Pose Estimation       | Keypoints                   |
| Tracking              | Persistent IDs across video |
| OCR                   | Text                        |
| Captioning            | Sentence                    |
| VQA                   | Answer to a question        |

---

# Which Models Solve Which Tasks?

| Model      | Classification |   Detection   |  Segmentation | Captioning | VQA |
| ---------- | :------------: | :-----------: | :-----------: | :--------: | :-: |
| ResNet     |        ✅       |       ❌       |       ❌       |      ❌     |  ❌  |
| ViT        |        ✅       |       ❌*      |       ❌*      |      ❌     |  ❌  |
| YOLO       |        ❌       |       ✅       | Some versions |      ❌     |  ❌  |
| Mask R-CNN |        ❌       |       ✅       |       ✅       |      ❌     |  ❌  |
| SAM        |        ❌       | Prompt-guided |       ✅       |      ❌     |  ❌  |
| Florence-2 |        ✅       |       ✅       |       ✅       |      ✅     |  ✅  |

> *ViT serves as a backbone in many detection and segmentation systems but is not, by itself, a complete detection or segmentation model.

---

# Real-World Example

Consider an autonomous vehicle.

A single camera frame may require multiple vision tasks simultaneously:

* **Classification** identifies traffic sign categories.
* **Object Detection** locates cars, pedestrians, and bicycles.
* **Semantic Segmentation** labels the road, sidewalks, and vegetation.
* **Instance Segmentation** separates one pedestrian from another.
* **Tracking** follows nearby vehicles across consecutive frames.
* **OCR** reads speed limit signs.
* **VQA** (in an assistant system) could answer questions like, "How many pedestrians are crossing?"

Modern autonomous systems often combine several of these tasks into one perception pipeline.

---

# Common Misconceptions

❌ **"Object Detection and Segmentation are the same."**

Detection predicts bounding boxes. Segmentation predicts pixel-level masks.

---

❌ **"Classification tells us where the object is."**

No. Classification only predicts *what* is present.

---

❌ **"SAM is an object detector."**

Not exactly. SAM is a segmentation model. It requires prompts (such as points, boxes, or masks) and predicts segmentation masks rather than directly detecting objects.

---

# Key Takeaways

* A Computer Vision task is defined by the information we want to extract from an image.
* Tasks become progressively more detailed, from classification to panoptic segmentation.
* Detection predicts boxes; segmentation predicts pixels.
* Pose estimation, tracking, OCR, captioning, and VQA extend vision beyond object recognition.
* Modern foundation models increasingly support multiple vision tasks within a single architecture.

---

# Practice Questions

## Conceptual

1. What distinguishes a Computer Vision task from the input image itself?
2. Compare image classification and object detection.
3. Explain the difference between semantic segmentation and instance segmentation.
4. Why is panoptic segmentation considered a combination of two earlier tasks?
5. Why is VQA considered a multimodal task?

## Scenario-Based

1. Which task would you choose to:

   * Count people entering a building?
   * Highlight every road pixel for autonomous driving?
   * Read text from scanned invoices?
   * Track a football player across an entire match?
   * Generate a natural-language description of a photograph?

2. Explain why SAM is well suited for interactive segmentation but is not a complete replacement for an object detector like YOLO.

---

# Chapter Summary

Computer Vision is not one problem—it is a collection of related tasks that answer different questions about visual data.

As the desired output becomes more detailed, models must understand increasingly complex relationships within the image. This progression—from classification to segmentation and beyond—explains why modern architectures have evolved from simple CNN classifiers to powerful multimodal foundation models like SAM and Florence-2.

Understanding these task categories provides the roadmap for the rest of this book, where you'll learn how different neural network architectures were designed to solve specific vision problems.

---

# Markdown File

Save this chapter as:

```text
Module-01/
└── Part-01/
    └── Chapter-10-Fundamental-Computer-Vision-Tasks.md
```

This Markdown chapter should preserve all explanations, diagrams, comparison tables, examples, and practice questions so it becomes the tenth chapter of your textbook.

---

# Next Chapter (Chapter 11 — The Complete Computer Vision Pipeline)

In the final chapter of Part 1, we'll bring everything together into one end-to-end workflow:

* From photons to pixels
* Camera capture
* Image preprocessing
* Data augmentation
* Feature extraction
* Classical Computer Vision pipeline
* Deep learning pipeline
* CNN pipeline
* Vision Transformer pipeline
* SAM segmentation pipeline
* Training vs. inference
* Where each concept from Chapters 1–10 fits into the complete system

This chapter will serve as the bridge from **Computer Vision Foundations** into **traditional image processing and feature extraction**, which we'll study in the next part of Module 1.
