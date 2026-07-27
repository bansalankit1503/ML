Excellent. Today we study one of the most misunderstood topics in Computer Vision.

Most beginners think:

> "RGB just means Red, Green, and Blue."

While that's true, it doesn't answer **why** those three colors were chosen, **how** they can represent millions of colors, or **why** almost every deep learning model—from CNNs to Vision Transformers and SAM—expects RGB images as input.

This chapter will answer those questions from first principles.

---

# Module 1 – Foundations of Computer Vision

## Part 1 – Computer Vision Foundations

# Chapter 5 — Understanding Color and the RGB Color Model

---

# Learning Objectives

By the end of this chapter, you will understand:

* What color really is
* Why humans perceive different colors
* Why RGB was chosen
* Additive vs. subtractive color mixing
* How millions of colors are represented
* Bit depth
* RGB image representation
* Alpha channel
* Why OpenCV uses BGR
* Introduction to other color spaces

---

# 5.1 What is Color?

Let's begin with a simple question.

**Does color exist in the real world?**

Surprisingly...

**Not exactly.**

Objects do not contain "redness" or "blueness."

Color is a **perception created by our brain** based on the wavelength of light entering our eyes.

Imagine sunlight.

```text
          ☀ Sun
             │
             ▼
      White Light
             │
             ▼
          🌈 Prism
             │
             ▼
 Violet Blue Green Yellow Orange Red
```

White sunlight contains many wavelengths.

Each wavelength corresponds to what we perceive as a different color.

Approximate visible wavelengths:

| Color  | Wavelength |
| ------ | ---------- |
| Violet | 380–450 nm |
| Blue   | 450–495 nm |
| Green  | 495–570 nm |
| Yellow | 570–590 nm |
| Orange | 590–620 nm |
| Red    | 620–750 nm |

So when we say,

> "This apple is red."

what we really mean is:

> "This apple reflects light whose dominant wavelength is around 620–750 nm."

---

# 5.2 Why Does a Banana Look Yellow?

Suppose sunlight falls on a banana.

```text
White Light
      │
      ▼
   🍌 Banana
      │
      ▼
Mostly Yellow Light Reflected
```

The banana absorbs many wavelengths and reflects wavelengths that our eyes interpret as yellow.

Notice:

The banana **does not create yellow light.**

It selectively reflects certain wavelengths.

This idea explains the color of every object around us.

---

# 5.3 Human Color Vision

Inside the retina are special cells called **cones**.

Humans usually have three types:

| Cone Type | Sensitive To |
| --------- | ------------ |
| S-cones   | Blue light   |
| M-cones   | Green light  |
| L-cones   | Red light    |

Notice something interesting.

We **don't have one sensor for every possible color.**

Instead, our brain estimates millions of colors by combining signals from only **three** cone types.

This biological fact inspired the RGB color model.

---

# 5.4 Why RGB?

This is one of the most common interview questions.

Why didn't engineers choose:

* Red, Yellow, Blue?
* Purple, Orange, Green?
* Seven rainbow colors?

Because humans perceive color primarily through **three types of cones**.

Therefore, recording the amount of:

* Red
* Green
* Blue

is sufficient to reproduce most colors that humans can perceive.

RGB is therefore **a perceptual model**, not a complete physical description of light.

---

# 5.5 The RGB Cube

Every RGB color is represented by three numbers.

```text
(R, G, B)
```

Each value usually ranges from:

```text
0 → 255
```

Example:

Pure Red

```text
(255,0,0)
```

Pure Green

```text
(0,255,0)
```

Pure Blue

```text
(0,0,255)
```

Black

```text
(0,0,0)
```

White

```text
(255,255,255)
```

Gray

```text
(128,128,128)
```

Yellow

```text
(255,255,0)
```

Cyan

```text
(0,255,255)
```

Magenta

```text
(255,0,255)
```

Every visible pixel in an RGB image stores one such triplet.

---

# 5.6 How Many Colors Can RGB Represent?

Each channel has:

```text
256 possible values
```

Three independent channels:

```text
256 × 256 × 256
```

Which equals:

```text
16,777,216 colors
```

This is why your phone can display millions of colors even though each pixel stores only three numbers.

---

# 5.7 Additive Color Mixing

RGB is an **additive color model**.

Think about three flashlights:

* Red
* Green
* Blue

Shining them together produces new colors.

```text
Red + Green = Yellow

Green + Blue = Cyan

Red + Blue = Magenta

Red + Green + Blue = White
```

Why?

Because adding more light increases the total light reaching your eyes.

This is exactly how:

* phone screens
* TVs
* monitors
* laptop displays

produce images.

Each screen pixel contains tiny red, green, and blue light emitters.

---

# 5.8 Subtractive Color Mixing

Printers work differently.

Ink **absorbs** light instead of emitting it.

They use:

* Cyan
* Magenta
* Yellow
* Black (CMYK)

Example:

```text
White Paper

↓

Add Cyan Ink

↓

Less Red Light Reflected

↓

Different Color Appears
```

Computer Vision mostly works with RGB because cameras and displays use emitted or measured light rather than ink.

---

# 5.9 Bit Depth

Each RGB channel usually uses:

```text
8 bits
```

Possible values:

```text
0–255
```

So one RGB pixel contains:

```text
Red   → 8 bits

Green → 8 bits

Blue  → 8 bits
```

Total:

```text
24 bits per pixel
```

This is why standard color images are often called **24-bit images**.

---

# 5.10 Memory Representation

Suppose we have a tiny 2×2 RGB image.

Visually:

```text
🟥 🟩

🟦 ⬜
```

Internally:

```text
[
 [(255,0,0), (0,255,0)],

 [(0,0,255), (255,255,255)]
]
```

This is how NumPy represents RGB images.

Shape:

```text
Height × Width × Channels

2 × 2 × 3
```

Example:

```python
image.shape

(224,224,3)
```

---

# 5.11 PyTorch Representation

PyTorch stores channels first.

Instead of:

```text
Height × Width × Channels
```

PyTorch uses:

```text
Channels × Height × Width
```

Example:

NumPy

```text
224 × 224 × 3
```

PyTorch

```text
3 × 224 × 224
```

For a batch:

```text
Batch × Channels × Height × Width

32 × 3 × 224 × 224
```

Understanding this layout is essential when training CNNs and Vision Transformers.

---

# 5.12 What is the Alpha Channel?

Sometimes images also store transparency.

Instead of:

```text
RGB
```

they store:

```text
RGBA
```

Where:

A = Alpha

Example:

```text
(255,0,0,255)
```

means:

Opaque red.

```text
(255,0,0,0)
```

means:

Completely transparent red.

PNG images often support an alpha channel.

JPEG does not.

---

# 5.13 Why Does OpenCV Use BGR?

This confuses many beginners.

Suppose you load an image:

```python
import cv2

image = cv2.imread("cat.jpg")
```

Many expect:

```text
(R,G,B)
```

But OpenCV stores:

```text
(B,G,R)
```

This is a historical design decision dating back to older Windows bitmap formats.

It is not mathematically better—it is simply the convention OpenCV follows.

When displaying images with libraries like Matplotlib, developers often convert BGR to RGB to avoid incorrect colors.

---

# 5.14 Other Color Spaces

RGB is not always the best representation.

Different tasks benefit from different color spaces.

| Color Space | Purpose                      |
| ----------- | ---------------------------- |
| RGB         | Display, deep learning input |
| HSV         | Color-based segmentation     |
| LAB         | Perceptually uniform colors  |
| YCbCr       | Video compression            |
| CMYK        | Printing                     |
| Grayscale   | Simpler image processing     |

We'll study each of these in detail later in the course.

---

# RGB in CNNs vs Vision Transformers

### CNN

Each convolution kernel processes all three color channels simultaneously.

For example, a 3×3 kernel on an RGB image actually operates on a **3 × 3 × 3** block of values.

---

### Vision Transformer (ViT)

Before patch embedding, each image patch contains RGB values.

Example:

```text
16 × 16 patch

↓

16 × 16 × 3 values

↓

Flatten

↓

Linear Projection

↓

Embedding Vector
```

So although ViTs reason over patches, the underlying information still begins as RGB pixel values.

---

# Real-World Example

Imagine taking a selfie outdoors.

The camera records the amount of red, green, and blue light reaching each photosite.

Image processing reconstructs the full RGB image.

That RGB image is then normalized and passed to a face recognition model, an object detector, or a Vision Transformer.

Without the RGB representation, the neural network would have no color information to learn from.

---

# Common Misconceptions

❌ **"RGB is based on the three primary colors taught in art class."**

Not exactly. RGB is based on **human visual perception** and additive light mixing, whereas art classes often discuss pigment mixing.

---

❌ **"Each pixel stores a color name."**

No. It stores three numerical intensities.

---

❌ **"OpenCV is wrong because it uses BGR."**

No. It simply follows a different storage convention.

---

# Key Takeaways

* Color is a perception created by the brain from different wavelengths of light.
* Humans have three cone types, which inspired the RGB model.
* RGB represents colors using three numerical channels.
* An 8-bit RGB image can represent over **16.7 million** colors.
* Screens use additive RGB mixing, while printers use subtractive CMYK mixing.
* NumPy typically stores images as **Height × Width × Channels**, whereas PyTorch uses **Channels × Height × Width**.
* Understanding channel order is essential when building deep learning pipelines.

---

# Practice Questions

### Conceptual

1. Why is RGB based on human vision?
2. Explain the difference between additive and subtractive color mixing.
3. Why can three channels represent millions of colors?
4. Why does PyTorch store channels first?
5. Why does OpenCV use BGR instead of RGB?

### Numerical

1. How many colors can be represented by a 10-bit RGB image per channel?
2. How much memory (in bytes) is required to store a **512 × 512 RGB** image with 8 bits per channel (ignore metadata and compression)?
3. What is the tensor shape of a batch of **64 RGB images**, each of size **224 × 224**, in PyTorch?

---

# Markdown File

Save this chapter as:

```text
Module-01/
└── Part-01/
    └── Chapter-05-Understanding-Color-and-the-RGB-Color-Model.md
```

This Markdown chapter should preserve all explanations, tables, examples, diagrams, and practice questions so it becomes the fifth chapter of your textbook.

---

## Next Chapter (Chapter 6)

In the next lesson, we'll study **Image Resolution and Image Size** in depth, including:

* What resolution really means
* Spatial resolution vs. intensity resolution
* DPI vs. PPI vs. pixel resolution
* Aspect ratio
* Image resizing
* Upsampling and downsampling
* Aliasing
* Anti-aliasing
* Why Vision Transformers resize images to **224 × 224**
* Why SAM uses **1024 × 1024**
* How resizing affects deep learning performance

This chapter will explain why image size is far more than just "width × height" and how it impacts every computer vision model.
