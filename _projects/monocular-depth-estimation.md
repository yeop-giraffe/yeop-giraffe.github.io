---
layout: page
title: Monocular Depth Estimation
description: Domain-transferred synthetic data generation for improving monocular depth estimation.
img:
importance: 3
category: research
display_category: Robot Vision / Monocular Depth Estimation
period: Nov.2023 - Feb.2024
summary: Domain-transferred synthetic data generation for improving monocular depth estimation using CycleGAN and LiDAR-based evaluation.
---

## Overview

Investigated how domain-transferred synthetic data can improve supervised monocular depth estimation. Simulation images were translated toward the visual appearance of real environments and then mixed with real data for model training.

**Period:** Nov.2023 – Feb.2024<br>
**Context:** Robotics research internship, Intelligent Machine Perception and Learning Lab, Drexel University<br>
**Focus:** Synthetic-to-real domain transfer for robot vision

---

## Problem

Supervised monocular depth models require images paired with reliable depth labels, but collecting and labeling real-world depth data is expensive. Simulation can produce labels efficiently, although the visual gap between synthetic and real images can reduce its value for training.

---

## Approach

### Synthetic-to-Real Image Translation

- Used CycleGAN to translate simulation images into a style resembling real-world environments
- Preserved synthetic depth labels while reducing the visual domain gap in the corresponding RGB images

### Mixed-Domain Training

- Combined real images with domain-transferred synthetic images
- Trained a supervised monocular depth estimation model on the mixed dataset

### LiDAR-Based Evaluation

- Compared predicted depth with ground-truth LiDAR point clouds
- Used the comparison to evaluate how domain-transferred data affected depth-estimation performance

---

## Outcome

Mixing real data with converted synthetic data improved supervised monocular depth estimation performance and demonstrated a practical route for using simulation data in real-world perception tasks.

---

## Skills

`CycleGAN` `Monocular Depth Estimation` `Domain Adaptation` `Deep Learning` `Computer Vision` `LiDAR` `Synthetic Data`
