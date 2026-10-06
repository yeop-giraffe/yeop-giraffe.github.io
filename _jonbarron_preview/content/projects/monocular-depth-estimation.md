---
title: Domain-Transferred Synthetic Data Generation for Improving Monocular Depth Estimation
description: CycleGAN domain transfer for synthetic RGB–depth training data and real-domain depth estimation.
importance: 3
standalone_html: true
publications: [lee2025domain]
display_category: Monocular Depth Estimation / CycleGAN / Synthetic Data
period: Sep. 2023 – Feb. 2024 (visiting research)
summary: Compared original and CycleGAN-translated synthetic training data across three depth models on NYU-Depth V2. AbsRel improved slightly; other metrics were mixed. CycleGAN training included NYU test RGB images.
---

## Overview

This ICCAS 2025 paper studies whether translating synthetic training images into a more realistic domain improves monocular depth estimation. Unreal Engine 4.27 and AirSim provide RGB images with dense simulated depth labels. CycleGAN changes image appearance before depth-model training, so the final depth model does not need CycleGAN during inference.

**Paper:** Domain-Transferred Synthetic Data Generation for Improving Monocular Depth Estimation  
**Authors:** Knut Peterson*†, Seungyeop Lee*, Solmaz Arezoomandan, David Han  
**Affiliations:** Drexel University (Peterson, Arezoomandan, Han); Korea University (Lee)  
**Contribution note:** On the website, * denotes equal contribution, and † denotes the corresponding author.  
**Venue:** 2025 25th International Conference on Control, Automation and Systems (ICCAS), November 4–7, 2025, Incheon, Korea, pp. 203–208.

[Read the paper (PDF)](../assets/monocular-depth-estimation/paper.pdf)

## Method and Evaluation

- Generated UE10k: 10,000 RGB–depth pairs from seven indoor Unreal Engine environments, at 640 × 480 pixels and 57° field of view. AirSim depth is clipped at 10 m and saved as 16-bit millimeter depth maps.
- Generated GAN10k: the same synthetic images translated by CycleGAN and paired with their simulated depth maps.
- Trained ZoeDepth, DepthAnything, and Marigold for 15 epochs and evaluated on the official NYU-Depth V2 test split. CycleGAN used 200 epochs and learning rate 0.0001.
- The paper notes that NYU test images were used for CycleGAN training. This is target-domain exposure in translation training, so the evaluation is not a completely unseen target-domain test.

## Results and Limitations

AbsRel changed from 0.320 to 0.313 for ZoeDepth, 0.321 to 0.316 for DepthAnything, and 0.082 to 0.079 for Marigold. These improvements are modest; not all other metrics improve. Marigold predicts affine-invariant depth and should not be directly ranked against the two metric-depth models from these numbers.

Translation artifacts and limited indoor-scene diversity are identified limitations. The paper proposes filtering translated samples, adding semantic/depth constraints, and broadening simulated environments as future work.

## My Role

I worked on CycleGAN-based synthetic image domain transfer and monocular depth model training during my visiting research at Drexel University's IMAPLE Lab, advised by Prof. David Han (September 2023 to February 2024). The method and experimental results belong to the four-author research team, with equal contribution credited to Knut Peterson and Seungyeop Lee.

## Skills

`Monocular Depth Estimation` `CycleGAN` `Domain Transfer` `Unreal Engine` `AirSim` `Synthetic Training Data`
