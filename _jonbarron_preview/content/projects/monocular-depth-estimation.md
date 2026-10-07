---
title: Domain-Transferred Synthetic Data Generation for Improving Monocular Depth Estimation
description: CycleGAN domain transfer for synthetic RGB–depth training data and real-domain depth estimation.
importance: 3
standalone_html: true
publications: [lee2025domain]
display_category: Monocular Depth Estimation / CycleGAN / Synthetic Data
period: Sep. 2023 – Feb. 2024 (visiting research)
summary: "Collected RGB–depth pairs in Unreal Engine simulation environments and applied CycleGAN for synthetic-to-real image translation. The translated images and simulated depth labels were used to train monocular depth estimation models."
role_summary: "Unreal Engine environment setup, synthetic data collection, CycleGAN domain transfer and depth model training."
project_brief:
  Research question: Does translating synthetic training images improve depth estimation on real indoor images?
  My contribution: Built Unreal Engine simulation environments, collected RGB–depth data, and worked on CycleGAN-based domain transfer and monocular depth model training.
  Key result: AbsRel decreased modestly for all three models; other metrics showed mixed changes.
  Evaluation: 10,000 paired scenes; three depth architectures; NYU-Depth V2 test evaluation. CycleGAN training included NYU
    test RGB images.
---

## Abstract

A major obstacle to the development of effective monocular depth estimation algorithms is the difficulty in obtaining high-quality metric depth data that corresponds to real-world RGB images. Collecting this data is time-consuming and costly, and even data collected by modern sensors has limited range or resolution, and is subject to inconsistencies and noise. Data generated in simulation avoids these problems with accurate depth information, but models trained on synthetic data often do not transfer well to real world applications. To combat this, we propose a method of data generation in simulation using 3D synthetic environments and CycleGAN domain transfer to increase the realism of simulated images. We analyze this data generation method by training multiple depth estimation models on different datasets, including synthetic and domain-transferred data. We evaluate the performance of the models on the NYU-Depth V2 dataset to verify the generalizability of the approach and show that GAN-transformed data effectively helps to bridge the gap between simulated and real-world data in depth estimation.

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

## My Contributions and Research Context

This research was conducted during a visiting appointment at Drexel University's IMAPLE Lab, supervised by Prof. David Han (September 2023 to February 2024).

- Built Unreal Engine simulation environments and collected paired synthetic RGB images and depth maps.
- Applied CycleGAN-based domain transfer to generate realistic training images from synthetic scenes.
- Worked on monocular depth model training with domain-transferred synthetic data.

## Skills

`Monocular Depth Estimation` `CycleGAN` `Domain Transfer` `Unreal Engine` `AirSim` `Synthetic Training Data`
