---
title: A Development of Intuitive Controller and Monocular Vision based Interface for Disaster Response UAV
description: Master's thesis on intuitive flight control and monocular vision interfaces for indoor disaster-response UAV teleoperation.
importance: 1.5
standalone_html: true
publications: [lee2025thesis]
display_category: UAV Teleoperation / Human-Robot Interaction / Monocular Vision
period: Sep. 2022 – Feb. 2025
summary: "Designed a UAV controller and a monocular-vision display with 3D reconstruction, obstacle highlighting and predicted paths. In a six-participant Gazebo interface study, System Usability Scale (SUS) scores increased by 28% and NASA-TLX mental demand decreased by approximately 43% versus RGB video alone."
role_summary: "Controller design, vision interface, prototype integration and user evaluation."
project_brief:
  Research question: Can control mapping and visual assistance make indoor UAV teleoperation easier for novice operators?
  My contribution: Designed the controller mapping and vision interface, integrated the prototype, and evaluated the controller
    and display in separate studies.
  Key result: 'SUS increased by 28%; NASA-TLX mental demand decreased by approximately 43% compared with RGB video alone.'
  Evaluation: Two Gazebo studies, each with six novice participants and one run per condition; separate indoor hardware checks.
---

## Overview

**Master's thesis:** A Development of Intuitive Controller and Monocular Vision based Interface for Disaster Response UAV  
**Author:** Seungyeop Lee  
**Degree:** Master of Science in Mechanical Engineering, Korea University, February 2025  
**Advisor:** Prof. Shinsuk Park

[Read the thesis (PDF)](../assets/masters-thesis/paper.pdf)

This thesis investigates both the operator's control input and the information presented during indoor UAV teleoperation. It combines a helicopter-inspired controller mapping with a monocular-vision interface for spatial reconstruction, nearby-obstacle highlighting, and predicted flight-path visualization.

The thesis includes hardware integration and indoor flight checks, a gamepad-versus-joystick comparison, and a separate interface usability/workload evaluation. It complements the ICROS navigation papers with a focus on human operation and user evaluation. A remotely operated prototype integrates a monocular RGB camera, an NVIDIA Jetson Nano embedded computer, and ROS-based communication with the operator station.

## Results

- Interface study: six participants with no drone-control experience; System Usability Scale (SUS) scores improved by 28%, from 62.5 to 80, and NASA-TLX mental demand decreased by 43%, from 57.5 to 32.5, compared with RGB video alone. Percentages are rounded to whole numbers.
- Controller study: six participants with no drone-control experience; mean path-smoothness score decreased from 1.2712 to 0.358 and mean minimum distance to gate centers from 0.24 m to 0.18 m compared with a gamepad.

## Skills

`UAV Teleoperation` `Human-Robot Interaction` `User Studies` `Monocular Depth Estimation` `ORB-SLAM3` `Visual-Inertial Odometry` `ROS / PX4` `Gazebo`
