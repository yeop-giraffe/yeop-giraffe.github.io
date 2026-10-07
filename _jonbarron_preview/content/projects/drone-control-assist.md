---
title: Vision-Based Indoor UAV Navigation
description: Indoor localization and obstacle avoidance using visual odometry and monocular depth estimation.
importance: 2
publications: [lee2024monocular, lee2023visual]
display_category: Visual Odometry / Monocular Depth Estimation / UAVs
period: Sep. 2022 – Dec. 2024
summary: Two studies on vision-based indoor UAV navigation, covering a ROS-based visual-odometry platform and monocular-depth-based obstacle avoidance.
publication_descriptions:
  lee2024monocular: Used MiDaS relative-depth maps and grid-based direction selection for UAV obstacle avoidance. In PX4/Gazebo simulation, depth-map generation ran at 41 FPS and the reported 10 m flight avoided the placed obstacles.
  lee2023visual: Built a ROS/MAVROS platform that feeds ZED 2i stereo visual odometry to PX4 for indoor localization without GPS. Integrated the physical hardware and demonstrated position-command flight in Gazebo simulation.
publication_categories:
  lee2024monocular: Monocular Depth Estimation / Obstacle Avoidance / UAVs
  lee2023visual: Visual Odometry / ROS / UAVs
affiliation: "Korea University, Human-Machine Systems Lab"
publication_contributions:
  lee2024monocular: Relative-depth processing and ROS/PX4 obstacle-avoidance integration.
  lee2023visual: Quadcopter platform and visual-odometry-to-flight-control integration.
project_brief:
  Research question: How can camera-based perception support indoor UAV flight where GPS is unavailable?
  My contribution: Built the ROS-based visual-odometry platform and explored monocular-depth-based flight-direction selection.
  Key result: '2023: integrated localization/control platform. 2024: 41-FPS depth processing and a collision-free 10-m simulated
    flight.'
  Evaluation: Physical platform integration and PX4/Gazebo demonstrations; the two studies use different camera configurations.
---

## Overview

At Korea University's Human-Machine Systems Lab, advised by Prof. Shinsuk Park, I studied vision-based navigation for indoor UAVs. These two ICROS papers address complementary challenges: estimating the UAV's position where GPS is unavailable, and selecting a flight direction to avoid obstacles using a single RGB camera.

<nav class="uav-section-nav" aria-label="Project sections"><a href="#visual-odometry">ICROS 2023 · Visual odometry</a><a href="#obstacle-avoidance">ICROS 2024 · Obstacle avoidance</a></nav>

<h2 id="visual-odometry">ICROS 2023 — Visual Odometry for Indoor Flight</h2>

[Paper (PDF)](../assets/pdf/ICROS2023_LSY.pdf)

### Hardware Platform

I developed a ROS-based quadcopter platform integrating a **Pixhawk 6C flight controller**, **NVIDIA Jetson Nano**, and **ZED 2i stereo camera**. The camera's visual odometry estimates position and orientation for indoor localization without GPS.

<figure class="uav-figure uav-figure-medium">
  <a href="../assets/images/uav/icros2023-platform.jpg" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2023-platform.jpg" width="582" height="389" alt="Indoor quadcopter with a ZED 2i stereo camera mounted at the front and stacked onboard electronics." decoding="async"></a>
  <figcaption><span>Quadcopter platform</span> Physical quadcopter platform integrating stereo vision, onboard computing, and flight control.</figcaption>
</figure>

### From Visual Odometry to Flight Control

The ROS/MAVROS pipeline transforms the camera's pose into the flight controller's coordinate frame and sends it to PX4 for position-based offboard control. The work combined a physical hardware platform and pose visualization in RViz with flight-control evaluation in **PX4/Gazebo simulation**.

<div class="uav-figure-grid">
  <figure class="uav-figure">
    <a href="../assets/images/uav/icros2023-ros-graph.jpg" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2023-ros-graph.jpg" width="517" height="291" alt="ROS rqt graph showing MAVROS topics and nodes connecting position commands to the flight controller." loading="lazy" decoding="async"></a>
    <figcaption><span>ROS control pipeline</span> ROS node and topic connections for offboard flight control.</figcaption>
  </figure>
  <figure class="uav-figure">
    <a href="../assets/images/uav/icros2023-rviz.jpg" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2023-rviz.jpg" width="640" height="388" alt="RViz screenshot visualizing the position and orientation estimated by stereo visual odometry." loading="lazy" decoding="async"></a>
    <figcaption><span>Pose visualization</span> Visual-odometry pose estimates displayed in RViz.</figcaption>
  </figure>
</div>

<figure class="uav-figure uav-figure-medium">
  <a href="../assets/images/uav/icros2023-gazebo.jpg" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2023-gazebo.jpg" width="582" height="327" alt="PX4 quadcopter model in a Gazebo simulation used for evaluating flight control." loading="lazy" decoding="async"></a>
  <figcaption><span>Flight-control simulation</span> PX4/Gazebo environment used to evaluate position-based flight control.</figcaption>
</figure>

The paper reports a physical platform and a simulated flight demonstration. It does not quantify localization errors or report physical autonomous-flight success rates.

<h2 id="obstacle-avoidance">ICROS 2024 — Monocular Depth for Obstacle Avoidance</h2>

[Paper (PDF)](../assets/pdf/ICROS2024_LSY.pdf) / [Poster (PDF)](../assets/pdf/ICROS2024-poster-en.pdf)

### Depth-Based Flight-Direction Selection

I explored a lightweight obstacle-avoidance approach using **MiDaS**, which estimates relative scene depth from a single RGB image. The system divides the depth map into a grid, compares the mean depth value of each cell, and selects a flight region with fewer obstacles. These are relative depth cues rather than measured metric distances.

<figure class="uav-figure">
  <div class="uav-image-pair">
    <a href="../assets/images/uav/icros2024-depth.jpg" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2024-depth.jpg" width="320" height="240" alt="MiDaS relative depth map of pillar obstacles in the simulated UAV camera view." loading="lazy" decoding="async"></a>
    <a href="../assets/images/uav/icros2024-grid-original.png" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2024-grid-original.png" width="320" height="240" alt="Depth map divided into five columns and three rows, with cell mean values and a selected region outlined in yellow." loading="lazy" decoding="async"></a>
  </div>
  <figcaption><span>Depth-based direction selection</span> MiDaS relative depth map and grid-based flight-direction selection. The yellow outline marks the selected region.</figcaption>
</figure>

**Grid configuration:** The method description specifies a 4 × 5 grid (20 cells), while the illustrated grid has 5 × 3 cells (15 cells).

The selected cell's horizontal position determines a yaw adjustment, followed by a **0.5 m forward step**. ROS/MAVROS sends the resulting position commands to PX4. New depth observations update the next target during flight.

<figure class="uav-figure uav-figure-medium">
  <a href="../assets/images/uav/icros2024-pipeline-original.png" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2024-pipeline-original.png" width="2861" height="1848" alt="Closed-loop system diagram connecting Gazebo RGB images, depth estimation, grid comparison, and PX4-Autopilot pose and position commands." loading="lazy" decoding="async"></a>
  <figcaption><span>Perception-to-control pipeline</span> The perception-to-control loop connects RGB images, depth estimation, grid comparison, and PX4 flight commands.</figcaption>
</figure>

### Simulation Demonstration

The PX4/Gazebo test environment placed **six 1 × 1 × 3 m rectangular pillars** within the forward **10 m** test area to evaluate obstacle avoidance.

<figure class="uav-figure uav-figure-medium">
  <a href="../assets/images/uav/icros2024-obstacles-original.png" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2024-obstacles-original.png" width="939" height="755" alt="Gazebo test scene containing rectangular pillar obstacles ahead of a small UAV." loading="lazy" decoding="async"></a>
  <figcaption><span>Obstacle environment</span> Gazebo obstacle environment used for the flight demonstration.</figcaption>
</figure>

Depth-map generation ran at **41 FPS**. After taking off to **1.5 m**, the reported trajectory completed a **10 m flight without colliding with the placed obstacles**. The 41 FPS result measures RGB-to-depth image processing. The plots below show the simulated trajectory in three dimensions and from above.

<figure class="uav-figure">
  <div class="uav-image-pair">
    <a href="../assets/images/uav/icros2024-path-3d-original.png" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2024-path-3d-original.png" width="570" height="420" alt="Three-dimensional flight trajectory curving around rectangular obstacles in the simulation." loading="lazy" decoding="async"></a>
    <a href="../assets/images/uav/icros2024-path-top-original.png" target="_blank" rel="noopener"><img src="../assets/images/uav/icros2024-path-top-original.png" width="560" height="420" alt="Top view of the simulated flight path passing between rectangular obstacles." loading="lazy" decoding="async"></a>
  </div>
  <figcaption><span>Obstacle-avoidance trajectory</span> Collision-free simulated trajectory shown in 3D and from above.</figcaption>
</figure>

This is a simulation demonstration, with no reported repeated-trial success rate or baseline comparison. Physical flight tests and altitude-aware path planning were identified as future work.

## Connecting the Two Studies

Both studies connect visual perception to flight control through ROS, MAVROS, and PX4. The 2023 work establishes a platform for indoor localization and position control with stereo visual odometry; the 2024 work investigates obstacle avoidance with monocular relative-depth cues. Together, they explore how camera-based perception can support indoor UAV navigation.

## Skills

`Visual Odometry` `Monocular Depth Estimation` `MiDaS` `ROS / MAVROS` `PX4` `Gazebo` `UAVs`
