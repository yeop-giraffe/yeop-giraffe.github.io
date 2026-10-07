---
title: Lighter-Than-Air Robot for a Light Drone Competition
description: A helium-supported robot with Raspberry Pi video streaming, YOLOv5 and tracking on a laptop, and ESP32 motor commands.
importance: 6
display_category: Aerial Robotics / Embedded Systems / Visual Tracking
period: Sep. 2023 – Feb. 2024 (visiting research)
affiliation: Drexel University, IMAPLE Lab
summary: "Integrated Raspberry Pi video streaming and ESP32 motor commands for a helium-supported competition robot. A laptop runs YOLOv5 and visual tracking; the onboard controller drives four DC motors."
project_brief:
  Engineering goal: Connect video-based tracking to motor commands on a helium-supported competition robot.
  Individual contribution: Integrated Raspberry Pi video streaming, laptop-based detection/tracking, and ESP32 motor control.
  Outcome: A physical prototype combining helium-supported flight, video streaming and motor commands in an indoor venue.
---

## Overview

This lighter-than-air robot was developed for the **Defend the Republic – Light Drone Competition** during visiting research at Drexel University's Intelligent Machine Perception and Learning (IMAPLE) Lab. Helium balloons support a lightweight structure built from balsa wood, carbon pipes, and 3D-printed parts.

<figure class="uav-figure">
  <a href="../assets/lighter-than-air/competition-demo.png" target="_blank" rel="noopener"><img src="../assets/lighter-than-air/competition-demo.png" width="1280" height="720" alt="Blue helium-supported blimp in an indoor competition venue, with a person standing beneath the platform." decoding="async"></a>
  <figcaption><span>Prototype in the competition venue</span> The helium-supported robot operating in the indoor venue.</figcaption>
</figure>

## Physical Platform

Helium buoyancy supports the lightweight frame, control electronics and motors.

<div class="uav-figure-grid">
  <figure class="uav-figure">
    <a href="../assets/lighter-than-air/platform-original.png" target="_blank" rel="noopener" style="aspect-ratio:1;display:grid;place-items:center"><img src="../assets/lighter-than-air/platform-original.png" width="4032" height="3024" alt="Full prototype photo showing blue helium balloons suspended above a lightweight frame in the indoor venue." loading="lazy" decoding="async" style="width:100%;height:100%;min-height:0;object-fit:contain;transform:rotate(90deg)"></a>
    <figcaption><span>Balloon-supported frame</span> Helium balloons support the lightweight structure.</figcaption>
  </figure>
  <figure class="uav-figure">
    <a href="../assets/lighter-than-air/electronics-original.png" target="_blank" rel="noopener" style="aspect-ratio:1;display:grid;place-items:center"><img src="../assets/lighter-than-air/electronics-original.png" width="5712" height="4284" alt="Underside of the blimp showing a balsa frame, wiring, battery, and mounted electronics beneath the helium balloons." loading="lazy" decoding="async" style="width:100%;height:100%;min-height:0;object-fit:contain;transform:rotate(90deg)"></a>
    <figcaption><span>Mounted electronics</span> Control electronics and wiring mounted beneath the balloons.</figcaption>
  </figure>
</div>

## Control Hardware and Data Flow

The blimp's control system integrates a **Raspberry Pi 4**, an **ESP32**, and **four DC motors**. The Raspberry Pi streams video to a **laptop running YOLOv5 and a tracker**. The laptop sends motor-value commands to the ESP32, which controls the motors.

This places object detection and tracking on the laptop while the onboard hardware handles video streaming and motor commands.

<figure class="uav-figure">
  <a href="../assets/lighter-than-air/control-system.png" target="_blank" rel="noopener"><img src="../assets/lighter-than-air/control-system.png" width="2643" height="1697" alt="Blimp control diagram showing Raspberry Pi 4 streaming video to a laptop running YOLOv5 and a tracker, laptop motor commands entering ESP32, and ESP32 driving four DC motors." loading="lazy" decoding="async"></a>
  <figcaption><span>Video streaming and motor-command flow</span> Video flows from the Raspberry Pi 4 to the laptop, and motor-value commands flow from the laptop to the ESP32.</figcaption>
</figure>

## Demonstration Scope

The prototype was demonstrated in an indoor competition venue, combining helium-supported flight with video streaming and motor control.

## Research Context

The visiting research appointment at Drexel University's IMAPLE Lab ran from **September 2023 to February 2024**, supervised by Prof. David Han.

## Skills

`Lighter-Than-Air Robotics` `Raspberry Pi` `ESP32` `YOLOv5` `Visual Tracking` `Embedded Systems`
