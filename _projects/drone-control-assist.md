---
layout: page
title: Drone Control Assist Interface
description: ROS/MAVLink based HMI system for indoor UAV control assistance in non-GPS environments.
img:
importance: 1
category: research
display_category: UAV / Human-Machine Interface / ROS
period: Sep.2022 - Dec.2024
summary: ROS/MAVLink based HMI system for indoor UAV control assistance in non-GPS environments to reduce crashes caused by human error.
---

## Overview

Developed a human-machine interface (HMI) that assists operators during indoor UAV flights where GPS is unavailable. The system was designed to reduce crashes caused by human error by translating assist logic into flight commands for the vehicle.

**Period:** Sep.2022 – Dec.2024<br>
**Context:** Master's research, Human-Machine Systems Lab, Korea University<br>
**Focus:** Indoor UAV control assistance in non-GPS environments

---

## Problem

Indoor drone operators must control the vehicle without reliable GPS positioning and often with limited situational awareness. Small input errors can therefore lead quickly to collisions, making an additional control-assistance layer valuable for safer flight.

---

## Approach

### Companion Computer Architecture

- Ran the control-assistance software with ROS on an Nvidia Jetson Nano
- Used the companion computer to generate and transmit assist commands to a Pixhawk flight controller

### Flight Controller Communication

- Built the communication pipeline with MAVLink and MAVROS
- Connected high-level HMI and assistance logic to the UAV's low-level flight-control system

### Simulation-Based Validation

- Implemented flight-assist algorithms as ROS components
- Tested control behavior in Gazebo before deployment to the flight platform

---

## System

**Companion computer:** Nvidia Jetson Nano<br>
**Flight controller:** Pixhawk<br>
**Communication:** MAVLink / MAVROS<br>
**Validation environment:** ROS / Gazebo

---

## Skills

`ROS` `Gazebo` `MAVLink` `MAVROS` `Nvidia Jetson` `Pixhawk` `UAV Control` `Human-Machine Interface`
