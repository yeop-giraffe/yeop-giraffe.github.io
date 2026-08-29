---
layout: page
title: Soft Exo-suit Controller
description: Embedded wearable robot controller based on Nvidia Jetson Orin Nano and CAN communication.
img:
importance: 2
category: research
display_category: Wearable Robotics / Embedded Control
period: Dec.2024 - Feb.2025
summary: Embedded controller development for a soft wearable exo-suit using Jetson Orin Nano, CAN communication, multi-threaded processing, and sensor integration.
---

## Overview

Developed an embedded controller for a soft wearable exo-suit. The controller combined motor communication and multi-sensor acquisition on an Nvidia Jetson Orin Nano, with an emphasis on low-latency data transfer for real-time wearable-robot control.

**Period:** Dec.2024 – Feb.2025<br>
**Context:** Research internship, Wearable Robotics Laboratory, Seoul National University<br>
**Focus:** Embedded control and sensor integration for wearable robotics

---

## Problem

A wearable robot controller must exchange motor and sensor data with low latency while handling multiple communication tasks at the same time. Delayed or serialized processing can limit control responsiveness and make it harder to integrate additional sensors.

---

## Approach

### Embedded Controller

- Built the controller around the Nvidia Jetson Orin Nano platform
- Organized the software to support both motor commands and sensor feedback

### Low-Latency CAN Communication

- Implemented a CAN-based motor and sensor communication pipeline
- Used multi-threaded data processing to transmit data in under 10 ms

### Sensor Integration Module

- Designed a module for connecting the exo-suit's sensing components to the controller
- Validated the module as part of the integrated embedded system

---

## Outcome

The resulting controller provided sub-10 ms CAN data transmission and a validated interface for integrating motor and sensor data on a single embedded platform.

---

## Skills

`Nvidia Jetson Orin Nano` `CAN` `Multi-threading` `Embedded Systems` `Sensor Integration` `Wearable Robotics` `Real-Time Control`
