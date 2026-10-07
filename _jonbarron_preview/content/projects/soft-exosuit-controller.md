---
title: Embedded Controller for a Tendon-Driven Soft Exosuit
description: A Jetson-based pipeline connecting wearable sensor modules to motor commands and feedback.
importance: 1
display_category: Wearable Robotics / Embedded Control / Sensor Integration
period: Dec. 2024 – Feb. 2025
affiliation: Seoul National University, Wearable Robotics Laboratory
summary: "Built sensor modules and a Jetson CAN pipeline for motor commands, feedback and logging. Sensor-to-command processing took 10 ms; bench tests identified motor-response delay during concurrent sensor communication."
project_brief:
  Research question: How can wearable sensors and a motor share an embedded control and logging pipeline?
  My contribution: Developed the Jetson program, assembled sensor modules, and connected and controlled the sensors and motor.
  Key result: Established the basic pipeline with 10-ms sensor-to-command processing; observed motor-response delay with concurrent
    sensor communication.
  Evaluation: Torque-profile bench tests and sensor timing checks. Walking assistance was outside the completed work; mechanical
    hardware was developed by another student.
---

## Overview

At Seoul National University's Wearable Robotics Laboratory, I worked with Prof. Jinsoo Kim on the embedded control infrastructure for a lower-limb, tendon-driven soft exosuit. I established a basic pipeline connecting wearable sensor modules to motor commands, feedback, and data logging on an **NVIDIA Jetson Orin Nano**.

The pipeline had a **10-ms end-to-end processing time** from the sensor module through motor-command generation. Motor response was evaluated separately through feedback measurements, which revealed delays during concurrent sensor communication.

<figure class="uav-figure">
  <a href="../assets/soft-exosuit/can-system-diagram.png" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/can-system-diagram.png" width="3011" height="1343" alt="Jetson Orin Nano connected through a CAN transceiver to a T-Motor and a Feather M4 CAN board interfacing with a load cell and an IMU." decoding="async"></a>
  <figcaption><span>Sensor-to-motor control pipeline</span> The Jetson exchanges motor commands and feedback over CAN while sensor modules provide load-cell and IMU measurements.</figcaption>
</figure>

## My Contributions

- Developed the Jetson control program, including CAN message decoding, motor commands, and feedback handling.
- Assembled sensor board modules and connected the load cells, IMUs, and motor to the control pipeline.
- Separated sensor reception and data logging from the motor-control loop using asynchronous callbacks, queues, and background threads.
- Ran torque-profile tests and investigated motor feedback timing with sensors connected to the shared CAN bus.

Another student developed the mechanical hardware. My work focused on the controller, sensor modules, and their integration with the motor.

## Sensor Modules and Communication

I first tested bidirectional Jetson–Teensy communication using Micro-ROS and UART, then developed a direct Jetson CAN interface for sensor acquisition and motor control. **Feather M4 CAN** boards handle the sensor interfaces, while the Jetson runs the application and motor-command logic.

The sensor receiver supports two node groups. It decodes load-cell values, orientation quaternions, angular velocity, and acceleration into a shared latest-data state. Distinct CAN identifiers separate the sensor channels and motor feedback.

<figure class="uav-figure uav-figure-medium">
  <a href="../assets/soft-exosuit/sensor-module.jpg" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/sensor-module.jpg" width="732" height="684" alt="Assembled sensor module with a Feather M4 CAN board, IMU, load-cell interface, and power module mounted on a perforated board." loading="lazy" decoding="async"></a>
  <figcaption><span>Assembled sensor module</span> Sensor interfaces and power components mounted together for CAN communication with the Jetson.</figcaption>
</figure>

## Control Software and Data Logging

The Python implementation uses **SocketCAN** for communication. An asynchronous receiver decodes sensor messages, and the motor loop packs torque commands into the motor's MIT-format CAN messages. Motor feedback provides position, velocity, torque, temperature, and error status.

Separate queues feed sensor and motor records to background CSV writers. This lets the control program enqueue timestamped records without writing each row directly in the motor loop. The logs retain the commanded torque alongside the received motor state for comparison.

The motor experiments also used a **1-kHz scheduling target**. This setting is distinct from the 10-ms end-to-end sensor-to-command processing time and does not establish the physical motor's response time.

## Torque-Profile Bench Tests

I used periodic torque commands to test the basic motor-control pipeline. An initial test applied a sinusoidal pulse during the first 40% of each one-second cycle, followed by zero commanded torque. The peak command was **1 Nm**, and the run lasted **3 seconds**.

<figure class="uav-figure">
  <a href="../assets/soft-exosuit/torque-1nm.png" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/torque-1nm.png" width="2048" height="1078" alt="Commanded and measured torque over a three-second bench test with three periodic pulses reaching approximately one newton meter." loading="lazy" decoding="async"></a>
  <figcaption><span>Periodic torque command and feedback</span> Three one-second cycles compare the 1-Nm target profile with measured motor torque.</figcaption>
</figure>

Later tests used **5-Nm peak commands**, a **5-second cycle**, and a **10-second run**. I compared feedback with the IMU connected and disconnected while disabling data saving in the control program. This helped separate the effect of sensor traffic from file-writing work.

<figure class="uav-figure">
  <a href="../assets/soft-exosuit/imu-connected.png" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/imu-connected.png" width="2194" height="1131" alt="Torque, position, and velocity during a ten-second motor test with the IMU connected; measured torque appears later than the commanded pulses." loading="lazy" decoding="async"></a>
  <figcaption><span>IMU connected</span> Torque feedback shows a delay relative to the target pulses. Position and velocity are recorded below the torque plot.</figcaption>
</figure>

<figure class="uav-figure">
  <a href="../assets/soft-exosuit/imu-disconnected.png" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/imu-disconnected.png" width="2194" height="1131" alt="Torque, position, and velocity during a ten-second motor test with the IMU disconnected; commanded and measured torque pulses align more closely." loading="lazy" decoding="async"></a>
  <figcaption><span>IMU disconnected</span> Target and feedback torque align more closely in this test. The comparison identified a timing issue for further investigation.</figcaption>
</figure>

## Outcome and Further Work

The project established the basic sensor-to-motor pipeline, including sensor modules, CAN communication, torque commands, feedback decoding, and concurrent logging. Timing checks combined application logs with a logic analyzer to inspect sensor transmission intervals.

<figure class="uav-figure">
  <a href="../assets/soft-exosuit/sensor-timing.png" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/sensor-timing.png" width="1635" height="715" alt="Logic-analyzer capture showing repeated sensor transmission pulses and interval measurements." loading="lazy" decoding="async"></a>
  <figcaption><span>Sensor transmission timing</span> Logic-analyzer measurements characterize sensor transmissions separately from the application-level control cycle.</figcaption>
</figure>

Motor response delay remained an integration issue. Further work includes optimizing sensor traffic and control timing before adding gait-phase estimation and evaluating assistance during walking.

## Skills

`Wearable Robotics` `Jetson Orin Nano` `Python` `SocketCAN` `CAN Communication` `Multithreading` `Sensor Integration` `Motor Control`
