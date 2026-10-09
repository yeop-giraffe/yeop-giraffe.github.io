---
title: Embedded Controller for a Tendon-Driven Soft Exosuit
description: A Jetson-based pipeline connecting wearable sensor modules to motor commands and feedback.
importance: 1
display_category: Wearable Robotics / Embedded Control / Sensor Integration
period: Dec. 2024 – Feb. 2025
affiliation: Seoul National University, Wearable Robotics Laboratory
summary: "Developed a Jetson-based control and sensing pipeline for a tendon-driven lower-limb soft exosuit, integrating load-cell and IMU measurements with CAN-based motor commands, feedback, and logging. Sensor-to-motor command processing took 10 ms."
role_summary: "Jetson control software, load-cell and IMU sensor modules, and sensor–motor integration."
project_brief:
  Engineering goal: Develop onboard sensing and motor-control infrastructure for a tendon-driven lower-limb soft exosuit.
  My contribution: Developed Jetson control software, assembled load-cell and IMU sensor modules, and integrated CAN-based motor commands, feedback, and logging.
  Key result: Established the basic sensor-to-motor control pipeline with 10-ms end-to-end command processing.
  Evaluation: Torque-profile bench tests and sensor timing checks.
---

## Overview

<div class="exosuit-overview-layout">
<figure class="uav-figure exosuit-representative">
  <a href="../assets/soft-exosuit/exosuit-control-sensing-overview.png" target="_blank" rel="noopener" aria-label="Open the full-size soft exosuit illustration"><img src="../assets/soft-exosuit/exosuit-control-sensing-overview.png" width="1254" height="1254" alt="Tendon-driven lower-limb soft exosuit alongside a Jetson controller, motor, and load-cell and IMU sensor interfaces." loading="lazy" decoding="async"></a>
  <figcaption><span>Soft exosuit and control hardware</span></figcaption>
</figure>
<div class="exosuit-overview-copy">
  <p>This project established a <strong>sensor-to-motor control pipeline</strong> for a tendon-driven soft exosuit intended for <strong>lower-limb assistance</strong>. The system integrates <strong>load-cell and IMU measurements</strong> with CAN-based motor commands, feedback, and data logging on an <strong>NVIDIA Jetson Orin Nano</strong>.</p>
  <p>The pipeline had a <strong>10-ms end-to-end processing time</strong>, measured from the sensor module through motor-command generation. Physical motor response was evaluated separately in the bench tests below.</p>
  <p>The work was conducted at Seoul National University's Wearable Robotics Laboratory, supervised by Prof. Jinsoo Kim.</p>
</div>
</div>

## My Contributions

- Developed the Jetson control program, including CAN message decoding, motor commands, and feedback handling.
- Assembled sensor board modules and connected the load cells, IMUs, and motor to the control pipeline.
- Separated sensor reception and data logging from the motor-control loop using asynchronous callbacks, queues, and background threads.
- Ran torque-profile tests and investigated motor feedback timing with sensors connected to the shared CAN bus.

The contribution scope covered the controller, sensor modules, and their integration with the motor.

## Sensor Modules and Communication

Initial communication tests used a bidirectional Jetson–Teensy connection over Micro-ROS and UART. A direct Jetson CAN interface was then developed for sensor acquisition and motor control. **Feather M4 CAN** boards handle the sensor interfaces, while the Jetson runs the application and motor-command logic.

The sensor receiver supports two node groups. It decodes load-cell values, orientation quaternions, angular velocity, and acceleration into a shared latest-data state. Distinct CAN identifiers separate the sensor channels and motor feedback.

<div class="exosuit-sensor-media">
<figure class="uav-figure">
  <a href="../assets/soft-exosuit/can-system-diagram.png" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/can-system-diagram.png" width="3011" height="1343" alt="Jetson Orin Nano connected through a CAN transceiver to a T-Motor and a Feather M4 CAN board interfacing with a load cell and an IMU." loading="lazy" decoding="async"></a>
  <figcaption><span>Sensor-to-motor control pipeline</span> The Jetson exchanges motor commands and feedback over CAN while sensor modules provide load-cell and IMU measurements.</figcaption>
</figure>
<figure class="uav-figure exosuit-sensor-module">
  <a href="../assets/soft-exosuit/sensor-module.jpg" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/sensor-module.jpg" width="732" height="684" alt="Assembled sensor module with a Feather M4 CAN board, IMU, load-cell interface, and power module mounted on a perforated board." loading="lazy" decoding="async"></a>
  <figcaption><span>Assembled sensor module</span> Sensor interfaces and power components mounted together for CAN communication with the Jetson.</figcaption>
</figure>
</div>

## Control Software and Data Logging

The Python implementation uses **SocketCAN** for communication. An asynchronous receiver decodes sensor messages, and the motor loop packs torque commands into the motor's MIT-format CAN messages. Motor feedback provides position, velocity, torque, temperature, and error status.

Separate queues feed sensor and motor records to background CSV writers. This lets the control program enqueue timestamped records without writing each row directly in the motor loop. The logs retain the commanded torque alongside the received motor state for comparison.

The motor experiments also used a **1-kHz scheduling target**. This setting is distinct from the 10-ms end-to-end sensor-to-command processing time and does not establish the physical motor's response time.

## Torque-Profile Bench Tests

Periodic torque commands tested the basic motor-control pipeline. An initial test applied a sinusoidal pulse during the first 40% of each one-second cycle, followed by zero commanded torque. The peak command was **1 Nm**, and the run lasted **3 seconds**.

<figure class="uav-figure">
  <a href="../assets/soft-exosuit/torque-1nm.png" target="_blank" rel="noopener"><img src="../assets/soft-exosuit/torque-1nm.png" width="2048" height="1078" alt="Commanded and measured torque over a three-second bench test with three periodic pulses reaching approximately one newton meter." loading="lazy" decoding="async"></a>
  <figcaption><span>Periodic torque command and feedback</span> Three one-second cycles compare the 1-Nm target profile with measured motor torque.</figcaption>
</figure>

Later tests used **5-Nm peak commands**, a **5-second cycle**, and a **10-second run**. Feedback was compared with the IMU connected and disconnected while data saving was disabled in the control program. This helped separate the effect of sensor traffic from file-writing work.

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
