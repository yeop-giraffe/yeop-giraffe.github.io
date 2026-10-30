---
title: HIL-Based Controller Refinement for Robot Vacuum Cleaners
description: Iterative controller refinement for robot vacuum cleaners using threshold-climbing simulation feedback, an LLM, and an Isaac Sim hardware-in-the-loop test environment.
importance: 4
display_category: Robot Control / LLM-Based Refinement / HIL
period: May 2026 – Present
affiliation: Samsung Electronics, Robot Vacuum Cleaner SW Lab
summary: Developing an LLM-based controller refinement loop on an Isaac Sim HIL test system. Simulation feedback from threshold-climbing evaluation guides iterative code revisions, with the goal of improving threshold-crossing reliability.
role_summary: "Built the HIL test environment with Samsung R&D Institute-Delhi; evaluating threshold climbing and developing the LLM-based refinement loop."
project_brief:
  Engineering goal: Improve threshold-crossing reliability through iterative controller refinement using simulation feedback.
  My contribution: HIL system integration, threshold-climbing evaluation, and development of the LLM-based refinement loop.
  Status: Built the HIL test system with Samsung R&D Institute-Delhi; controller refinement is ongoing.
---

## Overview

At Samsung Electronics' **Robot Vacuum Cleaner SW Lab**, this work uses an **NVIDIA Isaac Sim-based hardware-in-the-loop (HIL) test system** to support iterative controller refinement for threshold crossing.

The HIL system, built in collaboration with **Samsung R&D Institute-Delhi**, connects physical control hardware and production control software to simulated robot sensors. Current work develops an **LLM-based refinement loop** that uses threshold-climbing simulation feedback to guide controller-code revisions within this workflow.

## Controller Refinement Workflow

Threshold-climbing evaluations in **NVIDIA Isaac Sim** characterize the robot vacuum's traversal performance and failure modes. These observations provide simulation feedback for controller refinement.

An **LLM-based loop** is being developed to analyze this feedback and iteratively revise controller code. The goal is to improve threshold-crossing reliability through repeated evaluation and code refinement.

## HIL Test Environment

The HIL setup keeps the physical controller and production control software in the test loop, while Isaac Sim supplies simulated sensor observations.

| Component | Role in the HIL setup |
| --- | --- |
| Physical controller board | Connects the robot's control hardware to the test system. |
| Production control software | Integrates the robot-control software used in the product. |
| NVIDIA Isaac Sim | Provides simulated robot sensor inputs. |

## My Contributions

- Built the HIL test system by integrating the physical controller board, production control software, and Isaac Sim sensor inputs in collaboration with Samsung R&D Institute-Delhi.
- Evaluating the threshold-climbing mechanism in Isaac Sim to characterize traversal performance and failure modes as feedback for controller refinement.
- Developing the LLM-based loop to analyze this feedback and iteratively revise controller code.

## Skills

`LLM-Based Controller Refinement` `Robot Control` `NVIDIA Isaac Sim` `Hardware-in-the-Loop` `System Integration` `Simulation Evaluation`
