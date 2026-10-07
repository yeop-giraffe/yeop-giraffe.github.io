---
title: "DART: From Attentive Reasoning to Intuitive Execution for Experience-Driven Robot Intelligence"
description: Reusing task-acquired knowledge for object search and navigation on a robot vacuum.
importance: 0
standalone_html: true
publications: [dart2027]
display_category: Spatial AI / Scene Graphs / LLM & VLM
period: May 2026 – Present
summary: "Retains task-acquired scene-graph knowledge for repeated object search and navigation. Real-Home success was 87% versus 66% for a baseline that discards acquired knowledge, with 49.3% fewer VLM calls for visual acquisition and description generation (20 tasks × five iterations)."
role_summary: "Viewpoint selection, room classification, architecture and system integration."
project_brief:
  Research question: How can a home robot reuse what it learns from earlier tasks?
  My contribution: Contributed to the architecture; developed viewpoint selection and room classification; integrated perception,
    VLM and navigation in Isaac Sim.
  Key result: 'Real-Home task success: 87% vs. 66% for DART-Frozen, with 49.3% fewer VLM acquisition/description calls.'
  Evaluation: 20 tasks × five iterations in each of two homes (one simulated, one real). Language-model inference runs on
    an external server.
---

## Overview

DART (Deliberation-Adaptive Reasoning from Task Experience) is a dual-mode architecture inspired by theories of human cognition. An LLM parses instructions; Intuitive Mode then searches a persistent scene graph without invoking an LLM. When stored knowledge is insufficient, Attentive Mode uses LLM reasoning and targeted VLM observations to acquire the information needed for execution and retain it for later related tasks. The implementation uses lightweight on-device perception on a robot vacuum, with Qwen3.5 inference running on an external server.

**Paper:** DART: From Attentive Reasoning to Intuitive Execution for Experience-Driven Robot Intelligence<br>
**Authors:** Seungyeop Lee*, Yeeun Kim*, Sooho Park, Seong Oh Lee, and Jong Jin Park†<br>
**Status:** Submitted to ICRA 2027; under review<br>
**Contribution:** * Equal contribution (co-first authors); † Corresponding author

[Read the manuscript (PDF)]({{ '/assets/pdf/ICRA27.pdf' | relative_url }})

## My Contributions

- Contributed to the dual-process architecture for task-knowledge reuse and incremental scene-graph enrichment.
- Developed observation-viewpoint selection based on object visibility and viewpoint accessibility.
- Developed room classification using object occurrence likelihoods and visual evidence accumulated during robot navigation.
- Integrated VLM, perception, and navigation modules into the Spatial AI system and validated the system in NVIDIA Isaac Sim.

## Evaluation and Scope

The evaluation uses 20 object-search and navigation tasks in each of two residential environments, repeated as a complete sequence for five iterations. DART retains acquired knowledge across tasks and iterations; DART-Frozen acquires information for the current task but discards it afterward. Each task starts from the same docking-station pose and is limited to 20 VLM calls.

Overall success is 58.0% versus 55.0% for DART-Frozen in Sim-Home, and 87.0% versus 66.0% in Real-Home (Table IV). Averaged over five iterations, VLM token/call reductions are 47.6%/56.2% in Sim-Home and 52.6%/49.3% in Real-Home (Section V-D). These counts cover visual information acquisition and description generation, rather than all model inference.

The study addresses repeated tasks in the same or slowly changing environment. It does not evaluate manipulation or long-term deployment across diverse homes. Future work includes manipulation-capable platforms and temporal scene-graph updates (Section VI).

## Skills

`Scene Graphs` `Spatial AI` `LLM/VLM Integration` `Viewpoint Selection` `Room Classification` `NVIDIA Isaac Sim`
