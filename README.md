# Accel-Sim Test Project

## Overview

This project, `accelsim-test`, is a dedicated workspace for evaluating the **[Accel-Sim Framework](https://accel-sim.github.io/)** as a potential component for a larger **LLM Inference Simulation Framework**.

## Objective

The primary goal is to determine if Accel-Sim can accurately and efficiently estimate theoretical inference performance of Large Language Models (LLMs) across different hardware settings, specifically:

*   **Low-Level Estimation:** Leveraging Accel-Sim's detailed architecture modeling to predict performance metrics at the kernel and instruction level.
*   **Cross-Architecture Simulation:** validating performance on existing GPUs and projecting performance for unseen or future GPU architectures (e.g., hypothetical next-gen configurations).
*   **Integration Feasibility:** Assessing the effort required to integrate Accel-Sim's trace-driven or execution-driven modes into a high-level LLM performance modeling pipeline.

## Scope

This repository will contain:

*   Configuration files for simulating specific GPU architectures (e.g., NVIDIA A100, H100, and hypothetical specs).
*   Test kernels and micro-benchmarks relevant to LLM inference (e.g., GEMM, Attention mechanisms).
*   Scripts to drive Accel-Sim simulations and parse output metrics.
*   Documentation of findings regarding accuracy, simulation speed, and ease of use.

## References

*   [Accel-Sim Official Website](https://accel-sim.github.io/)
*   [Accel-Sim GitHub Repository](https://github.com/accel-sim/accel-sim-framework)
