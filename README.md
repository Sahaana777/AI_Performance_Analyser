# Comparative Performance Analysis of Cloud-Based AI Models

## Overview

This project evaluates the performance characteristics of cloud-based
AI models from different providers using a common execution environment.

The benchmark is designed to compare:

- API response time
- CPU utilization
- RAM utilization
- Network traffic
- Input token usage
- Output token usage
- Token throughput
- Request success/failure

The system is designed to run on an OpenStack virtual machine and
communicate with external AI provider APIs.

---

## Research Question

How do cloud-based AI models from different providers affect the
computational, network, and response-time characteristics of a common
execution environment?

---

## Architecture

```text
                    OpenStack VM
                         |
                         v
                Benchmark Engine
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       OpenAI         Anthropic       Google
          |              |              |
          +--------------+--------------+
                         |
                         v
                  Measurements
                         |
                         v
                   CSV Dataset
                         |
              +----------+----------+
              |                     |
              v                     v
          Analysis.py          Flask Dashboard
              |                     |
              v                     v
           Graphs              Web Interface
```
