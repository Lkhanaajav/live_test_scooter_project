# Autonomous Scooter -- Live Test Environment

Live testing harness for the autonomous scooter navigation system, featuring an AI-agent-driven development workflow with specialized agents for planning, debugging, code review, and execution.

## Overview

This repository is the live test environment for the autonomous scooter project. Beyond the navigation code itself, it introduces a multi-agent AI development pipeline powered by Claude Code -- purpose-built agents handle the full software lifecycle from planning new features to debugging failures on the scooter hardware.

## Tech Stack

- **Language:** Python
- **AI Agents:** Claude Code (.claude/agents/)
- **Agent Framework:** Custom agent manifests per role

## Key Features

- **Multi-agent AI development pipeline** - specialized agents coordinate to plan, implement, review, and debug code changes without manual hand-offs
- **Automated code review and debugging agents** - code-reviewer enforces quality standards; gsd-debugger isolates and explains failures from live test logs
- **Modular agent architecture** - each agent (gsd-planner, gsd-executor, build-error-resolver, etc.) has a single responsibility, making the pipeline easy to extend or swap out
- **Live test integration** - agents operate directly on hardware test output, closing the loop between development and real-world validation

## Agent Manifest

| Agent | Role |
|---|---|
| gsd-planner | Breaks down features into actionable tasks |
| gsd-executor | Implements planned tasks in code |
| gsd-debugger | Diagnoses errors from logs and test runs |
| code-reviewer | Reviews diffs for correctness and style |
| build-error-resolver | Fixes compilation and dependency errors |

## What Was Interesting

Using AI coding agents to accelerate autonomous systems development turned out to be qualitatively different from using them for web or backend software. The agents had to reason about hardware constraints, sensor timing, and control-loop latency -- not just code correctness. Designing the agent handoffs so that the planner's output was always executable by the executor, and that the debugger could read raw hardware logs, required careful prompt engineering and exposed gaps in how LLMs handle real-time system context.

## License

MIT