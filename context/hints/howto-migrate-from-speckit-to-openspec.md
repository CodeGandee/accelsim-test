# How to Migrate from Spec Kit to OpenSpec

This guide is for developers transitioning from **Spec Kit** (GitHub Next) to **OpenSpec** (Fission-AI). It maps your existing mental model of "Strict Phases" to OpenSpec's "Fluid Actions" and "Delta Specs".

## Core Concept Shifts

| Concept | Spec Kit | OpenSpec (OPSX) | The Shift |
| :--- | :--- | :--- | :--- |
| **Philosophy** | **Code serves Specs**. Strict, linear phases. | **Fluid & Iterative**. Actions over phases. Brownfield-first. | You are no longer locked into a phase. You can edit a spec or design at any time. |
| **Feature Isolation** | **Git Branches**. Each feature lives in a `feat/xyz` branch. | **Change Folders**. Features live in `openspec/changes/xyz/` folders *within* your main branch until archived. | Allows working on multiple features in parallel without context switching git branches. |
| **Specifications** | **Full Spec**. A complete `spec.md` generated for the feature. | **Delta Specs**. You define only *changes* (Added/Modified/Removed) relative to the main specs. | Prevents spec bloat. Merges intelligently into a "Source of Truth" (`openspec/specs/`) upon completion. |
| **Context/Rules** | `memory/constitution.md` | `openspec/config.yaml` | Rules are now injected into *every* prompt via `config.yaml`, rather than just referenced. |

## Command Mapping Cheat Sheet

| Action | Spec Kit Command | OpenSpec Command | Note |
| :--- | :--- | :--- | :--- |
| **Start** | `/speckit.specify "Build X..."` | `/opsx:new "build-x"` | OpenSpec creates a folder first. You define intent in the next step. |
| **Plan (One-Shot)** | (Auto-chained usually) | `/opsx:ff` (Fast-Forward) | Creates Proposal -> Specs -> Design -> Tasks in one go. |
| **Plan (Step-by-Step)**| `/speckit.plan`, `/speckit.tasks` | `/opsx:continue` | Creates one artifact at a time (Proposal -> Specs -> ...). |
| **Think/Brainstorm** | (No equivalent) | `/opsx:explore` | "Thinking partner" mode before committing to a change. |
| **Implement** | `/speckit.implement` | `/opsx:apply` | Iterates through `tasks.md` and implements them. |
| **Verify** | (Manual / Checklist) | `/opsx:verify` | Checks Completeness, Correctness, and Coherence. |
| **Finish** | Git Merge / PR | `/opsx:archive` | Merges delta specs into main specs and moves change folder to `archive/`. |

## The New Workflow: "Fluid Actions"

In Spec Kit, you were likely used to:
1.  `specify` -> Wait for huge output.
2.  `plan` -> Wait for huge output.
3.  `implement`.

In OpenSpec, the workflow is designed to be more granular and forgiving:

1.  **Initialize**: `openspec init` (Do this once).
2.  **Explore (Optional)**: Run `/opsx:explore` to discuss ideas without creating files.
3.  **Start Change**: `/opsx:new add-login`. This creates `openspec/changes/add-login/`.
4.  **Define**:
    *   **Fast Path**: Run `/opsx:ff`. It generates `proposal.md` (Intent), `specs/` (Deltas), `design.md` (Tech Plan), and `tasks.md`.
    *   **Controlled Path**: Run `/opsx:continue`. It asks "What do you want to build?", creates the Proposal, then stops. Run it again to create Specs, etc.
5.  **Refine**: **Crucial Step**. You can open `proposal.md` or `design.md` and *edit them manually* at any time. OpenSpec picks up the changes immediately.
6.  **Implement**: Run `/opsx:apply`. It reads your tasks and code.
7.  **Iterate**: Realized you missed a requirement?
    *   *Spec Kit*: Hard to go back.
    *   *OpenSpec*: Just edit `specs/ui/spec.md`, add the requirement, and run `/opsx:apply` again.
8.  **Finish**: `/opsx:archive`. This is the "Merge" step. It updates your global specs in `openspec/specs/` with the deltas you created.

## Agentic Coding Tips

### 1. The `config.yaml` is your new Constitution
Instead of a long markdown file, use `openspec/config.yaml`.
```yaml
schema: spec-driven
context: |
  Tech Stack: Python 3.12, C++17, CMake.
  Testing: Pytest for Python, Catch2 for C++.
  Style: Google C++ Style Guide.
rules:
  specs:
    - Use Given/When/Then for scenarios.
  design:
    - Always consider performance implications for LLM inference.
```
This context is **injected** into every request, ensuring the agent always knows your tech stack and constraints.

### 2. Delta Specs are Powerful
When working on `openspec/changes/my-feature/specs/`, you don't rewrite the whole system spec. You typically write:
```markdown
# Delta for Auth
## ADDED Requirements
- The system MUST support Magic Links.
```
This keeps context window usage low and focus high.

### 3. "Fast-Forward" (`/opsx:ff`) vs "Continue" (`/opsx:continue`)
*   Use **FF** when you know exactly what you want and trust the agent.
*   Use **Continue** when you are exploring or the feature is complex. It pauses after each artifact (Proposal -> Specs -> Design -> Tasks) allowing you to review and steer.
