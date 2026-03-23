# Guidelines on PIA-SAA subfolder

## General Chat rules

- **REFLECTION PROTOCOL:** If I ever point out a mistake, bug, or oversight you made, your response MUST begin with a `### Why This Happened` section. You MUST rigorously explain the internal limitation, attention lapse, or logic error that led you to make that specific mistake. Do not just apologize or blindly fix the problem without this analysis.
- **Continuous Improvement:** Constantly suggests improvements to this own instructions to make them better.
- **Self-Criticism:** Be always critical with your own answers, and point out possible limitations or errors.
- **Instruction Criticism:** Be also critical with my instructions. Don't hesitate to point out possible mistakes or improvements.
- **Paramount Rigour:** Rigour is paramount. It's important that all explanations are technically correct.

## Jupyter Notebooks Guidelines

- It's important to go step by step, using each cell to show its own output and using markdown cells to explain the steps.
- Headings should be alone in their own cells (Markdown) so they can easily fold/unfold.
- Don't number the headings, so they can be easily reordered.
- Imports should be done in the first cell that requires them, not in a separate cell at the top. That way, if a notebook is split in different parts, each part will have its own imports. It also helps with understanding dependencies.
- Explanations should be in markdown cells, never in prints or comments in python cells.
- Comments in code cells should be used to explain specific lines or blocks of code, not for general explanations.
- Small cells should be used to go step by step. Avoid big cells with long outputs. Particularly, avoid several figures as output of the same cell.

## Python guidelines

- Use in general the most modern practices for Python code (project requires Python ≥3.13).
- Type hints: Use modern syntax for Python 3.13+
- Use `str | None` (not `Optional[str]`).
- Built-in generics: list[str], dict[str, int], tuple[int, ...] (not List, Dict, Tuple)
- Collections: collections.abc.Callable, collections.abc.Iterator (not typing.Callable, typing.Iterator)
- Environment: use `uv` for virtual environments and dependency management (using `uv sync` to keep the environment updated).
- Treat warnings as errors as much as possible.
- Formatting: YAPF (config in pyproject.toml, 120 char limit)
- Imports: Explicit only, don't use "**init**.py" files
- Code should be self-documenting as much as possible, with verbose variable names. Use docstrings only when necessary.
- Use comments to explain only the most complex parts of the code.
- No logging: No logger statements


## Overview

This repository contains **notes and projects** designed for courses in **Artificial Intelligence (AI) and Machine Learning (ML) programming**. The goal is to provide students with rigorous, educational, and challenging materials that emphasize clarity and correctness. A pedagogical approach is appreciated, but it should not compromise the accuracy of the information.
Even when there will be some references and particular notes on terminology in Spanish, all materials must be written in **clear and correct English**.

## Notes Guidelines

Notes are meant to serve as **didactic reference material** for students. They must be:

- **Detailed and rigorous**: Every explanation should be precise and technically correct.
- **Example-driven**: Include practical, illustrative examples (code, diagrams, datasets) wherever possible. Main tool for that will be Jupyter notebooks with embedded explanations and python code.
- **Step-by-step**: Break down complex concepts into manageable parts.
- **Focused**: Present only the content itself. Avoid introductions (“this note explains…”) or summaries at the end.

## Development and Coding Standards

- Solutions will use notebooks, python scripts or simple projects (in their own folders).
- Jupyter Notebooks are preferred for notes and simpler projects, so explanations can be more interactive, while python projects are better for more complex implementations.
- If a markdown file includes code snippets, probably a notebook is preferable.
- Code must be well-documented, with clear comments explaining the purpose and functionality of each section.
- `uv` will be used for virtual environments and dependency management (using `uv sync` to keep the environment updated).
- The course will be online, so all materials must be self-contained and easy to follow without in-person guidance.

### Projects Guidelines

Every project should be designed as a **learning unit** that includes:

1. **Definition**:

   - A clear problem statement. Since this is the only part the student will see, it must be self-contained. It will be normally a single markdown file: README.md.
   - Specific requirements and constraints.
   - Expected deliverables (code, report, presentation).
   - Don't include technical requirements or libraries here; the student should decide that.
   - All work will be done in GitHub repositories.
   - All project should require a detailed README file with:
     - Personal detailed explanation of the code, design decisions, challenges, etc.

2. **Analysis** and **Solution Implementation**:

   - Background and theoretical context.
   - Expected challenges and reasoning steps students must follow.
   - A detailed solution outline or reference implementation.
   - Explanations for each step of the solution, not just the final code.

3. **Assessment Checklist**:

   - A set of criteria to evaluate student work.
   - Should cover correctness, robustness, code quality, and clarity of reasoning.
   - Maximum score should be 10 points, with a breakdown of weights for different aspects.
   - They should be simple checklists easy to follow, not complex rubrics.
   - Since the projects are open, the assessment will focus on understanding and implementation quality, not just final results. Teacher qualitative assessment is acceptable.

4. **Proposals for exam verification**:

   - Final exam will require students to demonstrate their understanding and authorship of the projects developed throughout the course. Each project should include proposals for such verification.
   - Projects can be solved with help of AI tools, but in the exam students won't have access to the Internet, only to everything they bring downloaded before.
   - Keeping that in mind, the exam will consist in:
     - Explanations about the projects
     - Small modifications or extensions, that don't require new coding from scratch but rather adapting or extending existing code.
     - Test questions following

5. **Exam Qualification Rubric**:
   - For the exam qualification, there should be a rubric with a maximum of 10 points, with a breakdown of weights for different aspects. This rubric should be very clear and not open to interpretation.

#### Core Principles

- **Paradigmatic Problems**: Projects must address problems, representative of real-world AI/ML applications. They should:

  - Involve multiple steps or components (data processing, model training, evaluation).
  - Require critical thinking and problem-solving, not just rote application of algorithms.

- **Complexity & Depth**: Projects should:

  - Go beyond simple introductory exercises.
  - Encourage exploration of open-ended questions, multi-stage pipelines, or real-world datasets.

- **Learning Outcomes**: Projects should bridge **theory and practice**, ensuring that students:
  - Understand the underlying concepts.
  - Can implement, test, and critically analyze solutions.

### Dataset & Problem Definition Guidelines

Datasets and problem statements should be designed to **challenge assumptions** and encourage robust solutions. To achieve this:

- **Introduce Variations & Noise**:

  - Rename or shuffle columns.
  - Add irrelevant features.
  - Inject noise into data.
  - Reverse or redefine target variables (with clear documentation).
  - Include missing or inconsistent values.
  - Mix data formats.

- **Purpose**:  
  These controlled “imperfections” are intentional. They train students to:
  - Analyze data carefully.
  - Question assumptions.
  - Develop resilient and generalizable solutions.
