# Contributing to ARC

Thanks for your interest in contributing to ARC! This guide covers everything you need to get started.

---

## Before You Start

- Open an issue describing your proposed feature, fix, or improvement
- Wait for the issue to be **assigned to you** before starting work
- PRs submitted without prior discussion or assignment may be closed

For questions or discussions, join the [ARC Discord](https://discord.gg/E6UvPWC8DW).

---

## Contribution Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Create a Branch

Use a clear, descriptive branch name:

```bash
git checkout -b feature/your-feature-name
```

Examples:
```
feature/failure-prediction-upgrade
fix/checkpoint-memory-leak
docs/installation-guide-update
```

### 4. Make Your Changes

Keep contributions:
- Focused on a single feature or fix
- Modular and consistent with the existing architecture
- Well documented with comments and docstrings where appropriate

### 5. Commit Clearly

```bash
git commit -m "Add: short description"
```

Examples:
```
Add: gradient anomaly recovery module
Fix: checkpoint rollback edge case
Docs: improve installation instructions
```

### 6. Push and Open a PR

```bash
git push origin feature/your-feature-name
```

In your PR, include:
- A clear explanation of the change
- Motivation behind it
- Reference to the related issue (e.g. `Closes #12`)
- Relevant logs, benchmarks, or screenshots if applicable

---

## PR Standards

- Keep PRs focused; leave out unrelated changes
- Ensure code runs correctly before submitting
- Maintain compatibility with the existing architecture
- Discuss major architectural changes in an issue before implementing

---

## Reporting a Bug

When opening a bug report, include:
- Clear problem description
- Expected vs actual behaviour
- Steps to reproduce
- Relevant logs or screenshots
- Your environment details (OS, Python version, PyTorch version)

---

## Development Principles

ARC prioritizes:
- Reliability over complexity
- Measured experimental validation
- Transparent benchmarking
- Modular system design
- Honest reporting of limitations

Contributions aligned with these principles are strongly encouraged.

---

## Community

The [Discord server](https://discord.gg/E6UvPWC8DW) is the primary hub for contributor discussions, feature proposals, and development updates. Join us before starting larger contributions.