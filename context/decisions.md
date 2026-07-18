# Decisions

Append-only log of non-obvious decisions made while building this project. One entry per decision.

## Log

- Backend venv must use Python 3.12 (or 3.11/3.13), not 3.14 — pydantic-core
  has no prebuilt wheel for 3.14 yet and fails to build from source (pyo3
  doesn't support 3.14). Use `python3.12 -m venv venv`.
