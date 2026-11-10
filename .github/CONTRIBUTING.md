
# Contributing to PerSent

First off — thanks for your interest in contributing to **PerSent**! 🎉  
This project is built for the Persian NLP community, and every contribution helps us improve.

## How to Contribute

### 1. Reporting Bugs
If you find a bug:
1. Check if it’s already reported in the [issues page](../../issues).
2. If not, open a **new issue** with:
   - Clear description of the problem.
   - Steps to reproduce.
   - Expected vs. actual results.
   - Your Python version and OS.

### 2. Suggesting Features
Got an idea?  
- Create a **feature request** issue explaining:
  - The problem your idea solves.
  - Example usage.
  - Any alternatives considered.

### 3. Code Contributions

#### Fork & Clone
Fork the repository on GitHub, then:
```bash
git clone https://github.com/<your-username>/PerSent.git
cd PerSent
```

#### Create a Branch
```bash
git checkout -b feature/my-feature
```

#### Install Development Dependencies
```bash
pip install -e ".[dev]"
```

#### Make Changes & Test
- Code style is enforced with [ruff](https://docs.astral.sh/ruff/) (PEP 8, line length 120).
- Write unit tests for new features (`tests/` folder).
- Run the checks before committing (CI runs the same commands):
  ```bash
  ruff check . && ruff format --check .
  pytest
  ```

#### Commit & Push
```bash
git commit -m "Add: short but clear description of changes"
git push origin feature/my-feature
```

#### Open a Pull Request
- Describe **what** you changed and **why**.
- Reference related issues if any (`Closes #123`).

## Code of Conduct
This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).  
By participating in this project, you agree to abide by its terms.

## Development Tips
- Keep PRs focused and minimal.
- Add docstrings to functions and classes.
- Update `README.md` / `README.fa.md` if your changes affect usage, and add an entry to `CHANGELOG.md`.

Thank you for making **PerSent** better for the Persian NLP community 💛

> RezaGooner
