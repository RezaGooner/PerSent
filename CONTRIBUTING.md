
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
```bash
git fork https://github.com/rezagooner/persent.git
cd persent
```

#### Create a Branch
```bash
git checkout -b feature/my-feature
```

#### Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

#### Make Changes & Test
- Follow [PEP 8](https://peps.python.org/pep-0008/) style.
- Write unit tests for new features (`tests/` folder).
- Run tests before committing:
  ```bash
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
- Update `README.md` if your changes affect usage.

Thank you for making **PerSent** better for the Persian NLP community 💛

> RezaGooner
