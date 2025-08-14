# Contributing to SNAFU

Thank you for your interest in contributing to SNAFU! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## 📝 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests
```bash
pytest tests/
```

### 4. Check Code Style
```bash
black snafu/
flake8 snafu/
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "Add feature: brief description"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No merge conflicts

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=snafu

# Run specific test file
pytest tests/test_core.py
```

### Writing Tests
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common setup
- Mock external dependencies

## 📚 Documentation

### Code Documentation
- Use docstrings for all public functions
- Follow Google docstring format
- Include examples in docstrings

### User Documentation
- Update README.md for user-facing changes
- Add examples to docs/examples.md
- Update API documentation

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- SNAFU version
- Minimal reproducible example
- Expected vs actual behavior

## 💡 Feature Requests

For feature requests:
- Describe the use case
- Explain the expected behavior
- Consider implementation complexity
- Check if it aligns with project goals

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/AusterweilLab/snafu-py/issues)
- **Discussion**: [Google Group](https://groups.google.com/forum/#!forum/snafu-fluency)
- **Email**: snafu-fluency [at] googlegroups [dot] com

## 🙏 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

Thank you for contributing to SNAFU! 🎉
