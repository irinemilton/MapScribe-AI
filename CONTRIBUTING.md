# Contributing to MapScribe AI

🎉 Thank you for your interest in contributing to MapScribe AI! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of computer vision and OCR
- Familiarity with Flask (for web development)

### Development Setup

1. **Fork the repository**
```bash
git clone https://github.com/YOUR_USERNAME/MapScribe-AI.git
cd MapScribe-AI
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Run tests to ensure everything works**
```bash
python -m pytest tests/ -v
```

## 🛠️ Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable and function names

### Code Formatting
We use Black for code formatting:
```bash
black src/ tests/ *.py
```

### Linting
We use flake8 for linting:
```bash
flake8 src/ tests/ *.py
```

### Type Checking
We use mypy for type checking:
```bash
mypy src/
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_processors.py -v

# Run tests matching pattern
python -m pytest -k "test_ocr" -v
```

### Writing Tests
- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Include both positive and negative test cases

Example test structure:
```python
def test_processor_categorizes_states_correctly():
    """Test that the processor correctly identifies state names."""
    processor = SmartGeoProcessor()
    result = processor.categorize_text("Andhra Pradesh")
    
    assert result['category'] == 'states'
    assert result['display_name'] == 'Andhra Pradesh'
    assert result['validated'] is True
```

## 📝 Documentation

### Docstrings
Use Google-style docstrings:
```python
def process_image(self, image_path: str, options: Dict = None) -> Dict:
    """Process historical map image with OCR and categorization.
    
    Args:
        image_path: Path to the image file to process
        options: Optional processing parameters
        
    Returns:
        Dictionary containing categorized results and metadata
        
    Raises:
        ValueError: If image file cannot be read
        ProcessingError: If OCR processing fails
    """
```

### README Updates
- Update README.md for new features
- Include code examples
- Update version numbers appropriately

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues to avoid duplicates
2. Test with the latest version
3. Provide minimal reproduction case

### Bug Report Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Upload image '...'
2. Select options '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Windows 10, Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- MapScribe AI version: [e.g. 1.0.0]

**Additional context**
Any other context about the problem.
```

## ✨ Feature Requests

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 🔄 Pull Request Process

### Before Submitting
1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Run code formatting and linting

### Pull Request Template
```markdown
**Description**
Brief description of changes made.

**Type of Change**
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

**Testing**
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process
1. Automated tests must pass
2. Code review by maintainers
3. Address feedback if requested
4. Merge after approval

## 🏗️ Architecture Guidelines

### Adding New Processors
1. Inherit from base processor class
2. Implement required methods
3. Add comprehensive tests
4. Update processor selection logic
5. Document new processor capabilities

### Adding New Features
1. Follow existing patterns
2. Maintain backward compatibility
3. Add configuration options
4. Include error handling
5. Write comprehensive tests

### Database Changes
1. Create migration scripts
2. Update data models
3. Test with existing data
4. Document schema changes

## 🌍 Internationalization

### Adding Language Support
1. Add language codes to supported list
2. Update OCR model configurations
3. Add language-specific categorization rules
4. Test with sample text in new language
5. Update documentation

## 📊 Performance Guidelines

### Optimization Principles
1. Profile before optimizing
2. Optimize for common use cases
3. Maintain code readability
4. Document performance characteristics
5. Add performance tests

### Memory Management
- Use generators for large datasets
- Clean up temporary files
- Monitor memory usage in tests
- Implement proper resource cleanup

## 🔒 Security Guidelines

### Input Validation
- Validate all user inputs
- Sanitize file uploads
- Check file types and sizes
- Prevent path traversal attacks

### Dependencies
- Keep dependencies updated
- Review security advisories
- Use virtual environments
- Pin dependency versions

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Code Reviews**: Technical discussions on PRs

### Maintainer Response Times
- Issues: Within 48 hours
- Pull Requests: Within 72 hours
- Security Issues: Within 24 hours

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MapScribe AI! 🗺️✨