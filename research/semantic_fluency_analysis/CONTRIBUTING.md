# Contributing to Semantic Fluency Analysis

Thank you for your interest in contributing to the Semantic Fluency Analysis project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Provide detailed descriptions of the problem
- Include steps to reproduce the issue
- Attach relevant data files (if possible)

### Suggesting Enhancements
- Open a feature request issue
- Describe the enhancement clearly
- Explain the scientific or practical benefits
- Provide examples if applicable

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 🏗️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- A code editor (VS Code, PyCharm, etc.)

### Local Development
```bash
# Clone your fork
git clone https://github.com/yourusername/semantic-fluency-analysis.git
cd semantic-fluency-analysis

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_md

# Run tests
python test_spacy_optimization.py
python test_basic.py
```

## 📝 Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

### Example
```python
def analyze_participant_data(participant_id: str, 
                           fluency_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze semantic fluency data for a single participant.
    
    Args:
        participant_id: Unique identifier for the participant
        fluency_data: DataFrame containing fluency responses
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        ValueError: If participant data is invalid
    """
    # Implementation here
    pass
```

### Documentation
- Update README.md for new features
- Add docstrings to all new functions
- Include examples in documentation
- Update configuration examples

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
python test_spacy_optimization.py
python test_basic.py
python test_migration.py

# Run specific analysis
python main.py
python phase_coherence_analysis.py
```

### Writing Tests
- Test both success and failure cases
- Use realistic test data
- Test edge cases and boundary conditions
- Ensure tests are independent and repeatable

### Example Test
```python
def test_phase_coherence_calculation():
    """Test phase coherence calculation with known data."""
    # Setup test data
    test_phases = [
        {'type': 'Exploitation', 'vectors': [vector1, vector2]},
        {'type': 'Exploration', 'vectors': [vector3, vector4]}
    ]
    
    # Run calculation
    result = compute_phase_coherence_metrics(test_phases)
    
    # Assert expected results
    assert 'exploitation_intra_mean' in result
    assert result['exploitation_intra_mean'] > 0
```

## 📊 Data Handling

### Data Privacy
- Never commit sensitive data to the repository
- Use .gitignore to exclude data files
- Provide example data for testing
- Document data format requirements

### Data Formats
- Support CSV format for input data
- Use standardized column names
- Validate data before processing
- Handle missing data gracefully

## 🔬 Scientific Contributions

### Research Applications
- Focus on cognitive neuroscience applications
- Consider clinical relevance
- Validate against established methods
- Document theoretical frameworks

### Methodological Improvements
- Propose new analysis methods
- Improve existing algorithms
- Add statistical validation
- Enhance visualization capabilities

## 📈 Performance Considerations

### Optimization
- Profile code for bottlenecks
- Use efficient data structures
- Implement caching where appropriate
- Consider parallel processing for large datasets

### Memory Management
- Handle large datasets efficiently
- Clean up resources properly
- Monitor memory usage
- Optimize for scalability

## 🎨 Visualization Contributions

### Figure Standards
- Use 600 DPI for publication quality
- Follow academic journal requirements
- Use consistent color schemes
- Include proper labels and legends

### Code Style
```python
def create_publication_figure(data: pd.DataFrame, 
                            output_path: str) -> None:
    """Create publication-quality figure."""
    # Set up matplotlib parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'figure.dpi': 600
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    ax.plot(data['x'], data['y'])
    
    # Format figure
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('Figure Title')
    
    # Save figure
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
```

## 📋 Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Update documentation
3. Check code style
4. Test with different data sets
5. Verify performance impact

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
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Documentation
- [ ] README updated
- [ ] Docstrings added
- [ ] Examples provided

## Scientific Impact
Description of scientific or clinical relevance
```

## 🏷️ Version Control

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, etc.)
- Reference issues when applicable
- Keep commits focused and atomic

### Branch Naming
- `feature/description` for new features
- `fix/description` for bug fixes
- `docs/description` for documentation
- `test/description` for test additions

## 📞 Getting Help

### Communication
- Use GitHub issues for discussions
- Be respectful and constructive
- Provide context for questions
- Share relevant code and data

### Resources
- Check existing documentation
- Review similar issues
- Consult scientific literature
- Ask for clarification when needed

## 🙏 Recognition

### Contributors
- All contributors will be listed in the README
- Significant contributions will be acknowledged
- Co-authorship considered for major scientific contributions

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Respect different perspectives
- Maintain professional standards

---

Thank you for contributing to advancing cognitive neuroscience research! 🧠
