# SNAFU: Semantic Network and Fluency Utility

[![Python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs42113--018--0003--7-blue.svg)](https://link.springer.com/article/10.1007/s42113-018-0003-7)

SNAFU is a comprehensive Python toolkit for analyzing semantic fluency data and generating semantic networks. It's widely used in psychology research and clinical practice for cognitive assessment.

## 🚀 Features

### Core Analysis
- **Cluster Analysis**: Count cluster switches and cluster sizes
- **Error Detection**: Identify perseverations and intrusions
- **Word Properties**: Calculate age-of-acquisition and word frequency
- **Network Generation**: Multiple network estimation methods

### Network Methods
- U-INVITE networks
- Pathfinder networks  
- Correlation-based networks
- Naive random walk networks
- Conceptual networks
- First Edge networks

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install pysnafu
```

### From Source
```bash
git clone https://github.com/AusterweilLab/snafu-py.git
cd snafu-py
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
import snafu

# Load your fluency data
data = snafu.load_data("your_fluency_data.csv")

# Analyze clusters
clusters = snafu.analyze_clusters(data)

# Generate network
network = snafu.generate_network(data, method='u-invite')

# Get network statistics
stats = snafu.network_statistics(network)
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Examples](docs/usage.md)
- [API Reference](docs/api.md)
- [Research Applications](docs/research.md)

## 🔬 Research Applications

SNAFU has been used in numerous research studies. See the [research/](research/) directory for:
- Semantic fluency analysis scripts
- Publication-ready figures
- Statistical analysis workflows
- Network comparison studies

## 🛠️ Development

### Setting up Development Environment
```bash
git clone https://github.com/AusterweilLab/snafu-py.git
cd snafu-py
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black snafu/
flake8 snafu/
```

## 📖 Examples

Check out the [examples/](examples/) directory for:
- Basic usage examples
- Network analysis tutorials
- Jupyter notebooks
- Real-world case studies

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/AusterweilLab/snafu-py/issues)
- **Discussion**: [Google Group](https://groups.google.com/forum/#!forum/snafu-fluency)

## 🙏 Acknowledgments

- Jeffrey C. Zemla and The Austerweil Lab at UW-Madison
- Contributors and users worldwide
- Research community for feedback and testing

## 📊 Citation

If you use SNAFU in your research, please cite:

```bibtex
@article{zemla2018semantic,
  title={Semantic fluency as a measure of cognitive reserve},
  author={Zemla, Jeffrey C and Austerweil, Joseph L},
  journal={Computational Brain \& Behavior},
  volume={1},
  number={1},
  pages={1--15},
  year={2018},
  publisher={Springer}
}
```
