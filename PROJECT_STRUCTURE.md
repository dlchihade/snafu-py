# SNAFU Project Structure

## Recommended Repository Organization

```
snafu-py/
├── README.md                    # Main project documentation
├── setup.py                     # Package installation
├── requirements.txt             # Python dependencies
├── LICENSE                      # Project license
├── .gitignore                   # Git ignore rules
├── CONTRIBUTING.md              # Contribution guidelines
├── CHANGELOG.md                 # Version history
├── docs/                        # Documentation
│   ├── installation.md
│   ├── usage.md
│   ├── api.md
│   └── examples.md
├── snafu/                       # Main package (keep as is)
│   ├── __init__.py
│   ├── core.py
│   ├── clustering.py
│   └── ...
├── examples/                    # Example scripts and demos
│   ├── basic_usage.py
│   ├── network_analysis.py
│   ├── fluency_analysis.py
│   └── jupyter_notebooks/
├── scripts/                     # Utility scripts
│   ├── data_processing/
│   ├── analysis/
│   └── visualization/
├── tests/                       # Test suite
│   ├── unit/
│   ├── integration/
│   └── test_data/
├── data/                        # Sample data and resources
│   ├── sample/
│   ├── schemes/
│   ├── spellfiles/
│   └── networks/
├── research/                    # Research-specific analysis
│   ├── semantic_fluency_analysis/
│   ├── publications/
│   └── experiments/
└── tools/                       # Additional tools
    ├── spell_checking/
    └── troyer_letter_functions/
```

## Key Principles

1. **Separation of Concerns**: Core library vs. research scripts vs. examples
2. **Documentation First**: Clear README and documentation structure
3. **Modular Organization**: Related functionality grouped together
4. **Research Preservation**: Keep research scripts but organize them clearly
5. **User-Friendly**: Easy for new users to find examples and get started
