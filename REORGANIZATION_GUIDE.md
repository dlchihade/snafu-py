# SNAFU Repository Reorganization Guide

This guide outlines the reorganization of the SNAFU repository to improve its structure for GitHub and make it more maintainable and user-friendly.

## 🎯 Goals

1. **Improve Discoverability**: Make it easy for new users to find examples and documentation
2. **Separate Concerns**: Distinguish between core library, research scripts, and examples
3. **Enhance Maintainability**: Better organization for easier maintenance
4. **Professional Presentation**: Follow GitHub best practices
5. **Preserve Research**: Keep all research work while organizing it clearly

## 📁 New Directory Structure

```
snafu-py/
├── README.md                    # Enhanced main documentation
├── setup.py                     # Package installation (existing)
├── requirements.txt             # Comprehensive dependencies
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
│   ├── schemes/                 # Moved from /schemes
│   ├── spellfiles/              # Moved from /spellfiles
│   └── networks/                # Moved from /snet
├── research/                    # Research-specific analysis
│   ├── semantic_fluency_analysis/  # Moved from root
│   ├── publications/
│   └── experiments/
└── tools/                       # Additional tools (keep as is)
    ├── spell_checking/
    └── troyer_letter_functions/
```

## 🔄 File Movements

### Data Files
- `schemes/` → `data/schemes/`
- `spellfiles/` → `data/spellfiles/`
- `snet/` → `data/networks/`
- `fluency_data/` → `data/sample/`

### Examples and Demos
- `demos/` → `examples/`
- Create new example files in `examples/`

### Research Work
- `semantic_fluency_analysis/` → `research/semantic_fluency_analysis/`

### Utility Scripts
- Move analysis scripts to `scripts/analysis/`
- Move data processing scripts to `scripts/data_processing/`
- Move visualization scripts to `scripts/visualization/`

## 🛠️ Implementation Steps

### Step 1: Run the Reorganization Script
```bash
python reorganize_repo.py
```

### Step 2: Review and Update Imports
After reorganization, you'll need to update import paths in your code:

```python
# Old imports
from schemes import animals_snafu_scheme
from spellfiles import animals_snafu_spellfile

# New imports
from data.schemes import animals_snafu_scheme
from data.spellfiles import animals_snafu_spellfile
```

### Step 3: Update Documentation
- Replace old README.md with README_NEW.md
- Update any file paths in documentation
- Create missing documentation files

### Step 4: Test Everything
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Test examples
python examples/basic_usage.py
```

### Step 5: Commit Changes
```bash
git add .
git commit -m "Reorganize repository structure for better GitHub organization"
git push origin main
```

## 📋 Checklist

### Before Reorganization
- [ ] Backup your current repository
- [ ] Review the reorganization plan
- [ ] Identify any custom import paths in your code

### During Reorganization
- [ ] Run `reorganize_repo.py`
- [ ] Update import paths in your code
- [ ] Test that everything still works
- [ ] Update documentation references

### After Reorganization
- [ ] Replace README.md with the new version
- [ ] Set up GitHub Actions (CI/CD)
- [ ] Update any external links or references
- [ ] Test installation from scratch
- [ ] Update any published documentation

## 🚨 Important Notes

### Preserving Research Work
- All research scripts are preserved in `research/semantic_fluency_analysis/`
- No functionality is lost
- All analysis workflows remain intact

### Backward Compatibility
- The core `snafu` package remains unchanged
- API compatibility is maintained
- Only file locations change, not functionality

### Data Files
- All data files are preserved and moved to appropriate locations
- File paths in code need to be updated
- Consider creating symbolic links for frequently accessed files

## 🔧 Customization

You can customize the reorganization by modifying `reorganize_repo.py`:

1. **Add more directories**: Modify the `directories` list
2. **Custom file movements**: Add to the `move_files()` function
3. **Additional examples**: Extend `create_example_files()`
4. **Custom documentation**: Modify `create_documentation_files()`

## 📞 Support

If you encounter issues during reorganization:

1. Check the error messages from `reorganize_repo.py`
2. Review the file movements in the script
3. Test individual components after reorganization
4. Update import paths systematically

## 🎉 Benefits

After reorganization, you'll have:

- **Better User Experience**: Clear examples and documentation
- **Easier Maintenance**: Logical file organization
- **Professional Appearance**: GitHub best practices
- **Improved Discoverability**: Easy to find relevant files
- **Research Preservation**: All work maintained and organized

This reorganization will make SNAFU more accessible to new users while preserving all existing functionality and research work.
