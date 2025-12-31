# Jupyter Notebooks

This directory contains Jupyter notebooks for data exploration, model analysis, and experimentation.

## Available Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)

Comprehensive exploration of the LaRoSeDa dataset:
- Dataset statistics and structure
- Class distribution analysis
- Text length and word count distributions
- Word frequency analysis (overall and by sentiment)
- Sample reviews from each class

**Run this notebook to:**
- Understand the dataset before training
- Identify potential biases or imbalances
- Explore Romanian text patterns
- Visualize data characteristics

### 2. Model Analysis (`02_model_analysis.ipynb`)

In-depth analysis of trained models:
- Load saved models and results
- Visualize performance metrics
- Confusion matrix analysis
- ROC and Precision-Recall curves
- Hyperparameter inspection
- Interactive prediction testing

**Run this notebook to:**
- Evaluate model performance
- Understand model predictions
- Test on custom Romanian reviews
- Compare different experiments

## Getting Started

### Prerequisites

Install Jupyter (if not already installed):
```bash
pip install jupyter
```

### Launch Jupyter

From the project root directory:
```bash
# Navigate to notebooks folder
cd notebooks

# Launch Jupyter
jupyter notebook
```

This will open Jupyter in your browser.

### Running Notebooks

1. Click on a notebook file (`.ipynb`)
2. Run cells sequentially with `Shift + Enter`
3. Modify code as needed for your experiments

## Notebook Usage Tips

### Data Exploration Notebook

- No prerequisites - runs directly with HuggingFace dataset
- Takes 1-2 minutes to download and load dataset
- Generates multiple visualizations
- Safe to re-run multiple times

### Model Analysis Notebook

**Prerequisites:**
1. Train a model first using one of the training scripts
2. Update `EXPERIMENT_NAME` variable to match your experiment

Example:
```python
EXPERIMENT_NAME = 'xgb_optimized'  # Change to your experiment folder name
```

**Location of experiments:**
```
../results/experiments/
├── xgb_optimized/
├── my_experiment/
└── ...
```

## Customization

### Adding Your Own Notebooks

Feel free to create additional notebooks for:
- Feature engineering experiments
- Model comparison studies
- Custom analysis scripts
- Visualization templates

### Saving Results

Notebooks can save outputs:
```python
# Save figure
plt.savefig('../results/my_analysis.png', dpi=300, bbox_inches='tight')

# Save DataFrame
df.to_csv('../results/my_results.csv', index=False)
```

## Common Issues

### Issue 1: Module Import Errors
```python
# Notebooks add src to path automatically
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

If imports fail:
- Ensure you're running from `notebooks/` directory
- Check that virtual environment is activated

### Issue 2: Experiment Not Found
```
FileNotFoundError: experiment directory not found
```

**Solution:**
- Verify experiment name matches folder in `results/experiments/`
- Train a model first if no experiments exist

### Issue 3: Out of Memory
```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce sample size in data exploration
- Close other notebooks
- Restart kernel: `Kernel → Restart & Clear Output`

## Best Practices

1. **Run cells in order** - Notebooks depend on previous cells
2. **Clear outputs before committing** - Keep notebook files small
3. **Document changes** - Add markdown cells explaining modifications
4. **Use descriptive names** - If creating new notebooks
5. **Test end-to-end** - Run full notebook before sharing

## Contributing

When adding new notebooks:
1. Use clear section headers with markdown
2. Include docstrings for functions
3. Add explanatory text between code cells
4. Generate meaningful visualizations
5. Update this README with notebook description

## Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Pandas Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)

---

**Note**: Notebooks are for exploration and analysis. For reproducible training, use the scripts in `scripts/` directory.
