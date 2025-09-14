# DAT200 CA5: Scoville Heat Unit Prediction

**Group 37**
- Jannicke Ådalen
- Marcus Dalaker Figenschou
- Rikke Sellevold Vegstein

## Project Overview

This project implements machine learning models to predict Scoville Heat Units (SHU) for peppers using a two-stage approach:
1. Binary classification to determine if a pepper is spicy (SHU > 0)
2. Regression to predict the exact SHU value for spicy peppers

The solution combines feature engineering, data preprocessing, and ensemble methods to handle the complex nature of pepper heat prediction.

**Final Result:** 23rd place out of 73 teams on Kaggle

## Technical Approach

- **Classification Model**: Identifies spicy vs non-spicy peppers
- **Regression Model**: Predicts SHU values for peppers classified as spicy
- **Feature Engineering**: Logarithmic scaling, outlier removal, correlation analysis
- **Model Selection**: Gradient boosting algorithms with hyperparameter tuning

## Technical Challenges

### Apple Silicon Compatibility Issues

This project was significantly impacted by a critical bug in Apple Silicon's matrix multiplication operations that caused "division by zero" errors in various ML libraries. This bug particularly affected:

- NumPy/SciPy matrix operations
- Scikit-learn model training and cross-validation
- Feature engineering pipelines
- Hyperparameter optimization processes

**Workarounds Implemented:**
- Modified computational approaches to avoid problematic matmul operations
- Used alternative algorithms where possible
- Implemented custom numerical stability fixes
- Adjusted model training procedures to work around the bug

**Note:** This bug was officially acknowledged and fixed by Apple during summer 2025. The workarounds in this codebase may no longer be necessary on updated systems with the latest macOS versions.

## Impact on Development

Due to the extensive time spent on compatibility issues and implementing workarounds, we were unable to fully optimize our models and feature engineering pipeline. This significantly limited our ability to:
- Conduct thorough hyperparameter tuning
- Implement advanced ensemble methods
- Perform comprehensive cross-validation
- Optimize feature selection processes

The technical constraints ultimately impacted our competitive performance, though the implemented solution demonstrates solid machine learning fundamentals despite these challenges.

## Files Structure

- `CA5.ipynb` - Main Jupyter notebook with complete analysis
- `A_C task.py` - Python script version of the solution
- `CA5_best_model.py` - Final model implementation
- `submission.csv` - Kaggle submission file
- `assets/` - Supporting files and visualizations
- `results/` - Model outputs and evaluation metrics

## Requirements

See `pyproject.toml` for dependencies. Note that some workarounds may require specific versions of NumPy and scikit-learn due to the Apple Silicon compatibility issues mentioned above.

## Usage

1. Install dependencies: `pip install -e .`
2. Run the main notebook: `jupyter notebook CA5.ipynb`
3. Or execute the Python script: `python "A_C task.py"`

**Warning:** If running on older macOS versions with Apple Silicon, you may encounter the matrix multiplication bug. Consider updating to the latest macOS version or use the workaround implementations in the code.
