# Financial Time Series Analysis

## Notebook

All analysis is performed in the **`first_lab`** notebook, which contains a detailed step-by-step exploration and regression analysis of the data.

## Data

To run the notebook, you will need the following CSV files:

* `CAC40.csv`
* `DowJones.csv`
* `EuroStoxx.csv`
* `SandP500.csv`

Additionally, the **`utils`** module (provided) is required for data processing and analysis.

## Utils Module

The `utils` module is a Python toolkit for **financial time series analysis**, including data loading, return calculations, statistics, regression, hypothesis testing, and visualization.

### Features

1. **Data Handling**

   * `get_dataframe()`: Load multiple CSVs, merge, index by date, and optionally plot time series.

2. **Returns Calculation**

   * `add_returns()`: Compute simple or log returns for price series.

3. **Descriptive Statistics**

   * `print_stats()`: Summary statistics, missing values, daily returns.
   * `yield_statistics()`: Statistics for yield columns with optional plotting and scatter matrix.

4. **Resampling**

   * `sample_average()`: Compute monthly or other period averages.

5. **Linear Regression**

   * `regression_matrix()` / `linear_regression_with_matrix()`: Generate a matrix of regression statistics (β₀, β₁, R²) for all pairs of yield columns.
   * `run_regression()`: Run simple OLS regression with statsmodels, optionally save results to a file.

6. **Hypothesis Testing**

   * `test_intercept()`: t-test and p-value for intercept β₀ = 0.
   * `test_beta1()`: t-test and p-value for slope β₁ = 0.

7. **Visualization**

   * `scatter_with_stats()`: Scatter matrix with regression lines and statistics for each pair of yield columns.

## Dependencies

* `pandas`
* `numpy`
* `matplotlib`
* `statsmodels`
* `scipy`

## Use Case

Designed for **quick exploratory analysis and regression** of multiple financial yield series, enabling both statistical testing and visual inspection of relationships between indices.

---
