import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.graphics.tsaplots import plot_pacf

####. Data Loading and Basic Stats Functions #########

def get_dataframe(folder_names, plot=False, autocorr=False, lags=30):
    """
    Import CSV files, merge them on DATE, optionally plot the time series
    and/or their autocorrelation.
    
    Parameters
    ----------
    folder_names : list of str
        Paths to CSV files to merge.
    plot : bool
        If True, plot the time series.
    autocorr : bool
        If True, plot autocorrelation for each series.
    lags : int
        Number of lags for autocorrelation plot.
        
    Returns
    -------
    pd.DataFrame
        Merged dataframe with DATE as index.
    """
    empty = False
    for folder_name in folder_names:
        df = pd.read_csv(folder_name, sep=';')
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        col_name = os.path.splitext(os.path.basename(folder_name))[0]
        df = df.rename(columns={'PX_LAST': col_name})

        if empty:
            df_all = pd.merge(df_all, df, how='outer', on='DATE')
        else:
            df_all = df
            empty = True

    # Set DATE as index and sort
    df_all = df_all.set_index('DATE').sort_index()

    # Plot time series
    if plot:
        df_all.plot(figsize=(12,5), title="Time Series of Prices")
        plt.show()

    # Plot autocorrelation
    if autocorr:
        for col in df_all.columns:
            plt.figure(figsize=(8,4))
            plot_pacf(df_all[col].dropna(), lags=lags, title=f"Autocorrelation of {col}")
            plt.show()

    return df_all
def print_stats(df):
    """Prints basic statistics about the DataFrame."""
    print("=== Info ===")
    print(df.info())
    # Basic descriptive statistics
    print("=== Stats ===")
    print(df.describe().T)

    # Start and end dates
    print("\n=== Time ===")
    print("Start:", df.index.min().date())
    print("End  :", df.index.max().date())

    # Number of missing values per column
    print("\n=== Missing Values ===")
    print(df.isna().sum())

    # Daily returns and stats
    returns = df.pct_change()
    print("\n=== Returns (mean, volatility) ===")
    print(returns.agg(['mean', 'std']).T)


def add_returns(df, columns=None, method="simple", suffix="_return"):
    """
    Compute returns and add them as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a time index and columns containing prices.
    columns : list or None
        Specific columns to compute returns for. 
        If None, all columns are used.
    method : str
        Method of return calculation:
        - "simple" : simple returns (default)
        - "log"    : log returns
    suffix : str
        String to append to the original column name.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new return columns.
    """
    if columns is None:
        columns = df.columns

    for col in columns:
        if method == "log":
            df[col + suffix] = np.log(df[col] / df[col].shift(1))
        else:
            df[col + suffix] = df[col].pct_change()
    return df




import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

def return_statistics(df, return_cols=None, plot=False, scatter=False, time_interval=None):
    """
    Compute descriptive statistics for return columns and optionally plot them.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing return columns (index should be datetime).
    return_cols : list or None
        List of return column names. If None, all columns ending with '_return' are used.
    plot : bool
        If True, plot the return time series.
    scatter : bool
        If True, plot a scatter matrix of the return columns (similar to R's pairs plot).
    time_interval : tuple of str or pd.Timestamp, optional
        Tuple specifying start and end dates for filtering the data, e.g. ('2022-01-01', '2023-01-01').

    Returns
    -------
    pd.DataFrame
        Table with statistics for each return column:
        - mean: average daily return
        - std: volatility (standard deviation)
        - min: minimum return
        - max: maximum return
        - cumulative_return: product of (1 + r) - 1
    """
    # Filter by time interval if provided
    if time_interval is not None:
        start, end = time_interval
        df = df.loc[start:end]

    # Select return columns
    if return_cols is None:
        return_cols = [col for col in df.columns if '_return' in col]

    # Compute statistics
    stats = pd.DataFrame(index=return_cols)
    stats["mean"] = df[return_cols].mean()
    stats["std"] = df[return_cols].std()
    stats["min"] = df[return_cols].min()
    stats["max"] = df[return_cols].max()
    stats["cumulative_return"] = (1 + df[return_cols]).prod() - 1

    # Plot time series
    if plot:
        plt.figure(figsize=(12, 6))
        for col in return_cols:
            plt.plot(df.index, df[col], label=col, linewidth=1.5)
        plt.title("Return Time Series", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Scatter matrix
    if scatter:
        scatter_matrix(df[return_cols], figsize=(10, 10), diagonal="kde", alpha=0.7)
        plt.suptitle("Scatter Matrix of Returns", fontsize=14, fontweight="bold")
        plt.show()

    return stats



def sample_average(df, sample_unit = "M"):
    """
    Group a time series DataFrame by month and compute the mean.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        New DataFrame resampled by month with mean values.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    
    return df.resample(sample_unit).sum()




######### Linear Regression Functions #########
def regression_matrix(df, return_cols=None):
    """
    Create a matrix of regression statistics for all pairs of return columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing return series.
    return_cols : list or None
        Columns to include. If None, all columns ending with '_return' are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - rows = dependent variable Y
        - columns = independent variable X
        - each cell = dict with sum, sumsq, cross, Beta0, Beta1, R2
    """
    if return_cols is None:
        return_cols = [c for c in df.columns if "_return" in c]

    # Initialize empty DataFrame
    matrix = pd.DataFrame(index=return_cols, columns=return_cols)

    for Y_col in return_cols:
        for X_col in return_cols:
            if X_col == Y_col:
                matrix.loc[Y_col, X_col] = None
                continue

            X = df[X_col]
            Y = df[Y_col]

            sum_X = X.sum()
            sum_Y = Y.sum()
            sumsq_X = (X ** 2).sum()
            sumsq_Y = (Y ** 2).sum()
            cross_XY = (X * Y).sum()
            mean_X = X.mean()
            mean_Y = Y.mean()
            # Beta1 and Beta0
            beta1 = ((X - X.mean()) * (Y - Y.mean())).sum() / ((X - X.mean())**2).sum()
            beta0 = Y.mean() - beta1 * X.mean()

            # R squared
            Y_pred = beta0 + beta1 * X
            ss_tot = ((Y - Y.mean()) ** 2).sum()
            ss_res = ((Y - Y_pred) ** 2).sum()
            R2 = 1 - ss_res / ss_tot

            # Fill cell with dict
            matrix.at[Y_col, X_col] = {
                "sum_Y": sum_Y,
                "sum_X": sum_X,
                "mean_Y": mean_X,
                "mean_X": mean_Y,
                "sumsq_Y": sumsq_Y,
                "sumsq_X": sumsq_X,
                "cross_XY": cross_XY,
                "Beta0": beta0,
                "Beta1": beta1,
                "R2": R2
            }

    return matrix


def linear_regression_with_matrix(df, return_cols=None):
    """
    Crée une matrice de régression pour toutes les paires de colonnes de rendements.
    Chaque cellule contient les stats et coefficients de régression (β0, β1, R², sommes, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les séries de rendement.
    return_cols : list or None
        Colonnes à inclure. Si None, toutes les colonnes se terminant par '_return' sont utilisées.

    Returns
    -------
    pd.DataFrame
        DataFrame avec :
        - lignes = variable dépendante Y
        - colonnes = variable indépendante X
        - chaque cellule = dict avec sum, sumsq, cross, Beta0, Beta1, R2
    """
    if return_cols is None:
        return_cols = [c for c in df.columns if "_return" in c]

    matrix = pd.DataFrame(index=return_cols, columns=return_cols)

    for Y_col in return_cols:
        for X_col in return_cols:
            if X_col == Y_col:
                matrix.at[Y_col, X_col] = None
                continue

            X = df[X_col].values
            Y = df[Y_col].values

            # Keep only finite values
            mask = np.isfinite(X) & np.isfinite(Y)
            X = X[mask]
            Y = Y[mask]
            n = len(X)
            if n <= 2:
                matrix.at[Y_col, X_col] = None
                continue

            # Matrice avec constante pour intercept
            X_mat = np.column_stack((np.ones(n), X))  # [1, X]
            beta = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ Y)
            beta0, beta1 = beta

            # Prédictions et R²
            Y_pred = beta0 + beta1 * X
            ss_tot = np.sum((Y - np.mean(Y))**2)
            ss_res = np.sum((Y - Y_pred)**2)
            R2 = 1 - ss_res / ss_tot

            # Remplir la cellule
            matrix.at[Y_col, X_col] = {
                "Beta0": beta0,
                "Beta1": beta1,
                "R2": R2
            }

    return matrix



def run_regression(df, y_col, x_col, outfile="IndexRegResults.txt", write_file=True):
    """
    Estimate a simple OLS regression: y = alpha + beta * x + error
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing return columns.
    y_col : str
        Dependent variable (response).
    x_col : str
        Independent variable (regressor).
    outfile : str
        File to save the regression summary.
    
    Returns
    -------
    results : statsmodels regression results
    """
    data = df[[y_col, x_col]].dropna()

    y = data[y_col]
    X = sm.add_constant(data[x_col])  # add intercept
    model = sm.OLS(y, X).fit(cov_type='HC0')

    # Save regression results into file
    if write_file:
        with open(outfile, "w") as f:
            f.write(model.summary().as_text())
    else: 
        print(model.summary())
    return model



###. Statistical Tests Functions #########

def test_intercept(df, matrix):
    """
    Manually compute t-statistic and p-value for H0: intercept = 0
    """

    for Y_col in matrix.index:
        for X_col in matrix.columns:
            if X_col == Y_col or matrix.loc[Y_col, X_col] is None:
                continue
            Y = df[Y_col].values
            X = df[X_col].values

            # Keep only rows where both X and Y are not NaN
            mask = (~np.isnan(X)) & (~np.isnan(Y))
            X = X[mask]
            Y = Y[mask]
            
            dict_regression = matrix.loc[Y_col, X_col]
            beta0 = dict_regression['Beta0']
            beta1 = dict_regression['Beta1']
            mean_X = dict_regression['mean_X']
            n = len(X)
            if n < 2:
                print("Not enough data points for regression.")
                return None

            # Residuals and residual variance
            residuals = Y - (beta0 + beta1 * X)
            s2 = np.sum(residuals**2) / (n - 2)
            
            # Standard error of intercept
            SE_beta0 = np.sqrt(s2 * (1/n + mean_X**2 / np.sum((X - mean_X)**2)))

            # t-statistic for H0: beta0 = 0
            t_stat = beta0 / SE_beta0

            # two-sided p-value
            p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n-2))
            if p_value < 0.05:
                result_intercept = "We reject H0: intercept != 0"
            if p_value > 0.05:
                result_intercept = "Fail to reject H0: intercept = 0"
            dict_regression['p_value_intercept'] = p_value
            dict_regression['t_stat_intercept'] = t_stat
            dict_regression['result_intercept'] = result_intercept
            
            matrix.at[Y_col, X_col] = dict_regression


    return matrix



def test_beta1(df, matrix):
    """
    Manually compute t-statistic and p-value for H0: beta1 = 0
    """
    for Y_col in matrix.index:
        for X_col in matrix.columns:
            if X_col == Y_col or matrix.loc[Y_col, X_col] is None:
                continue

            Y = df[Y_col].values
            X = df[X_col].values

            # Supprimer les NaN
            mask = (~np.isnan(X)) & (~np.isnan(Y))
            X = X[mask]
            Y = Y[mask]

            dict_regression = matrix.loc[Y_col, X_col]
            beta0 = dict_regression['Beta0']
            beta1 = dict_regression['Beta1']
            mean_X = dict_regression['mean_X']
            n = len(X)
            if n < 2:
                print("Not enough data points for regression.")
                return None

            # Résiduals and residual variance
            residuals = Y - (beta0 + beta1 * X)
            s2 = np.sum(residuals**2) / (n - 2)

            # stadard deviation of X
            Sxx = np.sum((X - mean_X)**2)

            # Standard error of beta1
            SE_beta1 = np.sqrt(s2 / Sxx)

            # t-statistic pour H0: beta1 = 0
            t_stat = beta1 / SE_beta1

            # p-value bilatérale
            p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n-2))
            if p_value < 0.05:
                test_result = "We reject H0: beta1 != 0"
            if p_value > 0.05:
                test_result = "Fail to reject H0: beta1 = 0"
            dict_regression['p_value_beta1'] = p_value
            dict_regression['t_stat_beta1'] = t_stat
            dict_regression['result_beta1'] = test_result
            matrix.at[Y_col, X_col] = dict_regression

    return matrix




###. Final Plot Function / Conclusion 

def scatter_with_stats(df, matrix_of_regression, return_cols=None, scatter=True):
    """
    Displays a scatter matrix of returns with associated regression statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series of returns.
    matrix_of_regression : pd.DataFrame or dict
        Contains regression results (Beta0, Beta1, R², p-values, etc.) for each pair of return columns.
    return_cols : list, optional
        List of columns to include in the scatter matrix. If None, selects columns containing '_return'.
    scatter : bool
        If True, plots the scatter matrix.

    Returns
    -------
    matrix_of_regression : same type as input
        Regression results matrix passed to the function.
    """

    # If return_cols not specified, use all columns with '_return' in their name
    if return_cols is None:
        return_cols = [col for col in df.columns if '_return' in col]

    if scatter:
        # Plot scatter matrix with KDE on diagonals
        axes = scatter_matrix(df[return_cols], figsize=(10, 10), diagonal="kde", alpha=0.7)

        n = len(return_cols)

        # Loop through each subplot in scatter matrix
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax = axes[i, j]
                    Y_col, X_col = return_cols[i], return_cols[j]

                    # Retrieve regression results for this pair of columns
                    reg = matrix_of_regression.at[Y_col, X_col]
                    beta0 = reg.get("Beta0", None)
                    beta1 = reg.get("Beta1", None)
                    R2    = reg.get("R2", None)
                    p_val_intercept = reg.get("p_value_intercept", None)
                    p_val_beta1 = reg.get("p_value_beta1", None)

                    # Format regression statistics text
                    text = f"β0={beta0:.2e}\nβ1={beta1:.2f}\nR²={R2:.2f}\np_val_beta1={p_val_beta1:.2f}\np_val_intercept={p_val_intercept:.2f}"

                    # Place stats text inside subplot
                    ax.text(
                        0.05, 0.95, text,
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                    )

                    # If coefficients are valid, plot regression line
                    if beta0 is not None and beta1 is not None:
                        x_vals = np.array(ax.get_xlim())
                        y_vals = beta0 + beta1 * x_vals
                        ax.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=1)

        # Add a title to the figure
        plt.suptitle("Scatter Matrix of Returns with Regression Stats", fontsize=14, fontweight="bold")
        plt.show()

    return matrix_of_regression



