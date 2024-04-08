'''
StatisticsGameChoice.py
Contains the code for the statistical tests of the Strategy and GameChoice chapter
'''


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def coevolution_risk_UV(data, variable, NH = False, normalizeGames =  True):
    '''
    Description: 
    Compute regression coefficients, p-values, and standard errors for the relationship between UV and a given variable.
    Inputs:
    - data (DataFrame): DataFrame containing the dataset.
    - variable (str): Name of the variable to analyze.
    Outputs:
    - Prints regression coefficients, p-values, standard errors, and summary of regression model.
    '''
    print("NH:", NH, "Normalized Games:", normalizeGames)
    print(f"Regression for {variable}")
    
    # Filter data for the last timestep
    last_timestep_data = data[data['Step'] == data['Step'].max()]


    # Extract predictor variables (U and V) and response variable (risk aversion)
    x = last_timestep_data['UV'].apply(lambda uv: pd.Series(uv))  # Extract U and V into separate columns
    y = last_timestep_data[variable]

    # Add constant term to predictor variables
    x = sm.add_constant(x)

    # Fit the regression model
    regression_model = sm.OLS(y, x).fit()

    # Get regression coefficients
    coef_u = regression_model.params[0]
    coef_v = regression_model.params[1]
    
    # Get p-values
    p_value_u = regression_model.pvalues[0]
    p_value_v = regression_model.pvalues[1]
    
    # Get standard errors
    se_u = regression_model.bse[0]
    se_v = regression_model.bse[1]

    # Print regression coefficients, p-values, and standard errors
    print(f"Coefficient for U: {coef_u}, p-value: {p_value_u}, SE: {se_u}")
    print(f"Coefficient for V: {coef_v}, p-value: {p_value_v}, SE: {se_v}")

    # Print regression summary
    print(regression_model.summary())


def distributions_statistics(data, NH, normalizeGames):
    '''
    Description: 
    Compute statistics for normalized wealth and recent wealth variables.
    Inputs:
    - data (DataFrame): DataFrame containing the dataset.
    Outputs:
    - Prints statistics for normalized wealth and recent wealth variables.
    '''
    print("NH:", NH, "Normalized Games:", normalizeGames)

    # Filter data for the last time step
    last_time_step_data = data[data['Step'] == data['Step'].max()]

    # Normalize the wealth and recent wealth variables
    last_time_step_data['Normalized_Wealth'] = (last_time_step_data['Wealth'] - last_time_step_data['Wealth'].min()) / (last_time_step_data['Wealth'].max() - last_time_step_data['Wealth'].min())
    last_time_step_data['Normalized_Recent_Wealth'] = (last_time_step_data['Recent Wealth'] - last_time_step_data['Recent Wealth'].min()) / (last_time_step_data['Recent Wealth'].max() - last_time_step_data['Recent Wealth'].min())

    # Compute statistics for normalized wealth
    wealth_statistics = {
        'Mean_Wealth': last_time_step_data['Normalized_Wealth'].mean(),
        'Median_Wealth': last_time_step_data['Normalized_Wealth'].median(),
        'Mode_Wealth': stats.mode(last_time_step_data['Normalized_Wealth'])[0][0],
        'Variance_Wealth': last_time_step_data['Normalized_Wealth'].var(),
        'Skewness_Wealth': last_time_step_data['Normalized_Wealth'].skew(),
        'Kurtosis_Wealth': last_time_step_data['Normalized_Wealth'].kurtosis()
    }

    # Compute statistics for normalized recent wealth
    recent_wealth_statistics = {
        'Mean_Recent_Wealth': last_time_step_data['Normalized_Recent_Wealth'].mean(),
        'Median_Recent_Wealth': last_time_step_data['Normalized_Recent_Wealth'].median(),
        'Mode_Recent_Wealth': stats.mode(last_time_step_data['Normalized_Recent_Wealth'])[0][0],
        'Variance_Recent_Wealth': last_time_step_data['Normalized_Recent_Wealth'].var(),
        'Skewness_Recent_Wealth': last_time_step_data['Normalized_Recent_Wealth'].skew(),
        'Kurtosis_Recent_Wealth': last_time_step_data['Normalized_Recent_Wealth'].kurtosis()
    }

    # Print statistics for normalized wealth
    print("Statistics for Normalized Wealth:")
    for key, value in wealth_statistics.items():
        print(f"{key}: {value:.5f}")

    # Print statistics for normalized recent wealth
    print("\nStatistics for Normalized Recent Wealth:")
    for key, value in recent_wealth_statistics.items():
        print(f"{key}: {value:.5f}")
