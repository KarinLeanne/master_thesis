from scipy.stats import f_oneway, kruskal
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pingouin as pg


def adf_test(series):
    '''
    Description: 
    Perform Augmented Dickey-Fuller test on a given time series.
    Inputs:
    - series (array-like): The time series data.
    Outputs:
    - result (tuple): A tuple containing ADF statistic, p-value, and other test statistics.
    '''
    result = adfuller(series)
    return result

def adf_test_for_combinations(dataframe):
    '''
    Description: 
    Perform Augmented Dickey-Fuller test for each combination of network and measure.
    Inputs:
    - dataframe (DataFrame): The DataFrame containing the time series data.
    Outputs:
    - Prints ADF test results for each combination of network and measure.
    '''

    results = []
    for network in dataframe['Network'].unique():
        for measure in ['Var of Degree', 'Avg Clustering', 'Avg Path Length']:
            series = dataframe.loc[dataframe['Network'] == network, f'M: {measure}']
            if series.var() != 0:  # Skip constant series
                adf_result = adf_test(series)
                results.append({
                    'Network': network,
                    'Measure': measure,
                    'ADF Statistic': adf_result[0],
                    'p-value': adf_result[1],
                    'Critical Values': adf_result[4]
                })

    results_df = pd.DataFrame(results)  # Convert list of dicts to DataFrame
    for _, row in results_df.iterrows():
        print(f"Network: {row['Network']}, Measure: {row['Measure']}")
        print(f"ADF Statistic: {row['ADF Statistic']}")
        print(f"p-value: {row['p-value']}")
        
        # Interpretation of p-value
        if row['p-value'] < 0.05:
            print("p-value is less than 0.05: Reject the null hypothesis (series is stationary)")
        else:
            print("p-value is greater than or equal to 0.05: Fail to reject the null hypothesis (series is non-stationary)")
        
        print("Critical Values:")
        for key, value in row['Critical Values'].items():
            print(f"\t{key}: {value}")
        print("\n")




def sliding_window_rmsd(data, window_size=24, thresholds={'M: Var of Degree': 0.2, 'M: Avg Clustering': 0.01, 'M: Avg Path Length': 0.05}):
    '''
    Description: 
    Calculate RMSD in a sliding window for each measure and check for convergence.
    Inputs:
    - data (DataFrame): DataFrame containing the time series data.
    - window_size (int): Size of the sliding window.
    - thresholds (dict): Dictionary of threshold values for each measure.
    Outputs:
    - Prints convergence information for each measure.
    '''
    # Convert 'Step' column to numeric
    data['Step'] = pd.to_numeric(data['Step'])
    
    # Initialize variables
    networks = data['Network'].unique()
    
    for network in networks:
        print(f"Calculating for Network: {network}")
        network_data = data[data['Network'] == network]
        
        time_series = network_data.iloc[:, 2:]  # Exclude 'Step' and 'Network' columns
        rmsd_values = {measure: [] for measure in thresholds.keys()}
        convergence_info = {measure: None for measure in thresholds.keys()}
        
        # Calculate RMSD in a sliding window for each measure
        for measure in thresholds.keys():
            threshold = thresholds[measure]
            for i in range(len(time_series) - window_size + 1):
                window = time_series.iloc[i:i+window_size, :]
                window_mean = window.mean()
                window_rmsd = np.sqrt(((window[measure] - window_mean[measure])**2).sum() / (window_size - 1))
                rmsd_values[measure].append(window_rmsd)
                if window_rmsd < threshold and convergence_info[measure] is None:
                    convergence_info[measure] = (i + window_size - 1, window_mean[measure])
        
        # Check for convergence for each measure
        for measure in thresholds.keys():
            if convergence_info[measure] is not None:
                index, value = convergence_info[measure]
                print(f"Convergence reached for {measure} at timestep {index} with value = {value} (below threshold {thresholds[measure]}).")
            else:
                print(f"No convergence reached for {measure}. Adjust threshold or window size.")



def compare_distributions(data, column_name):
    '''
    Description: 
    Compare distributions of a column across low, medium, and high game players.
    Inputs:
    - data (DataFrame): DataFrame containing the dataset.
    - column_name (str): Name of the column to compare distributions.
    Outputs:
    - Prints Kruskal-Wallis test results.
    '''
    # Divide players into low, medium, and high game players
    data = data[data['Step'] == data['Step'].max()]
    low_game_players = data[data['Games played'] <= data['Games played'].quantile(0.33)][column_name]
    medium_game_players = data[(data['Games played'] > data['Games played'].quantile(0.33)) & 
                               (data['Games played'] <= data['Games played'].quantile(0.66))][column_name]
    high_game_players = data[data['Games played'] > data['Games played'].quantile(0.66)][column_name]
    
    # Perform Kruskal-Wallis test
    statistic, p_value = kruskal(low_game_players, medium_game_players, high_game_players)
    
    # Calculate sample sizes
    n_low = len(low_game_players)
    n_medium = len(medium_game_players)
    n_high = len(high_game_players)
    total_n = n_low + n_medium + n_high
    
    # Calculate degrees of freedom
    dof = len([low_game_players, medium_game_players, high_game_players]) - 1
    
    # Calculate effect size (eta-squared)
    eta_squared = statistic / (total_n - 1)
    
    # Print results
    print("Kruskal-Wallis Test Results for", column_name)
    print("Statistic:", statistic)
    print("P-value:", p_value)
    print("Degrees of Freedom:", dof)
    print("Effect size (eta-squared):", eta_squared)
    
    return statistic, p_value, dof, eta_squared


def kruskal_wallis_test(data, independent_var, measure):
    '''
    Description: 
    Calculate Kruskal-Wallis test statistics for a given column.
    Inputs:
    - data (DataFrame): DataFrame containing the dataset.
    - independent_var (str): Name of the independent variable column.
    - measure (str): Name of the column for which to calculate the test.
    Outputs:
    - Prints Kruskal-Wallis test results.
    '''
    subset = data[data['IndependentVariable'] == independent_var]

    measure_values = []
    
    # Group the data by the independent variable column
    grouped_data = subset.groupby("IndependentValue")
    
    for _, group_data in grouped_data:
        # Get the measure values for the current group
        values = group_data[measure].tolist()
        measure_values.append(values)


    total_n = sum(len(sublist) for sublist in measure_values)
    
    # Perform Kruskal-Wallis test
    statistic, p_value = kruskal(*measure_values)
    
    
    # Calculate degrees of freedom
    dof = len(measure_values) - 1
    
    # Calculate effect size (eta-squared)
    eta_squared = statistic / (total_n - 1)
    
    # Print results
    print("Kruskal-Wallis Test Results for", measure, independent_var)
    print("Statistic:", statistic)
    print("P-value:", p_value)
    print("Degrees of Freedom:", dof)
    print("Effect size (eta-squared):", eta_squared)
    
    return statistic, p_value, dof, eta_squared
