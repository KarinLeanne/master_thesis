# External
import matplotlib.pyplot as plt

# Internal
import utils

params = utils.get_config()


def viz_speed_network_vs_speed_games_old(df, measure):
    plt.figure(figsize=(10, 6))

    # Create a scatter plot with colorbar
    scatter = plt.scatter(x=df['Speed_Ratio'], y=df[measure], c=df['rewiring_p'], cmap='viridis', s=100)
    
    plt.title(f'Scatter Plot of {measure}'.replace("_", " ") + " vs Speed Ratio with Colorbar'")
    plt.xlabel('Speed Ratio (e_g / e_n)')
    plt.ylabel(f"{measure}".replace("_", " "))
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Rewiring Probability')

    # Save Plots
    path = utils.make_path("Figures", "UpdatingMechanism", f"Plot of {measure} vs Speed Ratio")
    plt.savefig(path)
    plt.close()


def viz_time_series_y_varying_rewiring_p(df, values, y):

    rewiring_probabilities = values[::2]

    for rewiring_p in rewiring_probabilities:

        subset = df[df['rewiring_p'] == rewiring_p]
        df_grouped = subset.groupby(['Step', 'Round'])[y].mean().reset_index()

        # Calculate mean and SEM over simulations
        mean_df = df_grouped.groupby('Step')[y].mean().reset_index()
        sem_df = df_grouped.groupby('Step')[y].sem().reset_index()

        # Calculate upper and lower bounds of CI
        upper_bound = mean_df[y] + 1.96 * sem_df[y]
        lower_bound = mean_df[y] - 1.96 * sem_df[y]

        # Plot the mean speed ratio and fill between the bounds
        plt.plot(mean_df['Step'], mean_df[y], label=f'rewiring_p={rewiring_p}')
        plt.fill_between(mean_df['Step'], lower_bound, upper_bound, alpha=0.3)

    # Customize the plot
    plt.xlabel('Step')
    plt.ylabel(y)
    plt.title(f'{y} over Time for Different Rewiring Probabilities')
    plt.legend()
    plt.grid(True)

    # Save Plots
    path = utils.make_path("Figures", "UpdatingMechanism", f"{y} Time Series")
    plt.savefig(path)
    plt.close()