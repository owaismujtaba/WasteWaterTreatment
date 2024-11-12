import pandas as pd
import config as config
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corelation_parameters_and_distilled_metals():

    metals = metals = pd.read_csv(config.METALS_DIFF_DISTILLED_FILEPATH)
    metals = metals.drop('Unnamed: 0', axis=1)
    physical = pd.read_csv(config.PHYSIO_DIFFRENCE_FILEPATH)
    physical = physical.drop('Unnamed: 0', axis=1)

    physical_df = physical.set_index('Parameters')
    metals_df = metals.set_index('HeavyMetal')

    correlation_results = pd.DataFrame(index=physical_df.index, columns=metals_df.index)

    for param in physical_df.index:
        for metal in metals_df.index:
            correlation_results.loc[param, metal] = physical_df.loc[param].corr(metals_df.loc[metal])

    print(correlation_results)

    

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_results.astype(float), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    #plt.title('Correlation Between Physical Parameters and Heavy Metals')
    plt.tight_layout()
    plt.savefig('Images/correlation_dist_metals.png', dpi=config.DPI)

def plot_corelation_parameters_and_cetrophylum_metals():

    physical = pd.read_csv(config.PHYSIOCHEMICL_PARAMETERS_PATH)
    metals = pd.read_csv(config.METALS_DIFF_CETROPHYLUM_FILEPATH)

    physical_df = physical.set_index('Parameters')
    metals_df = metals.set_index('HeavyMetal')

    correlation_results = pd.DataFrame(index=physical_df.index, columns=metals_df.index)

    for param in physical_df.index:
        for metal in metals_df.index:
            correlation_results.loc[param, metal] = physical_df.loc[param].corr(metals_df.loc[metal])

    print(correlation_results)
    

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_results.astype(float), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    #plt.title('Correlation Between Physical Parameters and Heavy Metals in')
    plt.tight_layout()
    plt.savefig('Images/correlation_dist_metals.png', dpi=config.DPI)
