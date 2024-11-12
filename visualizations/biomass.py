import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.stats import linregress


import config as config
from utils import  calculate_t_static_and_p_values
from utils import get_significance_mark, array_mean



plt.rcParams.update({
        'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'axes.facecolor': '#f9f9f9',
        'figure.facecolor': '#ffffff'
        })


def calculate_biomass_reduction_rate():
    table = pd.read_csv(config.BIOMASS_FILEPATH)
    table = table.drop('Unnamed: 0', axis=1)

    for col in table.columns:
        table[col] = table[col].apply(array_mean)

    new_columns = ['RawSewage',  '1:1_Dilution',  '1:3_Dilution', '1:7_Dilution']
    new_column_count = 0
    columns = table.columns
    table_percentage_diff = pd.DataFrame()
    for index in range(0, len(columns), 2):
        W0 = table[columns[index]].values
        W10 = table[columns[index+1]].values
        table_percentage_diff[new_columns[new_column_count]] = (W10-W0)/W10 *100
        new_column_count += 1
    table_percentage_diff.to_csv('Data/biomass_difference_percentage.csv')
    return table_percentage_diff

def plot_biomass_distribution_and_percentage_difference(nrows=1, ncols=2):
    table = pd.read_csv(config.BIOMASS_FILEPATH)
    table = table.drop('Unnamed: 0', axis=1)
    ################### ist plot##############################
    

    color_palette = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', 
                        '#d62728', '#d62728', '#8c564b', '#8c564b']  
    ticks = ['W$_0$', 'W$_{10}$'] * 4
    condition_names = ['Raw Sewage', '1:1 Dilution', 
            '1:3 Dilution',  '1:7 Dilution'
        ]
    subplot_titles = ['a)', 'b)']

    title_index = 0
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
        
    index = 0
    values, labels = [], []
    
    data = table.iloc[index]
    col = 0
    element = 'Biomass'       
    for key, item in data.items():
        item = [float(x) for x in item.strip("[]").split()]
        labels = labels + [key] * len(item)
        values = values + list(item)

    
    plot_data = {'Condition':labels, 'Concentration':values}
    plot_data = pd.DataFrame(plot_data)  

    sns.boxplot(data=plot_data, x='Condition', y='Concentration', ax=axes[ col], palette=color_palette, showfliers=False)
    sns.stripplot(data=plot_data, x='Condition', y='Concentration', ax=axes[ col], 
                                color='black', jitter=True, dodge=True, size=5, alpha=0.6)
                    
    means = plot_data.groupby('Condition')['Concentration'].mean()  
     
    for i, (key, mean_value) in enumerate(means.items()):
        axes[ col].scatter(
            key, mean_value, color='#FF00FF', label='Mean' if i == 0 else "", 
            marker='s', s=20, zorder=3, edgecolor='black', linewidth=1
        )

    t_statistic, p_value = calculate_t_static_and_p_values(
    plot_data=plot_data,
    conditions=['RawSewage', '1:1_Dilution', '1:3_Dilution', '1:7_Dilution']
    )

    xticks = axes[col].get_xticks()
    axes[col].set_xticklabels(ticks)
    axes[col].spines['top'].set_visible(False)
    axes[col].spines['right'].set_visible(False)
    axes[col].set_xlabel('')
    axes[col].set_ylabel('')
    axes[col].text(-0.05, 1.1, subplot_titles[title_index], transform=axes[col].transAxes, 
                                        fontsize=14, ha='right', va='top', fontweight='bold')
    axes[col].set_title(element)
    title_index += 1
    if col == 0:
        legend_colors = [color_palette[index] for index in range(0, len(color_palette), 2)]
        handles = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, condition_names)]
        axes[col].legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 0.9))
        axes[col].set_ylabel('Concentration')
        axes[col].set_xlabel('Before(C$_0$) and after(C$_{10}$) incubation period')
    p_value_count = 0
                    
    for i in range(0, len(xticks) - 1, 2):
        midpoint = (xticks[i] + xticks[i+1]) / 2
        y_position = axes[col].get_ylim()[1] + (axes[col].get_ylim()[1] * -0.009)
        axes[col].text(midpoint, y_position, get_significance_mark(p_value[p_value_count]), ha='center', va='bottom', 
                                                transform=axes[col].transData)
                            
        xstart = xticks[i]- 0.2
        xend = xticks[i+1] + 0.2
                            
        line = patches.FancyArrowPatch(
                                (xstart, y_position), (xend, y_position), 
                                color='black', lw=1, arrowstyle='-'
                            )
        axes[col].add_patch(line)
                            
        vertical_line_start = patches.FancyArrowPatch(
                                (xstart, y_position - y_position * 0.01), (xstart, y_position),
                                color='black', lw=1, arrowstyle='-'
                            )
        vertical_line_end = patches.FancyArrowPatch(
                                (xend, y_position - y_position * 0.01), (xend, y_position),
                                color='black', lw=1, arrowstyle='-'
                            )

        axes[col].add_patch(vertical_line_start)
        axes[col].add_patch(vertical_line_end)
        
    p_value_count += 1

    index += 1


    #############second plot #################################
    difference_percentage = calculate_biomass_reduction_rate()

    #color_palette = ['#ff7f0e',  '#d62728', '#8c564b']  
    color_palette = ['#1f77b4', '#ff7f0e', 
                        '#d62728',  '#8c564b']          

    col = 1
    categories = ["RawSewage", "1:1 Dilution", "1:3 Dilution", "1:7 Dilution"]
    values = difference_percentage.T.values
    dilution_values = values[1:]  # 1:1, 1:3, 1:7 Dilution values
    dilution_indices = np.array([1, 3, 7]) # Indices for dilutions only

    slope, intercept, r_value, p_value, std_err = linregress(dilution_indices, dilution_values.reshape(3))
    regression_line = slope * np.array([1, 7]) + intercept

    axes[col].bar(categories, values.reshape(4), color=color_palette)

    # Plot regression line (for dilutions only)
    axes[col].plot([1, 3], regression_line, color='black', linestyle="--", linewidth=2)

    equation_text = f"y = {slope:.2f}x + {intercept:.2f}"

    axes[col].text(0, max(values) * 0.9, equation_text, color='black', fontsize=14)
    axes[col].spines['top'].set_visible(False)
    axes[col].spines['right'].set_visible(False)
    axes[col].set_ylabel('Concentration increase (%)')
    axes[col].set_xlabel('Dilution ratios')
    axes[col].text(-0.05, 1.1, subplot_titles[title_index], transform=axes[col].transAxes, 
                                        fontsize=14, ha='right', va='top', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Images/biomass_distribution_and_change.png', dpi=config.DPI) 
    
