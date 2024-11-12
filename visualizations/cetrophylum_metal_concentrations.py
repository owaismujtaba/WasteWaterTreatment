import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.patches as patches

import config as config
from utils import get_data_for_boxplot, calculate_t_static_and_p_values
from utils import get_significance_mark, calculate_reduction_rate_cetrophylum


from scipy.stats import linregress
import pdb

plt.rcParams.update({
        'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 14, 'axes.facecolor': '#f9f9f9',
        'figure.facecolor': '#ffffff'
        })


def plot_cetrophylum_reduction_percentage():
    path = config.CITROPHYLUM_METALS
    table1_percentage_diff = calculate_reduction_rate_cetrophylum(path)
    
    df = pd.DataFrame(table1_percentage_diff).set_index("HeavyMetal")

    x_labels = ["RawSewage", "1:1", "1:3", "1:7"]
    x_pos_full = np.arange(len(x_labels))
    width = 0.15  # Width of each bar

    color_map = {
        'Mn': 'blue',
        'Cu': 'green',
        'Zn': 'purple',
        'Co': 'orange',
        'Cd': 'brown',
        'Zn': 'cyan'
    }

    plt.figure(figsize=(10, 6))

    handles = []
    labels = []
    
    for i, metal in enumerate(df.index):
        raw_sewage = df.loc[metal, "RawSewage"]
        dilution_values = df.loc[metal, ["1:1_Dilution", "1:3_Dilution", "1:7_Dilution"]]
        
        color = color_map.get(metal, 'black')
        bar_positions = x_pos_full - 0.3 + i * width  # Adjust positions for each metal

        plt.bar(bar_positions[0], raw_sewage, color=color, width=width, label=metal if i == 0 else "")
        plt.bar(bar_positions[1:], dilution_values, color=color, width=width)

        slope_dil, intercept_dil, _, _, _ = linregress([1, 3, 7], dilution_values)
        
        dil_regression_line = slope_dil * np.array([1, 7]) + intercept_dil
        plt.plot([1, 3], dil_regression_line, color=color, linestyle='--', linewidth=2)

        dil_eq = f"{metal}: y = {slope_dil:.2f}x + {intercept_dil:.2f}"
        plt.text(0.5, 0.95 - i * 0.05, dil_eq, color=color, fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

        handles.append(plt.Line2D([0], [0], color=color, lw=6, marker='s', markersize=10))
        labels.append(f"{metal}")

    plt.xticks(x_pos_full, x_labels)
    plt.xlabel('Dilution ratios')
    plt.ylabel('Concentration decrease %')

    plt.legend(handles=handles, labels=labels, loc='upper right', frameon=False, bbox_to_anchor=(1, 1))

    plt.gca().invert_yaxis()  # Invert y-axis for negative values
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Images/cetrophylum_metals_percentage_difference.png', dpi=config.DPI)
    









def plot_cetrophylum_distributions_for_metals(nrows=2, ncols=3):
        table1 = pd.read_csv(config.CITROPHYLUM_METALS)
        table1.drop('Unnamed: 0', axis=1, inplace=True)
    
        color_palette = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', 
                        '#d62728', '#d62728', '#8c564b', '#8c564b']  
        ticks = ['C$_0$', 'C$_{10}$'] * 4
        condition_names = ['Raw Sewage', '1:1 Dilution', 
            '1:3 Dilution',  '1:7 Dilution'
        ]
        subplot_titles = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

        title_index = 0
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
        
        index = 0
        for row in range(nrows):
            for col in range(ncols):
                data = table1.iloc[index]
                plot_data, element = get_data_for_boxplot(data)
                
                sns.boxplot(data=plot_data, x='Condition', y='Concentration', ax=axes[row, col], palette=color_palette, showfliers=False)
                sns.stripplot(data=plot_data, x='Condition', y='Concentration', ax=axes[row, col], 
                            color='black', jitter=True, dodge=True, size=5, alpha=0.6)
                
                means = plot_data.groupby('Condition')['Concentration'].mean()  
                
                for i, (key, mean_value) in enumerate(means.items()):
                    axes[row, col].scatter(
                        key, mean_value, color='#FF00FF', label='Mean' if i == 0 else "", 
                        marker='s', s=20, zorder=3, edgecolor='black', linewidth=1
                    )

                t_statistic, p_value = calculate_t_static_and_p_values(
                    plot_data=plot_data,
                    conditions=['RawSewage', '1:1_Dilution', '1:3_Dilution', '1:7_Dilution']

                )

                xticks = axes[row, col].get_xticks()
                axes[row, col].set_xticklabels(ticks)
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)
                axes[row, col].set_xlabel('')
                axes[row, col].set_ylabel('')
                axes[row, col].text(-0.05, 1.1, subplot_titles[title_index], transform=axes[row, col].transAxes, 
                                    fontsize=14, ha='right', va='top', fontweight='bold')
                axes[row, col].set_title(element)
                title_index += 1
                if row == 0 and col == 0:
                    legend_colors = [color_palette[index] for index in range(0, len(color_palette), 2)]
                    handles = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, condition_names)]
                    axes[row, col].legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 0.9))
                    axes[row, col].set_ylabel('Concentration (ppm)')
                    axes[row, col].set_xlabel('Before(C$_0$) and after(C$_{10}$) incubation period')
                p_value_count = 0
                
                for i in range(0, len(xticks) - 1, 2):
                    midpoint = (xticks[i] + xticks[i+1]) / 2
                    y_position = axes[row, col].get_ylim()[1] + (axes[row, col].get_ylim()[1] * -0.09)
                    axes[row, col].text(midpoint, y_position, get_significance_mark(p_value[p_value_count]), ha='center', va='bottom', 
                                        transform=axes[row, col].transData)
                    
                    xstart = xticks[i]
                    xend = xticks[i+1]
                    
                    line = patches.FancyArrowPatch(
                        (xstart, y_position), (xend, y_position), 
                        color='black', lw=1, arrowstyle='-'
                    )
                    axes[row, col].add_patch(line)
                    
                    vertical_line_start = patches.FancyArrowPatch(
                        (xstart, y_position - y_position * 0.04), (xstart, y_position),
                        color='black', lw=1, arrowstyle='-'
                    )
                    vertical_line_end = patches.FancyArrowPatch(
                        (xend, y_position - y_position * 0.04), (xend, y_position),
                        color='black', lw=1, arrowstyle='-'
                    )

                    axes[row, col].add_patch(vertical_line_start)
                    axes[row, col].add_patch(vertical_line_end)

                    p_value_count += 1

                index += 1

        plt.tight_layout()  
        plt.savefig('Images/cetrophylum_metals_distributions.png', dpi=config.DPI)
        plt.clf()