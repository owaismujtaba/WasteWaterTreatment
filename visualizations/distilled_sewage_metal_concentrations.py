import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.patches as patches

import config as config
from utils import get_data_for_boxplot, calculate_t_static_and_p_values
from utils import get_significance_mark, calculate_reduction_rate_distilled
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from scipy.stats import linregress


plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 6,
    'axes.facecolor': '#f9f9f9',
    'figure.facecolor': '#ffffff',
    'font.family': 'STIXGeneral',  # Use STIXGeneral as an alternative to Times New Roman
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, t
from sklearn.metrics import mean_squared_error

def plot_distilled_sewage_reduction_percentage():
    path = config.DISTILLED_METALS_FILEPATH
    table1_percentage_diff = calculate_reduction_rate_distilled(path)

    data = table1_percentage_diff
    subplot_titles = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    title_index = 0

    df = pd.DataFrame(data).set_index("HeavyMetal")
    x_labels = ["RawSewage", "1:1", "1:3", "1:7"]
    x_pos_full = np.arange(len(x_labels))
    x_pos = np.arange(len(x_labels) - 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()

    for i, metal in enumerate(df.index):
        raw_sewage = df.loc[metal, "RawSewage"]
        dilution_values = df.loc[metal, ["1:1_Dilution", "1:3_Dilution", "1:7_Dilution"]]
        control_values = df.loc[metal, ["1:1_Control", "1:3_Control", "1:7_Control"]]

        X = np.array([1, 3, 7])
        line = np.array([1, 7])

        # Plot the dilution values and regression
        axes[i].plot(x_pos + 1, dilution_values, marker='s', color='b', label="Dilution", linestyle='-', linewidth=1)
        axes[i].scatter(0, raw_sewage, color='r', marker='D')
        slope_dil, intercept_dil, r_value_dil, _, std_err_dil = linregress(X, dilution_values)
        dil_regression_line = slope_dil * line + intercept_dil
        axes[i].plot([1, 3], dil_regression_line, color='b', linestyle='--', linewidth=1)

        # Calculate and plot 95% confidence interval for dilution regression
        conf_interval_dil = t.ppf(0.975, len(X)-2) * std_err_dil * np.sqrt(1/len(X) + (line - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
        axes[i].fill_between([1, 3], dil_regression_line - conf_interval_dil, dil_regression_line + conf_interval_dil, color='b', alpha=0.2)

        axes[i].plot([1, 3], dil_regression_line + conf_interval_dil, color='b', linestyle='--', linewidth=0.5)
        axes[i].plot([1, 3], dil_regression_line - conf_interval_dil, color='b', linestyle='--', linewidth=0.5)


        # Calculate RMSE for dilution regression and add to plot
        rmse_dil = np.sqrt(mean_squared_error(dilution_values, slope_dil * X + intercept_dil))
        dil_eq = f"y = {slope_dil:.2f}x + {intercept_dil:.2f}\nRMSE={rmse_dil:.2f}"
        axes[i].text(0.05, 0.95, dil_eq, color='b', fontsize=config.TITLE_FONT_SIZE, rotation=0, transform=axes[i].transAxes, ha='left', va='top')

        # Plot the control values and regression
        axes[i].plot(x_pos + 1, control_values, marker='s', color='g', label="Control", linestyle='-', linewidth=1)
        slope_ctrl, intercept_ctrl, r_value_ctrl, _, std_err_ctrl = linregress(X, control_values)
        ctrl_regression_line = slope_ctrl * line + intercept_ctrl
        axes[i].plot([1, 3], ctrl_regression_line, color='g', linestyle='--', linewidth=1)

        # Calculate and plot 95% confidence interval for control regression
        conf_interval_ctrl = t.ppf(0.975, len(X)-2) * std_err_ctrl * np.sqrt(1/len(X) + (line - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
        axes[i].fill_between([1, 3], ctrl_regression_line - conf_interval_ctrl, ctrl_regression_line + conf_interval_ctrl, color='g', alpha=0.2)
        axes[i].plot([1, 3], ctrl_regression_line + conf_interval_ctrl, color='g', linestyle='--', linewidth=0.5)
        axes[i].plot([1, 3], ctrl_regression_line - conf_interval_ctrl, color='g', linestyle='--', linewidth=0.5)


        # Calculate RMSE for control regression and add to plot
        rmse_ctrl = np.sqrt(mean_squared_error(control_values, slope_ctrl * X + intercept_ctrl))
        ctrl_eq = f"y = {slope_ctrl:.2f}x + {intercept_ctrl:.2f}\nRMSE={rmse_ctrl:.2f}"
        axes[i].text(0.05, 0.65, ctrl_eq, color='g', fontsize=config.TITLE_FONT_SIZE, rotation=0, transform=axes[i].transAxes, ha='left', va='top')

        # Add subplot titles, labels, etc.
        axes[i].text(-0.03, 1.1, subplot_titles[title_index], transform=axes[i].transAxes, 
                     fontsize=config.TITLE_FONT_SIZE, ha='right', va='top', fontweight='bold')
        axes[i].set_xticks(x_pos_full)
        axes[i].set_xticklabels(x_labels)
        axes[i].set_title(f"{metal}")
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Concentration decrease %')

        title_index += 1
        if i == 0:
            axes[i].set_xlabel('Dilution and control ratio')
            axes[i].set_ylabel('Concentration decrease %')

    axes[0].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('Images/distilled_sewage_metals_percentage_decrease.png', dpi=config.DPI)


def plot_distilled_sewage_reduction_percentage1():
    path = config.DISTILLED_METALS_FILEPATH
    table1_percentage_diff = calculate_reduction_rate_distilled(path)

    data = table1_percentage_diff
    subplot_titles = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    title_index = 0

    df = pd.DataFrame(data).set_index("HeavyMetal")

    x_labels = ["RawSewage", "1:1", "1:3", "1:7"]
    x_pos_full = np.arange(len(x_labels))
    x_pos = np.arange(len(x_labels) - 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()

    for i, metal in enumerate(df.index):
        raw_sewage = df.loc[metal, "RawSewage"]
        dilution_values = df.loc[metal, ["1:1_Dilution", "1:3_Dilution", "1:7_Dilution"]]
        control_values = df.loc[metal, ["1:1_Control", "1:3_Control", "1:7_Control"]]
        
        axes[i].plot(x_pos + 1, dilution_values, marker='s', color='b', label="Dilution", linestyle='-', linewidth=1)
        axes[i].plot(x_pos + 1, control_values, marker='s', color='g', label="Control", linestyle='-', linewidth=1)
        axes[i].scatter(0, raw_sewage, color='r', marker='D')
        X = np.array([1, 3, 7])
        slope_dil, intercept_dil, _, _, _ = linregress(X, dilution_values)
        line = np.array([1, 7])
        dil_regression_line = slope_dil * line + intercept_dil
        axes[i].plot([1, 3], dil_regression_line, color='b', linestyle='--', linewidth=1)
        
        angle_dil = np.degrees(np.arctan(slope_dil))
        
        dil_eq = f"y = {slope_dil:.2f}x + {intercept_dil:.2f}"
        axes[i].text(0.05, 0.95, dil_eq, color='b', fontsize=config.TITLE_FONT_SIZE, rotation=0, transform=axes[i].transAxes, ha='left', va='top')
        
        slope_ctrl, intercept_ctrl, _, _, _ = linregress(X, control_values)
        ctrl_regression_line = slope_ctrl * line + intercept_ctrl
        axes[i].plot([1, 3], ctrl_regression_line, color='g', linestyle='--', linewidth=1)
        
        
        ctrl_eq = f"y = {slope_ctrl:.2f}x + {intercept_ctrl:.2f}"
        axes[i].text(0.05, 0.85, ctrl_eq, color='g', fontsize=config.TITLE_FONT_SIZE, rotation=0, transform=axes[i].transAxes, ha='left', va='top')
        axes[i].text(-0.03, 1.1, subplot_titles[title_index], transform=axes[i].transAxes, 
                                    fontsize=config.TITLE_FONT_SIZE, ha='right', va='top', fontweight='bold')
        axes[i].set_xticks(x_pos_full)
        axes[i].set_xticklabels(x_labels)
        axes[i].set_title(f"{metal}")
        
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Concentration decrease %')
        
        title_index +=1
        if i == 0:
            axes[i].set_xlabel('Dilution and control ratio')
            axes[i].set_ylabel('Concentration decrease %')

    axes[0].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('Images/distilled_sewage_metals_percentage_decrease1.png', dpi=config.DPI)
    

def plot_distilled_distributions_for_metals(nrows=2, ncols=3):
        table1 = pd.read_csv(config.DISTILLED_METALS_FILEPATH)
        table1.drop('Unnamed: 0', axis=1, inplace=True)
        


        color_palette = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', 
                        '#2ca02c', '#2ca02c', '#d62728', '#d62728', 
                        '#9467bd', '#9467bd', '#8c564b', '#8c564b', '#e377c2', '#e377c2']  
        ticks = ['C$_0$', 'C$_{10}$'] * 7
        condition_names = ['Raw Sewage', '1:1 Dilution', '1:1 Control', 
            '1:3 Dilution', '1:3 Control', '1:7 Dilution', '1:7 Control'
        ]
        subplot_titles = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

        title_index = 0
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
        
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
                    conditions=['RawSewage', '1:1_Dilution', '1:1_Control', '1:3_Dilution', '1:3_Control', '1:7_Dilution', '1:7_Control']

                )

                xticks = axes[row, col].get_xticks()
                axes[row, col].set_xticklabels(ticks)
                axes[row, col].spines['top'].set_visible(False)
                axes[row, col].spines['right'].set_visible(False)
                axes[row, col].set_xlabel('')
                axes[row, col].set_ylabel('')
                axes[row, col].text(-0.03, 1.1, subplot_titles[title_index], transform=axes[row, col].transAxes, 
                                    fontsize=config.TITLE_FONT_SIZE, ha='right', va='top', fontweight='bold')
                axes[row, col].set_title(element)
                title_index += 1
                if row == 0 and col == 0:
                    legend_colors = [color_palette[index] for index in range(0, len(color_palette), 2)]
                    handles = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, condition_names)]
                    axes[row, col].legend(handles=handles, loc='upper right',fontsize=8, bbox_to_anchor=(1, 0.9))
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
        plt.savefig('Images/distilled_sewage_metal_distributions.png', dpi=config.DPI)
        plt.clf()