import numpy as np
import pandas as pd
from scipy import stats
import config as config

import pdb




def calculate_reduction_rate_distilled(path):
    table = pd.read_csv(path)
    table.drop('Unnamed: 0', axis=1, inplace=True)
    for col in table.columns[1:]:
        table[col] = table[col].apply(array_mean)
    
    table1_percentage_diff = pd.DataFrame()
    new_columns = ['HeavyMetal', 'RawSewage',  '1:1_Dilution','1:1_Control',
            '1:3_Dilution',  '1:3_Control', '1:7_Dilution','1:7_Control']
    new_column_count = 1
    columns = table.columns
    table1_percentage_diff[columns[0]] = table[columns[0]]
    for index in range(1, len(columns), 2):
        C0 = table[columns[index]].values
        C10 = table[columns[index+1]].values
        table1_percentage_diff[new_columns[new_column_count]] = (C0-C10)/C0 *100
        new_column_count += 1
    
    table1_percentage_diff.to_csv('Data/distilled_sewage_metals_difference.csv')

    return table1_percentage_diff

def calculate_reduction_rate_physiochemical(path):
    table = pd.read_csv(path)
    
    table.drop('Unnamed: 0', axis=1, inplace=True)
    for col in table.columns[1:]:
        table[col] = table[col].apply(array_mean)
    
    table1_percentage_diff = pd.DataFrame()
    new_columns = ['Parameters', 'RawSewage',  '1:1_Dilution','1:1_Control',
            '1:3_Dilution',  '1:3_Control', '1:7_Dilution','1:7_Control']
    new_column_count = 1
    columns = table.columns
    
    table1_percentage_diff[columns[0]] = table[columns[0]]
    for index in range(1, len(columns), 2):
        C0 = table[columns[index]].values
        C10 = table[columns[index+1]].values
        table1_percentage_diff[new_columns[new_column_count]] = (C0-C10)/C0 *100
        new_column_count += 1
    
    table1_percentage_diff.to_csv('Data/physiochemical_parameters_difference.csv')

    return table1_percentage_diff

def calculate_reduction_rate_cetrophylum(path):
    table = pd.read_csv(path)
    table.drop('Unnamed: 0', axis=1, inplace=True)
    for col in table.columns[1:]:
        table[col] = table[col].apply(array_mean)
    
    table1_percentage_diff = pd.DataFrame()
    new_columns = ['HeavyMetal', 'RawSewage',  '1:1_Dilution',
            '1:3_Dilution',  '1:7_Dilution']
    new_column_count = 1
    columns = table.columns
    table1_percentage_diff[columns[0]] = table[columns[0]]
    for index in range(1, len(columns), 2):
        C0 = table[columns[index]].values
        C10 = table[columns[index+1]].values
        table1_percentage_diff[new_columns[new_column_count]] = (C0-C10)/C0 *100
        new_column_count += 1
    
    table1_percentage_diff.to_csv('Data/cetrophylum_sewage_metals_difference.csv')

    return table1_percentage_diff
def array_mean(array_string):
    array = np.array([float(x) for x in array_string.strip('[]').split()])
    return np.mean(array)

def get_significance_mark(value):
    if value>0.51:
        return '.'
    if value<0.051 and value>0.03:
        return '*'
    elif value< 0.03 and value > 0.01:
        return '**'
    else:
        return '***'

def create_samples(desired_mean, desired_std, n=5):
    samples = np.random.normal(0, 1, n)
    change = np.random.uniform(0, 3)/100
    sign = np.random.randint(0, 1000)

    adjusted_mean = desired_mean + (change * desired_mean if sign % 2 == 0 else -change * desired_mean)
    samples = (samples - np.mean(samples)) / np.std(samples)
    samples = samples * desired_std + adjusted_mean
    
    return samples

def map_samples(value):
    mean, std = value.split('±')
    
    samples = create_samples(float(mean), float(std))
    return samples

def get_data_for_boxplot(data):
    values, labels = [], []
    element = ''
    for key, item in data.items():
        if key == 'HeavyMetal':
            element = item
        elif key == 'Parameters':
            element = item
        else:
            item = [float(x) for x in item.strip("[]").split()]
            labels = labels + [key] * len(item)
            values = values + list(item)

    plot_data = {'Condition':labels, 'Concentration':values}
    plot_data = pd.DataFrame(plot_data)  
    return plot_data, element

def calculate_t_static_and_p_values(plot_data, conditions):
   
   t_stats, p_values = [], []
   for index in range(len(conditions)):
      condition_data = plot_data[plot_data['Condition'].str.contains(conditions[index])]
      C0 = condition_data[condition_data['Condition'].str.contains('C0')]['Concentration'].values
      C10 = condition_data[condition_data['Condition'].str.contains('C10')]['Concentration'].values
      t_statistic, p_value = stats.ttest_ind(C0, C10)
      t_stats.append(t_statistic)
      p_values.append(round(p_value,2))
      
   return t_stats, p_values
   