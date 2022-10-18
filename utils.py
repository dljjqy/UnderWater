import numpy as np
import pandas as pd
    
def zscore(df, k, threshold=1.5):
    all_value = df[k].values.copy()
    indices = np.array(list(map(lambda x: not x, np.isnan(all_value))))
    true_value = all_value[indices]
    m = np.mean(true_value)
    s = np.std(true_value)
    
    all_value[indices] = np.abs((all_value[indices] - m) / s)
    all_value = pd.Series(all_value)
    return all_value > threshold

def modify_zscore(df, k, threshold=2):
    all_value = df[k].values.copy()
    indices = np.array(list(map(lambda x: not x, np.isnan(all_value))))
    true_value = all_value[indices]
    m = np.mean(true_value)
    diff = all_value[indices] - m
    median = np.median(np.abs(diff))
    
    all_value[indices] = np.abs(0.6745 * (diff) / median)
    all_value = pd.Series(all_value)
    return all_value > threshold

def standard_deviation(df, label, rate=25):
    all_values = df[label].values.copy()
    indices = np.array(list(map(lambda x: not x, np.isnan(all_values))))
    true_values = all_values[indices]
    
    Q1 = np.percentile(true_values, rate)
    Q3 = np.percentile(true_values, 100-rate)    
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    all_values = pd.Series(all_values)
    return (all_values < lower_limit) | (all_values > upper_limit)

def remove_outliers(df, method):
    for k in df.keys():
        vals = df[k].values.copy()
        outlier_idx = method(df, k)
        vals[outlier_idx] = None
        df.loc[:, k] = vals
    return df