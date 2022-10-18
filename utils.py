import numpy as np
import pandas as pd

def dataClean(excel_path, save_path ='./data2.csv',header=2, labels=['采集时间', '水温', 'pH', '溶解氧']):
    # Read the original data and copy one.
    # 'Date' is default as the first colume.
    df = pd.read_excel(excel_path, header=header, usecols=labels).copy()
    df.set_index(labels[0], inplace=True)
    
    # Clean the wrong data type and labeled them as 'None'
    for k in labels[1:]:
        df[k].mask(df[k] == '--', None, inplace=True)
    df = df.astype('float64')
    df = df.abs()
    
    # Clean the Outliers
    for k in labels[1:]:
        vals = df[k].values.copy()
        idxs = zscore(df, k)
#         idxs = detect_outlier(df, k, 4)
        vals[idxs] = None
        df.loc[:, k] = vals
    
    # Drop rows with nan
    df.dropna(inplace=True)
    
    #  Denoise by moving average
    for k in labels[1:]:
        vals = df[k].values.copy()
        smooth_vals = pd.Series(vals).rolling(window=18).mean()
        df.loc[:, k] = smooth_vals
        
    # Sort by the date 
    ls = []
    df = df.reset_index(drop=False)
    times = pd.to_datetime(arg=df[labels[0]], format='%Y-%m-%d %H:%M:%S')
    for group in df.groupby([times.dt.year, times.dt.month, times.dt.day]):
        if group[1].shape[0] == 6:
            dfn = group[1][::-1].values.copy()
            ls.append(dfn)
    arr = np.concatenate(ls, axis=0)
    new_data = pd.DataFrame(arr, columns=labels)
    new_data.to_csv(save_path, index=False)
    return new_data

    
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

def detect_outlier(df, label, rate=25):
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