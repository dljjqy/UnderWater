import argparse
import pandas as pd
import numpy as np
keys = ['采集时间', '水温', 'pH', '溶解氧', '电导率', '浊度', '高锰酸盐指数','氨氮', '总磷', '总氮']

parser = argparse.ArgumentParser(description='Process the origional data with noise.')
parser.add_argument('-data_path', type=str, default=None, required=True, help='Path of excel file.')
parser.add_argument('-save_path', type=str, default=None, required=True, help='Path for saving processed data.')
parser.add_argument('--start_date', type=str, default=None, required=False, help='Load excel from the start day')
parser.add_argument('--end_date', type=str, default=None, required=False, help='Load excel before the end day')
parser.add_argument('--keys', default=keys, nargs='+', required=False, help='Which keys you want load in excel file.')
args = parser.parse_args()

def standard_deviation(df, label, rate=5):
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

def remove_outliers(df, method, rate):
    for k in df.keys():
        vals = df[k].values.copy()
        outlier_idx = method(df, k, rate)
        vals[outlier_idx] = None
        df.loc[:, k] = vals
    return df

def patch_up(df, size=7):
    '''
    patch up the missing data by the mean value of previous and next seven days' data.
    '''
    step = 2 * size + 1
    group = df.groupby([df.index.hour])
    dfns = [pd.DataFrame(group.get_group(k)) for k in group.groups.keys()]
    dfns = [dfn.rolling(step, center=True, min_periods=1).mean() for dfn in dfns]
    # Fix the rows with no true value at all. 
    all_nan_idx = df.index[df.isna().all(axis=1)]
    for idx in all_nan_idx:
        k = idx.hour // 4
        df.loc[idx] =  dfns[k].loc[str(idx.date())].values
    
    # Fix the rows with true values and nan.
    for k in df.keys():
        nan_idx = df[k].index[df[k].isna()]
        for idx in nan_idx:
            hour = idx.hour // 4
            df[k].loc[idx] = dfns[hour][k].loc[str(idx.date())]
    return df

def smooth(df, size=60):
    for k in df.keys():
        vals = df[k].values.copy()
        smooth_vals = pd.Series(vals).ewm(size).mean().values    
        df.loc[:, k] = smooth_vals
    return df

def read_excel(excel_path, save_path, start_date, end_date, usecols=keys, header=2, index_col=0):
    '''
    This function is used for reading excel file and fix all the mistakes.
    '''
    df = pd.read_excel(excel_path, header=header, usecols=usecols, index_col=index_col)
    df = df.loc[::-1]
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df.replace(to_replace= '--', value = np.nan, inplace=True)
    df = df.astype('float64')
    if start_date is not None:
        df = df.drop(df[df.index < start_date].index)
    if end_date is not None:
        df = df.drop(df[df.index >= end_date].index)
    df = remove_outliers(df, standard_deviation, 25)
    df = df.resample('4H', 'index').mean()
    df = patch_up(df, 7)
    df = smooth(df, 3)
    mins = df.min().values
    maxs = df.max().values
    df = (df - mins)/(maxs - mins)
    df.loc['max'] = maxs
    df.loc['min'] = mins
    df.to_csv(save_path)
    return True

if __name__ == '__main__':
    read_excel(excel_path = args.data_path, 
            save_path = args.save_path,
            start_date = args.start_date,
            end_date = args.end_date,
            usecols = args.keys,
    )