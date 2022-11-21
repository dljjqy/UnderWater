import numpy as np
import pandas as pd

keys = ['采集时间', '水温', 'pH', '溶解氧', '电导率', '浊度', '高锰酸盐指数','氨氮', '总磷', '总氮']

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

def find_miss(df):
    '''
    Find which day's date is lost.
    '''
    dateIndex = df.index.normalize().drop_duplicates()
    myIndex = pd.date_range(dateIndex.min(), dateIndex.max(), freq='D')
    return myIndex.difference(dateIndex) 

def smooth(df, size=60):
    for k in df.keys():
        vals = df[k].values.copy()
        smooth_vals = pd.Series(vals).ewm(size).mean().values    
        df.loc[:, k] = smooth_vals
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

def my_read_excel(excel_path, save_path, start_date, usecols=keys, header=2, index_col=0):
    '''
    This function is used for reading excel file and fix all the mistakes.
    '''
    df = pd.read_excel(excel_path, header=header, usecols=usecols, index_col=index_col)
    df = df.loc[::-1]
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df.replace(to_replace= '--', value = np.nan, inplace=True)
    df = df.astype('float64')
    df = df.drop(df[df.index <= start_date].index)
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
    my_read_excel(excel_path = './origional_data/泸沽湖邛海鲁班水库水质数据/原始查询/原始查询（泸沽湖湖心-泸沽湖）.xls',
                  start_date = '2019-02-6',
                  save_path = './data/luguhuxin.csv')
    my_read_excel(excel_path = './origional_data/泸沽湖邛海鲁班水库水质数据/原始查询/原始查询（邛海湖心-邛海）.xls',
                  start_date = '2018-07-18',
                  save_path = './data/qionghai.csv')
    my_read_excel(excel_path = './origional_data/泸沽湖邛海鲁班水库水质数据/原始查询/原始查询（礼板湾(王妃岛)-泸沽湖）.xls',
                  start_date = '2018-08-07',
                  save_path = './data/wangfeidao.csv')
    my_read_excel(excel_path = './origional_data/泸沽湖邛海鲁班水库水质数据/原始查询/原始查询（鲁班岛-鲁班水库）.xls',
                  start_date = '2018-06-27',
                  save_path = './data/luban.csv')      

