import numpy as np
import padas as pd
from scipy.stats import zscore

def detect_outlier(df, label, rate=4):
    '''
    使用分位数来筛选数据，返回异常数据坐标
    '''
    dfvals = df[label]
    indices = np.array(list(map(lambda x: not x, np.isnan(dfvals))))
    values = dfvals[indices]
    Q1 = np.percentile(values, rate)
    Q3 = np.percentile(values, 100-rate)    
    
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outlier_list = dfvals[(dfvals < lower_limit) | (dfvals > upper_limit)].index
    return outlier_list

def compute_zscore(df, label, threshold=1.5):
    '''
    使用标准差来筛选数据，返回异常数据坐标
    '''
    data = df[label]
    all_values = data.values.copy()
    indices = np.array(list(map(lambda x: not x, np.isnan(all_values))))
    values = all_values[indices]
    z_value = zscore(values)
    all_values[indices] = z_value
    ls = data[np.abs(all_values) > threshold].index
    return ls

def myFilter(df, filter_method = compute_zscore, 
        labels = ['电导率', '高锰酸盐指数'], name='./data_filtered.csv'):
    '''
    df: DataFrame
    filter_method: 想要使用的筛选方法
    labels: 需要筛选的label
    name: 将筛选出的异常数据标记为nan后保存为'name'(csv)
    '''
    dfn = df.copy()
    for label in labels:
        dfvals = dfn[label].values.copy()
        out_list = filter_method(dfn, label, 1.5)
        dfvals[out_list] = None
        dfn[label] = dfvals
    dfn.to_csv(name)
    return True

def sort_by_date(df, name, date_label='时间', oneday_nums=6):
    '''
    Sort the data base on date.Remove incomplete data.
    '''
    DFList = []
    keys = df.keys()
    times = pd.to_datetime(arg=data[data_label], format='%Y-%m-%d %H:%M:%S')
    for group in df.groupby([times.dt.year, times.dt.month, times.dt.day]):
        if group[1].shape[0] == oneday_nums:
            df = group[1][::-1].values.copy()
            DFList.append(df)
    arr = np.concatenate(DFList, axis=0)
    new_data = pd.DataFrame(arr, columns=keys)
    new_data.to_csv(name, index=False)
    return new_data