import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt

keys = ['采集时间', '水温', 'pH', '溶解氧', '电导率', '浊度', '高锰酸盐指数','氨氮', '总磷', '总氮']

en_keys = ['WaterTemperature', 'PH' ,'dissolved oxygen', 'Conductivity','Turbidity','PermanganateIndex',
        'AmmoniaNitrogen','TP','TN', 'humidity','room temperature','chlorophyll','Algae density']
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
    smooth_df = df.ewm(size).mean()
    not_nan_idx = ~df.isna()
    df[not_nan_idx] = smooth_df[not_nan_idx]
    return df

def patch_up(df, r, limit=3):
    # Remove the top and end illegal rows
    start_date = df.first_valid_index().date() + timedelta(days=1)
    end_date = df.last_valid_index().date() - timedelta(days=1)
    df = df.loc[start_date: end_date]

    # # First insert data in front, end of the dataframe
    # front_date = df.index.min().date()
    # end_date = df.index.max().date()
    # for i in range(1, r+1):
    #     front_idx = df.loc[str(front_date)].index - timedelta(days=r+i)
    #     end_idx = df.loc[str(end_date)].index + timedelta(days=r+i)
        
    #     front_data = df.loc[str(front_date + timedelta(days=r+i))].values
    #     end_data = df.loc[str(end_date - timedelta(days=r+i))].values
        
    #     front_dfn = pd.DataFrame(front_data, index=front_idx, columns=df.columns)
    #     end_dfn = pd.DataFrame(end_data, index=end_idx, columns=df.columns)
        
    #     df = pd.concat((front_dfn, df, end_dfn))
        
    # Group by hour and fill nan
    group = df.groupby([df.index.hour])
    df_mean = group.transform(lambda x: x.rolling(2*r+1, 1, center=True).mean())
    dfn = df.fillna(df_mean, limit=limit)
    return dfn

def plot_df(dfn, keys=en_keys):
    index_nums = len(dfn.keys())
    print(dfn.keys())
    l, h = 18, 3
    fig, axis = plt.subplots(index_nums, 1, figsize=(l, h*index_nums), constrained_layout=True)
    for i in range(index_nums):
        name = keys[i]
        dfn.plot(y=dfn.keys()[i], ax=axis[i])
        axis[i].set_title(name, fontsize=20)
        axis[i].set_xlabel('', fontsize=15)
        axis[i].set_ylabel('', fontsize=15)
        axis[i].legend([name], fontsize=15)

def fujiang_factory(data_path, patch_up_r, patch_up_limit, smooth_step):
    df=pd.read_excel(data_path, header=2, usecols=keys, index_col=0)
    df.replace('\d*【已删除】|\d*/.\d*【已删除】',np.nan,regex=True,inplace=True)
    df.replace('--', np.nan, inplace=True)
    df.index=pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df=df.astype('float64')
    df=df.resample('4H').mean()
    df.loc[(df==0).all(axis=1)] = np.nan    
    df = remove_outliers(df, standard_deviation, 25)
    df[df < 0] = np.nan
    df = patch_up(df, patch_up_r, patch_up_limit)
    df = smooth(df, smooth_step)
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

def _gen_data(df, lGet, lPre, save_path=''):
    '''
    Parameters:
        df: The DataFrame came from data factory.
        lGet: How long old data you need.
        lPre: How long new data you predicted.
        save_path: Where to save the .npz file.
    '''
    step = lGet + lPre
    data = []
    for i in range(df.shape[0]-step):
        vals = df.iloc[i: i+step].values
        if (vals != np.nan).all():
            data.append(vals)
    data = np.stack(data, axis=0)
    if save_path:
        np.save(save_path, data)
    return np.stack(data, axis=0)

def dataHander(path, lGet, lPre, save_path, func, *args):
    p = Path(path)
    for file in p.iterdir():
        print(file.stem)
        save_file_name = f'{save_path}{file.stem}'
        describe_save_name = f'{save_path}{file.stem}_describe.csv'
        df = func(file, *args)
        _gen_data(df, lGet, lPre, save_file_name)
        df.describe().to_csv(describe_save_name)
    return 

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

