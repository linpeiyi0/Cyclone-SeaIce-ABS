#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 19:58:31 2026

@author: peiyilin
"""

#可选择滑动窗口、超前滞后步长进行要素间的超前滞后相关
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def adaptive_window_corr_fixed_valid(x, y,time_x, window, max_lag):
    """
    滑动窗口相关性计算，确保每个相关计算至少使用指定数量的有效值（min_valid），
    并通过扩展窗口直到满足要求。
    参数：
        x, y: 输入的一维数据序列
        window: 初始窗口长度
        max_lag: 最大滞后天数（包括正负）
        min_valid: 相关计算所需的最小有效值对数
    返回：
        corr_mat: 相关系数矩阵
        pval_mat: p值矩阵
        lags: 滞后数组
    """
    time_x=time_index
    window=40
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    max_expand = 10  # 最多扩展这么多天

    corr_mat = np.full((n - window + 1, len(lags)), np.nan)
    pval_mat = np.full((n - window + 1, len(lags)), np.nan)
    start_end_dates_x=np.full((n - window + 1, len(lags)), np.nan, dtype=object)
    used_dates_x = [[None for _ in range(len(lags))] for _ in range(n - window + 1)]

    for i in range(n - window + 1):#总窗口
        x_win_base = x[i:i + window]
        for j, lag in enumerate(lags):#总超前滞后
            y_start = i + lag
            y_end = y_start + window
            if y_start < 0 or y_end > len(y):
                continue
            y_win_base = y[y_start:y_end]
            if np.sum(~np.isnan(x_win_base))==np.sum(~np.isnan(y_win_base))==40:
                r, p = spearmanr(x_win_base, y_win_base)
                corr_mat[i, j] = r
                pval_mat[i, j] = p
                start_end_dates_x [i, j] = (time_x.iloc[i+ lag], time_x.iloc[i + window+ lag - 1])
                used_dates_x[i][j] = range(i+ lag,i + window+ lag)
                continue       
            # 扩展窗口直至满足有效值对数
            for expand in range(1,max_expand + 1):#如果存在nan需要扩展
                x_win = x[i:i + window + expand]
                y_start_adj = i + lag
                y_end_adj = y_start_adj + window + expand
                if y_start_adj < 0 or y_end_adj > len(y):
                    break
            
                y_win = y[y_start_adj:y_end_adj]
                min_len = min(len(x_win), len(y_win))
                x_win = x_win[:min_len]
                y_win = y_win[:min_len]
                valid_mask = (~np.isnan(x_win)) & (~np.isnan(y_win))
                if np.sum(valid_mask) == window:
                    r, p = spearmanr(x_win[valid_mask], y_win[valid_mask])
                    
                    indices = np.arange(i + lag, i + window + lag+ expand)
                    valid_indices = indices[valid_mask]
                    used_dates_x[i][j] =  valid_indices  
                    start_end_dates_x[i, j] = (time_x.iloc[valid_indices].iloc[0],time_x.iloc[valid_indices].iloc[-1])
                    corr_mat[i, j] = r
                    pval_mat[i, j] = p
                    break  # 已满足有效值要求，停止扩展
    return corr_mat, pval_mat, lags, start_end_dates_x,used_dates_x

#读取excel文件
df_2010=pd.read_csv('df_2010.csv')
date_values_2010=pd.to_datetime(df_2010["time"].values)

lag_2010 = pd.DataFrame({'序列A':df_2010['cyintensity'], '序列B': df_2010['SIEF'], '序列C': date_values_2010})
data1 = lag_2010['序列A']  #气旋强度
data2 = lag_2010['序列B']  #海冰日变化
data3 = lag_2010['序列C']  #日期

# 选择滑动窗口大小
window_size =40 
# 选择最大超前滞后步长
max_lag = 5 
#可选择从哪个时间点开始进行相关
time_index = data3 #[10:]
x = data1
y = data2
#计算超前滞后相关
correlation_matrix, correlationP_matrix, lags, start_dates, end_dates = adaptive_window_corr_fixed_valid(x, y, time_index,window_size, max_lag)
max_corr_index = np.nanargmax(abs(correlation_matrix))  # 忽略 NaN 取最大值索引
max_corr_row, max_corr_col = np.unravel_index(max_corr_index, correlation_matrix.shape)
# 对应的滞后步长和时间
best_lag = lags[max_corr_col]  # 滞后步长 (列索引)
best_time=start_dates[max_corr_row, max_corr_col-best_lag]
# 找出 p 值小于 0.01 的点
p_threshold = 0.01
sig_indices = np.where(correlationP_matrix < p_threshold)  # （窗口，lag）
# 转换索引为真实的时间 & 滞后步长
sig_times = time_index[window_size-1:].iloc[sig_indices[0]] # 真实end时间
sig_lags = lags[sig_indices[1]]  # 真实滞后步长
sig_times[180:]
# 最高相关系数
max_corr_value = correlation_matrix[max_corr_row, max_corr_col]
time_index.iloc[window_size-1]
# 输出结果
print(f"最高相关系数: {max_corr_value:.2f}")
print(f"发生在时间: {best_time}")
print(f"对应滞后步长: {best_lag}")
#最高相关系数: 0.86
#发生在时间: (Timestamp('2009-11-05 00:00:00'), Timestamp('2009-12-16 00:00:00'))
#对应滞后步长: -3

#可视化
plt.figure(figsize=(10, 6))
extent = [lags[0], lags[-1], time_index.iloc[window_size-1], time_index.iloc[-1]]
plt.imshow(correlation_matrix, aspect='auto', cmap='coolwarm', extent=extent,
            origin='lower', vmin=-0.7, vmax=0.7)
plt.colorbar(label="Pearson r")
# 在显著点位置打点
plt.scatter(sig_lags, sig_times, s=1, color='black', label="p < 0.05")
plt.xlabel("Lag (days)")
plt.ylabel("time")
plt.axvline(x=0, color="black", linestyle="--", lw=1, label="no offset")
plt.show()