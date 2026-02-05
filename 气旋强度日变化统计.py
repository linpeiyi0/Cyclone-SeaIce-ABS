#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:16:48 2026

@author: peiyilin
"""

 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
import scipy.interpolate
import datetime

# 定义网格范围
lon_min, lon_max = -130, -60
lat_min, lat_max = -80, -55
grid_resolution = 1

num_lon_bins = int((lon_max - lon_min) / grid_resolution)
num_lat_bins = int((lat_max - lat_min) / grid_resolution)

# 创建网格点的经纬度
grid_lons = lon_min + np.arange(num_lon_bins + 1) * grid_resolution
grid_lats = lat_min + np.arange(num_lat_bins + 1) * grid_resolution
grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)  # 生成网格点

ra = 750
lon_margins = ra / (111 * np.cos(np.radians(grid_lats)))
# 对每个纬度，扩展经度区间
grid_lons_2d = []
for ilat, lat in enumerate(grid_lats):
    lon_min_exp = lon_min - lon_margins[ilat]
    lon_max_exp = lon_max + lon_margins[ilat]
    grid_lons_row = np.arange(lon_min_exp, lon_max_exp + grid_resolution, grid_resolution)
    grid_lons_2d.append(grid_lons_row)
# 构建网格
max_nlon = max(len(row) for row in grid_lons_2d)
# 统一为矩形，短的补np.nan
grid_lon = np.full((num_lat_bins + 1, max_nlon), np.nan)
grid_lat = np.full((num_lat_bins + 1, max_nlon), np.nan)
for ilat, lons_row in enumerate(grid_lons_2d):
    grid_lon[ilat, :len(lons_row)] = lons_row
    grid_lat[ilat, :len(lons_row)] = grid_lats[ilat]

#选择年份
timetick_=timetick2_all[28][0]
lonss_area=lonsc2_all[28]
latss_area=latsc2_all[28]
mslpss_area=mslp2_all[28]
# def daily_cy(timetick_,lonss_area,latss_area,mslpss_area):
# **合并所有小时数据为每日数据**
# 这个代码可以得到每天的轨迹点位置和强度
daily_points = {}  # {日期: (lon_list, lat_list)}

for y in range(len(timetick_)):  # 月份
    print(f"Processing y: {y}")

    for i in range(len(timetick_[y])):  # 遍历气旋的轨迹点
        for k in range(len(timetick_[y][i])):

            timedata = pd.Timestamp(datetime.datetime(1979, 1, 1, 0, 0, 0) + timedelta(hours=timetick_[y][i][k]))
            t_point = pd.Timestamp(timedata.date())  # 取日期

# pd.Timestamp(datetime(1970, 1, 1, 0, 0, 0) + timedelta(seconds=1262300400))
            # 存储该日期下的 lon, lat
            if t_point not in daily_points:
                daily_points[t_point] = ([], [], [])  # 初始化
            if 230<=lonss_area[y][i][k]<=300:
                daily_points[t_point][0].append(lonss_area[y][i][k])  # 存 lon
                daily_points[t_point][1].append(latss_area[y][i][k])  # 存 lat
                daily_points[t_point][2].append(mslpss_area[y][i][k])

#集合所有年份的轨迹                   
# daily_points_all = {}  # {日期: (lon_list, lat_list)}

# for yr in range(44):
#     # for y in range(len(timetick2_all[yr][0])):  # 月份
#         y=4
#         print(f"Processing y: {y}")
       
#         for i in range(len(timetick2_all[yr][0][y])):  # 遍历气旋的轨迹点
#             for k in range(len(timetick2_all[yr][0][y][i])):
    
#                 timedata = pd.Timestamp(datetime.datetime(1979, 1, 1, 0, 0, 0) + timedelta(hours=timetick2_all[yr][0][y][i][k]))
#                 t_point = pd.Timestamp(timedata.date())  # 取日期
    
#     # pd.Timestamp(datetime(1970, 1, 1, 0, 0, 0) + timedelta(seconds=1262300400))
#                 # 存储该日期下的 lon, lat
#                 if t_point not in daily_points_all:
#                     daily_points_all[t_point] = ([], [], [])  # 初始化
#                 if 230<=lonsc2_all[yr][y][i][k]<=300:
#                     daily_points_all[t_point][0].append(lonsc2_all[yr][y][i][k])  # 存 lon
#                     daily_points_all[t_point][1].append(latsc2_all[yr][y][i][k])  # 存 lat
#                     daily_points_all[t_point][2].append(mslp2_all[yr][y][i][k])
                        
# in_avg_all = []
# for t_point, (lon_values, lat_values, intensities) in daily_points_all.items():
#     lat_values = np.array(lat_values)
#     intensities = np.array(intensities)
#     mask = lat_values <= -50
#     in_avg_all.append(np.nanmean(intensities[mask]))
# #求气旋强度偏离值
# np.nanmean(in_avg_all)-968.664468
               
# **对每天的数据计算受影响区域**
count_grid = np.zeros((len(daily_points), num_lat_bins + 1, max_nlon))  # 点数量
sum_intensity_grid = np.zeros((len(daily_points), num_lat_bins + 1, max_nlon))  # 强度和
# daily_points

c=0
time=[]
in_avg=[]
in_avg_max=[]
for t_point, (lon_values, lat_values, intensities) in daily_points.items():
    time.append(t_point)

    lon_values = [(lon - 360) if lon > 180 else lon for lon in lon_values]
    lat_values = np.array(lat_values)
    intensities = np.array(intensities)
    mask = lat_values <= -50
    in_avg.append(np.nanmean(intensities[mask]))
    if intensities[mask] != []:
        in_avg_max.append(np.nanmin(intensities[mask]))
    else:
        in_avg_max.append(np.nan)  
    c+=1

vtimesc = time

#%%
#气旋强度平均日变化
#2010
# 确保时间轴为 numpy 数组
start_date = datetime.datetime(2009, 10, 1)
end_date = datetime.datetime(2010, 3, 31)

date_values = np.array(pd.date_range(start=start_date, end=end_date, freq='D') ) # 'MS'表示每月的第1天

vtimesc = pd.to_datetime(vtimesc)  # 原始时间轴:-36
mask = (vtimesc >= start_date) & (vtimesc <= end_date)
vtimesc = vtimesc[mask]

#控制时间范围，不超过4月
in_avg_=in_avg[:-7]
in_avg_min=in_avg_max[:-7]#区域中的最大强度
# in_avg=data2[:-7]
# 创建一个与 date_values 匹配的新 count 数组，初始化为 0
in_avg_filled = np.zeros(len(date_values))  # (新时间轴, 经度)

indices = np.searchsorted(date_values, vtimesc)
in_avg_filled[indices] =in_avg_# counts[:-6]#
cintensity_2010 = np.where(in_avg_filled == 0,np.nan, in_avg_filled)

#1980
# 确保时间轴为 numpy 数组
start_date = datetime(1979, 10, 1)
end_date = datetime(1980, 3, 31)
date_values = np.array(pd.date_range(start=start_date, end=end_date, freq='D') ) # 'MS'表示每月的第1天

vtimesc = pd.to_datetime(vtimesc[:-5])  # 原始时间轴
count = np.array(count[:-5])  # 原始 count 数据，形状 (time, lon)
in_avg=np.array(in_avg[:-5])
# 创建一个与 date_values 匹配的新 count 数组，初始化为 0
da_fill= np.zeros((len(date_values))) 
# 找到 vtimesc 在 date_values 中的位置
indices = np.searchsorted(date_values, vtimesc)

da_fill[indices] =in_avg#in_avg 
cintensity_1980 = np.where(da_fill == 0,np.nan, da_fill)

