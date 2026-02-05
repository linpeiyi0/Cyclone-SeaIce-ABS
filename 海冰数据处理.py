#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 19:38:49 2026

@author: peiyilin
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime,timedelta
import os
import re
import glob
import scipy.stats as stats
import pyproj
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates   
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from collections import defaultdict
from matplotlib.colors import ListedColormap
from scipy.signal import detrend


#函数定义
#读取nc文件
def readnc(inputfile): 
    with xr.open_dataset(inputfile) as dat:
        return dat

#读取特殊年份的12月数据
def read_sic_12(years):
    septmber_datasets = []
    # 正则表达式匹配日期格式 YYYYMMDD
    date_pattern = re.compile(r'\d{8}')
    # 循环读取每个文件，提取对应年份的2月数据
    for file in file_list:
        try:
            # 提取文件名中的日期信息（文件名格式为"NSIDC0051_SEAICE_PS_S25km_YYYYMMDD_v2.0.nc"）
            date_str = file.split('_')[4][:8]
            
            if date_pattern.match(date_str):
                year = int(date_str[:4])
                
                if year in years:
                    ds = readnc(file)
                    
                    # 选择特定月的日数据
                    septmber_data = ds.sel(time=slice(f"{year}-12-01",f"{year}-12-31"))
                    # 检查是否有有效数据
                    data_variables = [septmber_data[data_var] for data_var in septmber_data.data_vars]
                    ice=data_variables[0]
                    
                    if septmber_data.time.size > 0 and ice.long_name=='Sea Ice Concentration':
                        print(date_str)
                        septmber_datasets.append(ice)
        except Exception as e:
            print(f"Error processing file {file}: {e}")    
    # 合并所有符合条件的数据，确保合并的维度一致
    if septmber_datasets:
        ice = xr.concat(septmber_datasets, dim="time")
        ice = ice.sortby(['time'])
        return ice
    else:
        print("No valid data found for the given years.")
        return None    
    return ice

#读取特殊年份的1月数据
def read_sic_1(years):
    septmber_datasets = []
    
    # 正则表达式匹配日期格式 YYYYMMDD
    date_pattern = re.compile(r'\d{8}')
    
    # 循环读取每个文件，提取对应年份的1月数据
    for file in file_list:
        try:
            # 提取文件名中的日期信息（文件名格式为 "NSIDC0051_SEAICE_PS_S25km_YYYYMMDD_v2.0.nc"）
            date_str = file.split('_')[4][:8]
            
            if date_pattern.match(date_str):
                year = int(date_str[:4])
                
                if year in years:
                    
                    ds = readnc(file)
                    
                    # 选择特定月的日数据
                    septmber_data = ds.sel(time=slice(f"{year}-01-01", f"{year}-01-31"))
        
                    # 检查是否有有效数据（避免选择空的数据集）
                    data_variables = [septmber_data[data_var] for data_var in septmber_data.data_vars]
                    ice=data_variables[0]
                    
                    if septmber_data.time.size > 0 and ice.long_name=='Sea Ice Concentration':
                        print(date_str)
                        septmber_datasets.append(ice)
        except Exception as e:
            print(f"Error processing file {file}: {e}")    
    # 合并所有1月数据，确保合并的维度一致
    if septmber_datasets:
        ice = xr.concat(septmber_datasets, dim="time")
        ice = ice.sortby(['time'])
        return ice
    else:
        print("No valid data found for the given years.")
        return None    
    return ice

#读取各个年份的月数据
def read_all_sic_data(years):
    # 创建一个空的列表来存储所有匹配年份的数据
    all_datasets = []

    # 正则表达式匹配日期格式 YYYYMM
    date_pattern = re.compile(r"NSIDC0079_SEAICE_PS_S25km_\d{6}_v4\.0\.nc")
    
    # 循环读取每个文件
    for file in file_list:
        file_name = os.path.basename(file)  
        if date_pattern.match(file_name):
            # 提取日期信息
            date_str = file_name.split('_')[4][:6]
            year = int(date_str[:4])
            
            # 检查年份是否符合条件
            if year in years:
                try: 
                    ds = xr.open_dataset(file)
                    for var in ds.data_vars:
                        var_data = ds[var]
                        if 'time' in var_data.dims and var_data.size > 0:
                            all_datasets.append(var_data)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
    
    # 合并所有年份和月份的数据
    if all_datasets:
        ice = xr.concat(all_datasets, dim="time").sortby("time")
        return ice
    else:
        print("No valid data found for the specified years.")
        return None

#可选：去趋势
#pd seires序列
def detrend_series(series):
    x = np.arange(len(series))
    slope, intercept, _, _, _ = stats.linregress(x, series)
    trend = slope * x + intercept
    return series - trend

#读取海冰文件
#地图格点面积数据
cell_area=readnc('/Volumes/Expansion/sicdata/NSIDC0771_CellArea_PS_S25km_v1.0.nc')['cell_area']
#区域划分数据
south_grid=readnc('/Volumes/Expansion/sicdata/NSIDC-0780_SeaIceRegions_PS-S25km_v1.0.nc')['sea_ice_region_NASA_surface_mask']
abs_grid=south_grid.where(south_grid==5) #ABS区域=5

# 定义数据目录和文件模式
data_directory = "/Volumes/Expansion/sicdata/"
file_pattern = data_directory + "NSIDC0079_SEAICE_PS_S25km_*_v4.0.nc"

# 获取所有匹配的文件列表
file_list = glob.glob(file_pattern)
file_list.sort()

# 选择要提取的年份范围，得到海冰月数据
years = np.linspace(1979, 2023,45)
icemean=read_all_sic_data(years)
#1987年12月和1988年1月的月数据缺失，需要通过日数据计算月数据再合并
ice198712=read_sic_12([1987])
ice198801=read_sic_1([1988])
ice198712_t=ice198712.mean(dim='time').expand_dims(time=[pd.Timestamp("1987-12-01")])
ice198801_t=ice198801.mean(dim='time').expand_dims(time=[pd.Timestamp("1988-01-01")])
icemean_al = xr.concat([icemean, ice198712_t], dim="time").sortby("time")
icemean_all = xr.concat([icemean_al, ice198801_t], dim="time").sortby("time")	

#海冰日数据
ice_all=xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/ice_all.nc', chunks={"time": 100, "y": 100, "x": 100})['N07_ICECON']
#转换经纬度格点
# x 坐标数据（投影坐标，单位：米）
x_data = icemean_all['x']
# y 坐标数据（投影坐标，单位：米）
y_data = icemean_all['y']
# 创建网格
xx, yy = np.meshgrid(x_data, y_data)
# 定义投影（EPSG:3412，WGS 84 / NSIDC Sea Ice Polar Stereographic South）
proj= pyproj.Proj('epsg:3412')
proj2 = pyproj.Proj('epsg:4326')
# 转换为经纬度
lon, lat = proj(xx, yy, inverse=True)
# 创建经纬度 DataArray
lon_da = xr.DataArray(lon,coords=(y_data,x_data), dims=("y", "x"), name='longitude')
lat_da = xr.DataArray(lat,coords=(y_data,x_data), dims=("y", "x"), name='latitude')
# 这是sic的经纬度格点信息
print(lon_da)
print(lat_da)

# 44年每个月的海冰范围序列，海冰范围SIE = SIC中格点大于等于0.15小于等于1的格点面积相加
sie_all=[]
for i in range(len(icemean_all)):#filled_icemean_mon
    mask = (icemean_all[i] >= 0.15) & (icemean_all[i] <= 1) & (south_grid==5)
    filtered_cell_area = cell_area.where(mask)
    # 计算符合条件的总面积（单位：平方米）
    # 将面积从平方米转换为平方公里
sie_all.append(filtered_cell_area.sum(skipna=True).item() / 1e6  )

# 构造一个与 sie_all 对应的时间索引 (1979/01 到 2023/12, 540 个月)
time_index = pd.date_range(start='1979-01', periods=len(sie_all), freq='MS') 

# 创建一个 pandas Series 来保存月度海冰范围
sie_series = pd.Series(data=sie_all, index=time_index, name='SIE')
yearly_min = sie_series.groupby(sie_series.index.year).min()  # 每年的月最小值
# 找到每年月最小值的日期
min_dates = sie_series.groupby(sie_series.index.year).idxmin() 

#可选：把44年的sie序列去趋势
detrend_sie=detrend_series(yearly_min[1:])

#%%
#可以选取年份以及月份，求某年某月的海冰密集度SIC，这里选择了极小年2010年（在2月达极小）以及极大年1980年（在3月达到极小）
ice_2010 = icemean_all.sel(time=(icemean_all.time.dt.year == 2010) & ( icemean_all.time.dt.month == 2))
ice_1980=icemean_all.sel(time=(icemean_all.time.dt.year == 1980) & ( icemean_all.time.dt.month == 3))
#自1980年后的SIC气候态
icemean_mon_80=icemean_all.sel(time=(icemean_all.time.dt.year >= 1980)).groupby('time.month').mean(dim='time')

#选取年份以及月份，求某年某月的海冰密集度SIC异常
#求各年各月SIC异常
anomaly_all=icemean_all.groupby('time.month') - icemean_mon_80
ice_2010_monano=anomaly_all.sel(time=(icemean_all.time.dt.year == 2010) & ( icemean_all.time.dt.month == 2))
ice_1980_monano=anomaly_all.sel(time=(icemean_all.time.dt.year == 1980) & ( icemean_all.time.dt.month == 3))

#选取月份，求该月份的标准差，便于对选择的年份和月份做显著性检验
ice_mon_2=icemean_all.sel(time=(icemean_all.time.dt.year >= 1980) & ( icemean_all.time.dt.month == 2))
ice_mon_3=icemean_all.sel(time=(icemean_all.time.dt.year >= 1980) & ( icemean_all.time.dt.month == 3))
clim_2_std=ice_mon_2.std(dim='time')
clim_3_std=ice_mon_3.std(dim='time')
extreme_mask_10 = np.abs(ice_2010_monano) > 2 * clim_2_std
extreme_mask_80 = np.abs(ice_1980_monano) > 2 * clim_3_std

#确定每年的极小值在哪个月份
yearly_min_2=[]
yearly_min_3=[]
for i in range(len(min_dates)):
    if min_dates.iloc[i].month ==2:
        yearly_min_2.append(yearly_min.index[i])
    elif min_dates.iloc[i].month ==3:
        yearly_min_3.append(yearly_min.index[i])

#得到某时间段的SIE日数据，单位：平方公里）
#日SIC
ice_daily_2010=ice_all.sel(time=slice('2009-09-26','2010-03-31'))
ice_daily_1980=ice_all.sel(time=slice('1979-09-29','1980-03-31'))

def to_fake_year(time_array):
    real_time = pd.to_datetime(time_array.values)
    fake_time = []
    for t in real_time:
        if t.month >=5:
            fake_time.append(pd.Timestamp(year=2007, month=t.month, day=t.day))
        else:
            fake_time.append(pd.Timestamp(year=2008, month=t.month, day=t.day))
    return pd.to_datetime(fake_time)

# 将不同年份的时间统一转换为 2001 年的日期，便于将不同年份放在同个图上
x_2010 = to_fake_year(ice_daily_2010['time'])
x_1980 = to_fake_year(ice_daily_1980['time'])

#计算SIE
def sie_cal(sic_):
    sie_all=[]
    for i in range(len(sic_)):
        print(i)
        mask = (sic_[i] >= 0.15) & (sic_[i] <= 1) & (south_grid==5)
        filtered_cell_area = cell_area.where(mask& sic_[i].notnull())
        # 计算符合条件的总面积（单位：平方公里）
        sie_all.append(filtered_cell_area.sum(skipna=True) / 1e6  )
    return sie_all
#此处限制了ABS海区（-130度到-60度）
sie_daily_1980=sie_cal(ice_daily_1980.where((lon_da>=-130) & (lon_da<=-60)).compute())
sie_daily_2010=sie_cal(ice_daily_2010.where((lon_da>=-130) & (lon_da<=-60)).compute())

#求气候态SIE，以及前90%/10%上下界
ice_period=ice_all.sel(time=ice_all['time'].dt.month.isin([8,9,10,11,12,1,2,3,4]))
sie_period=sie_cal(ice_period.where((lon_da>=-130) & (lon_da<=-60)).compute())
sie_daily_all=xr.concat(sie_period, dim='time')

p10 = sie_daily_all.groupby('month_day').reduce(np.nanpercentile, q=10, method='nearest')
p90 = sie_daily_all.groupby('month_day').reduce(np.nanpercentile, q=90, method='nearest')
meansie = sie_daily_all.groupby('month_day').mean()

#2、3月海冰气候态（1980-2023）
ice_clim_2=icemean_mon_80.sel(month=2)
ice_clim_3=icemean_mon_80.sel(month=3)
#%%
#可视化：年循环最小SIE序列；选择年份的SIC异常、SIE日演变；SIE日气候态，以及前90%/10%上下界


ice_monano=[ ice_2010_monano[0],ice_1980_monano[0]]
ice_sic_ex=[ice_2010[0],ice_1980[0]]
ice_sic_clim=[ice_clim_2,ice_clim_3]
ex_p=[extreme_mask_10[0],extreme_mask_80[0]] 

icename=['Feb. 2010','Mar. 1980']
iceclimname=['Feb. Climatology','Mar. Climatology']
ti=['(b)Minimum SIE Year (2010.02)','(c)Maximum SIE Year (1980.03)']
minyear = 1980
maxyear = 2022

density_d=[np.linspace(0, 50,11),np.linspace(-5, 5,11)]
density_d=np.linspace(-100, 100,21)

fig = plt.figure(figsize=(12, 8), dpi=500)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 4,1]) 
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.09, wspace=-0.25, hspace=0.5)

ax_top =fig.add_axes([0.18, 0.7, 0.7, 0.25]) # Span across two columns for the top plot
ax_top.plot(yearly_min.index[1:], yearly_min[1:]/1000000,color='k',marker='o')
ax_top.scatter(yearly_min_3[1:], yearly_min[yearly_min_3[1:]]/1000000,facecolors='none', edgecolors='k', marker='o',s=100)

# 回归
slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_min.index[1:], yearly_min[1:])
y_fit = intercept/1000000 + slope/1000000 * yearly_min.index[1:]

# 回归方程字符串
equation_str = f"${slope:.0f} ± {std_err:.0f} km^2 / yr$"
r_value_str = f"$R^2 = {r_value**2:.2f}$"
p_value_str = f"$p < 0.01$"

ax_top.plot(np.unique(yearly_min.index[1:]),y_fit , color='k',linestyle='--',linewidth=1)        
correlation_coefficient, p_value = pearsonr(yearly_min.index[1:-1], yearly_min[1:-1])

ax_top.text(0.7, 0.14, '2010', transform=ax_top.transAxes,  verticalalignment='top')
ax_top.text(0.02, 0.94, '1980', transform=ax_top.transAxes,  verticalalignment='top')
# 标注斜率、R² 和 p 值
textstr = f'{equation_str}\n{r_value_str}\n{p_value_str}'
ax_top.text(0.7, 0.95, textstr, transform=plt.gca().transAxes, verticalalignment='top', fontsize=14,bbox=dict(facecolor='white', alpha=1,edgecolor='none'))

top5_max_indices_2 = np.argsort(yearly_min.values[1:])[-5:]
top5_min_indices_2 = np.argsort(yearly_min.values[1:])[:5]
# 标记最大和最小的前5个点
ax_top.scatter(yearly_min[1:].index[top5_max_indices_2], yearly_min[1:].values[top5_max_indices_2]/1000000, color='#4682B4', s=40,  label='Top 5 Max Sep',zorder=2)
ax_top.scatter(yearly_min[1:].index[top5_min_indices_2],yearly_min[1:].values[top5_min_indices_2]/1000000, color='#A0522D', s=40, label='Top 5 Min Sep',zorder=2)
x_top.text(0., 1.12, '(a)Annual Minimum Monthly SIE in the ABS', transform=ax_top.transAxes,  va='top', ha='left')
ax_top.set_ylabel("SIE (10$^6$ km$^2$)")
ax_top.set_xbound(lower=minyear-1, upper=maxyear+1)
ax_top.set_xticks(np.arange(minyear, maxyear, 4), minor=False)
ax_top.set_xticks(np.arange(minyear, maxyear+1, 1), minor=True)
ax_top.set_ylim(0.0,1.3)#(0.1,3.5)
ax_top.set_xlim(1979,2024)#(0.1,3.5)
ax_top.tick_params(axis='y', colors='k')

ax_top.yaxis.label.set_color('k')
ax_top.yaxis.set_tick_params(color='k')

# **手动设置下方两个子图的位置**
axe = [
    fig.add_axes([0.1, 0.02, 0.7, 0.7], projection=ccrs.SouthPolarStereo()),  # 左图
    fig.add_axes([0.43, 0.02, 0.7, 0.7], projection=ccrs.SouthPolarStereo())  # 右图
]

for i,ax in enumerate(axe):
    ax.text(-0.1, 0.82, ti[i], transform=ax.transAxes,   va='top', ha='left')
    # 地图的经纬度范围
    leftlon, rightlon, lowerlat, upperlat = (-180, 180, -62, -90)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    ax.set_extent(img_extent, ccrs.PlateCarree())
    # 添加地图特征
    ax.add_feature(cfeature.LAND, facecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='none')
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=2)
    im=ax.contourf(xx, yy, ice_monano[i].where(south_grid==5).where(ice_monano[i]!=0)*100,density_d, cmap='RdBu_r', transform=ccrs.SouthPolarStereo(),extend='both')#
    contour1=ax.contour(xx, yy,ice_sic_ex[i].where(lat_da<-58),levels=[0.15],colors='orange',transform=ccrs.SouthPolarStereo(),linewidths=3,linestyles='solid',label=icename[i])
    contour2=ax.contour(xx, yy,ice_sic_clim[i].where(lat_da<-58),levels=[0.15],colors='k',transform=ccrs.SouthPolarStereo(),linewidths=3,linestyles='--',label=iceclimname[i])
    ax.contourf(xx, yy,ex_p[i].where(lat_da<-58), levels=[0.5, 1],colors='none',transform=ccrs.SouthPolarStereo(),hatches=['...'])
    # 设置扇形边界
    theta = np.linspace(-13/18*np.pi, -np.pi/3, 100)
    radius_outer, radius_inner = 0.48, 0.15
    center = [0.5, 0.5]
    outer_arc = np.vstack([np.sin(theta) * radius_outer, np.cos(theta) * radius_outer]).T + center
    inner_arc = np.vstack([np.sin(theta[::-1]) * radius_inner, np.cos(theta[::-1]) * radius_inner]).T + center
    verts = np.vstack([outer_arc, inner_arc, outer_arc[0]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts)-2) + [mpath.Path.CLOSEPOLY]
    ax.set_boundary(mpath.Path(verts, codes), transform=ax.transAxes)
    # 添加网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.ylocator = mticker.FixedLocator([-60, -70, -80])
    gl.xlocator = mticker.FixedLocator([-120, -90, -60, -30, 0])
    
    # 添加纬度标签在边缘方向
    add_lat_labels_on_sector(ax, [ 70, 80], label_lon=-60, offset=0)
    add_lon_labels_on_sector(ax, [ 120, 90, 60], label_lat=-62, offset=0)

    line1 = plt.Line2D([0], [0], color='orange', linewidth=3, linestyle='solid', label=icename[i])
    line2 = plt.Line2D([0], [0], color='k', linewidth=3, linestyle='--', label=iceclimname[i])
    
    # 添加图例，并自定义位置（可以调整 loc 和 bbox_to_anchor）
    axe[i].legend(handles=[line1, line2], loc='upper left', bbox_to_anchor=(0.18, 0.28), frameon=False, fontsize=14)

plt.subplots_adjust(hspace=-0.6)
cbar_ax = fig.add_axes([0.84, 0.13, 0.01, 0.4])  # 颜色条位置
cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label="SIC Ano-mean (%)")#'SIC Anomalies (%)'

ax1 =fig.add_axes([0.18, -0.2, 0.7, 0.25])
ax1.text(0., 1.12, '(d)Daily SIE', transform=ax1.transAxes,  va='top', ha='left')

ax1.plot(x_2010[5:], pd.Series(np.array(sie_daily_2010[5:])).rolling(window = 3, center=True, min_periods=1).mean()/1e6, color='#A0522D', alpha=0.8, label='2010 ',linewidth=2.5)
ax1.tick_params(axis='y', labelcolor='black')
ax1.plot(x_1980[1:], pd.Series(np.array(sie_daily_1980[1:])).rolling(window = 3, center=True, min_periods=1).mean()/1e6, color='#4682B4',alpha=0.8,  label='1980 ',linewidth=2.5)

# 计算最小值及其对应日期索引
min_idx_1980 = np.argmin(sie_daily_1980[1:])
min_val_1980 = sie_daily_1980[1:][min_idx_1980] / 1e6
min_day_1980 = x_1980[1:][min_idx_1980]

min_idx_2010 = np.argmin(sie_daily_2010[1:])
min_val_2010 = sie_daily_2010[1:][min_idx_2010] / 1e6
min_day_2010 = x_2010[1:][min_idx_2010]

ax1.scatter(min_day_1980, min_val_1980, color='green', alpha=0.8,marker='^', s=80, zorder=2)
ax1.scatter(min_day_2010, min_val_2010, color='green',alpha=0.8, marker='^', s=80, zorder=2)
ax1.set_ylabel('SIE (10$^6$ km$^2$)', color='black')
ax1.set_ylim(0,3.)

month_days = pd.to_datetime(x_2010[5:]).strftime('%m-%d')  # e.g. ['10-01', '10-02', ...]

mean_selected = meansie.sel(month_day=month_days)
mean_selected = mean_selected.rolling(month_day = 3, center=True, min_periods=1).mean()

ax1.plot(x_2010[5:],np.array(mean_selected)/1e6, color='k',  label='1980-2023 climatology',linewidth=2.5)

p10_selected = p10.sel(month_day=month_days)
p90_selected = p90.sel(month_day=month_days)
p10_selected = p10_selected.rolling(month_day = 3, center=True, min_periods=1).mean()
p90_selected = p90_selected.rolling(month_day = 3, center=True, min_periods=1).mean()

ax1.fill_between(x_2010[5:], p10_selected/1e6, p90_selected/1e6, color='gray', alpha=0.3)
ax1.set_xlim(pd.Timestamp('2007-10-01'), pd.Timestamp('2008-03-31') ) 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
# 设置每月1号为刻度
ticks = pd.date_range("2007-09-30", "2008-04-01", freq="MS")
ax1.set_xticks(ticks)

lines1, labels1 = ax1.get_legend_handles_labels()
lines = lines1 
labels = labels1

# 创建合并图例，并放在图外右上角
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.75, 0.06), fontsize = 12, ncol=1, frameon=False)
plt.show()