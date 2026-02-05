#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 11:42:41 2026

@author: peiyilin
"""


#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyproj
import pandas as pd
from datetime import datetime,timedelta
import os
import re
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import scipy.stats as stats

import matplotlib.path as mpath
import matplotlib.ticker as mticker

from collections import defaultdict
from datetime import datetime, timedelta

from matplotlib.colors import ListedColormap

from scipy.signal import detrend

def readnc(inputfile): 
    with xr.open_dataset(inputfile) as dat:
        
        return dat
    
# readnc('/Volumes/Expansion/sicdata/NSIDC0079_SEAICE_PS_S25km_197901_v4.0.nc')#.sel(time=slice(f"1980-01-01", f"1980-1-31"))
              
###########提取需要的月份sic数据##########################

lat_f=readnc('/Volumes/Expansion/sicdata/NSIDC0771_LatLon_PS_S25km_v1.0.nc')['latitude']
lon_f=readnc('/Volumes/Expansion/sicdata/NSIDC0771_LatLon_PS_S25km_v1.0.nc')['longitude']
cell_area=readnc('/Volumes/Expansion/sicdata/NSIDC0771_CellArea_PS_S25km_v1.0.nc')['cell_area']
south_grid=readnc('/Volumes/Expansion/sicdata/NSIDC-0780_SeaIceRegions_PS-S25km_v1.0.nc')['sea_ice_region_NASA_surface_mask']
abs_grid=south_grid.where(south_grid==5)
rs_grid=south_grid.where(south_grid==4)
abs_grid.plot()
a=cell_area.values
rabs_grid=south_grid.where((south_grid == 4) | (south_grid == 5))
rabs_grid.plot()
# 定义数据目录和文件模式（假设文件名包含年份信息）
data_directory = "/Volumes/Expansion/sicdata/"
file_pattern = data_directory + "NSIDC0079_SEAICE_PS_S25km_*_v4.0.nc"

# 获取所有匹配的文件列表
file_list = glob.glob(file_pattern)
file_list.sort()


def read_sic_12(years):
    septmber_datasets = []
    
    # 正则表达式匹配日期格式 YYYYMMDD
    date_pattern = re.compile(r'\d{8}')
    
    # 循环读取每个文件，提取对应年份的2月数据
    for file in file_list:
        try:
            # 提取文件名中的日期信息（假设文件名格式为 "NSIDC0051_SEAICE_PS_S25km_YYYYMMDD_v2.0.nc"）
            date_str = file.split('_')[4][:8]
            
            if date_pattern.match(date_str):
                year = int(date_str[:4])
                
                if year in years:
                    # 打开 NetCDF 文件
        
                    ds = readnc(file)
                    
                    # 选择特定年份的9月数据
                    septmber_data = ds.sel(time=slice(f"{year}-12-01", f"{year}-12-31"))
                    
        
                    # 检查是否有有效数据（避免选择空的数据集）
                    # Optionally, if you want to exclude coordinates (like time, latitude, longitude) and keep only data variables:
                    data_variables = [septmber_data[data_var] for data_var in septmber_data.data_vars]
                    ice=data_variables[0]
                    
                    if septmber_data.time.size > 0 and ice.long_name=='Sea Ice Concentration':
                        print(date_str)
                        septmber_datasets.append(ice)
        except Exception as e:
            print(f"Error processing file {file}: {e}")    
    # 合并所有年份的2月数据，确保合并的维度一致
    if septmber_datasets:
        ice = xr.concat(septmber_datasets, dim="time")
        ice = ice.sortby(['time'])
        return ice
    else:
        print("No valid data found for the given years.")
        return None    
    return ice

# 创建一个空的列表来存储各年份的2月数据
def read_sic_1(years):
    septmber_datasets = []
    
    # 正则表达式匹配日期格式 YYYYMMDD
    date_pattern = re.compile(r'\d{8}')
    
    # 循环读取每个文件，提取对应年份的2月数据
    for file in file_list:
        try:
            # 提取文件名中的日期信息（假设文件名格式为 "NSIDC0051_SEAICE_PS_S25km_YYYYMMDD_v2.0.nc"）
            date_str = file.split('_')[4][:8]
            
            if date_pattern.match(date_str):
                year = int(date_str[:4])
                
                if year in years:
                    # 打开 NetCDF 文件
        
                    ds = readnc(file)
                    
                    # 选择特定年份的9月数据
                    septmber_data = ds.sel(time=slice(f"{year}-01-01", f"{year}-01-31"))
                    
        
                    # 检查是否有有效数据（避免选择空的数据集）
                    # Optionally, if you want to exclude coordinates (like time, latitude, longitude) and keep only data variables:
                    data_variables = [septmber_data[data_var] for data_var in septmber_data.data_vars]
                    ice=data_variables[0]
                    
                    if septmber_data.time.size > 0 and ice.long_name=='Sea Ice Concentration':
                        print(date_str)
                        septmber_datasets.append(ice)
        except Exception as e:
            print(f"Error processing file {file}: {e}")    
    # 合并所有年份的2月数据，确保合并的维度一致
    if septmber_datasets:
        ice = xr.concat(septmber_datasets, dim="time")
        ice = ice.sortby(['time'])
        return ice
    else:
        print("No valid data found for the given years.")
        return None    
    return ice




data_directory = "/Volumes/Expansion/sicdata/"
file_pattern = data_directory + "NSIDC0079_SEAICE_PS_S25km_*_v4.0.nc"

# 获取所有匹配的文件列表
file_list = glob.glob(file_pattern)

def read_all_sic_data(years):
    # 创建一个空的列表来存储所有匹配年份的数据
    all_datasets = []

    # 正则表达式匹配日期格式 YYYYMM
    date_pattern = re.compile(r"NSIDC0079_SEAICE_PS_S25km_\d{6}_v4\.0\.nc")
    
    # 循环读取每个文件
    for file in file_list:
        file_name = os.path.basename(file)  # 获取文件名
        if date_pattern.match(file_name):
            # 提取日期信息
            date_str = file_name.split('_')[4][:6]
            year = int(date_str[:4])
            
            # 检查年份是否符合条件
            if year in years:
                try: 
                    # 使用 xarray 打开 NetCDF 文件
                    ds = xr.open_dataset(file)
                    
                    # 动态选择数据变量（忽略非数据变量）
                    for var in ds.data_vars:
                        var_data = ds[var]
                    
                        # 确保数据变量的维度包含时间 (time) 并非空
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

#%%
#极端年2月/9月海冰范围趋势
ice=[]
for year in [1987]:
    february_data=readnc('/Volumes/Expansion/sicdata/NSIDC0079_SEAICE_PS_S25km_'+str(int(year))+'12_v4.0.nc')
    data_variables = [february_data[data_var] for data_var in february_data.data_vars]
    ice.append(data_variables[-1])
    
ice2_mean=xr.concat(ice, dim='time')


ice=[]
for year in years:
    february_data=readnc('/Volumes/Expansion/sicdata/NSIDC0051_SEAICE_PS_S25km_'+str(int(year))+'09_v2.0.nc')
    data_variables = [february_data[data_var] for data_var in february_data.data_vars]
    ice.append(data_variables[-1])
    
ice9_mean=xr.concat(ice, dim='time')

ice2min_mean=ice2_mean[:5].mean(dim='time')
ice2max_mean=ice2_mean[5:].mean(dim='time')
ice9min_mean=ice9_mean[:5].mean(dim='time')
ice9max_mean=ice9_mean[5:].mean(dim='time')


ice2_mean[1].plot()

from scipy import stats
import cmaps
import matplotlib.colors as mcolors

trend_2 = np.zeros((ice9_mean.y.shape[0],ice9_mean.x.shape[0]))
p_value_2 = np.zeros((ice9_mean.y.shape[0],ice9_mean.x.shape[0]))

for i in range (0,ice2_mean.y.shape[0]):
    for j in range (0,ice2_mean.x.shape[0]):
        trend_2[i,j], intercept, r_value, p_value_2[i,j], std_err=stats.linregress(np.arange(1980,2023),ice2_mean[:,i,j])

trend=[trend_2,trend_9]
p_value=[p_value_2,p_value_9]

fig,axes = plt.subplots(1, 2,figsize=(20,10),dpi=300,subplot_kw={'projection': ccrs.SouthPolarStereo()})#画布
# axes = fig.add_axes([2,2,2,2],projection = ccrs.SouthPolarStereo())#画层
for i in range(2):
    
    
    
    level = np.arange(-2,2,0.001)
    midnorm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    
    plot = axes[i].contourf(ice2_mean.x, ice2_mean.y,trend[i]*100, zorder=0,  extend = 'both', 
                          cmap='RdBu_r',
                           norm = midnorm,
                         levels = level,
                         transform=ccrs.SouthPolarStereo())
    c1b = axes[i].contourf(ice2_mean.x, ice2_mean.y,  p_value[i],[np.min(p_value[i]),0.05,np.max(p_value[i])], hatches=['.', None],zorder=1,colors="none", transform=ccrs.SouthPolarStereo())
    
    axes[i].gridlines()
    
    axes[i].add_feature(cfeature.LAND, facecolor='darkgray',zorder=2)
    axes[i].add_feature(cfeature.OCEAN, facecolor='lightblue')
    axes[i].add_feature(cfeature.COASTLINE,edgecolor='none',linewidth=3)
    # axes[i].set_title(title[p])
    bar=fig.colorbar(plot,ax=axes[i],orientation="horizontal",shrink=0.5,aspect=20,pad=0.02)  #aspect代表长宽比
    
    plt.tight_layout()
    
    # cax = fig.add_axes([axes[2].get_position().x1-0.47,
    #                    axes[0].get_position().y0-0.04, 0.45, 0.02])
    # plt.colorbar(contourf1,cax = cax,orientation='horizontal',
    #             label="Sea Ice Concentration(%)")  # Similar to fig.colorbar(im, cax = cax)

plt.show()

#%%
# 定义要提取的年份范围

years = np.linspace(1979, 2023,45)
# ice_all=read_sic_years(years,file_list)  
ice_all=xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/ice_all.nc', chunks={"time": 100, "y": 100, "x": 100})['N07_ICECON']#.where(abs_grid==5)#
ice_daily_avg=xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/ice_all_dailymean.nc')['N07_ICECON']#.where(abs_grid==5)

#全年月度

icemean=read_all_sic_data(years)
icemean.time.data

#1987.12和1988.01月度数据缺失，使用日数据平均补全
ice198712=read_sic_12([1987])
ice198801=read_sic_1([1988])

ice198712_t=ice198712.mean(dim='time').expand_dims(time=[pd.Timestamp("1987-12-01")])
ice198801_t=ice198801.mean(dim='time').expand_dims(time=[pd.Timestamp("1988-01-01")])

icemean_al = xr.concat([icemean, ice198712_t], dim="time").sortby("time")
icemean_all = xr.concat([icemean_al, ice198801_t], dim="time").sortby("time")

#气候态
icemean_mon=icemean_all.groupby('time.month').mean(dim='time')
# 创建目标时间坐标（44年，每年12个月，总计528个月）
start_year = 1979
end_year = 2023
time_coords = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")

# 1. 通过扩展数据，将 12 个月的数据重复填充为 528 个月
repeated_data = np.tile(icemean_mon.values, (45, 1, 1))  # 重复 44 次

# 2. 创建新的 DataArray
filled_icemean_mon = xr.DataArray(
    data=repeated_data,
    dims=["time", "y", "x"],
    coords={
        "time": time_coords,  # 新的时间坐标
        "y": icemean_all["y"],  # 原来的 y 坐标
        "x": icemean_all["x"],  # 原来的 x 坐标
    },
    attrs=icemean_all.attrs  # 保留原数据的属性
)

# 查看结果
# print(filled_icemean_mon)
# print(np.allclose(filled_icemean_mon.isel(time=0), filled_icemean_mon.isel(time=12)))
icemean_mon_80=icemean_all.sel(time=(icemean_all.time.dt.year >= 1980)).groupby('time.month').mean(dim='time')
ice_mon_2=icemean_all.sel(time=(icemean_all.time.dt.year >= 1980) & ( icemean_all.time.dt.month == 2))
ice_mon_3=icemean_all.sel(time=(icemean_all.time.dt.year >= 1980) & ( icemean_all.time.dt.month == 3))
clim_2_std=ice_mon_2.std(dim='time')
clim_3_std=ice_mon_3.std(dim='time')

clim_3_std.plot()

ice_2010_monano=icemean_all.sel(time=(icemean_all.time.dt.year == 2010) & ( icemean_all.time.dt.month == 2))-icemean_mon.sel(month=2)
ice_2010=icemean_all.sel(time=(icemean_all.time.dt.year == 2010) & ( icemean_all.time.dt.month == 2))
ice_1980=icemean_all.sel(time=(icemean_all.time.dt.year == 1980))
ice_clim_2=icemean_mon_80.sel(month=2)
ice_clim_3=icemean_mon_80.sel(month=3)
anomaly_all=icemean_all.groupby('time.month') - icemean_mon_80
ice_2010_monano=anomaly_all.sel(time=(icemean_all.time.dt.year == 2010) & ( icemean_all.time.dt.month == 2))
ice_1980_monano=anomaly_all.sel(time=(icemean_all.time.dt.year == 1980) & ( icemean_all.time.dt.month == 3))

extreme_mask_10 = np.abs(ice_2010_monano) > 2 * clim_2_std
extreme_mask_80 = np.abs(ice_1980_monano) > 2 * clim_3_std

ice_2022=icemean_all.sel(time=(icemean_all.time.dt.year == 2022) & ( icemean_all.time.dt.month == 2))
ice_2023=icemean_all.sel(time=(icemean_all.time.dt.year == 2023) & ( icemean_all.time.dt.month == 2))
ice_sic_ex_=[ice_2022[0],ice_2023[0]]
#%%
##########################转换经纬度格点
# 示例 x 坐标数据（投影坐标，单位：米）
x_data = ice_all['x']
# -3937500+3912500
# 示例 y 坐标数据（投影坐标，单位：米）
y_data = ice_all['y']
# 4337500-4312500 25
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

# t=icemin2['time']
is_equal = lon_f.equals(lon_da)
is_close = np.allclose(lon_f.values, lon_da.values, atol=1e-6)

coords_equal = lon_f.coords["x"].equals(lon_da.coords["x"]) and \
               lon_f.coords["y"].equals(lon_da.coords["y"])
   

#%%
##每月海冰变化
# 选择每年2月的第一天和最后一天
def sicmeanchange(ice):
  
    first_days = ice.groupby('time.year').first().where(ice[0]<=1.000)
    last_days = ice.groupby('time.year').last().where(ice[0]<=1.000)

    # 计算差异.mean(dim='year')
    diff_sie = (last_days - first_days).where((first_days-0.15)*(last_days-0.15)<0)#.mean(dim='year')
    
    return diff_sie

diff_icemin2=sicmeanchange(icemin2.where((lon_da<=-60)&(lon_da>=-130)))
diff_icemax2=sicmeanchange(icemax2.where((lon_da<=-60)&(lon_da>=-130)))
diff_icemin9=sicmeanchange(icemin9.where((lon_da<=-60)&(lon_da>=-130)))
diff_icemax9=sicmeanchange(icemax9.where((lon_da<=-60)&(lon_da>=-130)))

diff_icechange=[diff_icemin2,diff_icemax2,diff_icemin9,diff_icemax9]


for i in range(4):
    print(diff_icechange[i].where((diff_icechange[i]<=-0.15)|(diff_icechange[i]>=0.15)).mean(dim='year').mean())


b=diff.where(diff!=0,drop=True).where((lon_da<=-60)&(lon_da>=-150)).values.flatten()
a=sample_count.where(diff!=0,drop=True).where((lon_da<=-60)&(lon_da>=-150)).values.flatten()
a = a[~np.isnan(a)]
b = b[~np.isnan(b)]
plt.scatter(a,b,s=2.5)

density_levels1=np.linspace(0, 55,11)
density_sic=np.linspace(-1,1,11)

icemin2

diff_icechange[0].mean(dim='year').max()

a=diff_icechange[0].mean(dim='year')[114,52:54].data

#[121:125,60:72]
#%%
sic_clim = ice_all.groupby("time.dayofyear")
mean = sic_clim.mean("time")
std = sic_clim.std("time")

# 创建完整年份的时间序列（适用于非闰年）
full_dates = pd.date_range("2012-01-01", "2012-12-31", freq="D")  # 任意年都行，只要是365天
month_days = full_dates.strftime('%m-%d')

# 给 std 添加坐标
std = std.assign_coords(month_day=("dayofyear", month_days))
std = std.swap_dims({'dayofyear': 'month_day'})

ice_period=ice_all.sel(time=ice_all['time'].dt.month.isin([8,9,10,11,12,1,2,3,4]))
sie_period=sie_cal(ice_period.where((lon_da>=-130) & (lon_da<=-60)).compute())
sie_daily_all=xr.concat(sie_period, dim='time')

#计算为sie后再求百分位和平均
p10 = sie_daily_all.groupby('month_day').reduce(np.nanpercentile, q=5, method='nearest')
p90 = sie_daily_all.groupby('month_day').reduce(np.nanpercentile, q=95, method='nearest')
mean = sie_daily_all.groupby('month_day').mean()

p10.to_netcdf("daily_sie_upper10.nc")
p90.to_netcdf("daily_sie_lower90.nc")
mean.to_netcdf("daily_sie_clim.nc")
sie_daily_all.to_netcdf("daily_sie_10_3.nc")


p10=xr.open_dataarray("daily_sie_upper10.nc")
p90=xr.open_dataarray("daily_sie_lower90.nc")
mean=xr.open_dataarray("daily_sie_clim.nc")

#%%
#海冰日变化率：前一天海冰-后一天
#1980年日ano,气候态
ice_daily_1980=ice_all.sel(time=slice('1979-09-29','1980-03-31'))
ice_daily_1980.time
#计算sie
def sie_cal(sic_):
    sie_all=[]
    for i in range(len(sic_)):
        print(i)
        mask = (sic_[i] >= 0.15) & (sic_[i] <= 1) & (south_grid==5)
        filtered_cell_area = cell_area.where(mask& sic_[i].notnull())

        # 计算符合条件的总面积（单位：平方公里）
        sie_all.append(filtered_cell_area.sum(skipna=True) / 1e6  )# 将面积从平方米转换为平方公里
    return sie_all


sie_daily_1980=sie_cal(ice_daily_1980.where((lon_da>=-130) & (lon_da<=-60)).compute())
#先算dailymean的sic，再算sie
daily_clim=ice_daily_avg.sel(month_day=ice_daily_1980['month_day'])
std_daily_1980 = std.sel(month_day=ice_daily_1980['month_day'])

sie_daily_1980_clim=sie_cal(daily_clim.where((lon_da>=-130) & (lon_da<=-60)))

#sie异常
sie_ano_1980 = np.array(sie_daily_1980) - np.array(sie_daily_1980_clim) 
min(sie_daily_2010)
#日变化率:当天ano-前两天的ano，1980年两天一数据
sie_2_daychange = np.array(sie_daily_1980[1:]) - np.array(sie_daily_1980[:-1]) 
sief_1980 = sie_2_daychange/np.array(sie_daily_1980[:-1]) #9.27

#找到当年的气旋
date_values = ice_daily_1980.time[:]
start_date = datetime.datetime(1979, 11, 1)
end_date = datetime.datetime(1980, 3, 31)

# 用于存储每天有哪些气旋
daily_cyclone_ids = defaultdict(set)
for month_idx in range(len(timetick[1])):  # 遍历所有月份的数据 timetick[1]为1980年
    for cyclone_idx in range(len(timetick[1][month_idx])):  # 遍历每个气旋
        for point_idx in range(len(timetick[1][month_idx][cyclone_idx])):
            if 230<=lonss[1][month_idx][cyclone_idx][point_idx]<=300 and latss[1][month_idx][cyclone_idx][point_idx]<=-50:
                
                t_hour = timetick[1][month_idx][cyclone_idx][point_idx]
                t_day = (datetime.datetime(1979, 1, 1) + timedelta(hours=t_hour)).date()
                
                # 把气旋 ID（比如 month_idx + '_' + cyclone_idx）记入该日期
                cyclone_id = f"{month_idx}_{cyclone_idx}"
                daily_cyclone_ids[t_day].add(cyclone_id)

# 统计每天的气旋数量
daily_cyclone_counts = {day: len(cyclones) for day, cyclones in daily_cyclone_ids.items()}
dates = sorted(daily_cyclone_counts.keys())
# 按排序后的日期提取对应的计数
counts_1980 = [daily_cyclone_counts[date] for date in dates]
#%%
#2010
ice_daily_2010=ice_all.sel(time=slice('2009-09-26','2010-03-31'))
ice_daily_2010.time

sie_daily_2010=sie_cal(ice_daily_2010.where((lon_da>=-130) & (lon_da<=-60)).compute())
daily_clim=ice_daily_avg.sel(month_day=ice_daily_2010['month_day'])
sie_daily_2010_clim=sie_cal(daily_clim.where((lon_da>=-130) & (lon_da<=-60)))
#sie异常
sie_ano_2010 = np.array(sie_daily_2010) - np.array(sie_daily_2010_clim) 
std_daily_2010 = sie_cal(std.sel(month_day=ice_daily_2010['month_day']).where((lon_da>=-130) & (lon_da<=-60)).compute())

#日变化率:当天ano-前一天的ano，2010为逐日数据
sie_2_daychange = np.array(sie_daily_2010[2:]) - np.array(sie_daily_2010[:-2]) 
sief_2010 = sie_2_daychange/np.array(sie_daily_2010[:-2]) #9.27

#找到当年的气旋
# 用于存储每天有哪些气旋
daily_cyclone_ids = defaultdict(set)

for month_idx in range(len(timetick[0])):  # 遍历所有月份的数据 timetick[0]为2010年
    for cyclone_idx in range(len(timetick[0][month_idx])):  # 遍历每个气旋
        for point_idx in range(len(timetick[0][month_idx][cyclone_idx])):
            if 230<=lonss[0][month_idx][cyclone_idx][point_idx]<=300 and latss[0][month_idx][cyclone_idx][point_idx]<=-50:
                t_hour = timetick[0][month_idx][cyclone_idx][point_idx]
                t_day = (datetime.datetime(1979, 1, 1) + timedelta(hours=t_hour)).date()
                
                # 把气旋 ID（比如 month_idx + '_' + cyclone_idx）记入该日期
                cyclone_id = f"{month_idx}_{cyclone_idx}"
                daily_cyclone_ids[t_day].add(cyclone_id)

# 统计每天的气旋数量
daily_cyclone_counts = {day: len(cyclones) for day, cyclones in daily_cyclone_ids.items()}
dates = sorted(daily_cyclone_counts.keys())
# 按排序后的日期提取对应的计数
counts_2010 = [daily_cyclone_counts[date] for date in dates]
#%%

# # 创建 DataFrame
saved = pd.DataFrame({
    'time': date_values_1980,
    'sief': sief_1980,
    'cintensity': cintensity_1980,
    'count': counts_1980,
})



#%% 
#气旋相关的sic异常       
def cyclone_impact_day(ice, timetick_,lonss_area,latss_area):
    impact_mask = {}  # 用字典存储受影响的区域，避免创建大数组
    ice_times = ice['time'].values  # 获取所有时间点

    # **合并所有小时数据为每日数据**
    daily_points = {}  # {日期: (lon_list, lat_list)}

    for y in range(len(timetick_)):  # 月份
        print(f"Processing y: {y}")

        for i in range(len(timetick_[y])):  # 遍历气旋的轨迹点
            for k in range(len(timetick_[y][i])):

                timedata = pd.Timestamp(datetime(1979, 1, 1, 0, 0, 0) + timedelta(hours=timetick_[y][i][k]))
                t_point = pd.Timestamp(timedata.date())  # 取日期

                # 存储该日期下的 lon, lat
                if t_point not in daily_points:
                    daily_points[t_point] = ([], [])  # 初始化
                if 230<=lonss_area[y][i][k]<=300 and latss_area[y][i][k]<=-50:
                    daily_points[t_point][0].append(lonss_area[y][i][k])  # 存 lon
                    daily_points[t_point][1].append(latss_area[y][i][k])  # 存 lat
                    
    # **计算受影响和未受影响的区域**
    excyc_impact_days = ice.where((ice > 0) & (ice <= 1))
    no_excyc_impact_days = excyc_impact_days.copy()
    
    # **对每天的数据计算受影响区域**
    for t_point, (lon_values, lat_values) in daily_points.items():
        print(t_point)
        if np.datetime64(t_point) not in ice_times:
            continue  # 如果时间不在海冰数据中，则跳过
        
        

        ice_time_slice = ice.sel(time=t_point)  # 选取当天的海冰数据
        
        lon_values = [(lon - 360) if lon > 180 else lon for lon in lon_values]
        
        # 初始化一个全False的掩码
        combined_within_radius = np.zeros_like(lon_da, dtype=bool)  # 假设lon_da是海冰网格的经度坐标
        
        # 遍历所有轨迹点，合并掩码
        for i in range(len(lon_values)):
            distances = haversine(lon_da, lat_da, lon_values[i], lat_values[i])
            within_radius = distances <= 750
            combined_within_radius = combined_within_radius | within_radius  # 逻辑或操作
        
        excyc_impact_days.loc[{"time": t_point}] = ice_time_slice.where(combined_within_radius)
        no_excyc_impact_days.loc[{"time": t_point}] = ice_time_slice.where(~combined_within_radius)
        print(i)

    return excyc_impact_days, no_excyc_impact_days


excyc_impact_days_icemin2,no_excyc_impact_days_icemin2=cyclone_impact_day(ice_daily_2010, timetick[0],lonss[0],latss[0])

excyc_impact_days_icemax2,no_excyc_impact_days_icemax2=cyclone_impact_day(ice_daily_1980, timetick[1],lonss[1],latss[1])


def cyclone_impact_day(ice, timetick_, lonss_area, latss_area):
    impact_mask = {}  # 用字典存储受影响的区域，避免创建大数组
    ice_times = ice['time'].values  # 获取所有时间点

    # **合并所有小时数据为每日数据**
    daily_points = {}  # {日期: (lon_list, lat_list)}

    for y in range(len(timetick_)):  # 遍历年份
        print(f"Processing y: {y}")
        for j in range(len(timetick_[y][0])):  # 遍历不同气旋
            # print(f"Processing j: {j}")

            for i in range(len(timetick_[y][0][j])):  # 遍历气旋的轨迹点
                for k in range(len(timetick_[y][0][j][i])):

                    timedata = pd.Timestamp(datetime(1979, 1, 1, 0, 0, 0) + timedelta(hours=timetick_[y][0][j][i][k]))
                    t_point = pd.Timestamp(timedata.date())  # 取日期

                    # 存储该日期下的 lon, lat
                    if t_point not in daily_points:
                        daily_points[t_point] = ([], [])  # 初始化
                    if 230<=lonss_area[y][j][i][k]<=300 and latss_area[y][j][i][k]<=-50:
                        daily_points[t_point][0].append(lonss_area[y][j][i][k])  # 存 lon
                        daily_points[t_point][1].append(latss_area[y][j][i][k])  # 存 lat
                    
    # **计算受影响和未受影响的区域**
    excyc_impact_days = ice.where((ice > 0) & (ice <= 1))
    no_excyc_impact_days = excyc_impact_days.copy()
    
    # **对每天的数据计算受影响区域**
    for t_point, (lon_values, lat_values) in daily_points.items():
        print(t_point)
        if np.datetime64(t_point) not in ice_times:
            continue  # 如果时间不在海冰数据中，则跳过

        ice_time_slice = ice.sel(time=t_point)  # 选取当天的海冰数据
        
        lon_values = [(lon - 360) if lon > 180 else lon for lon in lon_values]
        
        # 初始化一个全False的掩码
        combined_within_radius = np.zeros_like(lon_da, dtype=bool)  # 假设lon_da是海冰网格的经度坐标
        
        # 遍历所有轨迹点，合并掩码
        for i in range(len(lon_values)):
            distances = haversine(lon_da, lat_da, lon_values[i], lat_values[i])
            within_radius = distances <= 750
            combined_within_radius = combined_within_radius | within_radius  # 逻辑或操作
        
        excyc_impact_days.loc[{"time": t_point}] = ice_time_slice.where(combined_within_radius)
        no_excyc_impact_days.loc[{"time": t_point}] = ice_time_slice.where(~combined_within_radius)
        print(i)
    return excyc_impact_days, no_excyc_impact_days

excyc_impact_days_ice2,no_excyc_impact_days_ice2=cyclone_impact_day(ice_all, timetick2_all,lonsc2_all,latsc2_all)

excyc_impact_days_icemin2[10].plot()
allyearno_days_mean_ice2.sel(month_day='10-06').plot()

def selected_cycsic(sic_days):
    # 添加 'month_day' 坐标
    # sic_days=excyc_impact_days_icemin2.copy()
    sic_days = sic_days.assign_coords(month_day=('time', sic_days['time'].dt.strftime('%m-%d').data))
    
    # 对每个 'month_day' 的数据，按年份求平均，然后再对所有年份求平均
    allyear_days_mean = sic_days.groupby('month_day').map(
        lambda x: x.groupby('time.year').mean(dim='time')
    ).mean(dim='year')
    
    return allyear_days_mean

# no_excyc_impact_days_ice2.assign_coords(month_day=('time',  no_excyc_impact_days_ice2['time'].dt.strftime('%m-%d').data)).groupby('month_day').map(
#     lambda x: x.groupby('time.year').mean(dim='time')
# )

#有气旋影响的日数
excyc_impact_days_icemax2_whole=excyc_impact_days_icemax2.reindex(time=no_excyc_impact_days_icemax2.time.values,method='nearest',tolerance=0)#补齐时间维
excyc_impact_days_icemin9_whole=excyc_impact_days_icemin9.reindex(time=no_excyc_impact_days_icemin9.time.values,method='nearest',tolerance=0)#补齐时间维

allyear_days_mean_icemin2=selected_cycsic(excyc_impact_days_icemin2)
allyear_days_mean_icemax2=selected_cycsic(excyc_impact_days_icemax2)
# allyear_days_mean_icemin9=selected_cycsic(excyc_impact_days_icemin9_whole)
# allyear_days_mean_icemax9=selected_cycsic(excyc_impact_days_icemax9) 

# allyearno_days_mean_ice9=selected_cycsic(no_excyc_impact_days_ice9)
allyearno_days_mean_ice2=selected_cycsic(no_excyc_impact_days_ice2)


diff_icemin2=allyear_days_mean_icemin2-allyearno_days_mean_ice2
diff_icemax2=allyear_days_mean_icemax2-allyearno_days_mean_ice2#[:-1]时间维不同，早期海冰数据为两天一数据，需要去掉气候态的最后一天

allyear_days_mean_icemin2
icemean_all.groupby('time.month') - icemean_mon_80
# 创建 day->month 的映射
month_from_day = allyear_days_mean_icemin2['month_day'].str.slice(0, 2).astype(int).data
allyear_days_mean_icemin2.coords['month'] = ('month_day', month_from_day)

month_from_day = allyear_days_mean_icemax2['month_day'].str.slice(0, 2).astype(int).data
allyear_days_mean_icemax2.coords['month'] = ('month_day', month_from_day)

# 广播 monthly_climatology 到 day 维度
clim_matched_10 = xr.apply_ufunc(
    lambda m: icemean_mon_80.sel(month=m),
    allyear_days_mean_icemin2['month'],
    vectorize=True,
    dask='parallelized',
    input_core_dims=[[]],
    output_core_dims=[['y', 'x']],
    output_dtypes=[allyear_days_mean_icemin2.dtype]
)

clim_matched_80 = xr.apply_ufunc(
    lambda m: icemean_mon_80.sel(month=m),
    allyear_days_mean_icemax2['month'],
    vectorize=True,
    dask='parallelized',
    input_core_dims=[[]],
    output_core_dims=[['y', 'x']],
    output_dtypes=[allyear_days_mean_icemax2.dtype]
)
# 得到气旋对应时间的 SIC 异常（C_SIC_anom）
c_sic_anom_min = allyear_days_mean_icemin2.where((clim_matched>0) & (clim_matched <1)) - clim_matched.where((clim_matched>0) & (clim_matched <1))
c_sic_anom_max = allyear_days_mean_icemax2.where((clim_matched>0) & (clim_matched <1)) - clim_matched.where((clim_matched>0) & (clim_matched <1))


diff_4=[diff_icemin2.mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130)),diff_icemax2.mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130)),diff_icemin9.mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130)),diff_icemax9.mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130))]


diff_icemin2_n=diff_icemin2.sel(month_day=slice('02-01','02-29')).mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130))
diff_icemax2_n=diff_icemax2.sel(month_day=slice('03-01','03-31')).mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130))

diff_icemin2_n_=c_sic_anom_min.sel(month_day=slice('02-01','02-29')).mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130))
diff_icemax2_n_=c_sic_anom_max.sel(month_day=slice('03-01','03-31')).mean(dim='month_day').where((lon_da<=-60) & (lon_da>=-130))

diff_2=[diff_icemin2_n.compute(),diff_icemax2_n.compute()]
diff_2_=[diff_icemin2_n_.compute(),diff_icemax2_n_.compute()]


# ano_diff=allyear_days_mean_icemin2-allyear_days_mean_ice2     #.where((diff_icemin9.mean(dim='time')>0.15)|(diff_icemin9.mean(dim='time')<-0.15))
#显著性检验
allyear_days_mean_ice=[allyear_days_mean_icemin2.sel(month_day=slice('02-01','02-29')),allyear_days_mean_icemax2.sel(month_day=slice('03-01','03-31'))]
allyearno_days_mean_ice=[allyearno_days_mean_ice2.sel(month_day=allyear_days_mean_ice[0].month_day),allyearno_days_mean_ice2.sel(month_day=allyear_days_mean_ice[1].month_day)]
    

labels = ['Feb. SIEmin', 'Feb. SIEmax', 'Sept. SIEmin', 'Sept. SIEmax']    
allyear_days_mean_ice[1].month_day


# 定义 t 检验函数
def ttest_func(cyclone_values, non_cyclone_values):
    t_stat, p_value = stats.ttest_ind(
        cyclone_values,
        non_cyclone_values,
        
        equal_var=False,
        nan_policy='omit'
    )
    return p_value

# 使用 xr.apply_ufunc 进行 t 检验

p_values = np.full((2, 332, 316), np.nan)
for k in range(2):
    allyear_days_mean_ice[k]=allyear_days_mean_ice[k].chunk({"month_day": -1})
    allyearno_days_mean_ice[k]=allyearno_days_mean_ice[k].chunk({"month_day": -1})
    p_values[k, :, :] = xr.apply_ufunc(
        ttest_func,
        allyear_days_mean_ice[k],
        allyearno_days_mean_ice[k],
        input_core_dims=[['month_day'], ['month_day']],
        output_core_dims=[[]],
        exclude_dims=set(('month_day',)),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    ).data
    
p_val= xr.DataArray(
    p_values,
    coords=[ ('type', labels[:2]),('y', y_data.data), ('x', x_data.data)],
    name='p_values'
)
significance_mask = p_val < 0.05
# 如果需要，可以将 significance_mask 保存或进一步处理

density_levels1=np.linspace(-50, 50,21)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

plot(diff_2, density_levels1)

plot(no_excyc_impact_days_ice2[0], density_levels1)
plot(diff_icemin9.mean(dim='time'), density_levels1)
plot(ice2[0], density_levels1)

diff_2_da=xr.concat(diff_2, dim='time')
diff_2_da.to_netcdf("/Volumes/Elements/大气要素/diff_2.nc")
a=xr.open_dataset('/Volumes/Elements/大气要素/diff_2.nc')['N07_ICECON']
diff_2[1].where(diff_2[1]!=0).plot()


diff_2[0].where((clim_matched[0]>0) & (clim_matched[0] <1)& (south_grid==5)).mean()



#%%sie

#44年每个月的sie序列
sie_all_clim=[]
# sic_mon_abs=[]
for i in range(len(icemean_all)):#filled_icemean_mon
    mask = (filled_icemean_mon[i] >= 0.15) & (filled_icemean_mon[i] <= 1) & (south_grid==5)
    filtered_cell_area = cell_area.where(mask)
    # sic_mon_abs.append(filtered_cell_area)
    # 计算符合条件的总面积（单位：平方公里）
    sie_all_clim.append(filtered_cell_area.sum(skipna=True).item() / 1e6  )# 将面积从平方米转换为平方公里
 
sic_ex_abs=icemean_all.where((south_grid==5)& (filled_icemean_mon[i] <= 1)).sel(time=min_dates[1:].values)


#去趋势
def detrend_series(series):
    x = np.arange(len(series))
    slope, intercept, _, _, _ = stats.linregress(x, series)
    trend = slope * x + intercept
    return series - trend

detrend_sic = xr.apply_ufunc(
    safe_detrend,
    sic_ex_abs,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[sic_ex_abs.dtype],
)

# 1) 构造一个与 sie_all 对应的时间索引 (1980/01 到 2023/12, 528 个月)
time_index = pd.date_range(start='1979-01', periods=len(sie_all), freq='MS') 
time_index[12]
# freq='MS' 表示每个月开始那一天

# 2) 创建一个 pandas Series 来保存月度海冰范围
sie_series = pd.Series(data=sie_all, index=time_index, name='SIE')
# groupby(sie_series.index.year) 可以按年份分组
yearly_max = sie_series.groupby(sie_series.index.year).max()  # 每年的月最大值
yearly_min = sie_series.groupby(sie_series.index.year).min()  # 每年的月最小值

plt.plot(range(44), detrend_sie)
# 确保 sie_all_clim 是 numpy 数组或 pandas Series
sie_all_clim = np.array(sie_all_clim[:12])  # 月度气候态数据


# 找到每年最大值和最小值的月份索引
max_dates = sie_series.groupby(sie_series.index.year).idxmax()  # 最大值对应的时间点
min_dates = sie_series.groupby(sie_series.index.year).idxmin()  # 最小值对应的时间点

ice_monmin=icemean_all.sel(time = icemean_all['time'].isin(min_dates[1:]))

ice_minano=[i-icemean_mon_80.sel(month = i.time.dt.month) for i in ice_monmin]
ice_minano=xr.concat(ice_minano, dim='time')
ice_minnaom=ice_minano.mean(dim='time')
ice_minano[0].plot()



ice_monminstd=ice_monmin.std(dim='time')
ice_monminstd.plot()


min_dates.dt.month.value

yearly_min_2=[]
yearly_min_3=[]
for i in range(len(min_dates)):
    if min_dates.iloc[i].month ==2:
        yearly_min_2.append(yearly_min.index[i])
    elif min_dates.iloc[i].month ==3:
        yearly_min_3.append(yearly_min.index[i])
       
# 2. 计算每年最大值和最小值的异常偏差
yearly_max_anomaly = yearly_max - sie_all_clim[max_dates.dt.month - 1]
yearly_min_anomaly = yearly_min - sie_all_clim[min_dates.dt.month - 1]

# 3. 按偏差排序，获取最大和最小的偏差年份
# 年最大值偏高的5年
top_5_max_high = yearly_max_anomaly.nlargest(10)
# 年最大值偏低的5年
top_5_max_low = yearly_max_anomaly.nsmallest(5)

# 年最小值偏高的5年
top_5_min_high = yearly_min_anomaly.nlargest(5)
# 年最小值偏低的5年
top_5_min_low = yearly_min_anomaly.nsmallest(5)


top5_min = yearly_min.nlargest(10)

bottom5_min = yearly_min.nsmallest(10)

# 年最小值的前 5 大：
# 1980    987545.577726
# 1983    953709.516340
# 1982    794710.355688
# 1987    794358.166660
# 1986    791372.944967
# Name: SIE, dtype: float64

# 年最小值的前 5 小：
# 2010    167202.785242
# 2023    235597.576390
# 2013    239479.023991
# 2017    261026.029010
# 2008    306186.435122
# Name: SIE, dtype: float64
