#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:46:01 2026

@author: peiyilin
"""

#%%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

#eof分析
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
import xarray as xr
from eofs.standard import Eof
import numpy as np
import xarray as xr
import matplotlib.patches as mpatches

def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_lat_labels_on_sector(ax, lat_values, label_lon, 
                             fontsize=14, fontweight='normal',
                             ha='left', va='bottom', offset=0):
    """
    添加纬度标签到极区扇形地图中。

    参数：
    - ax: cartopy axes
    - lat_values: 纬度值列表（负值，例如 [-60, -70, -80]）
    - label_lon: 标签放-置在哪个经度（推荐扇形中轴）
    - offset: 纬度微调（正值往外移，单位°）
    """
    for lat in lat_values:
        ax.text(
            label_lon, -lat + offset,
            f"{lat}°S",
            transform=ccrs.PlateCarree(),
            fontsize=fontsize,
            fontweight=fontweight,
            ha=ha,
            va=va,
            zorder=10
        )

def add_lon_labels_on_sector(ax, lon_values, label_lat,
                             fontsize=14, fontweight='normal',
                             ha='right', va='top', offset=0):
    """
    添加经度标签到极区扇形地图中。

    参数：
    - ax: cartopy axes
    - lon_values: 经度值列表（例如 [-120, -90, -60, -30, 0]）
    - label_lat: 标签在什么纬度圈（通常是 -60、-65）
    - offset: 纬度微调（正值更远离南极，单位°）
    """
    for lon in lon_values:
        ax.text(
            -lon, label_lat + offset,
            f"{lon}°W",
            transform=ccrs.PlateCarree(),
            fontsize=fontsize,
            fontweight=fontweight,
            ha=ha,
            va=va,
            zorder=10
        )
def add_rectangle(ax, lon_center, lat_center, dlon=10, dlat=5, color='red', label=None):
    rect = mpatches.Rectangle(
        (lon_center - dlon, lat_center - dlat),   # 左下角 (lon, lat)
        2 * dlon,                                 # 宽度
        2 * dlat,                                 # 高度
        linewidth=2, edgecolor=color, facecolor='none',
        linestyle='-', transform=ccrs.PlateCarree(), zorder=4, label=label
    )
    ax.add_patch(rect)
    ax.plot(lon_center, lat_center, 'o', color=color, transform=ccrs.PlateCarree(), markersize=6)

#%%
dlatss2_all = []  # 存储5个月结果
for j in range(len(latsc2_all)):  # 遍历5个月
    monthly_dlat = []  #每个月结果
    for m in range(len(latsc2_all[j])):  # 遍历44年
        yearly_dlat= []  # 每年的轨迹结果
        for k in range(len(latsc2_all[j][m])):  # 遍历轨迹
            lon_array = np.array(lonsc2_all[j][m][k])
            lat_array = np.array(latsc2_all[j][m][k])
            t = np.array(timetick2_all[j][0][m][k])
            # 检查是否为空轨迹
            # if len(lon_array) == 0 or len(lat_array) == 0 or len(t) == 0:
            #     continue
            # # 筛选经度范围内的数据
            # # valid_indices = np.where((lon_array >= minlo_) & (lon_array <= maxlo_))[0]

            # # 检查有效索引是否存在
            # if len(valid_indices) > 0:
            #     first_pidx = valid_indices[0]
            #     last_pidx = valid_indices[-1]

            #     # 确保范围内有变化
            #     if lat_array[first_pidx] != lat_array[last_pidx]:
            #         # 计算纬度变化速率
            dlat = []
            for l in range(len(lon_array)):
                
                if l<len(lon_array)-1: 
                    time_diff = t[l + 1] - t[l]
                    if time_diff > 0: # 确保时间差非零
                        # 使用 haversine 计算距离并归一化时间差
                        distance = haversine(lon_array[l], lat_array[l], 
                                             lon_array[l + 1], lat_array[l + 1])
                        dlat.append(distance / time_diff)
                else:
                    dlat.append(np.nan)
            # 如果计算出有效速率，保存结果
            if dlat:
                yearly_dlat.append(dlat)
        
        # 计算年度平均值，如果无结果，则返回 NaN
        monthly_dlat.append(yearly_dlat if yearly_dlat else np.nan)
    
    # 将每月的结果存入最终结果
    dlatss2_all.append(monthly_dlat)

# 44年

timetick_area_tina2_ = []
for i in range(44):
    monthly_valid_points = defaultdict(list)
    for j in range(len(lonsc2_all[i])):
       
        for k in range(len(lonsc2_all[i][j])):

            for l in range(len(lonsc2_all[i][j][k])):
               tin = datetime.datetime(1979, 1, 1, 0, 0, 0)+timedelta(hours=timetick2_all[i][0][j][k][l])
               month=tin.month
               # if 230<=lonsc2_all[i][j][k][l]<=300 :
                   # 可以只存时间，也可以加上经纬度等信息
               monthly_valid_points[month].append({
    
                    "lon": lonsc2_all[i][j][k][l],
                    "lat": latsc2_all[i][j][k][l],
                    # 'Intensity':mslp2_all[i][j][k][l],
                    # 'Speed':dlatss2_all[i][j][k][l], 
                    # "cyclone_id": f"{j}_{k}",
                    "time": tin
                })
             
    timetick_area_tina2_.append(monthly_valid_points)



#%%
# 500/haversine(230,-50,231,-50)
# 定义网格范围
lon_min, lon_max = 230, 300
lat_min, lat_max = -80, -55
grid_resolution = 1

num_lon_bins = int((lon_max - lon_min) / grid_resolution)
num_lat_bins = int((lat_max - lat_min) / grid_resolution)

# 创建网格点的经纬度
grid_lons = lon_min + np.arange(num_lon_bins + 1) * grid_resolution
grid_lats = lat_min + np.arange(num_lat_bins + 1) * grid_resolution
# grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)  # 生成网格点
rr=750
lon_margins = rr / (111 * np.cos(np.radians(grid_lats)))
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



sepmon=[10, 11, 12, 1, 2,3]

def calden(latsc_all,timetick_a):
    # 新增网格变量：存储强度和移速的总和及计数

    sum_intensity_grid = np.zeros((6, len(latsc_all), num_lat_bins + 1, max_nlon))  # 强度总和
    sum_speed_grid = np.zeros((6, len(latsc_all), num_lat_bins + 1, max_nlon))  # 移速总和
    count_grid = np.zeros((6, len(latsc_all), num_lat_bins + 1, max_nlon))  # 点数量
    
    
    for i in range(0,len(latsc_all)):
        m=0 
        for mon,records in timetick_a[i].items():
            
            if mon in sepmon:
                
                
                # 拆解轨迹点属性
                lons = np.array([r['lon'] for r in records])
                lats = np.array([r['lat'] for r in records])
                intensities = np.array([r['Intensity'] for r in records])
                speeds = np.array([r['Speed'] for r in records])
                # cids = [r['cyclone_id'] for r in records]  # 若后续想用
        
                # 广播格点和轨迹点，计算所有距离（假设 haversine 支持广播）
                dist = haversine(grid_lon[..., np.newaxis], grid_lat[..., np.newaxis], lons, lats)
        
                # 计算哪些轨迹点在750km范围内
                in_range = dist < rr  # shape: (grid_lat, grid_lon, n_points)
        
                # 统计格点内的数量、强度、速度
                count = np.sum(in_range, axis=-1)
                intensity_sum = np.sum(in_range * np.nan_to_num(intensities), axis=-1)
                speed_sum = np.sum(in_range * np.nan_to_num(speeds), axis=-1)
        
                # 累加到网格
                count_grid[m, i] += count
                print(mon,m)
                sum_intensity_grid[m, i] += intensity_sum
                sum_speed_grid[m, i] += speed_sum    
                m+=1
       
        
    # 计算每个格点的平均值（避免除零错误）
    average_intensity_grid = np.where(
        count_grid > 0, 
        sum_intensity_grid / count_grid, 
        np.nan  # 没有点的格点填充为 NaN
    )
    
    average_speed_grid = np.where(
        count_grid > 0, 
        sum_speed_grid / count_grid, 
        np.nan
    )
    #没有气旋的地方设置为nan
    count_grid = np.where(count_grid == 0, np.nan, count_grid)
    return average_intensity_grid,average_speed_grid,count_grid

average_intensity_grid_2,average_speed_grid_2,count_grid_2=calden(latsc2_all,timetick_area_tina2)
# average_intensity_grid_9,average_speed_grid_9,count_grid_9=calden(latsc9_all, tracksy9_all)#(5,44,n,m)
b = min_dates[1:].dt.month + 2
m = min_dates[1:].dt.month 
# average_intensity_grid_2_a= average_intensity_grid_2-np.mean(average_intensity_grid_2[b.values, np.arange(44),:,:], axis=0, keepdims=True)  # 计算每个空间点的平均值
#%%
#去趋势
def detrend_3d_field(X):
    """
    X: (T, Y, X) numpy array
    return: X_dt same shape, linear detrended along time at each grid point
    """
    T, Ny, Nx = X.shape
    t = np.arange(T, dtype=float)

    # reshape to (T, P)
    X2 = X.reshape(T, -1)
    mask = np.isfinite(X2)

    X_dt = np.full_like(X2, np.nan, dtype=float)

    # 对每个格点做回归：y = a*t + b，仅用有限值
    for p in range(X2.shape[1]):
        m = mask[:, p]
        if m.sum() < 10:   # 44年数据，10 就够了；你也可以设 30 更严格
            continue
        y = X2[m, p]
        tt = t[m]
        a, b = np.polyfit(tt, y, 1)
        trend = a * t + b
        X_dt[m, p] = X2[m, p] - trend[m]  # 只在有效值处写回

    return X_dt.reshape(T, Ny, Nx)

average_intensity_grid_2 = detrend_3d_field(average_intensity_grid_2[b.values, np.arange(44),:,:])
average_speed_grid_2 = detrend_3d_field(average_speed_grid_2[b.values, np.arange(44),:,:])
count_grid_2 = detrend_3d_field(count_grid_2[b.values, np.arange(44),:,:])

average_intensity_grid_2_a= average_intensity_grid_2-np.mean(average_intensity_grid_2, axis=0, keepdims=True)
average_speed_grid_2_a= average_speed_grid_2-np.mean(average_speed_grid_2, axis=0, keepdims=True)
count_grid_2_a= count_grid_2-np.mean(count_grid_2, axis=0, keepdims=True)


coslat = np.cos(np.deg2rad(np.array(grid_lats)))
wgts = np.sqrt(coslat)[..., np.newaxis]


#%%

#去趋势
solver1 = Eof(average_intensity_grid_2_a,weights = wgts)#自动做距平

eof = solver1.eofs(neofs=3, eofscaling=2)#AsCorrelation

pc = solver1.pcs(npcs=3, pcscaling=1)
# eof1 = solver.eofsAsCorrelation(neofs=1)
# eof
var = solver1.varianceFraction()
errors = solver1.northTest(neigs=5, vfscaled=True)
sum(var[:2])
# count_grid_2_a[4][9]

#计算特征值和NorthTest的结果
eigenvalues =solver1.eigenvalues(neigs=40)
errors = solver1.northTest(neigs=40)
# 绘制Error Bar图
plt.figure(figsize=(8, 6),dpi=200)
plt.errorbar(np.arange(1, 41), eigenvalues, yerr=errors, fmt='o-', capsize=5)
plt.xlabel('EOF modes')
plt.ylabel('Eigenvalues')
plt.title('NorthTest(with error bars)')
plt.grid()
plt.show()


#%%定义偶极子

def _gauss2d_keepnan(da2d, sigma=1.5):
    """2D 高斯平滑但保持 NaN 为空（加权归一）"""
    arr = da2d.values
    m = np.isnan(arr)
    a0 = arr.copy(); a0[m] = 0.0
    w0 = (~m).astype(float)
    a = ndi.gaussian_filter(a0, sigma=sigma, mode='nearest')
    w = ndi.gaussian_filter(w0, sigma=sigma, mode='nearest')
    out = a / np.maximum(w, 1e-12)
    out[m] = np.nan
    return xr.DataArray(out, coords=da2d.coords, dims=da2d.dims)

# 假设 eof[1, :, :] 与 grid_lat、grid_lon 形状完全一致
arr_ = xr.DataArray(
    eof[1, :, :],
    dims=("y", "x"),                      # 先给数据起两个维度名
    name="EOF2",
)
# 把 2D 的经纬度作为坐标挂到这两个维度上
arr_ = arr_.assign_coords(
    lat=(("y", "x"), grid_lat),
    lon=(("y", "x"), grid_lon),
)



#重新定义一个区域来计算气旋强度（存疑
I_da = xr.DataArray(
    average_intensity_grid_2_a,#[b.values, np.arange(44),:,:],   #average_intensity_grid_2_a,#
    dims=("time", "y", "x"),                      # 先给数据起两个维度名
    name="ci",
)
# 把 2D 的经纬度作为坐标挂到这两个维度上
I_da = I_da.assign_coords(
    lat=(("y", "x"), grid_lat),
    lon=(("y", "x"), grid_lon),
)

lat2d = arr_['lat']
lon2d = arr_['lon']

# 在 230–300E 内找 EOF2 的正/负极值（避免全 NaN 报错）
roi = (lon2d % 360 >= 230) & (lon2d % 360 <= 300)
#%%

from scipy.stats import t
from numpy.linalg import lstsq

def lag1_autocorr(a):
    """返回一维序列的 lag-1 自相关（忽略 NaN）。"""
    a = np.asarray(a, float)
    m = np.isfinite(a)
    a = a[m]
    if a.size < 3:
        return np.nan
    a0, a1 = a[:-1], a[1:]
    a0 -= a0.mean(); a1 -= a1.mean()
    denom = (a0.std(ddof=1) * a1.std(ddof=1))
    if denom == 0:
        return np.nan
    return np.dot(a0, a1) / ((a0.size-1) * denom)

def effective_dof_ar1(pc, y):
    """
    Bretherton 等的 Neff 估计：Neff = N * (1 - r1x*r1y) / (1 + r1x*r1y)
    pc: (T,)
    y : (T,) 单格点时间序列
    """
    N = np.isfinite(pc).sum()
    r1x = lag1_autocorr(pc)
    r1y = lag1_autocorr(y)
    if not np.isfinite(r1x): r1x = 0.0
    if not np.isfinite(r1y): r1y = 0.0
    num = (1 - r1x*r1y)
    den = (1 + r1x*r1y)
    if den <= 0:
        return N  # 退化时回退
    Neff = N * num/den
    # 合理边界
    return max(3, min(N, Neff))

def regress_grid(field, pc2, use_neff=True):
    """
    对 field(time, y, x) 用 pc2(time) 做逐格点回归/相关并计算 p 值。
    返回：beta(回归斜率), r(相关), p(p值), Neff(有效自由度)
    """
    # 对齐时间
    if 'time' not in field.dims:
        raise ValueError("field 必须含有 time 维度")
    # 若 pc2 是 xarray，确保时间对齐
    if isinstance(pc2, xr.DataArray):
        # 内联合并
        field, pc2 = xr.align(field, pc2, join='inner')
        pc = pc2.values
    else:
        pc = np.asarray(pc2)

    T, ny, nx = field.sizes['time'], field.sizes[field.dims[1]], field.sizes[field.dims[2]]
    beta   = np.full((ny, nx), np.nan)
    corr   = np.full((ny, nx), np.nan)
    pval   = np.full((ny, nx), np.nan)
    neff   = np.full((ny, nx), np.nan)

    pc_std = np.nanstd(pc, ddof=1)
    pc_c   = pc - np.nanmean(pc)

    # 遍历格点（矢量化复杂，稳妥起见用循环；44×(ny*nx)通常很快）
    fld = field.values  # (T, ny, nx)
    for j in range(ny):
        y = fld[:, j, :]
        # 掩蔽全 NaN 列
        valid_col = np.isfinite(y).sum(axis=0) >= 3
        if not np.any(valid_col):
            continue
        for i in np.where(valid_col)[0]:
            yi = y[:, i]
            m  = np.isfinite(yi) & np.isfinite(pc_c)
            if m.sum() < 3:
                continue
            yi_c = yi[m] - yi[m].mean()
            xi   = pc_c[m]
            # 斜率（最小二乘）：beta = cov(x,y)/var(x)
            varx = np.dot(xi, xi) / (xi.size - 1)
            cov  = np.dot(xi, yi_c) / (xi.size - 1)
            if varx == 0:
                continue
            b = cov / varx
            beta[j, i] = b

            # 相关 r
            sy = yi_c.std(ddof=1)
            if pc_std > 0 and sy > 0:
                r = cov / (pc_std * sy)
                corr[j, i] = np.clip(r, -1, 1)
            else:
                corr[j, i] = np.nan

            # 有效自由度与 p 值
            if use_neff:
                Neff = effective_dof_ar1(pc_c[m], yi_c)
            else:
                Neff = m.sum()
            neff[j, i] = Neff

            if np.isfinite(corr[j, i]) and Neff > 3:
                rr = corr[j, i]
                tstat = rr * np.sqrt((Neff - 2) / (1 - rr**2 + 1e-12))
                p = 2 * (1 - t.cdf(abs(tstat), df=Neff - 2))
                pval[j, i] = p

    # 打包为 DataArray（复制坐标）
    coords2d = {}
    if 'lat' in field.coords and field['lat'].ndim == 2:
        coords2d['lat'] = (('y','x'), field['lat'].values)
    if 'lon' in field.coords and field['lon'].ndim == 2:
        coords2d['lon'] = (('y','x'), field['lon'].values)

    da_beta = xr.DataArray(beta, dims=('y','x'), coords=coords2d, name='beta')
    da_r    = xr.DataArray(corr, dims=('y','x'), coords=coords2d, name='r')
    da_p    = xr.DataArray(pval, dims=('y','x'), coords=coords2d, name='p')
    da_neff = xr.DataArray(neff, dims=('y','x'), coords=coords2d, name='Neff')
    return xr.Dataset(dict(beta=da_beta, r=da_r, p=da_p, Neff=da_neff))

def plot_reg_with_significance(out, title="PC2 regression on FIELD",
                               vcenter=True, clim=None, p_thr=0.05,
                               extent=(230, 300, -80, -50)):
    """
    out: regress_grid 的输出 Dataset
    vcenter: True 则色标以 0 为中心（发散色图）
    clim: (vmin, vmax)，不传自动用 98 百分位
    extent: (lon_min, lon_max, lat_min, lat_max)
    """
    proj = ccrs.SouthPolarStereo()
    data_crs = ccrs.PlateCarree()

    beta = out['beta']   # 也可换成 out['r']
    p    = out['p']

    # 值域
    fld = beta.values
    if clim is None:
        vmax = np.nanpercentile(np.abs(fld), 98)
        vmin = -vmax if vcenter else np.nanpercentile(fld, 2)
        vmax =  vmax if vcenter else np.nanpercentile(fld, 98)
    else:
        vmin, vmax = clim

    fig = plt.figure(figsize=(7.8, 6.5), dpi=220)
    ax  = plt.subplot(1,1,1, projection=proj)

    ax.add_feature(cfeature.LAND, facecolor='#E7D9B5', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#444', zorder=2)
    ax.set_extent(extent, crs=data_crs)

    # 主图：回归斜率
    pcm = ax.contourf(beta['lon'], beta['lat'], beta,
                      levels=21, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                      transform=data_crs, extend='both')

    # 显著性：p < 0.05 的点打小点（也可 stippling）
    sig = (p < p_thr)
    # 为避免 NaN 经纬度报错
    m = sig.values & np.isfinite(beta['lon'].values) & np.isfinite(beta['lat'].values)
    ax.scatter(beta['lon'].values[m], beta['lat'].values[m],
               s=1.2, color='k', alpha=0.6, transform=data_crs)

    # 网格 + 色标
    ax.gridlines(draw_labels=False, crs=data_crs, color='#666', ls='--', lw=0.4, alpha=0.35)
    cb = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.04, shrink=0.90)
    cb.set_label('Regression slope (FIELD per PC2 unit)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# 示例调用
# plot_reg_with_significance(out, title="PC2 → Cyclone Intensity Anom (regression)")
# 1) 逐格点回归 / 相关 / p 值（考虑 AR(1) 有效自由度）
out = regress_grid(field=I_da, pc2=pc[:, 1], use_neff=True)
out  # 里有 out['beta'], out['r'], out['p'], out['Neff']
plot_reg_with_significance(
    out,
    title="PC2 → Cyclone Intensity anomaly (regression)",
    vcenter=True,                 # 以 0 居中，发散色标
    p_thr=0.01,                   # 显著性阈值（格点打点）
    extent=(230, 300, -80, -50)   # 你的 ABS 区域
)


p_val = out['p']
arr_.plot()
#%%

# arr = _gauss2d_keepnan(arr_, sigma=3)
A = arr_.where(roi)
A.plot()
if np.all(np.isnan(A)):
    raise ValueError("ROI 内全为 NaN，无法定位极值")

# 正极值（限制在 ROI 内）
pos_flat = np.nanargmax(A.values)
neg_flat = np.nanargmin(A.values)
pos_idx = np.unravel_index(pos_flat, A.shape)
neg_idx = np.unravel_index(neg_flat, A.shape)

lat_pos, lon_pos = lat2d[pos_idx], lon2d[pos_idx]
lat_neg, lon_neg = lat2d[neg_idx], lon2d[neg_idx]


latE_ref, lonE_ref = lat2d[pos_idx], lon2d[pos_idx]
latW_ref, lonW_ref = lat2d[neg_idx], lon2d[neg_idx]

# ---- 区域与掩膜  ---
lat2d = I_da['lat']  # (y,x)
lon2d = I_da['lon']  # (y,x)

w2d = xr.apply_ufunc(lambda x: np.cos(np.deg2rad(x)), I_da['lat'])


def reg_mean(field, mask,  min_frac=0.8):
    """
    计算区域加权平均值，如果有效格点比例 < min_frac，则返回 NaN
    """
    # 总格点数（区域内）
    total_pts = np.count_nonzero(mask)

    # 有效格点数（非 NaN 且 mask=True）
    valid_pts = np.count_nonzero(mask & np.isfinite(field))

    # 有效比例
    frac = valid_pts / total_pts if total_pts > 0 else 0.0

    if frac < min_frac:
        return np.nan  # 有效值太少，直接返回 NaN

    # 否则正常计算加权平均
    num = (field.where(mask) * w2d).sum(dim=('y','x'))
    den = (w2d.where(mask)).sum(dim=('y','x'))
    return num / den


mask_pos = ((I_da['lon']>=lon_pos-10)&(I_da['lon']<=lon_pos+10) &
                (I_da['lat']>= lat_pos-5)&(I_da['lat']<=lat_pos+5))
mask_neg = ((I_da['lon']>=lon_neg-10)&(I_da['lon']<=lon_neg+10) &
      (I_da['lat']>= lat_neg-5)&(I_da['lat']<=lat_neg+5))
 
mask_neg.plot()

left = reg_mean(I_da, mask_pos)
right = reg_mean(I_da, mask_neg)

dci = left - right

dci.plot()

r, p = pearsonr(dci, detrend_sie)
pearsonr(dci, pc[:,1])


#%%
#figure8

df = pd.DataFrame({"year": range(1980,2024), "DCI": dci, "SIE": detrend_sie/1e+5}).dropna().reset_index(drop=True)

# 标准化 DCI，用于阈值与叠画
dci_mean, dci_std = df["DCI"].mean(), df["DCI"].std(ddof=0)
df["DCI_z"] = (df["DCI"] - dci_mean) / dci_std

# ============== A) 箱形图：DCI>0 vs DCI<0 的 SIE 分布 ==============
df["DCI_sign"] = np.where(df["DCI"] >= 0, "DCI > 0", "DCI < 0")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.),sharey=True, dpi=600,gridspec_kw={'width_ratios': [1., 5]})

# ========= A) 箱形图 =========
ax0 = axes[0]
sns.boxplot(
    data=df, x="DCI_sign", y="SIE",
    palette="Pastel2",
    showfliers=False,
    width=0.5,
    ax=ax0,
    boxprops=dict(edgecolor="black", linewidth=1.5),   # 箱体边框
    whiskerprops=dict(color="black", linewidth=1.5),                       # 胡须
    capprops=dict(color="black", linewidth=1.5),                           # 顶端横线
    medianprops=dict(color="black", linewidth=2)                           # 中位数线
)
sns.stripplot(
    data=df, x="DCI_sign", y="SIE",
    color="k", alpha=0.6,
    jitter=0.2, size=3,
    ax=ax0
)
ax0.set_xlabel("")
ax0.set_ylabel("Detrended SIE (10$^5$ km$^2$)")
# ax0.set_title("SIE under positive vs negative DCI")
ax0.grid(axis="y", ls=":", alpha=0.4)

# ========= B) 时间序列 =========
ax1 = axes[1]
ax2 = ax1.twinx()

# SIE
l1, = ax1.plot(df["year"], df["SIE"], "-o", ms=3., lw=1.2, 
               color="k", label="Detrended SIE (10$^6$ km$^2$)")

# DCI_z
l2, = ax2.plot(df["year"], df["DCI"], "-o", ms=3., lw=1.2, 
               color="grey", label="DCI")

for spine in ax0.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)   # 控制粗细，比如 1.5 或 2
    

for spine in ax2.spines.values():
    spine.set_edgecolor("k")
    spine.set_linewidth(1.2)
    
# 相关标注
r, p = pearsonr(df["DCI"], df["SIE"])
ax2.text(0.98, 0.04, fr"$r={r:.2f},\ p={p:.03f}$",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.7))

ax2.tick_params(axis='y', labelcolor='grey', color='grey')
ax2.spines['right'].set_color('grey')
# ax2.tick_params(axis='y', colors="#e74c3c")
# 坐标轴设置
# ax1.set_xlabel("Year")
ax1.set_ylim(-4,4)
ax2.set_ylim(-25,25)
# ax1.set_ylabel("SIE", color="#2b8a3e"); ax1.tick_params(axis='y', colors="#2b8a3e")
ax2.set_ylabel("DCI", color="grey"); ax2.tick_params(axis='y', colors="grey")
# ax1.set_title("Time series with |DCI| ≥ 0.5σ highlighted")
ax1.legend([l1, l2], ["Detrended SIE", "DCI"], loc="lower left", frameon=False)
ax1.grid(axis="y", ls=":", alpha=0.4)

plt.tight_layout()

plt.show()


#%%
#eof 可视化 figure 7

fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=600, 
                         subplot_kw={'projection': ccrs.SouthPolarStereo()})

# EOF 颜色设置
clevels = np.arange(-4, 4.25, 0.25)
cmap = plt.cm.RdBu_r
# clevels = np.arange(-20, 22, 2)
# 标题标签
titles = ["(a) EOF1", "(b) EOF2", "(c) EOF3", "(d) EOF4", "(e) EOF5"]
pc_titles = ["(c) PC1", "(d) PC2", "(g) PC3", "(h) PC4", "(i) PC5"]
colors = ["b", "r", "g", "k","g"]  # PC1, PC2, PC3,

# 开启 constrained_layout
fig.subplots_adjust( top=0.9,bottom=0.2,wspace=0.09, hspace=0.6)

for i in range(2):
    # 绘制 EOF 模态
    ax = axes[i]
    ax.text(0.01, 0.89, f"{titles[i]}  {var[i]*100:.1f}%", transform=ax.transAxes,   va='top', ha='left')
    
    # ax.set_title(f"{titles[i]}  {var[i]*100:.2f}%")
    
    # 替换 contourf 为 pcolormesh
    
    c = ax.contourf(grid_lon, grid_lat, eof[i, :, :], levels=clevels, extend='both',
                    transform=ccrs.PlateCarree(), cmap=cmap, alpha = 0.8 )


    # 右侧：PC 时间序列（普通 2D 轴）
    ax_pc = fig.add_subplot(2, 2, 2*i+2)  # **普通坐标系，不使用投影**\
    
    
    if i == 1:

         # 显著性：p < 0.05 的点打小点（也可 stippling）
         sig = (p_val < 0.05)
           # 为避免 NaN 经纬度报错
         m = sig.values & np.isfinite(grid_lon) & np.isfinite(grid_lat)
         ax.scatter(grid_lon[m][::3], grid_lat[m][::3],
                      s=0.4, color='k', alpha=0.5, transform=ccrs.PlateCarree())
         # ax.contour(grid_lon, grid_lat,sig,level=[0.05],
         #             color='k', alpha=0.5, transform=ccrs.PlateCarree())
         # 用法：每个 ax 上调用一次，不要重复用同一个 rect 对象
         add_rectangle(ax, lon_pos, lat_pos, dlon=10, dlat=5, color='k', label="Positive region")
         add_rectangle(ax, lon_neg, lat_neg, dlon=10, dlat=5, color='k', label="Negative region")
         
         ax_dci = ax_pc.twinx()
         ax_dci.plot(range(1980,2024), dci, 'k', label="SIE", markersize=5)
         # ax_dci.set_ylim(-0.5, 0.5)
          # ax_pc.set_title(pc_titles[i])
         ax_dci.tick_params(axis='y', labelcolor='k')
         ax_dci.set_ylabel('DCI', color='k')
          # ax_pc.set_ylim(-4, 4)
         r, p = pearsonr(pc[:, i], dci)
          
          
         r_value_str = f"$r = {r:.2f}$"
         p_value_str = f"$p = {p:.2f}$"
          
         text_str = f'{f"$r = {r:.2f}$"}  {f"$p < 0.01$"}'
         ax_dci.text(0.03, 0.12, text_str, transform=ax_pc.transAxes, va='top',fontsize=12)
         ax_dci.tick_params(axis='y', labelcolor='k', color='k')
         ax_dci.spines['left'].set_color('k')

    # 创建第二个y轴，绘制标准化的 AO/BO 指数
    # ax2 = ax_pc.twinx()
    # ax_pc.plot(range(1980,2024), yearly_min_std[:], 'ko', label="SIE", markersize=3)
    
    # 设置第二个y轴标签
    ax_pc.set_ylabel('PC ', color='black')
   
    ax_pc.set_ylim(-3, 3)
    ax_pc.axhline(0, linestyle="--", color="k")
    ax_pc.bar(np.arange(1980, 2024, 1), pc[:, i], color=colors[i], alpha = 0.5)


    # text_str = f"{f'$r= {r:.2f}\nP= {p:.2f}$'}"
    ax_pc.text(0.75, 0.81, f"{pc_titles[i]}", transform=ax.transAxes,   va='top', ha='left')

    
    ax_pc.set_position([0.52, 0.64-i*0.37, 0.4, 0.25])
    
    
    
    # 地图的经纬度范围
    leftlon, rightlon, lowerlat, upperlat = (-180, 180, -50, -90)
    ax.set_extent([leftlon, rightlon, lowerlat, upperlat], ccrs.PlateCarree())

    # 添加地图特征
    ax.add_feature(cfeature.LAND, facecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='none')
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=2)
    

    # 设置扇形边界
    theta = np.linspace(-14/18*np.pi, -np.pi / 4, 100)
    radius_outer, radius_inner = 0.43, 0.11
    center = [0.5, 0.5]
    outer_arc = np.vstack([np.sin(theta) * radius_outer, np.cos(theta) * radius_outer]).T + center
    inner_arc = np.vstack([np.sin(theta[::-1]) * radius_inner, np.cos(theta[::-1]) * radius_inner]).T + center
    verts = np.vstack([outer_arc, inner_arc, outer_arc[0]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 2) + [mpath.Path.CLOSEPOLY]
    ax.set_boundary(mpath.Path(verts, codes), transform=ax.transAxes)

    # 添加网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    # gl.ylocator = mticker.FixedLocator([-60, -70, -80])
    gl.ylocator = mticker.FixedLocator([-60, -70, -80])
    gl.xlocator = mticker.FixedLocator([-120, -90, -60, -30, 0])
    
    # 添加纬度标签在边缘方向
    add_lat_labels_on_sector(ax, [60, 70, 80], label_lon=-46, offset=0)
    
    # 添加经度标签贴在 -60° 纬线上
    add_lon_labels_on_sector(ax, [ 120, 90, 60], label_lat=-54, offset=0)
    
axes[0].set_position([0.2, 0.55, 0.45, 0.45])
axes[1].set_position([0.2, 0.18, 0.45, 0.45])

# ax_pc.set_position([0.5, 0.55, 0.6, 0.4])
# ax_pc[1,1].set_position([0.5, 0.05, 0.6, 0.4])
# 添加统一色标
cb_ax = fig.add_axes([0.23, 0.23, 0.2, 0.013])  # 颜色条位置
cb=fig.colorbar(c, cax=cb_ax, orientation='horizontal', format='%.0f',label='Intensity Anomalies (hPa)')
ticks = cb.get_ticks()  # 获取原始刻度数组
cb.set_ticks(ticks[::2])  # 仅设置首尾两个刻度

# plt.savefig('/Users/peiyilin/Desktop/研组/极端海冰与气旋/CYCINTEN_EOF_dci_detrend.pdf', format='pdf',bbox_inches='tight')

plt.show()

