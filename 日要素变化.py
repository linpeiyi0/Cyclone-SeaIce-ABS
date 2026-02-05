#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:54:52 2026

@author: peiyilin
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import glob
import string
from matplotlib import font_manager
import xarray as xr


#%%
#每日要素变化

#读取文件
def caldaiyva(file_1,file_2,va,ti):

    da_2010 = xr.open_dataset(file_1, chunks={'latitude': 100, 'longitude': 100})
    da_2009 = xr.open_dataset(file_2, chunks={'latitude': 100, 'longitude': 100})
    da = xr.concat([da_2009,da_2010], dim='valid_time').sortby('valid_time')
    va = da[va]
    va = va.sel(valid_time = ti).sel(longitude = slice(-130,-60), latitude = slice(-55,-75))
    
    va_daily = va.mean(dim=['latitude','longitude'])

    va_daily = va_daily.resample(valid_time='1D').mean('valid_time')#sel(valid_time=va_daily['valid_time.hour'].isin([0, 6, 12, 18]))

    return va_daily



# —— 1) 工具：把累积量(J/m^2)转成率(W/m^2)，并处理归零重置 —— #
def to_wm2(da, accum_hours=1):
    out = da / (accum_hours * 3600.0)
    out.attrs['units'] = 'W m**-2'
    return out


def caldaiyva_HF(file_1,file_2,ti):

    da_2010 = xr.open_dataset(file_1, chunks={'latitude': 100, 'longitude': 100})
    da_2009 = xr.open_dataset(file_2, chunks={'latitude': 100, 'longitude': 100})
    da = xr.concat([da_2009,da_2010], dim='valid_time').sortby('valid_time').sel(valid_time = ti).sel(longitude = slice(-130,-60), latitude = slice(-55,-75))
    
    slhf =  da['slhf']
    sshf =  da['sshf']
    ssr =  da['ssr']
    slr =  da['str']
    
    nshf = to_wm2(slhf + sshf + ssr + slr)
    
    nshf = nshf.mean(dim=['latitude','longitude'])
    
    nshf_ = nshf.resample(valid_time='1D').mean('valid_time')
    
    return nshf_

#VIWVN
viwvn_2010 = caldaiyva(r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_2010.nc',r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_2009.nc', 'viwvn', slice('2009-10','2010-04'))
viwvn_1980 = caldaiyva(r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_1980.nc',r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_1979.nc', 'viwvn', slice('1979-10','1980-04'))

#气候
viwvn_cl = xr.open_zarr("clim_daily_MMF_OctMar.zarr")['viwvn'].sel(longitude = slice(-130,-60), latitude = slice(-55,-75)).mean(dim=['latitude','longitude'])

viwvn_clim = viwvn_cl.sel(month = viwvn_2010['valid_time.month'], day = viwvn_2010['valid_time.day'])
viwvn_clim = viwvn_cl.sel(month = viwvn_1980['valid_time.month'], day = viwvn_1980['valid_time.day'])


viwvn_ano_2010 = viwvn_2010 - viwvn_clim
viwvn_ano_1980 = viwvn_1980 - viwvn_clim

viwvn_ano_2010.plot()
viwvn_ano_2010.to_netcdf('dailyano_viwvn_2010_5575.nc')

viwvn_ano_1980.plot()
viwvn_ano_1980.to_netcdf('dailyano_viwvn_1980_5575.nc')

#NSHF
target_year = 1980
start = f"{target_year-1}-10-01"
end   = f"{target_year}-03-31"

da = xr.open_dataset(r'/Volumes/Expansion/大气数据/ERA5_HF_1980.nc', chunks={'latitude': 100, 'longitude': 100}).sortby('valid_time')
da = da.sel(valid_time=slice(start, end)).sel(longitude = slice(-130,-60), latitude = slice(-55,-75))

slhf =  da['slhf']
sshf =  da['sshf']
ssr =  da['ssr']
slr =  da['str']

nshf = to_wm2(slhf + sshf + ssr + slr)
nshf = nshf.mean(dim=['latitude','longitude'])
nshf_1980 = nshf.resample(valid_time='1D').mean('valid_time')

nshf_1980.plot()


nshf_2010 = caldaiyva_HF(r'/Volumes/Elements/HF/ERA5_HF_2010.nc',r'/Volumes/Elements/HF/ERA5_HF_2009.nc', slice('2009-10','2010-04'))

nshf_2010.plot()

# 逐日异常

nshf_clim_daily = xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/clim_daily_nshf_OctMar.nc', chunks={'latitude': 100, 'longitude': 100})

# 用时间坐标的 month/day 直接“矢量化选择”对应的气候态
nshf_cl = nshf_clim_daily['clim_daily'].sel(longitude = slice(-130,-60), latitude = slice(-55,-75)).mean(dim=['latitude','longitude'])

nshf_clim = nshf_cl.sel(month = nshf_2010['valid_time.month'], day = nshf_2010['valid_time.day'])
nshf_clim = nshf_cl.sel(month = nshf_1980['valid_time.month'], day = nshf_1980['valid_time.day'])

nshf_ano_2010 = nshf_2010 - nshf_clim
nshf_ano_1980 = nshf_1980 - nshf_clim


nshf_ano_2010.plot()
nshf_ano_2010.to_netcdf('dailyano_nshf_2010_5575.nc')


nshf_ano_1980.plot()
nshf_ano_1980.to_netcdf('dailyano_nshf_1980_5575.nc')

#sst
sst_2010 = caldaiyva(r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_2010.nc',r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_2009.nc', 'sst', slice('2009-10','2010-04'))
sst_1980 = caldaiyva(r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_1980.nc',r'/Volumes/Expansion/大气数据/ERA5_SST_MMF_1979.nc', 'sst', slice('1979-10','1980-04'))

#气候
viwvn_cl = xr.open_dataset("clim_daily_SST_OctMar.nc")['sst'].sel(longitude = slice(-130,-60), latitude = slice(-55,-75))

viwvn_clim = viwvn_cl.sel(month = sst_2010['valid_time.month'], day = sst_2010['valid_time.day']).mean(dim=['latitude','longitude'])
viwvn_clim = viwvn_cl.sel(month = sst_1980['valid_time.month'], day = sst_1980['valid_time.day']).mean(dim=['latitude','longitude'])
viwvn_clim.plot()

viwvn_ano_2010 = sst_2010 - viwvn_clim
viwvn_ano_1980 = sst_1980 - viwvn_clim

viwvn_ano_2010.plot()
viwvn_ano_2010.to_netcdf('dailyano_sst_2010.nc')

viwvn_ano_1980.plot()
viwvn_ano_1980.to_netcdf('dailyano_sst_1980.nc')

#t2m
sst_2010 = caldaiyva(r'/Volumes/Elements/T2M/ERA5_T2M_2010.nc',r'/Volumes/Elements/T2M/ERA5_T2M_2009.nc', 't2m', slice('2009-10','2010-04'))
sst_1980 = caldaiyva(r'/Volumes/Elements/T2M/ERA5_T2M_1980.nc',r'/Volumes/Elements/T2M/ERA5_T2M_1979.nc', 't2m', slice('1979-10','1980-04'))

sst_2010.plot()



#气候
viwvn_cl = xr.open_dataset("clim_daily_T2M_OctMar.nc")['t2m'].sel(longitude = slice(-130,-60), latitude = slice(-55,-75))

viwvn_clim = viwvn_cl.sel(month = sst_2010['valid_time.month'], day = sst_2010['valid_time.day']).mean(dim=['latitude','longitude'])
viwvn_clim = viwvn_cl.sel(month = sst_1980['valid_time.month'], day = sst_1980['valid_time.day']).mean(dim=['latitude','longitude'])
viwvn_clim.plot()

viwvn_ano_2010 = sst_2010 - viwvn_clim
viwvn_ano_1980 = sst_1980 - viwvn_clim

viwvn_ano_2010.plot()
viwvn_ano_2010.to_netcdf('dailyano_t2m_2010.nc')

viwvn_ano_1980.plot()
viwvn_ano_1980.to_netcdf('dailyano_t2m_1980.nc')


viwvn_ano_2010 = xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/dailyano_viwvn_2010.nc', chunks={'latitude': 100, 'longitude': 100})['viwvn']
sst_ano_2010 = xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/dailyano_sst_2010.nc', chunks={'latitude': 100, 'longitude': 100})['sst']
t2m_ano_2010 = xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/dailyano_t2m_2010.nc', chunks={'latitude': 100, 'longitude': 100})['t2m']
nshf_ano_2010 = xr.open_dataset('/Users/peiyilin/Desktop/研组/极端海冰与气旋/dailyano_nshf_2010.nc', chunks={'latitude': 100, 'longitude': 100})['__xarray_dataarray_variable__'].rename('nshf')


#%%

df_2010 = pd.DataFrame({
    'SIEF': pd.Series(dsieloss_2010.values, index=date_values_2010),#dsieloss_i_2010.values*100
    'SIE': pd.Series(xr.concat(sie_daily_2010[5:],dim='time').to_series().values , index=date_values_2010),
    'SIEano': pd.Series(sieano_2010.values , index=date_values_2010),
    'nshf': pd.Series(nshf_ano_2010.values, index=date_values_2010),
    'viwvn': pd.Series(viwvn_ano_2010.values, index=date_values_2010),
    'sst': pd.Series(sst_ano_2010.values, index=date_values_2010),
    't2m': pd.Series(t2m_ano_2010.values, index=date_values_2010),
    'cyintensity': pd.Series(in_avg_2010.values, index=date_values_2010),
    'cycount': pd.Series(count_2010.values, index=date_values_2010),
})

df_1980 = pd.DataFrame({
    'SIEF': pd.Series(dsieloss_1980, index=date_values_1980),#dsieloss_i_1980.values*100
    'SIE': pd.Series(xr.concat(sie_daily_1980[1:],dim='time').to_series().values, index=date_values_1980[::2]),
    'SIEano': pd.Series(sieano_1980.values , index=date_values_1980),
    'nshf': pd.Series(nshf_ano_1980.values, index=date_values_1980),
    'viwvn': pd.Series(viwvn_ano_1980.values, index=date_values_1980),
    'sst': pd.Series(sst_ano_1980.values, index=date_values_1980),
    't2m': pd.Series(t2m_ano_1980.values, index=date_values_1980),
    'cyintensity': pd.Series(in_avg_1980.values, index=date_values_1980),
    'cycount': pd.Series(count_1980.values, index=date_values_1980),
})

mean=xr.open_dataarray("daily_sie_clim.nc")

sie_clim = pd.Series(mean.sel(month_day=df_1980.index.strftime('%m-%d')), index=date_values_1980)
sieano_1980 = df_1980['SIE'] - sie_clim

sie_clim = pd.Series(mean.sel(month_day=df_2010.index.strftime('%m-%d')), index=date_values_2010)
sieano_2010 = df_2010['SIE'] - sie_clim

# df_2010.to_csv('df_2010.csv', index=False)

#%%

# —— 准备 & 对齐 —— #
def rolling_with_preserve_nan(s, window=3):
    # 正常平滑
    smoothed = s.rolling(window, center=True, min_periods=1).mean()
    # 把原始 NaN 的位置强制改回 NaN
    smoothed[s.isna()] = np.nan
    return smoothed


cols_to_smooth = ['SIE', 'SIEF', 'SIEano', 'nshf', 'viwvn', 'sst', 't2m','cyintensity']
df_1980[cols_to_smooth] = df_1980[cols_to_smooth].rolling(3, center=True, min_periods=1).mean()
df_1980['cyintensity'][df_1980['cycount'].isna()] = np.nan
df = df_1980
std_1980_ = pd.Series(std_1980_.values, index=date_values_1980)
std_1980 = pd.Series(std_1980.values, index=date_values_1980)

std_1980_r = std_1980_.rolling(3, center=True, min_periods=1).mean()
std_1980r = std_1980.rolling(3, center=True, min_periods=1).mean()
# SIEF 超出 ±2σ 的日期（仅当上下界都非 NaN 时才判断）
exceed_mask = (std_1980r.notna() & std_1980_r.notna() &
               ((df['SIEF'] > std_1980r) | (df['SIEF'] < std_1980_r)))



# df =  df_2010.apply(lambda col: rolling_with_preserve_nan(col, window=3))
cols_to_smooth = ['SIE', 'SIEF', 'SIEano', 'nshf', 'viwvn', 'sst', 't2m','cyintensity']
# cols_to_smooth = ['SIE', 'SIEF', 'nshf', 'viwvn', 'sst', 't2m','cyintensity']
df_2010[cols_to_smooth] = df_2010[cols_to_smooth].rolling(3, center=True, min_periods=1).mean()
# df_2010['cyintensity'][df_2010['cycount'].isna()] = np.nan
df_2010.loc[df_2010["cycount"].isna(), "cyintensity"] = np.nan

df = df_2010
std_2010_ = pd.Series(std_2010_.values, index=date_values_2010)
std_2010 = pd.Series(std_2010.values, index=date_values_2010)

std_2010_r = std_2010_.rolling(3, center=True, min_periods=1).mean()
std_2010r = std_2010.rolling(3, center=True, min_periods=1).mean()

df_1980[170:180]['cyintensity'].min()


# —— 画布 —— #
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 15,          # 基础字号
    "axes.labelsize": 14,     # 轴标签
    "axes.titlesize": 14,     # 子图标题（你不需要标题就无所谓）
    "xtick.labelsize": 12,    # x刻度
    "ytick.labelsize": 12,    # y刻度
    "legend.fontsize": 11,    # 图例
    "axes.linewidth": 1.2,    # 坐标轴线更清晰
})
# —— 配色 —— #


# 推荐配色（8 个变量）
c_sief = '#2E2E2E'   # SIEF：深炭黑，权威、对比强
c_sie  = '#6c757d'   # SIE：天空蓝，清晰且不刺眼
c_nshf = '#D55E00'   # NHF：暖橙红，直观“热通量”感
c_viwv = '#009E73'   # VIWVN：蓝绿色，和热量区分明显
c_sst  = '#E69F00'   # SST：琥珀橙，温度概念直觉
c_t2m  = '#CC79A7'   # T2m：品红紫，和 SST 分得开
c_int  = '#0072B2'   # Intensity：深蓝，稳重醒目
c_cnt  = '#8C8C8C'   # Count 柱：中性灰，降低视觉权重


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

def find_extremes(df, start, end, col="SIEF", topn=3):
    """
    查找指定时间段内某列数据的极值
    
    参数:
        df     : DataFrame，索引必须是DatetimeIndex
        start  : str，起始日期 (例如 '2009-12-01')
        end    : str，结束日期 (例如 '2010-03-31')
        col    : str，指定要分析的列名 (默认 'SIEF')
        topn   : int，返回前 n 个极大值和极小值 (默认 3)
        
    返回:
        dict，包含 topn 极大值和极小值的时间和值
    """
    # 提取时间段
    subset = df.loc[start:end, col]

    # 极大值
    max_vals = subset.nlargest(topn)
    # 极小值
    min_vals = subset.nsmallest(topn)

    return {
        "max": max_vals,
        "min": min_vals
    }

len(df_2010[92:])*0.1= 9
len(df_1980[92:])*0.1= 9
# 使用示例
extremes = find_extremes(df, "2010-01-01", "2010-03-31", col="SIEF", topn=9)
print("极大值：\n", extremes["max"])
print("\n极小值：\n", extremes["min"])

extremes = find_extremes(df, "1980-01-01", "1980-03-31", col="SIEF", topn=9)

#%%

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 10), sharex=True)
ax1, ax2, ax3, ax4, ax5 = axes

# ========== 子图1：SIEF（左） + SIE（右） ==========
ax1r = ax1.twinx()

# 阴影（±2σ），严格使用你的写法与切片
# ax1.fill_between(
#     df.index, std_2010_r, std_2010r,
#     where=~std_2010r.isna(), color='black', alpha=0.10, label="±2σ Range"
# )

l1, = ax1.plot(df['SIEF'].dropna().index, df['SIEF'].dropna(), lw=2, color=c_sief, label='SIEF')
# l1r, = ax1r.plot(df.index, df['SIE'],  lw=2, color=c_sie,  label='SIE')
l1r, = ax1r.plot(df['SIEano'].dropna().index, df['SIEano'].dropna()/1e+6,
                 lw=2, color=c_sie, label='SIE anomaly')

ax1.set_ylim(-10, 10)
ax1r.set_ylim(-0.6, 0)
ax1.set_ylim(-30, 80)
ax1.set_ylabel('SIEF (%)', color=c_sief)
ax1r.set_ylabel('SIE anomaly \n(10$^6$ km$^2$)',     color=c_sie)
ax1.tick_params(axis='y', colors=c_sief)
ax1r.tick_params(axis='y', colors=c_sie)

# 合并图例
h, lab = [l1], ['SIEF']
h2, lab2 = [l1r], ['SIE anomaly']
ax1.legend(h+h2, lab+lab2, loc='upper left', frameon=False, ncols=2)

# ========== 子图2：nshf（左） + viwvn（右） ==========
ax2r = ax2.twinx()
l2,  = ax2.plot(df.index, df['nshf'],  lw=2, color=c_nshf, label='nshf')
l2r, = ax2r.plot(df.index, df['viwvn'], lw=2, color=c_viwv, label='viwvn')

ax2.set_ylim(-80, 80)
ax2r.set_ylim(-80, 80)

ax2.set_ylim(-40, 40)
ax2r.set_ylim(-70, 70)

ax2.set_ylabel('NHF anomaly (W m$^{-2}$)', color=c_nshf)
ax2r.set_ylabel('VIWVN anomaly\n (kg m$^{-1}$ s$^{-1}$)',            color=c_viwv)
ax2.tick_params(axis='y', colors=c_nshf)
ax2r.tick_params(axis='y', colors=c_viwv)

ax2.legend([l2, l2r], ['NHF', 'VIWVN'], loc='upper left', frameon=False, ncols=2)

# ========== 子图3：sst（左） + t2m（右） ==========
ax3r = ax3.twinx()
l3,  = ax3.plot(df.index, df['sst'], lw=2, color=c_sst, label='sst')
l3r, = ax3r.plot(df.index, df['t2m'], lw=2, color=c_t2m, label='t2m')
ax3r.set_ylim(-12, 0)
ax3.set_ylim(-0.5, 0.5)

ax3r.set_ylim(-3, 3)
ax3.set_ylim(-0.5, 0.5)

ax3.set_ylabel('SST anomaly (K)', color=c_sst)
ax3r.set_ylabel('T2m anomaly (K)', color=c_t2m)
ax3.tick_params(axis='y', colors=c_sst)
ax3r.tick_params(axis='y', colors=c_t2m)

ax3.legend([l3, l3r], ['SST', 'T2m'], loc='upper left', frameon=False, ncols=2)

# ========== 子图4：intensity（左） + count（右，柱状） ==========
ax4r = ax4.twinx()
l4, = ax4.plot(df.index, df['cyintensity'], lw=2, color=c_int, label='intensity')
bars = ax4r.bar(df.index, df['cycount'].fillna(0), width=0.6, color=c_cnt, alpha=0.35,
                edgecolor='none', label='count')
ax4.set_ylim(940, 1010)
ax4r.set_ylim(0, 6)
ax4.set_ylabel('Intensity (hPa)', color=c_int)
ax4r.set_ylabel('Count',    color=c_cnt)
ax4.tick_params(axis='y', colors=c_int)
ax4r.tick_params(axis='y', colors=c_cnt)

ax4.legend([l4, bars], ['Intensity', 'Count'], loc='upper left', frameon=False)
# 为次轴也加简单图例（可选）
# ax4r.legend([bars], ['count'], loc='upper right', frameon=False)


# ========== 通用修饰：零线 + 垂直事件线 + 网格 + 外观 ==========
twin_axes = [(ax1, ax1r), (ax2, ax2r), (ax3, ax3r), (ax4, ax4r)]
for ax, axr in twin_axes:
 
    for t, v in extremes['max'].items():
        ax.axvline(t, ls='--', lw=0.7, color='r', alpha=0.45, zorder=0)
    for t, v in extremes['min'].items():
        ax.axvline(t, ls='--', lw=0.7, color='k', alpha=0.45, zorder=0)

    # 外观：去掉上边框；主轴保留左，次轴保留右
    for spine in ['top']:
        ax.spines[spine].set_visible(False)
        axr.spines[spine].set_visible(False)
    ax.spines['right'].set_visible(False)   # 主轴右侧不要
    # 次轴左侧不要
    axr.spines['left'].set_visible(False)

    # 网格（只在主轴）
    ax.grid(True, axis='y', ls='--', lw=0.6, color='#b0b0b0', alpha=0.35)

end_date = datetime.datetime(2010, 4, 1)
start_date= datetime.datetime(2009, 12, 31)


# ====== 子图5：时间-滞后相关性等值图（横轴=时间，纵轴=lag） ======

levels = np.linspace(-0.8, 0.8, 25)
cf = ax5.contourf(end_dates, lags, filtered_corr.T, levels=levels,
                  cmap='RdBu_r', extend="both")

# 显著性点（注意横纵互换：x=时间，y=lag）
ax5.scatter(sig_times, sig_lags, s=8, marker='o', color='k', label='p < 0.01', zorder=3)

# 参考线：lag=0
ax5.axhline(0, ls='--', color='grey', lw=1)

# 轴标签
ax5.set_ylabel("Lag (days)")
ax5.set_ylim(min(lags), max(lags))
ax5.set_yticks([min(lags), 0, max(lags)])


for idx, ax in enumerate(axes.flat):
    ax.text(0.0, 1.13, f'({labels[idx]})', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='top', ha='left')
    
# 放一个水平的 colorbar（不挤占主图高度可用 inset，也可共用右侧竖直 cbar）
cax = fig.add_axes([0.9, 0.05, 0.01, 0.14])  # [left, bottom, width, height] 可微调
cb_b = fig.colorbar(cf, cax=cax)
cb_b.set_label(label='Correlation', labelpad=-15)
cb_b.set_ticks([-0.8,0.8])

# 统一时间轴范围与刻度（复用你前面设置）
axes[-1].set_xlim(start_date, end_date)
monthly_ticks = pd.date_range(start=start_date, end=end_date, freq='SMS') #+ pd.Timedelta(days=1)
axes[-1].set_xticks(monthly_ticks)
axes[-1].set_xticklabels([t.strftime("%d %b %Y") for t in monthly_ticks], rotation=0)

fig.autofmt_xdate(rotation=0, ha='center')

plt.tight_layout()

# fig.savefig(f"figure2_2010_co_1dlag.pdf", dpi=400, bbox_inches="tight")
plt.show()


#%%
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 10), sharex=True)
ax1, ax2, ax3, ax4, ax5 = axes

# ========== 子图1：SIEF（左） + SIE（右） ==========
ax1r = ax1.twinx()


l1, = ax1.plot(df['SIEF'].dropna().index, df['SIEF'].dropna(), lw=2, color=c_sief, label='SIEF')
# l1r, = ax1r.plot(df.index, df['SIE'],  lw=2, color=c_sie,  label='SIE')
l1r, = ax1r.plot(df['SIE'].dropna().index, df['SIEano'].dropna()/1e6,
                 lw=2, color=c_sie, label='SIE')

ax1.set_ylim(-10, 10)
ax1r.set_ylim(0.3, 0.6)

ax1.set_ylabel('SIEF (%)', color=c_sief)
ax1r.set_ylabel('SIE anomaly \n(10$^6$ km$^2$)',     color=c_sie)
ax1.tick_params(axis='y', colors=c_sief)
ax1r.tick_params(axis='y', colors=c_sie)

# 合并图例
h, lab = [l1], ['SIEF']
h2, lab2 = [l1r], ['SIE anomaly']
ax1.legend(h+h2, lab+lab2, loc='upper left', frameon=False, ncols=2)

# ========== 子图2：nshf（左） + viwvn（右） ==========
ax2r = ax2.twinx()
l2,  = ax2.plot(df.index, df['nshf'],  lw=2, color=c_nshf, label='nshf')
l2r, = ax2r.plot(df.index, df['viwvn'], lw=2, color=c_viwv, label='viwvn')

ax2.set_ylim(-60, 90)
ax2r.set_ylim(-60, 90)


ax2.set_ylabel('NHF anomaly (W m$^{-2}$)', color=c_nshf)
ax2r.set_ylabel('VIWVN anomaly\n (kg m$^{-1}$ s$^{-1}$)',            color=c_viwv)
ax2.tick_params(axis='y', colors=c_nshf)
ax2r.tick_params(axis='y', colors=c_viwv)

ax2.legend([l2, l2r], ['NHF', 'VIWVN'], loc='upper left', frameon=False, ncols=2)

# ========== 子图3：sst（左） + t2m（右） ==========
ax3r = ax3.twinx()
l3,  = ax3.plot(df.index, df['sst'], lw=2, color=c_sst, label='sst')
l3r, = ax3r.plot(df.index, df['t2m'], lw=2, color=c_t2m, label='t2m')
ax3r.set_ylim(-12, 0)
ax3.set_ylim(-0.5, 0.5)

ax3r.set_ylim(-3, 3)
ax3.set_ylim(-0.1, 0.5)

ax3.set_ylabel('SST anomaly (K)', color=c_sst)
ax3r.set_ylabel('T2m anomaly (K)', color=c_t2m)
ax3.tick_params(axis='y', colors=c_sst)
ax3r.tick_params(axis='y', colors=c_t2m)

ax3.legend([l3, l3r], ['SST', 'T2m'], loc='upper left', frameon=False, ncols=2)

# ========== 子图4：intensity（左） + count（右，柱状） ==========
ax4r = ax4.twinx()
l4, = ax4.plot(df.index, df['cyintensity'], lw=2, color=c_int, label='intensity')
bars = ax4r.bar(df.index, df['cycount'].fillna(0), width=0.6, color=c_cnt, alpha=0.35,
                edgecolor='none', label='count')
ax4.set_ylim(945, 1010)
ax4r.set_ylim(0, 6)
ax4.set_ylabel('Intensity (hPa)', color=c_int)
ax4r.set_ylabel('Count',    color=c_cnt)
ax4.tick_params(axis='y', colors=c_int)
ax4r.tick_params(axis='y', colors=c_cnt)

ax4.legend([l4, bars], ['Intensity', 'Count'], loc='upper left', frameon=False)
# 为次轴也加简单图例（可选）
# ax4r.legend([bars], ['count'], loc='upper right', frameon=False)

# ========== 通用修饰：零线 + 垂直事件线 + 网格 + 外观 ==========
twin_axes = [(ax1, ax1r), (ax2, ax2r), (ax3, ax3r), (ax4, ax4r)]
for ax, axr in twin_axes:

    for t, v in extremes['max'].items():
        ax.axvline(t, ls='--', lw=0.7, color='r', alpha=0.45, zorder=0)
    for t, v in extremes['min'].items():
        ax.axvline(t, ls='--', lw=0.7, color='k', alpha=0.45, zorder=0)

    # 外观：去掉上边框；主轴保留左，次轴保留右
    for spine in ['top']:
        ax.spines[spine].set_visible(False)
        axr.spines[spine].set_visible(False)
    ax.spines['right'].set_visible(False)   # 主轴右侧不要
    # 次轴左侧不要
    axr.spines['left'].set_visible(False)

    # 网格（只在主轴）
    ax.grid(True, axis='y', ls='--', lw=0.6, color='#b0b0b0', alpha=0.35)


start_date= datetime.datetime(1979, 12, 31)
end_date = datetime.datetime(1980, 4, 1)


# ====== 子图5：时间-滞后相关性等值图（横轴=时间，纵轴=lag） ======

levels = np.linspace(-0.8, 0.8, 25)
cf = ax5.contourf(end_dates, lags, filtered_corr.T, levels=levels,
                  cmap='RdBu_r', extend="both")

# 显著性点（注意横纵互换：x=时间，y=lag）
ax5.scatter(sig_times, sig_lags, s=8, marker='o', color='k', label='p < 0.01', zorder=3)

# 参考线：lag=0
ax5.axhline(0, ls='--', color='grey', lw=1)

# 轴标签
ax5.set_ylabel("Lag (days)")
ax5.set_ylim(min(lags), max(lags))
ax5.set_yticks([min(lags), 0, max(lags)])

for idx, ax in enumerate(axes.flat):
    ax.text(0.0, 1.13, f'({labels[idx]})', transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='top', ha='left')
    
# 放一个水平的 colorbar（不挤占主图高度可用 inset，也可共用右侧竖直 cbar）
cax = fig.add_axes([0.9, 0.05, 0.01, 0.14])  # [left, bottom, width, height] 可微调
cb_b = fig.colorbar(cf, cax=cax)
cb_b.set_label(label='Correlation', labelpad=-15)
cb_b.set_ticks([-0.8,0.8])

# 统一时间轴范围与刻度（复用你前面设置）
axes[-1].set_xlim(start_date, end_date)
monthly_ticks = pd.date_range(start=start_date, end=end_date, freq='SMS') 
axes[-1].set_xticks(monthly_ticks)
axes[-1].set_xticklabels([t.strftime("%d %b %Y") for t in monthly_ticks], rotation=0)

fig.autofmt_xdate(rotation=0, ha='center')

plt.tight_layout()

fig.savefig(f"figure2_1980_co_1dlag.pdf", dpi=400, bbox_inches="tight")
plt.show()

