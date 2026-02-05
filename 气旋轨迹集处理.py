#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:36:24 2026

@author: peiyilin
"""


# %%
from collections import deque, defaultdict
from scipy.stats import t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
import xarray as xr

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import statsmodels.api as sm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

import os
import pandas as pd  # Import pandas for concat
import math
from netCDF4 import Dataset
from datetime import datetime,  timedelta
import datetime
import cftime
from shapely.geometry import LineString, Point

import matplotlib.path as mpath
import matplotlib.ticker as mticker
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
import cmocean
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string
import metpy.calc as mpcalc


#计算由纬度和经度指定的两点之间的距离
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

#%%
##############################文件读取############################################

path1 = '/Users/Desktop/Shared/' 
path2='/Users/Desktop/1979track/' 

file1 = 'EC_detail_msg(-35).txt'
file2 = 'EC_whole_msg(-35).txt'
file3 = 'EC_line(-35).txt'


def readcy(path):

    f1 = open(path+file1, 'r')  # num
    f2 = open(path+file2, 'r')  # detail
    f3 = open(path+file3, 'r')  # whole
    
  
    data1 = f1.readlines()
    f1.close()
    data2 = f2.readlines()
    f2.close()
    data3 = f3.readlines()
    f3.close()


    data11 = []
    for line in data1:
        data11.append(line.split())
    whole = np.array(data11)  # detail
    
    data22 = []
    for line in data2:
        data22.append(line.split())
    wholeh = np.array(data22)  # whole
    
    data33 = []
    for line in data3:
        data33.append(line.split())
    ecline = np.array(data33)
    
    return whole,wholeh,ecline

whole1,wholeh1,ecline1=readcy(path1)
whole2,wholeh2,ecline2=readcy(path2)

#合并两个文件

whole=np.vstack((whole2, whole1)) 
wholeh=np.vstack((wholeh2, wholeh1)) 
ecline=np.vstack((ecline2, ecline1)) 

wholeh1[-10]


####################轨迹时间,气压,经纬度统计######################################

# 气旋生成月份/年份
mon = [] 
year = []
for i in range(len(ecline)):
    year.append(float(ecline[i][2]))
    mon.append(float(ecline[i][7]))
year = np.array(year)
mon = np.array(mon)

mslp = []  # 气旋最强中心气压
vor = []  # 气旋最强涡度
wind925 = []  # 气旋最强 925hpa wind
wind10 = []  # 气旋最强 10m wind
mons = []  # 气旋生成月份
monm = []  # 气旋最小mslp月份
lons = []  # 气旋生成的经度
lats = []  # 气旋生成的纬度
lony = []  # 气旋消亡时的经度
laty = []  # 气旋消亡时的纬度

for i in range(len(whole)):
    mslp.append(float(whole[i][31]))
    vor.append(float(whole[i][19]))
    wind925.append(float(whole[i][23]))
    wind10.append(float(whole[i][27]))
    y=int(whole[i][1])
    if y ==1979:
        a = datetime.datetime(1979, 1, 1, 0, 0, 0) + \
            timedelta(hours=float(whole[i][2])-1)
        mons.append(a.month)  
        a = datetime.datetime(1979, 1, 1, 0, 0, 0) + \
            timedelta(hours=float(whole[i][28])-1)    
        monm.append(a.month)
    else:
        a = datetime.datetime(1980, 1, 1, 0, 0, 0) + \
            timedelta(hours=float(whole[i][2])-1)
        mons.append(a.month)
        a = datetime.datetime(1980, 1, 1, 0, 0, 0) + \
            timedelta(hours=float(whole[i][28])-1)
        monm.append(a.month)
    lons.append(float(whole[i][3]))
    lats.append(float(whole[i][4]))
    lony.append(float(whole[i][10]))
    laty.append(float(whole[i][11]))

mslp = np.array(mslp)
vor = np.array(vor)
wind925 = np.array(wind925)
wind10 = np.array(wind10)
lats = np.array(lats)
lons = np.array(lons)
lony = np.array(lony)
laty = np.array(laty)

len(mslp)
len(lony)

#将时刻转为日期时间格式，时刻timetick为 从1970年1月1日开始计算的小时数
a = datetime.datetime(1970, 1, 1, 0, 0, 0)+timedelta(hours=16855)


mslph = []  # 每一时刻的气旋中心气压
lonmh = []  # 每一时刻的气旋涡度经度
latmh = []  # 每一时刻的气旋涡度纬度
wind10h = [] # 每一时刻的气旋风速
timeh = [] # 每一时刻timetick
vor850h=[]
for i in range(len(wholeh)):
    mslph.append(float(wholeh[i][8]))
    vor850h.append(float(wholeh[i][5]))
    wind10h.append(float(wholeh[i][15]))
    lonmh.append(float(wholeh[i][3]))
    latmh.append(float(wholeh[i][4]))
    sicw.append(float(wholeh[i][18]))
    y=int(wholeh[i][0])
    base_time = pd.Timestamp('1979-01-01 00:00:00')
    # 从1979年1月1日开始累积的小时数
    offset_hours = (pd.Timestamp(f'{y}-01-01') - base_time).total_seconds() / 3600 
    timeh.append( offset_hours + float(wholeh[i][2]))
    # if  y==2024:  
    #     timeh.append(385704+float(wholeh[i][2]))

mslph = np.array(mslph)
lonmh = np.array(lonmh)
latmh = np.array(latmh)

timeh = np.array(timeh)
vor850h=np.array(vor850h)
wind10h = np.array(wind10h)
len(mslph)
len(sicw)
timeh[127243]

#%%
###################################筛选ABS海区气旋#####################################

radius_km=500
#至少有一个轨迹点到达60度以南

absc, t = [], []
lon, lat = [], []
cmslp = []
lonii, latii = [], []
sic = []
tim = []
maxp = []
lonmax,latmax=[],[]
win10=[]
for i in range(len(whole)):  # len(whole)
    tt = []
    if i<=1053:
        cstart = int(ecline[i][3])-1
        cend = int(ecline[i][4])
    else:
        cstart = int(ecline[i][3])-1+127244
        cend = int(ecline[i][4])+127244
 
    # 每个气旋轨迹
    lo = lonmh[cstart:cend]
    la = latmh[cstart:cend]
    mslpp = list(mslph[cstart:cend])
    ti = timeh[cstart:cend]
    wi10 = wind10h[cstart:cend]
    
    loni, lati = [], []
    
    count = 0
    for j in range(len(lo)):
        if lonmh[j+cstart] >= 230 and lonmh[j+cstart] <= 300 and latmh[j+cstart] <= -60 and latmh[j+cstart] >= -75:#  #sicw[j+cstart] > 0. and sicw[j+cstart] <= 1: # and mslpp[j] == max(mslpp):  # 选定在as区域内且靠近冰区的气旋
           # if  latmh[j+cstart]<latmh[j+cstart-3]:
                count+=1            
                # tt.append(float(ti[j]))  # 气旋符合条件的轨迹点时刻
                # loni.append(lonmh[j+cstart])
                # lati.append(latmh[j+cstart])
        
            # lonmax.append(lo[j])
            # latmax.append(la[j])
        
        #     count +=1
        #     if count ==24:
        #         break
        # else:
        #     count=0
     # 至少在限制条件内存在1天
    
    if count >=1:#!= 0:
        print(count)
        lon.append(lo)
        lat.append(la)
        
        cmslp.append(mslpp)
        absc.append(i)
        tim.append(ti)
        win10.append(wi10)
        maxp.append(mslpp.index(min(mslpp)))
        # sicc = 0
        # for k in range(len(lo)):
        #     if lonmh[k+cstart] >= 230 and lonmh[k+cstart] <= 300 and latmh[k+cstart] >= -75 and sicw[k+cstart] > 0. and sicw[j+cstart] <= 1:
        #         tt.append(float(wholeh[k+cstart][1]))  # 气旋符合条件的轨迹点时刻
        #         loni.append(lonmh[k+cstart])
        #         lati.append(latmh[k+cstart])
        #         sicc = sicc+1
        # 在限制条件里的气旋经纬度，不一定是气旋整条轨迹
        
        # sic.append(count)  # 经过海冰的轨迹点数
        # lonii.append(loni)
        # latii.append(lati)
        # t.append(tt)


#至少有一个轨迹点到达60度以南

absc, t = [], []
lon, lat = [], []
cmslp = []
lonii, latii = [], []
sic = []
tim = []
maxp = []
lonmax,latmax=[],[]
win10=[]
for i in range(len(whole)):  # len(whole)
    tt = []
    if i<=1053:
        cstart = int(ecline[i][3])-1
        cend = int(ecline[i][4])
    else:
        cstart = int(ecline[i][3])-1+127244
        cend = int(ecline[i][4])+127244
 
    # 每个气旋轨迹
    lo = lonmh[cstart:cend]
    la = latmh[cstart:cend]
    mslpp = list(mslph[cstart:cend])
    ti = timeh[cstart:cend]
    wi10 = wind10h[cstart:cend]
    
    loni, lati = [], []
    
    count = 0
    for j in range(len(lo)):
        if lonmh[j+cstart] >= 230 and lonmh[j+cstart] <= 300 and latmh[j+cstart] <= -60 and latmh[j+cstart] >= -75:#  #sicw[j+cstart] > 0. and sicw[j+cstart] <= 1: # and mslpp[j] == max(mslpp):  # 选定在as区域内且靠近冰区的气旋
           # if  latmh[j+cstart]<latmh[j+cstart-3]:
                count+=1            
                # tt.append(float(ti[j]))  # 气旋符合条件的轨迹点时刻
                # loni.append(lonmh[j+cstart])
                # lati.append(latmh[j+cstart])
        
            # lonmax.append(lo[j])
            # latmax.append(la[j])
        
        #     count +=1
        #     if count ==24:
        #         break
        # else:
        #     count=0
     # 至少在限制条件内存在1天
    
    if count >=1:#!= 0:
        print(count)
        lon.append(lo)
        lat.append(la)
        
        cmslp.append(mslpp)
        absc.append(i)
        tim.append(ti)
        win10.append(wi10)
        maxp.append(mslpp.index(min(mslpp)))
        # sicc = 0
        # for k in range(len(lo)):
        #     if lonmh[k+cstart] >= 230 and lonmh[k+cstart] <= 300 and latmh[k+cstart] >= -75 and sicw[k+cstart] > 0. and sicw[j+cstart] <= 1:
        #         tt.append(float(wholeh[k+cstart][1]))  # 气旋符合条件的轨迹点时刻
        #         loni.append(lonmh[k+cstart])
        #         lati.append(latmh[k+cstart])
        #         sicc = sicc+1
        # 在限制条件里的气旋经纬度，不一定是气旋整条轨迹
        
        # sic.append(count)  # 经过海冰的轨迹点数
        # lonii.append(loni)
        # latii.append(lati)
        # t.append(tt

#%%
###################################ABS海区SIE极端小值年气旋提取#####################################
#年循环最小的极小5年，也可继续添加减少
y2min = [2010, 1992, 1997, 1991, 2013]
#年循环最小的极大5年，也可继续添加减少
y3max = [2021, 1980, 1983, 2019, 2012]

def choosecyclone2a(absw, yr, tw, mslpm):
    c, ct, mslpmax = [], [], []
 
    for y in range(len(yr)):  # 考虑消融期（上一年10，11，12月和当年1，2，3月）
        for i in range(len(absw)):
            if int(year[absw[i]]) == yr[y]:
                # print(y)
                for m in [1, 2,3]:
                    if int(mon[absw[i]]) == m:

                        c.append(absw[i])
                        ct.append(tw[i])
                        mslpmax.append(mslpm[i])
            if int(year[absw[i]]) == yr[y]-1:
                for m in [10, 11, 12]:
                    if int(mon[absw[i]]) == m:
                        c.append(absw[i])
                        ct.append(tw[i])
                        mslpmax.append(mslpm[i])

    return c, ct, mslpmax

def seperate_mon(c, ct, mo):
    abscm = []
    abscmt = []
    for l, k in enumerate(c):  # [c2min,c2max][c9min,c9max]
        mm, tt = [], []
        for i in mo:  # [10,11,12,1,2][5,6,7,8,9]
            m, t = [], []
            for g, j in enumerate(k):
                if mon[j] == i:
                    m.append(j)
                    t.append(ct[l][g])

            mm.append(m)
            tt.append(t)
        abscm.append(mm)
        abscmt.append(tt)
    return abscm, abscmt

def findtrackma(absw, c, low, law):
    lonsc, latsc = [], []
    maxtt = []
    siecc = []
    mslp_c = []
    ind=[]
    wind=[]
    for i in range(len(c)):
        lonscx, latscx = [], []
        maxt = []
        siec = []
        mslp_ = []
        index=[]
        wi=[]
        for k in range(len(c[i])):
            for j in range(len(absw)):
                if c[i][k] == absw[j]:
                    lonscx.append(low[j])
                    latscx.append(law[j])
                    maxt.append(maxp[j])
                    # siec.append(sic[j])
                    mslp_.append(cmslp[j])
                    index.append(absw[j])
                    wi.append(win10[j])
        lonsc.append(lonscx)
        latsc.append(latscx)
        maxtt.append(maxt)
        wind.append(wi)
        # siecc.append(siec)
        mslp_c.append(mslp_)
        ind.append(index)
    return lonsc, latsc, maxtt,  mslp_c,ind,wind

def cyc_catcha(absw,tim,maxp,lon,lat):
    
    c2min, c2mint, c2minmp = choosecyclone2a(absw, y2min, tim, maxp)   
    c2max, c2maxt, c2maxmp = choosecyclone2a(absw, y3max, tim, maxp)
    #年循环最小的极小年
    c2m, c2mt = seperate_mon([c2min], [c2mint], [10, 11, 12, 1, 2,3])
    c2minm,c2minmt= c2m[0], c2mt[0]
    #年循环最小的极大年
    c2m, c2mt = seperate_mon([c2max], [c2maxt], [ 10,11, 12, 1, 2,3])
    c2maxm,c2maxmt= c2m[0], c2mt[0]
   
    lonsc2x, latsc2x, maxt2x, mslp2x,ind2x,wind2x = findtrackma(absw, c2maxm, lon, lat)
    lonsc2i, latsc2i, maxt2i,  mslp2i,ind2i,wind2i = findtrackma(absw, c2minm, lon, lat)
  
    lonss = [lonsc2i, lonsc2x]
    latss = [latsc2i, latsc2x]
    maxtss = [maxt2i, maxt2x]
    timetick = [c2minmt, c2maxmt]
    mslpss = [mslp2i, mslp2x]
    indss=[ind2i,ind2x]
    windss=[wind2i,wind2x]
return lonss,latss,maxtss,timetick,mslpss,indss,windss

#得到选择年份的气旋轨迹经纬度、最小mslp时刻、各个轨迹时刻、mslp、在whole中的索引、风速，每个数组结构为（年类型，月，轨迹数，轨迹点）
lonss, latss, maxtss, timetick, mslpss, indss, windss = cyc_catcha(absc,tim,maxp,lon,lat)

#以上代码可以选择多个年份，得到多年份的集合轨迹，若要得到每个年份分离的轨迹信息，则需要得到每年sie达到最大/小的月份（通过yearly_max、min_dates确定，代码在海冰章节），使用以下代码：

def choosecyclone_yeara(absw, yr, tw, mslpm,ex_mon):   
    c, ct, mslpmax = [], [], []
    if ex_mon >3: # 获取每年最小值对应的月份
       for i in range(len(absw)):
           if int(year[absw[i]]) == yr:
               for m in [ex_mon-4,ex_mon-3,ex_mon-2,ex_mon-1,ex_mon]:
                   if int(mon[absw[i]]) == m:
                       c.append(absw[i])
                       ct.append(tw[i])
                       mslpmax.append(mslpm[i])    
    if ex_mon ==3:
        for i in range(len(absw)):
            if int(year[absw[i]]) == yr:
                for m in [1, 2,3]:
                    if int(mon[absw[i]]) == m:   
                        c.append(absw[i])
                        ct.append(tw[i])
                        mslpmax.append(mslpm[i])
            if int(year[absw[i]]) == yr-1:
                for m in [10, 11,12]:
                    if int(mon[absw[i]]) == m:
                        c.append(absw[i])
                        ct.append(tw[i])
                        mslpmax.append(mslpm[i])               
    else:
        for i in range(len(absw)):
            if int(year[absw[i]]) == yr:
                for m in [1, 2,3]:
                    if int(mon[absw[i]]) == m:
                        c.append(absw[i])
                        ct.append(tw[i])
                        mslpmax.append(mslpm[i])
            if int(year[absw[i]]) == yr-1:
                for m in [ 10,11,12]:
                    if int(mon[absw[i]]) == m:
                        c.append(absw[i])
                        ct.append(tw[i])
                        mslpmax.append(mslpm[i])

    return c, ct, mslpmax

def seperate_mon(c, ct, mo):
    abscm = []
    abscmt = []
    for l, k in enumerate(c): 
        mm, tt = [], []
        for i in mo:  # [10,11,12,1,2]
            m, t = [], []
            for g, j in enumerate(k):
                if mon[j] == i:
                    m.append(j)
                    t.append(ct[l][g])
            mm.append(m)
            tt.append(t)
        abscm.append(mm)
        abscmt.append(tt)
    return abscm, abscmt

def findtrackma(absw, c, low, law):
    lonsc, latsc = [], []
    maxtt = []
    siecc = []
    mslp_c = []
    ind=[]
    wind=[]
    for i in range(len(c)):
        lonscx, latscx = [], []
        maxt = []
        siec = []
        mslp_ = []
        index=[]
        wi=[]
        for k in range(len(c[i])):
            for j in range(len(absw)):
                if c[i][k] == absw[j]:
                    lonscx.append(low[j])
                    latscx.append(law[j])
                    maxt.append(maxp[j])
                    mslp_.append(cmslp[j])
                    index.append(absw[j])
                    wi.append(win10[j])
        lonsc.append(lonscx)
        latsc.append(latscx)
        maxtt.append(maxt)
        wind.append(wi)
        mslp_c.append(mslp_)
        ind.append(index)
    return lonsc, latsc, maxtt, mslp_c,ind,wind

def cyc_catch_yearlya(absw):
    ind2_all=[]
    mslp2_all=[]
    sic2_all=[]
    maxt2_all=[]
    latsc2_all=[]
    lonsc2_all=[]
    wind2_all=[]
    timetick2_all=[]
    for y in min_dates.index[1:]:
        min_mon= min_dates[y].month
        c2, c2t, c2mp = choosecyclone_yeara(absw, y, tim, maxp,min_mon) 
        if min_mon>3:
            c2m,c2mt=seperate_mon([c2],[c2t],[min_mon-4,min_mon-3,min_mon-2,min_mon-1,min_mon])    

        if min_mon==3:
            c2m,c2mt=seperate_mon([c2],[c2t],[10,11,12,1,2,3])    

        else:
            c2m,c2mt=seperate_mon([c2],[c2t],[10,11,12,1,2,3])    

        lonsc2, latsc2, maxt2, mslp2,ind2,wind2 = findtrackma(absw, c2m[0], lon, lat)    
        lonsc2_all.append(lonsc2)
        latsc2_all.append(latsc2)
        maxt2_all.append(maxt2)
        mslp2_all.append(mslp2)
        ind2_all.append(ind2)
        wind2_all.append(wind2)
        timetick2_all.append(c2mt)
    return timetick2_all,lonsc2_all,latsc2_all,maxt2_all,mslp2_all,ind2_all,wind2_all

#按年提取每条轨迹各个轨迹点时刻、经纬度、最小气压时刻、气压值、在absc中的索引、风速，数据格式（年份，月，轨迹数，轨迹点）
timetick2_all,lonsc2_all,latsc2_all,maxt2_all,mslp2_all,ind2_all,wind2_all=cyc_catch_yearlya(absc)

#%%
#求气旋移速，每个轨迹点的距离/时间，单位为千米/每小时
#计算由纬度和经度指定的两点之间的距离
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
return R * c

dlatss2_all = []  # 存储6个月结果
for j in range(len(latsc2_all)):  # 遍历6个月
    monthly_dlat = []  #每个月结果
    for m in range(len(latsc2_all[j])):  # 遍历44年
        yearly_dlat= []  # 每年的轨迹结果
        for k in range(len(latsc2_all[j][m])):  # 遍历轨迹
            lon_array = np.array(lonsc2_all[j][m][k])
            lat_array = np.array(latsc2_all[j][m][k])
            t = np.array(timetick2_all[j][0][m][k])

            dlat = []
            for l in range(len(lon_array)):
                
                if l<len(lon_array)-1: 
                    time_diff = t[l + 1] - t[l]
                    if time_diff > 0: # 确保时间差非零
                        # 使用 haversine 计算距离并归一化时间差
                        distance = haversine(lon_array[l], lat_array[l], lon_array[l + 1], lat_array[l + 1])
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

# 构建气旋特征网格化数据
#将所有轨迹点按年份和月份分配，数据结构为44年，7个月（有些轨迹点会越过月份），并有包含经纬度、强度、速度、气旋id 5个变量储存在字典中。
timetick_area_tina2 = []
for i in range(44):
    monthly_valid_points = defaultdict(list)
    for j in range(len(lonsc2_all[i])):
        for k in range(len(lonsc2_all[i][j])):
            for l in range(len(lonsc2_all[i][j][k])):
               tin = datetime(1979, 1, 1, 0, 0, 0)+timedelta(hours=timetick2_all[i][0][j][k][l])
               month=tin.month
               # if 230<=lonsc2_all[i][j][k][l]<=300 :
                   # 可以只存时间，也可以加上经纬度等信息
               monthly_valid_points[month].append({
    
                    "lon": lonsc2_all[i][j][k][l],
                    "lat": latsc2_all[i][j][k][l],
                    'Intensity':mslp2_all[i][j][k][l],
                    'Speed':dlatss2_all[i][j][k][l], 
                    "cyclone_id": f"{j}_{k}"
                })
             
    timetick_area_tina2.append(monthly_valid_points)

#空间范围，可调整
lon_min, lon_max =200, 300 #经度
lat_min, lat_max = -80, -40 #纬度
grid_resolution = 1 #格点精度，单位为度
num_lon_bins = int((lon_max - lon_min) / grid_resolution)
num_lat_bins = int((lat_max - lat_min) / grid_resolution)

# 创建网格点的经纬度
grid_lons = lon_min + np.arange(num_lon_bins + 1) * grid_resolution
grid_lats = lat_min + np.arange(num_lat_bins + 1) * grid_resolution

#挑选气旋影响半径，每个格点为中心以ra为半径的圆内出现气旋轨迹点，则记录为该格点的气旋信息，可调整
ra = 750 #单位为千米
lon_margins = ra / (111 * np.cos(np.radians(grid_lats))) #计算不同纬度的距离
# 对每个纬度，向外扩展气旋影响半径的边界经度区间
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
#月份
sepmon=[10, 11, 12, 1, 2,3]
#计算气旋的轨迹密度、平均强度和移速
def calden(latsc_all,timetick_a):
    # 新增网格变量：存储强度和移速的总和及计数
    sum_intensity_grid = np.zeros((6, len(latsc_all), num_lat_bins + 1, max_nlon)) # 强度总和
    sum_speed_grid = np.zeros((6, len(latsc_all), num_lat_bins + 1, max_nlon))  # 移速总和
    count_grid = np.zeros((6, len(latsc_all), num_lat_bins + 1, max_nlon))  # 轨迹点数量
    
    for i in range(0,len(latsc_all)):
        m=0 
        for mon,records in timetick_a[i].items():
            if mon in sepmon:
                # 拆解轨迹点属性
                lons = np.array([r['lon'] for r in records])
                lats = np.array([r['lat'] for r in records])
                intensities = np.array([r['Intensity'] for r in records])
                speeds = np.array([r['Speed'] for r in records])
                dist = haversine(grid_lon[..., np.newaxis], grid_lat[..., np.newaxis], lons, lats)
                # 计算哪些轨迹点在750km范围内
                in_range = dist < ra  # shape: (grid_lat, grid_lon, n_points)
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
#强度/速度/轨迹密度分布
average_intensity_grid_2,average_speed_grid_2,count_grid_2=calden(latsc2_all,timetick_area_tina2)

#取出每一年SIE极小值对应的月份索引
b = min_dates[1:].dt.month + 2
#求空间异常分布，原场-平均场
count_grid_2_a= count_grid_2-np.mean(count_grid_2[b.values, np.arange(44),:,:], axis=0, keepdims=True)  
average_intensity_grid_2_a= average_intensity_grid_2-np.mean(average_intensity_grid_2[b.values, np.arange(44),:,:], axis=0, keepdims=True)  
average_speed_grid_2_a= average_speed_grid_2-np.mean(average_speed_grid_2[b.values, np.arange(44),:,:], axis=0, keepdims=True)

#%%
##可选：对格点去年代际趋势
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
        if m.sum() < 10:   
            continue
        y = X2[m, p]
        tt = t[m]
        a, b = np.polyfit(tt, y, 1)
        trend = a * t + b
        X_dt[m, p] = X2[m, p] - trend[m]  # 只在有效值处写回

    return X_dt.reshape(T, Ny, Nx)

#若需要去趋势可选以下代码
average_intensity_grid_2 = detrend_3d_field(average_intensity_grid_2[b.values, np.arange(44),:,:])
average_speed_grid_2 = detrend_3d_field(average_speed_grid_2[b.values, np.arange(44),:,:])
count_grid_2 = detrend_3d_field(count_grid_2[b.values, np.arange(44),:,:])
#求异常
average_intensity_grid_2_a= average_intensity_grid_2-np.mean(average_intensity_grid_2, axis=0, keepdims=True)
average_speed_grid_2_a= average_speed_grid_2-np.mean(average_speed_grid_2, axis=0, keepdims=True)
count_grid_2_a= count_grid_2-np.mean(count_grid_2, axis=0, keepdims=True)

#%%
#对选定年份的轨迹和轨迹密度进行可视化，并显示海冰范围
ice_2010 =icemean_all.sel(time=((icemean_all.time.dt.year == 2010)&(icemean_all.time.dt.month.isin([1, 2,3]))) | ((icemean_all.time.dt.year == 2009)&(icemean_all.time.dt.month.isin([10,11,12]))) )

ice_1980=icemean_all.sel(time=((icemean_all.time.dt.year == 1980)&(icemean_all.time.dt.month.isin([1, 2,3]))) | ((icemean_all.time.dt.year == 1979)&(icemean_all.time.dt.month.isin([10,11,12]))) )
ice_ex=[ice_2010,ice_1980]

min_ice_months2 = ['(a) 2009-12','(b) 2010-01', '(c) 2010-02', '(d) 2010-03']
max_ice_months3 = [ '(e) 1979-12','(f) 1980-01',  '(g) 1980-02', '(h) 1980-03']
colors = [ '#8c564b','#8c564b','#1f77b4', '#9467bd','#2ca02c', '#ff7f0e']
# 创建 4 行 2 列的图表
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6, 12), dpi=300,subplot_kw={'projection': ccrs.SouthPolarStereo()})

# 遍历每行的子图
for g in range(4):
    g=g+2
    # 极小海冰气旋数据
    ax_min = axes[g-2,0]
    ax_min.set_extent([120, 350, -85, -35], ccrs.PlateCarree())
    ax_min.gridlines(alpha=0.4)
    ax_min.add_feature(cfeature.LAND, facecolor='none', edgecolor='none')
    ax_min.add_feature(cfeature.OCEAN, facecolor='none')
    ax_min.add_feature(cfeature.COASTLINE, facecolor='none')
    ax_min.set_title(f"{min_ice_months2[g-2]}", fontsize=15)

    p=0
    
    num_cyclones_min = len(lonss[p][g])  # 气旋数量
    for i in range(num_cyclones_min):
        ax_min.plot(lonss[p][g][i], latss[p][g][i],transform=ccrs.Geodetic(),linewidth=1.5, color=colors[p], zorder=3,alpha=0.5)
        # 起点
        ax_min.plot(lonss[p][g][i][0], latss[p][g][i][0], 'o', color='green', markersize=3,transform=ccrs.Geodetic())
        # 终点
        ax_min.plot(lonss[p][g][i][-1], latss[p][g][i][-1], 'x', color='red', markersize=3,transform=ccrs.Geodetic())

    # 标注气旋数量
    density_levels = np.arange(-400, 420, 20)
    density_levels1 = np.arange(-35,36, 1)
    Z = mpcalc.smooth_n_point(count_grid_2_a[g, 30], 9, 2)
    valid_xy = np.isfinite(grid_lon) & np.isfinite(grid_lat)
    Z = ma.masked_invalid(Z)                 # mask Z 中 nan
    Z = ma.masked_where(~valid_xy, Z)        # mask 坐标无效处
    
    contourf = ax_min.contourf(
        grid_lon, grid_lat, Z,
        levels=density_levels,
        transform=ccrs.PlateCarree(),
        extend="both",
        cmap="seismic",
    )
    # 标注气旋数量
    ax_min.text(0.6, 0.95, f'Num: {num_cyclones_min}', transform=ax_min.transAxes,fontsize=15, color='k', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    # 极大海冰气旋数据
    ax_max = axes[g-2,1]
    ax_max.set_extent([120, 350, -85, -35], ccrs.PlateCarree())
    ax_max.gridlines(alpha=0.4)
    ax_max.add_feature(cfeature.LAND, facecolor='none', edgecolor='none')
    ax_max.add_feature(cfeature.OCEAN, facecolor='none')
    ax_max.add_feature(cfeature.COASTLINE, facecolor='none')
    ax_max.set_title(f"{max_ice_months3[g-2]}", fontsize=15)
    
    p=1
    # 绘制轨迹并统计气旋数量
    num_cyclones_max = len(lonss[1][g])  # 气旋数量
    for i in range(num_cyclones_max):
        ax_max.plot(lonss[1][g][i],latss[1][g][i],transform=ccrs.Geodetic(),linewidth=1.5, color=colors[p], zorder=3, alpha=0.5)
        # 起点
        ax_max.plot(lonss[p][g][i][0], latss[p][g][i][0], 'o', color='green',markersize=3,transform=ccrs.Geodetic())
        # 终点
        ax_max.plot(lonss[p][g][i][-1], latss[p][g][i][-1], 'x', color='red', markersize=3,transform=ccrs.Geodetic())
    density_levels = np.arange(-400, 420, 20)
    density_levels1 = np.arange(-35,36, 1)
    
    Z = mpcalc.smooth_n_point(count_grid_2_a[g, 0], 9, 2)
    valid_xy = np.isfinite(grid_lon) & np.isfinite(grid_lat)
    Z = ma.masked_invalid(Z)                 # mask Z 中 nan
    Z = ma.masked_where(~valid_xy, Z)        # mask 坐标无效处
    
    contourf = ax_max.contourf(
        grid_lon, grid_lat, Z,
        levels=density_levels,
        transform=ccrs.PlateCarree(),
        extend="both",
        cmap="seismic",
    )
    # 标注气旋数量
    ax_max.text(0.6, 0.95, f'Num: {num_cyclones_max}', transform=ax_max.transAxes,
                fontsize=15, color='k', verticalalignment='top', bbox=dict(facecolor='white',alpha=0.7))
    
    contour1=ax_min.contour(xx,yy,ice_ex[0][g].where(lat_da<-55),levels=[0.15],colors='green',transform=ccrs.SouthPolarStereo(), zorder=2,linewidths=2,linestyles='solid')
    contour1=ax_max.contour(xx,yy,ice_ex[1][g].where(lat_da<-55),levels=[0.15],colors='green',transform=ccrs.SouthPolarStereo(),linewidths=2,linestyles='solid')
   
cbar_ax1 = fig.add_axes([0.15, -0.01, 0.7, 0.012])  # 设置颜色条位置 [x, y, width, height]
cb1 = fig.colorbar(contourf, cax=cbar_ax1, orientation='horizontal',label='Track Density Anomalies') 
# 调整布局
plt.tight_layout()
# 显示图表
plt.show()

#%%
#求极端年差值合成空间分布（SIE极小5年异常-极大5年异常）
#取出挑选年份海冰范围达到最小的月份索引，每一年达到极小的月份不同
ymin_ind = [y - 1980 for y in y2min]
ymax_ind = [y - 1980 for y in y3max]
b_min = (min_dates[1:][y2min].dt.month+2).values
b_max = (min_dates[1:][y3max].dt.month+2).values

#对齐各个月份并求异常
def calano_min(cyc_, b_,y_ind):
    #44 year average
    mean_all_0 = np.mean(cyc_[b, range(44),:,:], axis = 0)
    mean_all_m1 = np.mean(cyc_[b-1, range(44),:,:], axis = 0)
    mean_all_m2 = np.mean(cyc_[b-2, range(44),:,:], axis = 0)
    mean_all_m3 = np.mean(cyc_[b-3, range(44),:,:], axis = 0)
    # 合成最小月数据
    composite_0 = np.array([cyc_[b_[i],y_ind[i], :, :] for i in range(len(b_))])
    # 合成最小前1月
    composite_m1 = np.array([cyc_[b_[i]-1, y_ind[i], :, :] for i in range(len(b_))])
    # 合成最小前2月
    composite_m2 = np.array([cyc_[b_[i]-2, y_ind[i], :, :] for i in range(len(b_))])
    # 合成最小前3月
    composite_m3 = np.array([cyc_[b_[i]-3, y_ind[i], :, :] for i in range(len(b_))])
    #计算异常
    ano_0 = np.mean(composite_0,axis = 0) - mean_all_0
    ano_1 = np.mean(composite_m1,axis = 0) - mean_all_m1
    ano_2 = np.mean(composite_m2,axis = 0) - mean_all_m2
    ano_3 = np.mean(composite_m3,axis = 0) - mean_all_m3

    anomaly= np.array([ano_3, ano_2, ano_1, ano_0])
return anomaly

#SIE极大/小年轨迹密度异常
anomaly_ymax_c = calano_min(count_grid_2,b_max,ymax_ind)
anomaly_ymin_c = calano_min(count_grid_2,b_min,ymin_ind)
#两类型年异常差
dmin_max_c = anomaly_ymin_c - anomaly_ymax_c 
#SIE极大/小年强度异常
anomaly_ymax_in = calano_min(average_intensity_grid_2,b_max,ymax_ind)
anomaly_ymin_in = calano_min(average_intensity_grid_2,b_min,ymin_ind)
dmin_max_in = anomaly_ymin_in - anomaly_ymax_in
#SIE极大/小年移速异常
anomaly_ymax_sp = calano_min(average_speed_grid_2,b_max,ymax_ind)
anomaly_ymin_sp = calano_min(average_speed_grid_2,b_min,ymin_ind)
dmin_max_sp = anomaly_ymin_sp - anomaly_ymax_sp

# ##可选：对每个格点去趋势
# def detrend_3d_field(X):
#     """
#     X: (T, Y, X) numpy array
#     return: X_dt same shape, linear detrended along time at each grid point
#     """
#     T, Ny, Nx = X.shape
#     t = np.arange(T, dtype=float)

#     # reshape to (T, P)
#     X2 = X.reshape(T, -1)
#     mask = np.isfinite(X2)

#     X_dt = np.full_like(X2, np.nan, dtype=float)

#     # 对每个格点做回归：y = a*t + b，仅用有限值
#     for p in range(X2.shape[1]):
#         m = mask[:, p]
#         if m.sum() < 10:   
#             continue
#         y = X2[m, p]
#         tt = t[m]
#         a, b = np.polyfit(tt, y, 1)
#         trend = a * t + b
#         X_dt[m, p] = X2[m, p] - trend[m]  # 只在有效值处写回
#     return X_dt.reshape(T, Ny, Nx)

# #若需去趋势选择此代码
# def calano_min(cyc_, b_,y_ind):
#     #44 year average
#     mean_all_0 = np.mean(detrend_3d_field(cyc_[b, range(44),:,:]), axis = 0)
#     mean_all_m1 = np.mean(detrend_3d_field(cyc_[b-1, range(44),:,:]), axis = 0)
#     mean_all_m2 = np.mean(detrend_3d_field(cyc_[b-2, range(44),:,:]), axis = 0)
#     mean_all_m3 = np.mean(detrend_3d_field(cyc_[b-3, range(44),:,:]), axis = 0)
#     # 合成最小月数据
#     composite_0 = detrend_3d_field(cyc_[b, range(44),:,:])[y_ind]
#     # 合成最小前1月
#     composite_m1 = detrend_3d_field(cyc_[b-1, range(44),:,:])[y_ind]
#     # 合成最小前2月
#     composite_m2 = detrend_3d_field(cyc_[b-2, range(44),:,:])[y_ind]
#     # 合成最小前3月
#     composite_m3 = detrend_3d_field(cyc_[b-3, range(44),:,:])[y_ind]
    
#     #计算异常
#     ano_0 = np.mean(composite_0,axis = 0) - mean_all_0
#     ano_1 = np.mean(composite_m1,axis = 0) - mean_all_m1
#     ano_2 = np.mean(composite_m2,axis = 0) - mean_all_m2
#     ano_3 = np.mean(composite_m3,axis = 0) - mean_all_m3

#     anomaly= np.array([ano_3, ano_2, ano_1, ano_0])
#     return anomaly

#对差值场做显著性检验 t-test
def calano_min_yearly(cyc_, b_,y_ind):
    #44 year average
    mean_all_0 = np.mean(cyc_[b, range(44),:,:], axis = 0)
    mean_all_m1 = np.mean(cyc_[b-1, range(44),:,:], axis = 0)
    mean_all_m2 = np.mean(cyc_[b-2, range(44),:,:], axis = 0)
    mean_all_m3 = np.mean(cyc_[b-3, range(44),:,:], axis = 0)
    # 合成最小月数据
    composite_0 = np.array([cyc_[b_[i],y_ind[i], :, :] for i in range(len(b_))])
    # 合成最小前1月
    composite_m1 = np.array([cyc_[b_[i]-1, y_ind[i], :, :] for i in range(len(b_))])
    # 合成最小前2月
    composite_m2 = np.array([cyc_[b_[i]-2, y_ind[i], :, :] for i in range(len(b_))])
    # 合成最小前3月
    composite_m3 = np.array([cyc_[b_[i]-3, y_ind[i], :, :] for i in range(len(b_))])
    
    #计算异常
    ano_0 = [composite_0[i] - mean_all_0 for i in range(len(composite_0))]
    ano_1 = [composite_m1[i] - mean_all_m1 for i in range(len(composite_m1))]
    ano_2 = [composite_m2[i] - mean_all_m2 for i in range(len(composite_m2))]
    ano_3 = [composite_m3[i] - mean_all_m3 for i in range(len(composite_m3))]

    anomaly= np.array([ano_3, ano_2, ano_1, ano_0])
return anomaly

#5年各自的异常
anomaly_ymax_c_y = calano_min_yearly(count_grid_2,b_max,ymax_ind)
anomaly_ymin_c_y = calano_min_yearly(count_grid_2,b_min,ymin_ind)
dmin_max_c_y = anomaly_ymin_c_y - anomaly_ymax_c_y

anomaly_ymax_in_y = calano_min_yearly(average_intensity_grid_2,b_max,ymax_ind)
anomaly_ymin_in_y = calano_min_yearly(average_intensity_grid_2,b_min,ymin_ind)
dmin_max_in_y = anomaly_ymin_in_y - anomaly_ymax_in_y

anomaly_ymax_sp_y = calano_min_yearly(average_speed_grid_2,b_max,ymax_ind)
anomaly_ymin_sp_y = calano_min_yearly(average_speed_grid_2,b_min,ymin_ind)
dmin_max_sp_y = anomaly_ymin_sp_y - anomaly_ymax_sp_y

#显著性检验
from scipy.stats import ttest_ind
_, p_val_sp = ttest_ind(anomaly_ymin_sp_y, anomaly_ymax_sp_y, axis=1, equal_var=False, nan_policy='omit')
sig_sp = p_val_sp < 0.05

_, p_val_in = ttest_ind(anomaly_ymin_in_y, anomaly_ymax_in_y, axis=1, equal_var=False, nan_policy='omit')
sig_in = p_val_in < 0.05

_, p_val_c = ttest_ind(anomaly_ymin_c_y, anomaly_ymax_c_y, axis=1, equal_var=False, nan_policy='omit')
sig_c = p_val_c < 0.05

# #若需要去趋势，选择此代码
# def calano_min_yearly(cyc_, b_,y_ind):
#     #44 year average
#     mean_all_0 = np.mean(detrend_3d_field(cyc_[b, range(44),:,:]), axis = 0)
#     mean_all_m1 = np.mean(detrend_3d_field(cyc_[b-1, range(44),:,:]), axis = 0)
#     mean_all_m2 = np.mean(detrend_3d_field(cyc_[b-2, range(44),:,:]), axis = 0)
#     mean_all_m3 = np.mean(detrend_3d_field(cyc_[b-3, range(44),:,:]), axis = 0)
#     # 合成最小月数据
#     composite_0 = detrend_3d_field(cyc_[b, range(44),:,:])[y_ind]
#     # 合成最小前1月
#     composite_m1 = detrend_3d_field(cyc_[b-1, range(44),:,:])[y_ind]
#     # 合成最小前2月
#     composite_m2 = detrend_3d_field(cyc_[b-2, range(44),:,:])[y_ind]
#     # 合成最小前3月
#     composite_m3 = detrend_3d_field(cyc_[b-3, range(44),:,:])[y_ind]
#     #计算异常
#     ano_0 = [composite_0[i] - mean_all_0 for i in range(len(composite_0))]
#     ano_1 = [composite_m1[i] - mean_all_m1 for i in range(len(composite_m1))]
#     ano_2 = [composite_m2[i] - mean_all_m2 for i in range(len(composite_m2))]
#     ano_3 = [composite_m3[i] - mean_all_m3 for i in range(len(composite_m3))]
    
#     anomaly= np.array([ano_3, ano_2, ano_1, ano_0])
#     return anomaly

#%%
#可视化海冰达到极小的月份及其前三月差值场

def add_lat_labels_on_sector(ax, lat_values, label_lon,fontsize=14, fontweight='normal', ha='left', va='bottom', offset=0):
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

def add_lon_labels_on_sector(ax, lon_values, label_lat, fontsize=14, fontweight='normal', ha='right', va='top', offset=0):
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

min_ice_months2 = ['lag = -3','lag = -2', 'lag = -1', 'lag = 0']
max_ice_months9 = ['2015-05', '2015-06', '2015-07', '2015-08', '2015-09']
min_ice_months8 = ['1988-04', '1988-05', '1988-06', '1988-07', '1988-08']
max_ice_months3 = [ '(e) lag=-3','(f) lag=-2',  '(g) lag=-1', '(h) lag=0']
max_ice_months4 = [ '(i) lag =-3','(j) lag=-2',  '(k) lag=-1', '(l) lag=0']

ymin_ind=[y - 1980 for y in y2min]
ymax_ind=[y - 1980 for y in y3max]
b_min = (min_dates[1:][y2min].dt.month+2).values
b_max = (min_dates[1:][y3max].dt.month+2).values

num_min=[]
num_max=[]
labels = list(string.ascii_lowercase)
for i in reversed(range(4)):
    num_min.append( sum([len(lonsc2_all[y][m-i]) for m, y in zip(b_min, ymin_ind)]))
    num_max.append( sum([len(lonsc2_all[y][m-i]) for m, y in zip(b_max, ymax_ind)]))

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 12), dpi=300,
                         subplot_kw={'projection': ccrs.SouthPolarStereo()})
# 遍历每行的子图
for g in range(4):
    # 设置扇形边界
    theta = np.linspace(-14/18*np.pi, -np.pi / 4, 100)
    radius_outer, radius_inner = 0.43, 0.11
    center = [0.5, 0.5]
    outer_arc = np.vstack([np.sin(theta) * radius_outer, np.cos(theta) * radius_outer]).T + center
    inner_arc = np.vstack([np.sin(theta[::-1]) * radius_inner, np.cos(theta[::-1]) * radius_inner]).T + center
    verts = np.vstack([outer_arc, inner_arc, outer_arc[0]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts)-2) + [mpath.Path.CLOSEPOLY]

    ax_min = axes[g,0]
    pos = axes[g,0].get_position()
    axes[g,0].set_position([
        pos.x0+ 0.01, pos.y0 - 0.2, pos.width + 0.12, pos.height + 0.3
    ])
    pos = axes[g,1].get_position()
    axes[g,1].set_position([
        pos.x0+ 0.01, pos.y0 - 0.2, pos.width + 0.13, pos.height + 0.3    ])   
    pos = axes[g,2].get_position()
    axes[g,2].set_position([
        pos.x0+ 0.01,pos.y0 - 0.2,pos.width + 0.13,  pos.height + 0.3    ])
    
    leftlon, rightlon, lowerlat, upperlat = (-180, 180, -50, -90)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    ax_min.set_extent(img_extent, ccrs.PlateCarree())
    # 添加地图特征
    ax_min.add_feature(cfeature.LAND, facecolor='none')
    ax_min.add_feature(cfeature.OCEAN, facecolor='none')
    ax_min.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=2)

    p=0
    
    num_cyclones_min = num_min[g]  # 气旋数量
    density_levels = np.arange(-250, 250, 10)
    density_levels1 = np.arange(-8,8.2, 0.2)
    density_levels2 = np.arange(-25, 25, 0.5)
    
    contourf1 = ax_min.contourf(
        grid_lon, grid_lat,dmin_max_c[g], 
        levels=density_levels, transform=ccrs.PlateCarree(), 
        extend="both", cmap=cmocean.cm.balance#"seismic"
        )
    # 显著性点
    if np.any(sig_c[g]):
        sig_f = np.where(sig_c[g], 1.0, np.nan)  # 非显著设为 NaN
        ax_min.contourf(
            grid_lon, grid_lat, sig_f,
            levels=[0.5, 1.5],             
            colors='none', hatches=['..'],
            transform=ccrs.PlateCarree(),
            zorder=6
        )
    ax_min.set_boundary(mpath.Path(verts, codes), transform=ax_min.transAxes)

    # 添加网格线
    gl = ax_min.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.ylocator = mticker.FixedLocator([-50,-60, -70, -80])
    gl.xlocator = mticker.FixedLocator([-180,-150,-120, -90, -60, -30, 0])
    
    # 添加纬度标签在边缘方向
    add_lat_labels_on_sector(ax_min, [ 60,  80], label_lon=-46, offset=0)
    
    # 添加经度标签贴在 -54° 纬线上
    add_lon_labels_on_sector(ax_min, [ 120, 90, 60], label_lat=-54, offset=0)
    ax_max = axes[g,1]
    ax_max.set_extent(img_extent, ccrs.PlateCarree())
    # 添加地图特征
    ax_max.add_feature(cfeature.LAND, facecolor='none')
    ax_max.add_feature(cfeature.OCEAN, facecolor='none')
    ax_max.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=2)  
    ax_max.set_boundary(mpath.Path(verts, codes), transform=ax_max.transAxes)

    # 添加网格线
    gl = ax_max.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.ylocator = mticker.FixedLocator([-50,-60, -70, -80])
    gl.xlocator = mticker.FixedLocator([-180,-150,-120, -90, -60, -30, 0])
    # 添加纬度标签在边缘方向
    add_lat_labels_on_sector(ax_max, [ 60,  80], label_lon=-46, offset=0)
    
    # 添加经度标签贴在 -60° 纬线上
    add_lon_labels_on_sector(ax_max, [ 120, 90, 60], label_lat=-54, offset=0)
    p=1
    contourf2 = ax_max.contourf(
        grid_lon, grid_lat,dmin_max_in[g], 
        levels=density_levels2, transform=ccrs.PlateCarree(), 
        extend="both", cmap=cmocean.cm.balance#"seismic"
    ) 
    # 显著性点
    lon = grid_lon
    lat = grid_lat
    valid_xy = np.isfinite(lon) & np.isfinite(lat)  # True 表示坐标有效
    sig_mask = np.where(sig_in[g], 1.0, np.nan)
    
    # 坐标无效的地方强制 nan
    sig_mask = np.where(valid_xy, sig_mask, np.nan)
    ax_max.contourf(lon, lat, sig_mask,levels=[0.5, 1.5],
        colors="none",
        hatches=[".."],
        transform=ccrs.PlateCarree(),
    )
 
    # 差值
    ax_d = axes[g,2]
    ax_d.set_extent(img_extent, ccrs.PlateCarree())
    # 添加地图特征
    ax_d.add_feature(cfeature.LAND, facecolor='none')
    ax_d.add_feature(cfeature.OCEAN, facecolor='none')
    ax_d.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=2)  
    
    ax_d.set_boundary(mpath.Path(verts, codes), transform=ax_d.transAxes)

    # 添加网格线
    gl = ax_d.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.ylocator = mticker.FixedLocator([-50,-60, -70, -80])
    gl.xlocator = mticker.FixedLocator([-180,-150,-120, -90, -60, -30, 0])
    # 添加纬度标签在边缘方向
    add_lat_labels_on_sector(ax_d, [ 60,  80], label_lon=-46, offset=0)
    # 添加经度标签贴在 -54° 纬线上
    add_lon_labels_on_sector(ax_d, [120, 90, 60], label_lat=-54, offset=0)
    p=0
    anomaly_masked = np.ma.masked_where(np.isnan(grid_lat), dmin_max_c[g])
    
    contourf3 = ax_d.contourf(
        grid_lon, grid_lat,dmin_max_sp[g]/3.6, 
        levels=density_levels1, transform=ccrs.PlateCarree(), 
        extend="both", cmap=cmocean.cm.balance#"seismic"
    )  
    # 显著性点
    ax_d.contourf(grid_lon, grid_lat, sig_sp[g], levels=[0.5, 1], colors='none', hatches=['..'], transform=ccrs.PlateCarree())

for idx, ax in enumerate(axes.flatten(order='F')):
    ax.text(0.05, 0.85, f'({labels[idx]})', transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

cbar_ax1 = fig.add_axes([0.12, 0.02, 0.2, 0.01])  
cb1=fig.colorbar(contourf1,cax=cbar_ax1, orientation='horizontal',label="Track Count")
cb1.ax.set_xlabel("Track Count", labelpad=-10) 
cb1.set_ticks([-250, 250])   
cbar_ax2 = fig.add_axes([0.4, 0.02, 0.2, 0.01]) 
cb2 = fig.colorbar(contourf2, cax=cbar_ax2, orientation='horizontal',label='Intensity (hPa)')
cb2.set_ticks([-25, 25])
cb2.ax.set_xlabel('Intensity (hPa)', labelpad=-10)  
cbar_ax3 = fig.add_axes([0.67, 0.02, 0.2, 0.01])  
cb3 = fig.colorbar(contourf3, cax=cbar_ax3, orientation='horizontal',label='Speed (m s$^{-1}$)')
cb3.set_ticks([-8,8])
cb3.ax.set_xlabel('Speed (m s$^{-1}$)', labelpad=-10)  

# 显示图表
plt.show()

#%%
#EOF分解
from eofs.standard import Eof
#计算纬度权重
coslat = np.cos(np.deg2rad(np.array(grid_lats)))
wgts = np.sqrt(coslat)[..., np.newaxis]
#输入（年份，x，y）数据类型，这里的b表示44年里每一年的极小值对应月份
solver1 = Eof(average_intensity_grid_2[b.values, np.arange(44),:,:],weights = wgts)
eof = solver1.eofs(neofs=3, eofscaling=2)
pc = solver1.pcs(npcs=3, pcscaling=1)
var = solver1.varianceFraction()
errors = solver1.northTest(neigs=5, vfscaled=True)
#计算特征值和NorthTest的结果
eigenvalues =solver1.eigenvalues(neigs=40)
errors = solver1.northTest(neigs=40)
# 绘制Error Bar图，挑选显著模态
plt.figure(figsize=(8, 6),dpi=200)
plt.errorbar(np.arange(1, 41), eigenvalues, yerr=errors, fmt='o-', capsize=5)
plt.xlabel('EOF modes')
plt.ylabel('Eigenvalues')
plt.title('NorthTest(with error bars)')
plt.grid()
plt.show()

#对EOF空间分布显著性检验
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

    # 打包为 DataArray
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

# 计算每个空间点的强度异常值
average_intensity_grid_2_a=average_intensity_grid_2-np.mean(average_intensity_grid_2[b.values, np.arange(44),:,:], axis=0, keepdims=True)  

I_da = xr.DataArray(
    average_intensity_grid_2_a[b.values, np.arange(44),:,:],
    dims=("time", "y", "x"), name="ci",
)
# 把 2D 的经纬度作为坐标挂到这两个维度上
I_da = I_da.assign_coords(
    lat=(("y", "x"), grid_lat),
    lon=(("y", "x"), grid_lon),
)

out = regress_grid(field=I_da, pc2=pc[:, 1], use_neff=True)
p_val = out['p']

#可视化
fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=600, subplot_kw={'projection': ccrs.SouthPolarStereo()})

# EOF 颜色设置
clevels = np.arange(-4, 4.25, 0.25)
cmap = plt.cm.RdBu_r
# 标题标签
titles = ["(a) EOF1", "(b) EOF2", "(c) EOF3", "(d) EOF4", "(e) EOF5"]
pc_titles = ["(c) PC1", "(d) PC2", "(g) PC3", "(h) PC4", "(i) PC5"]
colors = ["b", "r", "g", "k","g"]  # PC1, PC2, PC3,

# 开启 constrained_layout
fig.subplots_adjust( top=0.9,bottom=0.2,wspace=0.09, hspace=0.6)

for i in range(2):
    # 绘制 EOF 模态
    ax = axes[i]
    ax.text(0.01, 0.89, f"{titles[i]}  {var[i]*100:.1f}%", transform=ax.transAxes, va='top', ha='left')
    
    c = ax.contourf(grid_lon, grid_lat, eof[i, :, :], levels=clevels, extend='both',
                    transform=ccrs.PlateCarree(), cmap=cmap, alpha = 0.8 )

    # 右侧：PC 时间序列（普通 2D 轴）
    ax_pc = fig.add_subplot(2, 2, 2*i+2)      
    if i == 1:
         # 显著性：p < 0.05 的点打小点
         sig = (p_val < 0.05)
           # 为避免 NaN 经纬度报错
         m = sig.values & np.isfinite(grid_lon) & np.isfinite(grid_lat)
         ax.scatter(grid_lon[m][::3], grid_lat[m][::3],
                      s=0.4, color='k', alpha=0.5, transform=ccrs.PlateCarree())

    # 设置第二个y轴标签
    ax_pc.set_ylabel('PC ', color='black')
    ax_pc.set_ylim(-3, 3)
    ax_pc.axhline(0, linestyle="--", color="k")
    ax_pc.bar(np.arange(1980, 2024, 1), pc[:, i], color=colors[i], alpha = 0.5)
    ax_pc.text(0.75, 0.81, f"{pc_titles[i]}", transform=ax.transAxes,  va='top', ha='left')
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
    gl.ylocator = mticker.FixedLocator([-60, -70, -80])
    gl.xlocator = mticker.FixedLocator([-120, -90, -60, -30, 0])
    # 添加纬度标签在边缘方向
    add_lat_labels_on_sector(ax, [60, 70, 80], label_lon=-46, offset=0)
    # 添加经度标签贴在 -54° 纬线上
    add_lon_labels_on_sector(ax, [ 120, 90, 60], label_lat=-54, offset=0)
axes[0].set_position([0.2, 0.55, 0.45, 0.45])
axes[1].set_position([0.2, 0.18, 0.45, 0.45])

# 添加统一色标
cb_ax = fig.add_axes([0.23, 0.23, 0.2, 0.013])  # 颜色条位置
cb=fig.colorbar(c, cax=cb_ax, orientation='horizontal', format='%.0f',label='Intensity (hPa)')
ticks = cb.get_ticks()  # 获取原始刻度数组
cb.set_ticks(ticks[::2])  # 仅设置首尾两个刻度
plt.show()