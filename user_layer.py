# -*- coding:utf-8 -*-
"""这里是要按照用户行为和用户类型将用户进行分层处理"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#数据提取
df = pd.read_csv('/Users/caoshishi/Downloads/linshi.txt', sep='\t',header=0,
                 names=['type','uid','view_pv','view_anchors','shot_pv','shot_anchors','duration'])
#查看数据及类型
#df.head(5)
#df.info()

#数据处理
data = df.copy()
data['duration'] = data.duration.apply(lambda x:math.log(x,10)) --方差较大该列取对数
data1 = data.iloc[:,2:]
##指标归一化
data.iloc[:,2:] = (data1-0)/(data1.max()-0)
##计算综合活跃评分
data['score'] = (data.view_pv + data.view_anchors + data.shot_pv + data.shot_anchors + data.duration)*200

#查看数据综合活跃评分密度分布
fig = plt.figure(figsize=(12,8))
plt.rcParams['font.sans-serif'] = ['SimHei']
data.score.plot(kind='kde')
plt.xlim(0,1000)
plt.xlabel(u'评分值')
plt.ylabel(u'频率')
plt.title(u'用户近一周的评分值分布情况',fontsize=20,color='black',fontweight='bold', alpha=0.8)
fig.savefig('/Users/caoshishi/Downloads/data_score.png', transparent=False, dpi=80, bbox_inches="tight")

#按活跃评分将用户分成高、中、低三个等级
bins = [0,60,120,1000]
data['level']  = pd.cut(data.score, bins=bins, labels=['low','middle','high'])

#将用户分层数据和原表进行合并，并存储到本地
dt = pd.concat([df,data[['level']]], axis=1)
dt = dt.groupby(['type','level']).agg({'view_pv':'sum','shot_pv':'sum','duration':'sum','uid':'count'})
dt.to_csv('/Users/caoshishi/Downloads/data_result1.csv')


"""以留存为目标，确定相关指标权重，相关指标有点击、短停和关注"""
###方法一：相关系数法
#数据获取
#select t1.uid,
#nvl(view_pv/show_pv,0) ctr,
#nvl(shot_pv,0) shot_pv,
#case when follow_pv>0 then 1 else 0 end followed,
#case when t3.uid is not null then 1 else 0 end reten
#from
#(select uid,count(distinct token,live_id) show_pv from hallone.mid_user_show where ymd='20200607' group by uid) t1
#left join
#(select uid,
#count(distinct case when duration>=60 then concat(token,live_id) end) shot_pv,
#count(distinct token,live_id) view_pv,
#sum(follow_pv) follow_pv
#from hallone.u_user_live_act_detail
#where ymd='20200607' and enter='首页'
#group by uid) t2
#on t1.uid=t2.uid
#left join
#(select uid from hallone.u_user_live_act_detail where ymd='20200608' group by uid) t3
#on t1.uid=t3.uid
import math
def get_corr(df):
    data = df.iloc[:, 1:].corr()['isreten'][:-1]
    s = (data*10000).map(lambda x:round(math.log(x,20),0)) #指数平滑获得权重
    return s

###方法二：熵权法
class cul_weight():
    def get_df(self,df):
        """处理数据，列名分别为组别、指标1、指标2、指标3，每行数据为每组数据统计量"""

        def str2ft(x):
            x = x[:-1]
            return float(x) * 0.01

        df['ctr'] = df.ctr.map(str2ft) #将带%的字符串转化成浮点数
        df['followed'] = df.followed.map(str2ft)
        data = df.copy()
        data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:, 1:].min()) / (
                    data.iloc[:, 1:].max() - data.iloc[:, 1:].min())  # 标准化
        data.iloc[:, 1:] = (data.iloc[:, 1:]) / (data.iloc[:, 1:].sum())  # 统计概率p
        return data

    def comentropy(self,s):
        """计算某一列的信息熵"""
        n = len(s)

        def get_list(x):
            if x > 0:
                x = x * math.log(x, math.e)
            else:
                x = 0
            return x

        s = s.map(get_list)
        E = (s.sum()) * (-1) / math.log(n, math.e)
        return E

    def get_weight(self,df):
        """根据信息熵来计算每个变量的权重"""
        self.data = self.get_df(df)
        columns = df.iloc[:, 1:].columns
        k = len(columns)
        s = []
        for x in columns:
            self.value = self.comentropy(self.data[x])
            s.append(self.value)
        wl = []
        for i in range(k):
            weight = (1-s[i])/(k-sum(s))
            wl.append(weight)
        res = pd.Series(wl, index=columns)
        return res
