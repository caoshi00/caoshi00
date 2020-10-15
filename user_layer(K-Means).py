# -*- coding:utf-8 -*-
"""
使用K-Means算法聚类用户活跃行为数据
业务关键行为：聊天、查看个人主页、进直播间
业务聚类类别：潜力新用户、直播发展用户、直播稳定用户、社交发展用户、社交稳定用户和双稳定用户6类
选取的特征有：有效活跃天数(日活跃时长超过1分钟)、浏览用户卡片数、浏览用户主页数、聊天对象数、聊天回合数、浏览直播卡片数、观看直播间数、观看时长
固定周期：1周
结果呈现：潜力新用户18%、直播发展用户27%、直播稳定用户16%、社交发展用户16%、社交稳定用户9%和双稳定用户15%
结果说明：为了固定分层结果，调度使用传入固定簇中心点，定期进行模型簇中心更新。
"""
import pandas as pd
import numpy as np
import math
"""数据准备"""
inputfile = '/Users/caoshishi/Downloads/linshi.txt'

df = pd.read_csv(inputfile, sep='\t', header=None, names=['uid','yx_days','show_uids','homepage_uids','chat_uids','round_pv','show_lives','view_lives','duration'], index_col='uid')
data = df[df.duration<36000]
def lg(x):
    return math.log(x + 1)
data.iloc[:,1:] = data.iloc[:,1:].applymap(lg) #对变量进行对数变换
data_zs = (data-data.mean())/(data.std()) #数据标准化


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#为了保证算法的收敛性，该模型只支持欧式距离

def deploy(index):
    """数据分布探查，查看对数变换后每一列的分布情况"""
    #list(np.linspace(0,200,11))+[4100]
    def lg(x):
        return math.log(x+1)
    p = pd.cut(data[index].map(lg),bins=list(np.linspace(0,11,12))).value_counts()
    p.plot(kind='bar')
    plt.show()

def k_draw(data):
    """聚类模型k值判断"""
    inertia=[]
    for k in range(2,10):
        model = KMeans(n_clusters=k, n_jobs=4, max_iter=500, n_init=10, init='k-means++') #分为k类，并发数4
        km = model.fit(data)
        center = km.cluster_centers_ #聚类的中心
        sse = km.inertia_ #sse，距离之和，勇于评估簇的个数是否合适
        inertia.append([k,sse])
    inertia = np.array(inertia)
    print (inertia)
    plt.plot(list(inertia[:,0]), list(inertia[:,1]))
    plt.title('kmeans-k')
    plt.show()



def r_cnt(k):
    """聚类结果打印"""
    #init = np.array([[-0.812291886,-1.466825492,-1.029891498,-0.627192359,-0.405749304,-0.574093109,-1.126955343,-1.321206946],[-0.282670325,-0.249581311,-0.675337093,-0.53745334,-0.391871787,-0.426475378,0.138678005,0.374757468],[0.064406041,0.50733544,0.653251039,0.199789954,-0.166006227,-0.54183605,0.801083427,0.766316067],[0.087836557,0.354454511,0.29294183,0.081085711,-0.135711258,-0.225864093,-0.827110255,-0.917137449],[0.283421019,0.613614085,0.527809823,0.174156425,-0.094625008,1.652117053,0.919807265,0.806773431],[1.71968911,1.077559169,1.511773827,2.104690448,2.743571714,0.748160092,0.31679771,0.429377792]])
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=500, n_init=10, init='k-means++').fit(data_zs)
    r1 = pd.Series(model.labels_).value_counts()
    r2 = pd.DataFrame(model.cluster_centers_, columns=data.columns)
    #r2 = r2*(data.std()) + data.mean() #重新变换成未标准化的值
    print(model.cluster_centers_)
    def ex(x):
        return (math.e)**x-1
    #r2.iloc[:,1:] = r2.iloc[:,1:].applymap(ex) #重新变换成未对数变换的值
    r = pd.concat([r2,r1],axis=1)  #簇中心坐标及类别数量统计
    r.columns = list(data.columns) + ['count']
    r.to_excel('/Users/caoshishi/Downloads/leibie2.xls')

    result = pd.concat([df[df.duration<36000],pd.Series(model.labels_, index=data.index)], axis=1) #输出原始数据及分类结果
    result.columns = list(data.columns) + ['leibie']
    #result.to_csv('/Users/caoshishi/Downloads/data_result.csv')
    return result

def r_show(result):
    """聚类结果可视化"""
    from sklearn.manifold import TSNE
    tsne = TSNE()
    data = data_zs.sample(n=10000)
    print (data.head(5))
    tsne.fit_transform(data) #进行数据降维
    tsne = pd.DataFrame(tsne.embedding_, index=data.index) #转换数据格式
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #不同类别用不同颜色和样式绘图
    d = tsne[result['leibie']==0]
    print (d.head(5))
    print (d[0][:5])
    plt.plot(d[0],d[1],'r.')
    d = tsne[result['leibie'] == 1]
    plt.plot(d[0], d[1], 'go')
    d = tsne[result['leibie'] == 2]
    plt.plot(d[0], d[1], 'b*')
    d = tsne[result['leibie'] == 3]
    plt.plot(d[0], d[1], 'y>')
    d = tsne[result['leibie'] == 4]
    plt.plot(d[0], d[1], 'm<')
    d = tsne[result['leibie'] == 5]
    plt.plot(d[0], d[1], 'k+')
    plt.show()

def cluster_gap(k):
    """找出每个变量分群的边界点"""
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=500, n_init=10, init='k-means++').fit(data_zs)
    r = pd.DataFrame(model.cluster_centers_, columns=data.columns)
    gap = []
    for i in data.columns:
        c = r[i].sort_values() #对某一变量聚类中心排序
        w = c.rolling(2).mean().iloc[1:] #求两簇中心点的中点，作为边界点
        w = pd.Series([data_zs[i].min()] + list(w) + [data_zs[i].max()],index=list(range(0,7)), name=i) #加上首末边界点
        w = w * (data[i].std()) + data[i].mean()
        def ex(x):
            return (math.e) ** x - 1
        if i != 'yx_days':
            w = w.map(ex)
        print (w)
        gap.append(w)
    edge = pd.concat(gap,axis=1)
    return  edge



if __name__ == '__main__':
    """根据sse-k的趋势，并结合业务选择k=6"""
    #k_draw(data_zs)
    result = r_cnt(6)
    #r_show(result)
    #print (data.describe())
    #cluster_gap(6).to_excel('/Users/caoshishi/Downloads/data_cnt3.xls')
    #print (result.loc[12030570])
