# -*- coding:utf-8 -*-
#计算等额本金和等额本息的借贷方式，当持有y年全部还清时，分别需要的利息是多少
import numpy as np
import matplotlib.pyplot as plt
class Tax():
    def taxes(self, a, n, w, y):
        """等额本金"""
        i = 0.01*w/12 #月利率
        s = a/n/12 #月还款本金
        d = y*12 #预备还多少个月
        total = 0
        for x in range(d):
            tax = (a-s*x)*i + s
            la = a-s*x
            total += tax
        return total+la-a
    def taxes2(self, a, n, w,y):
        """等额本息"""
        i = 0.01*w/12 #月利率
        d = y*12
        e = a*i*((1+i)**(n*12))/((1+i)**(n*12)-1) #月还款金额
        print (e, e*n*12, e*n*12-a)
        total = e*d
        for x in range(d):
            a1 = a*((1+i)**x)
            if x < 2:
                a2 = e*x
            else:
                a2 = np.sum([e*((1+i)**j) for j in range(x)])
            la = a1-a2
        return  total+la-a
    def draw_taxes(self, a, n, w):
        x = range(1,31)
        y1 = [self.taxes(a,n,w,y) for y in x]
        y2 = [self.taxes2(a,n,w,y) for y in x]
        plt.figure(figsize=(12,8))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(x,y1,marker='o',color='lightblue',label='等额本金')
        plt.plot(x,y2,marker='o',color='pink',label='等额本息')
        plt.legend()
        plt.title(u'持有年限的还款利息',fontsize=14,fontweight='bold')
        plt.xlabel(u'贷款N年后全部还清')
        plt.ylabel(u'支付的利息')
        return plt.show()

t = Tax()
#借款100w，时限30年，年利率5.12%时，计算持有年限提前还清与总利息之间的关系
t.draw_taxes(1000000, 30, 5.12)