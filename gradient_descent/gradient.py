# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

n_iter = 10000
def f(x):
    x1 = x[0]
    y1 = x[1]

    term1 = (1.5 - x1 + x1*y1)**2
    term2 = (2.25 - x1 + x1*y1**2)**2
    term3 = (2.625 - x1 + x1*y1**3)**2
    z = term1 + term2 + term3;
    return z

def g(x):
    x1 = x[0]
    y1 = x[1]
    g1 = 2*(1.5 - x1 + x1*y1)*(y1-1)+2*(2.25-x1+x1*y1**2)*(y1**2-1)+2*(2.625-x1+x1*y1**3)*(y1**3-1)
    g2 = 2*(1.5 - x1 + x1*y1)*x1 + 2*(2.25-x1+x1*y1**2)*(2*y1*x1)+2*(2.625-x1+x1*y1**3)*(3*y1**2*x1)
    return np.array([g1, g2])

#matplotlib inline
def contour(X,Y,Z, arr = None):
    plt.figure(figsize=(15,7))
    plt.contour(X, Y, Z.T,levels=np.logspace(0, 5, 35),norm=LogNorm(),cmap=plt.cm.jet)
    plt.plot(0,0,marker='*')
    plt.plot(3, 0.5, marker='*',markersize=10)
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i+2,0],arr[i:i+2,1])


#contour(X,Y,Z)

def plot3d(X,Y,Z):
    fig = plt.figure()
    # 创建3d图形的两种方式
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    # rstride:行之间的跨度  cstride:列之间的跨度
    # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
    #vmax和vmin  颜色的最大值和最小值
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    xminima=np.array([3.])
    yminima = np.array([.5])
    zminima=np.array([0.0])
    ax.plot(xminima,yminima,zminima,'*',markersize=10)
    # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
    # offset : 表示等高线图投射到指定页面的某个刻度
    ax.contourf(X,Y,Z,zdir='z',offset=-2)
    # 设置图像z轴的显示范围，x、y轴设置方式相同
    ax.set_zlim(-2,200000)

def gd(x_start, step, g):   # Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(n_iter):
        grad = g(x)
        x -= grad * step
        passing_dot.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

def momentum(x_start, step, g, discount = 0.9):   # momentum
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(n_iter):
        grad = g(x)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot


def nesterov(x_start, step, g, discount=0.9):   #Nesterov accelerated gradient
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(n_iter):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

def adam(x_start, step, g):             #Adaptive Moment Estimation
    x = np.array(x_start, dtype='float64')
    beta1=0.9
    beta2=0.999
    mt = np.zeros_like(x)
    vt = np.zeros_like(x)
    dp = np.zeros_like(x)
    passing_dot = [x.copy()]
    for i in range(n_iter):
        beta1t=beta1**(i+1)
        beta2t=beta2**(i+1)
        temp=g(x)
        mt=beta1*mt + (1-beta1) * g(x)
        vt=beta2*vt + (1-beta2) * g(x)**2
        mt1=mt/(1-beta1t)
        vt1=vt/(1-beta2t)
        dp[0]= step * mt1[0]/(math.sqrt(vt1[0])+0.00000001)
        dp[1] = step * mt1[1] / (math.sqrt(vt1[1]) + 0.00000001)
        x -= dp
        passing_dot.append(x.copy())
        if abs(sum(mt1)) < 1e-6:
            break;
    return x, passing_dot



xi = np.linspace(-4.5,4.5,1000)
yi = np.linspace(-4.5,4.5,1000)
X,Y = np.meshgrid(xi, yi)
Z = np.empty([len(xi),len(yi)], dtype = float)
for i in range(len(xi)):
    for j in range(len(yi)):
        x =(xi[i],yi[j])
        Z[i,j]=f(x)



plot3d(X,Y,Z)

contour(X,Y,Z)
res, x_arr = gd([3,4], 0.00005, g)
contour(X,Y,Z, x_arr)
print ('a')
print (res)
res, x_arr = momentum([3,4], 0.0000005, g)
contour(X,Y,Z, x_arr)
print (res)
res, x_arr = nesterov([3,4], 0.0000005, g)
contour(X,Y,Z, x_arr)
print (res)
res, x_arr = adam([3,4], 1, g)
print (res)
contour(X,Y,Z, x_arr)
print ('a')
print ('a')
plt.show()
