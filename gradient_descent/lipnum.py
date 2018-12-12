import numpy as np
import math
def g(x):
    x1 = x[0]
    y1 = x[1]
    g1 = 2*(1.5 - x1 + x1*y1)*(y1-1)+2*(2.25-x1+x1*y1**2)*(y1**2-1)+2*(2.625-x1+x1*y1**3)*(y1**3-1)
    g2 = 2*(1.5 - x1 + x1*y1)*x1 + 2*(2.25-x1+x1*y1**2)*(2*y1*x1)+2*(2.625-x1+x1*y1**3)*(3*y1**2*x1)
    g0=math.sqrt(g1**2+g2**2)
    return g0

xi = np.linspace(-4.5,4.5,1000)
yi = np.linspace(-4.5,4.5,1000)
X,Y = np.meshgrid(xi, yi)
Z = np.empty([len(xi),len(yi)], dtype = float)
for i in range(len(xi)):
    for j in range(len(yi)):
        x =(xi[i],yi[j])
        Z[i,j]=g(x)

print (np.amax(Z))