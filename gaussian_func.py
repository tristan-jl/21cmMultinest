import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def Gaussian_2D(coords, centre, width):
    x0=centre[0]
    y0=centre[1]
    w0=width[0]
    w1=width[1]
    
    norm=1.
    
    for i in range(len(width)):
        norm*=1./(width[i]*(2*np.pi)**0.5)
    
    result=norm*np.exp(-0.5*(((coords[0]-x0)/w0)**2 + ((coords[1]-y0)/w1)**2))
    
    return result

x=np.linspace(-5.,5.,1001)
y=x

xx,yy=np.meshgrid(x,y)

result2=Gaussian_2D(np.array([xx,yy]), (0.,0.), (1.,1.)) + Gaussian_2D(np.array([xx,yy]), (3.,-2.), (1.,1.))

num=8
centres=np.random.uniform(-5.,5.,(num,2))
widths=np.random.uniform(0.9,1.1,(num,2))
#widths=np.ones((num,2))

print centres
print widths

result2=np.zeros((1001,1001))

for i in range(num):
    result2+=Gaussian_2D(np.array([xx,yy]), centres[i], widths[i])

plt.pcolormesh(x,y,result2)
plt.show()