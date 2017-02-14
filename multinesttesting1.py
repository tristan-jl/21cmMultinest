import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pymultinest
import json

def Gaussian_2D(coords, centre, width, height):
    norm=1.
    
    for i in range(len(width)):
        norm*=1./(width[i]*(2*np.pi)**0.5)
    
    result_gaussian=height*norm*np.exp(-0.5*(((coords[0]-centre[0])/width[0])**2 + ((coords[1]-centre[1])/width[1])**2))
    
    return result_gaussian

array_size=101

x=np.linspace(-5.,5.,array_size)
y=x
xx,yy=np.meshgrid(x,y)

num=8
centre_list=np.random.uniform(-5.,5.,(num,2))
width_list=np.random.uniform(0.9,1.1,(num,2))
height_list=np.random.uniform(0.,2.,num)
#widths=np.ones((num,2))

print centre_list
print width_list

data=np.zeros((array_size,array_size))

for i in range(num):
    data+=Gaussian_2D(np.array([xx,yy]), centre_list[i], width_list[i], height_list[i])

plt.pcolormesh(x,y,data)

#Multinest time
number=5 #number of gaussians

def Model(number, centres, widths, heights):
    result=np.zeros((array_size,array_size))
    
    for i in range(number):
        result+=Gaussian_2D(np.array([xx,yy]), centre_list[i], width_list[i], height_list[i])
    
    return result

def Prior(cube, ndim, nparams):
    for i in range(ndim):
        cube[i]=cube[i]*10. - 5.

def Loglike(cube, ndim, nparams):
        i=range(ndim)
        centres=np.zeros((number,2))
        widths=centres
        heights=np.zeros(number)
        
        for i in i:
            if 0<=i<=(number*2 - 1):
                centres[i][0]=cube[2i]
                centres[i][1]=cube[2i+1]
            elif (number*2)<=i<=(number*2-1 + 2*number):
                widths[i][0]=cube[2i]
                widths[i][1]=cube[2i+1]
            elif (number*2+3*number)<=i<=(number-1 + 3*number):
                heights[i]=cube[i]
            else:
                print "i wrong index"
        
        loglikelihood=np.log(Model(number, centres, widths, heights))
        return loglikelihood

parameters=["number", "centres", "widths", "heights"]
n_params=len(parameters)

# run MultiNest
pymultinest.run(Loglike, Prior, n_params)
json.dump(parameters) # save parameter names

plt.show()