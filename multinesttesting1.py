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

array_size=1001

x=np.linspace(-5.,5.,array_size)
y=x
xx,yy=np.meshgrid(x,y)

num=8
centre_list=np.random.uniform(-5.,5.,(num,2))
width_list=np.random.uniform(0.9,1.1,(num,2))
height_list=np.random.uniform(0.,1.,num)
#widths=np.ones((num,2))

print centre_list
print width_list

data=np.zeros((array_size,array_size))

for i in range(num):
    data+=Gaussian_2D(np.array([xx,yy]), centre_list[i], width_list[i], height_list[i])

plt.pcolormesh(x,y,data)



#Multinest time

def Model(number, centres, widths, heights):
    '''
    Sum of [number] of gaussians.
    '''
    result=np.zeros((array_size,array_size))
    
    for i in range(number):
        result+=Gaussian_2D(np.array([xx,yy]), centre_list[i], width_list[i], height_list[i])
    
    return result

def Prior(cube, ndim, nparams):
    for i in range(2*number):  #centres
        cube[i]=cube[i]*10. - 5.
    
    for i in range(2*number, 4*number): #widths
        cube[i]=cube[i]*2.
    
    for i in range(4*number, 5*number): #heights
        cube[i]=cube[i]
        
def Loglike(cube, ndim, nparams):
    '''
    Log likelihood function.
    '''
    centres=np.zeros((number,2))
    widths=np.zeros((number,2))
    heights=np.zeros(number)
        
    for i in range(number):
        centres[i][0] = cube[2*i]
        centres[i][1] = cube[2*i+1]
        widths[i][0] = cube[2*i + 2*number]
        widths[i][1] = cube[2*i+1 + 2*number]
        heights[i] = cube[i + 4*number]    
        
    loglikelihood=np.log(Model(number, centres, widths, heights)).sum()   #not sure about this line
    return loglikelihood

def Create_parameter_list(empty_list=None):
    '''
    Creates list of parameters by appending to list.
    '''
    if empty_list == None:
        empty_list = []

    for i in range(2*number):
        if i%2 == 0:
            xy="x"
        else:
             xy="y"
        empty_list.append("centre_" + xy + str(int(i/2)))
    
    for i in range(2*number):
        if i%2 == 0:
            xy="x"
        else:
             xy="y"
        empty_list.append("width_" + xy + str(int(i/2)))
        
    for i in range(number):
        empty_list.append("height_" + str(int(i)))
    
    return empty_list

number = 5 #number of gaussians
parameters = Create_parameter_list()
n_params = len(parameters)

# run MultiNest
pymultinest.run(Loglike, Prior, n_params)
with open('chains/1-parameter_list.txt', 'w') as outfile:  
    json.dump(parameters, outfile, 'w')

plt.show()