import numpy as np
import pyprosail; prosail = pyprosail.run; spherical = pyprosail.Spherical
from scipy.io import loadmat
import pickle
import sys
sys.path.append('/home/daniel/projects/sample4Acause/tools')

###############################
#### PROSAIL FORWARD MODEL ####
###############################

# Load and normalize response matrix
with open('/home/daniel/projects/sample4Acause/tools/L8response.pkl', 'rb') as handle:
    L8response = pickle.load(handle, encoding='latin1')

L8RESP = L8response['responsemat']*(1/L8response['responsemat'].sum(axis=1)).reshape(9,1)

# Chloro    0-80
# LAI       0-10
# LMA       0-0.02 (0-20 as I divide by 1000 later) 

pro2d = lambda chloro, LAI : prosail(N = 1.5, chloro = chloro, caroten = 8, brown = 0, EWT = 0.01,
                  LMA = 0.01, psoil = 1, LAI = LAI, hot_spot = 0.01,
                  solar_zenith = 30, solar_azimuth = 10, view_zenith = 0, view_azimuth = 0,
                  LIDF = spherical)[:,1]

pro3d = lambda chloro, LAI, LMA : prosail(N = 1.5, chloro = chloro, caroten = 8, brown = 0, EWT = 0.01,
                  LMA = LMA, psoil = 1, LAI = LAI, hot_spot = 0.01,
                  solar_zenith = 30, solar_azimuth = 10, view_zenith = 0, view_azimuth = 0,
                  LIDF = spherical)[:,1]

pro3d_alvaro = lambda LMA, EWT, Chloro : prosail(N = 1.5, chloro = Chloro, caroten = 8, brown = 0, EWT = EWT,
                  LMA = LMA, psoil = 1, LAI = 4, hot_spot = 0.01,
                  solar_zenith = 30, solar_azimuth = 10, view_zenith = 0, view_azimuth = 0,
                  LIDF = spherical)[:,1]

pro4d = lambda chloro, LAI, LMA, EWT : prosail(N = 1.5, chloro = chloro, caroten = 8, brown = 0, EWT = EWT,
                  LMA = LMA, psoil = 1, LAI = LAI, hot_spot = 0.01,
                  solar_zenith = 30, solar_azimuth = 10, view_zenith = 0, view_azimuth = 0,
                  LIDF = spherical)[:,1]

pro4d_alvaro = lambda LMA, EWT, Chloro, LAI : prosail(N = 1.5, chloro = Chloro, caroten = 8, brown = 0, EWT = EWT,
                  LMA = LMA, psoil = 1, LAI = LAI, hot_spot = 0.01,
                  solar_zenith = 30, solar_azimuth = 10, view_zenith = 0, view_azimuth = 0,
                  LIDF = spherical)[:,1]



def prosail_2d_L8(parameters):
    chloro = parameters[:,0]
    LAI = parameters[:,1]
    N = parameters.shape[0]
    if N > 1:
        spec = np.zeros([N,2101])
        for n in range(N):
            spec[n,:] = pro2d(chloro[n],LAI[n])
    else:
        spec = pro2d(chloro,LAI)

    return (np.dot(L8RESP, spec.T)).T

def prosail_3d(parameters):
    chloro = parameters[:,0]
    LAI = parameters[:,1]
    LMA = parameters[:,2] / 1000 # To make the values on the same order of magnitude!
    N = parameters.shape[0]
    if N > 1:
        spec = np.zeros([N,2101])
        for n in range(N):
            spec[n,:] = pro3d(chloro[n],LAI[n],LMA[n])
    else:
        spec = pro3d(chloro,LAI,LMA)
    
    return(spec)

def prosail_3d_L8_alvaro(parameters):
    parameters = np.maximum(parameters, 0.01)
    parameters = np.minimum(parameters, 130)
    LMA = parameters[:,0] / 1000 # To make the values on the same order of magnitude!
    EWT = parameters[:,1] / 1000
    Chloro = parameters[:,2]
    N = parameters.shape[0]
    if N > 1:
        spec = np.zeros([N,2101])
        for n in range(N):
            spec[n,:] = pro3d_alvaro(LMA[n],EWT[n],Chloro[n])
    else:
        spec = pro3d_alvaro(LMA,EWT,Chloro)
    
    return (np.dot(L8RESP, spec.T)).T

def prosail_4d_L8_alvaro(parameters):
    parameters = np.maximum(parameters, 0.01)
    parameters = np.minimum(parameters, 125)
    LMA = parameters[:,0] / 1000 # To make the values on the same order of magnitude!
    EWT = parameters[:,1] / 1000
    Chloro = parameters[:,2]
    LAI = parameters[:,3]
    N = parameters.shape[0]
    if N > 1:
        spec = np.zeros([N,2101])
        for n in range(N):
            spec[n,:] = pro4d_alvaro(LMA[n],EWT[n],Chloro[n],LAI[n])
    else:
        spec = pro4d_alvaro(LMA,EWT,Chloro,LAI)
    
    return (np.dot(L8RESP, spec.T)).T

def prosail_3d_L8(parameters):
    chloro = parameters[:,0]
    LAI = parameters[:,1]
    LMA = parameters[:,2] / 1000 # To make the values on the same order of magnitude!
    N = parameters.shape[0]
    if N > 1:
        spec = np.zeros([N,2101])
        for n in range(N):
            spec[n,:] = pro3d(chloro[n],LAI[n],LMA[n])
    else:
        spec = pro3d(chloro,LAI,LMA)
    return (np.dot(L8RESP, spec.T)).T
        
def prosail_4d_L8(parameters):
    chloro = parameters[:,0]
    LAI = parameters[:,1]
    LMA = parameters[:,2] / 1000 # To make the values on the same order of magnitude!
    EWT = parameters[:,3] / 1000
    N = parameters.shape[0]
    if N > 1:
        spec = np.zeros([N,2101])
        for n in range(N):
            spec[n,:] = pro4d(chloro[n],LAI[n],LMA[n],EWT[n])
    else:
        spec = pro4d(chloro,LAI,LMA,EWT)

    return (np.dot(L8RESP, spec.T)).T

def prosail_2d_L8_posconv_flag(parameters,flag=0):
    negs = parameters < 0
    if flag*np.sum(negs):
        print('negatives in the soup ', parameters)
    #parameters[negs] = 1e-9
    
    chloro = parameters[:,0]
    LAI = parameters[:,1]
    N = parameters.shape[0]
    if N > 1:
        spec = np.zeros([N,2101])
        for n in range(N):
            spec[n,:] = pro2d(chloro[n],LAI[n])
    else:
        spec = pro2d(chloro,LAI)

    return (np.dot(L8RESP, spec.T)).T




############################
#### TOY FORWARD MODELS ####
############################

def directexp(x):
    return(np.exp(x))
