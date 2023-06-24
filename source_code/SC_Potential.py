import numpy as np

def Coulomb(grid, l, z):
    """ This is a simple coulomb potential with nuclear charge z and l specifying the quantum number 
        which is related to the centrifugal term in the potential"""
    return -z*np.power(grid, -1.0) + 0.5*l*(l+1)*np.power(grid, -2.0)

def Helium(grid, l, z):
    """ This is a simple SAE potential for helium with the centrifugal term of the potential added in spherical coordinate.
        z is not used but is included for uniformity with the Coulomb funcion"""
    return -1.0*np.power(grid, -1.0) + -1.0*np.exp(-2.0329*grid)/grid  - 0.3953*np.exp(-6.1805*grid) + 0.5*l*(l+1)*np.power(grid, -2.0)

def Argon(grid, l, z):
    """ This is a simple SAE potential for helium with the centrifugal term of the potential added in spherical coordinate.
        z is not used but is included for uniformity with the Coulomb funcion"""
    return -1.0*np.power(grid, -1.0) + -17.0*np.exp(-0.8103*grid)/grid  - (-15.9583)*np.exp(-1.2305*grid)\
        - (-27.7467)*np.exp(-4.3946*grid) - (2.1768)*np.exp(-86.7179*grid)  + 0.5*l*(l+1)*np.power(grid, -2.0)

def Short_Range(grid, l, z):
    # Zc, c = 3.65, 3.0
    Zc, c = 4.17, 3.0
    return  -Zc*np.exp(-c*grid)/grid + 0.5*l*(l+1)*np.power(grid, -2.0)

def Short_Range2(grid, l, Zc, c):
    return  -Zc*np.exp(-c*grid)/grid + 0.5*l*(l+1)*np.power(grid, -2.0)