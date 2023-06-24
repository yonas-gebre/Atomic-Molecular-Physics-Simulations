import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, pi, cos
from matplotlib.colors import LogNorm
from scipy.special import sph_harm
from scipy.special import spherical_in as In
from scipy.special import spherical_kn as Kn
from spherical import Wigner3j


def Cartesian_to_Spherical(x, y, z):

    r = sqrt(pow(x,2) + pow(y,2) + pow(z,2))

    if x > 0 and y > 0:
        phi = np.arctan(y/x)
    elif x > 0 and y < 0:
        phi = np.arctan(y/x) + 2*pi
    elif x < 0 and y > 0:
        phi = np.arctan(y/x) + pi
    elif x < 0 and y < 0:
        phi = np.arctan(y/x) + pi
    elif x == 0 and y == 0:
        phi = 0
    elif x == 0 and y > 0:
        phi = pi / 2
    elif x == 0 and y < 0:
        phi = 3*pi / 2
    elif y == 0 and x > 0:
        phi = 0
    elif y == 0 and x < 0:
        phi = pi

    theta = np.arccos(z/r)

    return phi, theta

def Cartessian_Potential():

    R = 2.3
    Co = 1
    Zc = 15
    c = 1.326
    co = 3.248
    crho = 2.778
    d = 1.643
    a1 = 10.804
    b1 = 0.466
    cc = (1/co)**2 - (1/crho)**2
    cbar = (4*pi)/3*cc

    resolution = 0.1
    x_axis_point = np.arange(-5, 5 + resolution, resolution)
    y_axis_point = np.arange(-5, 5 + resolution, resolution)
    z_axis_point = np.arange(-5, 5 + resolution, resolution)

    y_axis_point = np.array([0])
    potential_cartessian = np.zeros(z_axis_point.size)
    potential_spherical = np.zeros(z_axis_point.size)
    potential_spherical_taylor = np.zeros(z_axis_point.size)

    def Spherical_Potential(r):
        return -Co/(2*r)+ -Zc*exp(-r/c)/(2*r)+ a1*exp(-r/b1)

    def Cylindrical_Potential(z, rho):
        return d*exp(-1*(z/co)**2 - (rho/crho)**2)

    def Spherical_Potential_Expanded(r, theta, phi):
        r_less = np.amin(np.array([r, R/2]))
        r_greater = np.amax(np.array([r, R/2]))

        v1, v2, v3 = 0, 0, 0
        
        for L in range(0, 120, 2):
            v1 += sqrt(4*pi/(2*L+1)) * pow(r_less,L)/pow(r_greater,L+1)*sph_harm(0, L, phi, theta).real
            v2 +=  sqrt((2*L+1)/(4*pi))*In(L, (1.0/c)*r_less)*Kn(L, (1.0/c)*r_greater)*sph_harm(0, L, phi, theta).real
            
            v3a = In(L, (1.0/b1)*r_less)*Kn(L, (1.0/b1)*r_greater)
            v3b = (1/b1)*(In(L, (1.0/b1)*r_less, True)*Kn(L, (1.0/b1)*r_greater)*r_less + In(L, (1.0/b1)*r_less)*Kn(L, (1.0/b1)*r_greater, True)*r_greater)
            v3 += sqrt((2*L+1)/(4*pi))*(v3a + v3b)*sph_harm(0, L, phi, theta).real

        return   -Co*v1 + -Zc*8*(1.0/c)*v2 + -2*a1*8*v3

    def Cylindrical_Potential_Expanded(r, theta):
        return d*exp(-pow(r*cos(theta),2)*cc)*exp(-pow(r/crho, 2))
    
    def Cylindrical_Potential_Expanded_Taylor(r, theta, phi):
        N = 10
        spherical_harmonic_coefficents = Cylindrical_Potential_Spherical_Harmonic_Taylor(N)
        return_value = 0.0
        for i, coeff_list in enumerate(spherical_harmonic_coefficents):
            ith_order_term = 0
            for pair in coeff_list:
                ith_order_term +=  pair[1]*sph_harm(0, pair[0], phi, theta).real
            ith_order_term *= pow(-1*cbar*pow(r,2), i)/np.math.factorial(i)
            
            return_value += ith_order_term
             
        return d*exp(-pow(r/crho, 2))*return_value
        # return d*(1 - pow(r*cos(theta),2)*cc + pow(r*cos(theta),4)*(cc**2)/2 - pow(r*cos(theta),6)*(cc**3)/6)*exp(-pow(r/crho, 2))

    def Taylor_Term(n, phi, theta):
        return_value = 0.0
        for i in range(n+1):
            return_value += pow(-1*cbar*pow(r*sph_harm(0, 1, phi, theta).real,2), i)/np.math.factorial(i)
        return return_value    
    
    for i, z in enumerate(z_axis_point):
        print(round(z,3))
        potential_cartessian_temp = 0.0
        potential_spherical_temp = 0.0
        potential_spherical_temp_taylor = 0.0
        for j, x in enumerate(x_axis_point):
            

            for l, y in enumerate(y_axis_point):

                r1 = sqrt(pow(x,2) + pow(y,2) + pow(z + (R/2),2))
                r2 = sqrt(pow(x,2) + pow(y,2) + pow(z - (R/2),2))
                r = sqrt(pow(x,2) + pow(y,2) + pow(z,2))
                rho = sqrt(pow(x,2) + pow(y,2))
                phi, theta = Cartesian_to_Spherical(x, y, z)

                potential_cartessian_temp += Cylindrical_Potential(z, rho)#Spherical_Potential(r1) + Spherical_Potential(r2)#
                potential_spherical_temp += Cylindrical_Potential_Expanded_Taylor(r, theta, phi)#Spherical_Potential_Expanded(r, theta, phi)#Cylindrical_Potential_Expanded(r, theta)#
                potential_spherical_temp_taylor += Cylindrical_Potential_Expanded_Taylor(r, theta, phi)
                
                
        potential_cartessian[i] = potential_cartessian_temp
        potential_spherical[i] = potential_spherical_temp
        potential_spherical_taylor[i] = potential_spherical_temp_taylor
        
    return potential_cartessian, potential_spherical, potential_spherical_taylor

def Spherical_Harmonic_Product(l1, l2, c1, c2):
    coefficent_array = []
    for L in range(abs(l1-l2), l1+l2+2, 2):
        coeff = c1*c2*(Wigner3j(l1,l2,L,0,0,0)**2)*sqrt((2*l1+1)*(2*l2+1)*(2*L+1)/(4*pi))
        coefficent_array.append((L, coeff))
        
    return coefficent_array
     
def Cylindrical_Potential_Spherical_Harmonic_Taylor(N):
    zero_term = [(0,sqrt(4*pi))]
    base_term = Spherical_Harmonic_Product(1, 1, 1, 1)
    spherical_harmonic_coefficents = [zero_term, base_term]
    
    for n in range(2, N):
        nth_order_expansion = []
        nth_order_expansion_organized = []
        
        for pair_a in spherical_harmonic_coefficents[n-1]:
            for pair_b in base_term:
                nth_order_expansion.append(Spherical_Harmonic_Product(pair_a[0], pair_b[0], pair_a[1], pair_b[1]))
        

        for L in range(0, 2*n+2, 2):
            
            coeff = 0
            for pair_list in nth_order_expansion:
                for pair in pair_list:
                    if pair[0] == L:
                        coeff += pair[1]
            
            nth_order_expansion_organized.append((L,coeff))          
            
        
        spherical_harmonic_coefficents.append(nth_order_expansion_organized)
        
    return spherical_harmonic_coefficents
    
    
if __name__ == "__main__":

    spherical_harmonic_coefficents = Cylindrical_Potential_Spherical_Harmonic_Taylor(3)

    for list_x in spherical_harmonic_coefficents:
        print(list_x)
    potential_cartessian, potential_spherical, potential_spherical_taylor = Cartessian_Potential()

    print(potential_spherical-potential_cartessian)

    error = np.abs(potential_spherical-potential_cartessian)
    print(error.max())
    resolution = 0.1
    z_axis_point = np.arange(-5, 5 + resolution, resolution)

    plt.plot(z_axis_point, -1*potential_cartessian, label="cartessian")
    plt.plot(z_axis_point, -1*potential_spherical, label="spherical_expansion")
    # plt.plot(z_axis_point, -1*potential_spherical_taylor, label="spherical_taylor")
    plt.legend()
    plt.savefig("Z_Pot_Cyli.png")
    
    # plt.show()

    
    
    # potential_cartessian = -1*potential_cartessian
    # potential_cartessian = potential_cartessian / potential_cartessian.max()


    # potential_spherical = -1*potential_spherical
    # potential_spherical = potential_spherical / potential_spherical.max()
    
    # # diff = np.max(np.abs(potential_cartessian - potential_spherical))
    # diff = (potential_cartessian / potential_spherical)
    # print(diff)
    
    # f, axarr = plt.subplots(1,2)

    # axarr[0].imshow(potential_cartessian, extent=[-4, 4, -4, 4], norm=LogNorm(vmin=1e-2, vmax=1))
    # axarr[1].imshow(potential_spherical, extent=[-4, 4, -4, 4], norm=LogNorm(vmin=1e-2, vmax=1))

    # # plt.colorbar(pos)
    

    # #0.07051956326162168
    # #0.020953796975708427