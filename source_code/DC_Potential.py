from TDSE_Module import Check_Dictionary_Key_For_W3J_Symbol
from numpy import exp, sqrt, pi, cos
from scipy.special import spherical_in as In
from scipy.special import spherical_kn as Kn
import numpy as np
# from spherical import Wigner3j

def  Hydrogen_Ion( r,  l,  l_prime,  m,  Ro,  z, wigner_3j_dict):

    """ Calculates a  center potential coupling at centers Ro/2 and -Ro/2. For input files
        ones specifies Ro. 
        r  is the radial po in space
        l, l_prime are angular momentum quantum numbers that are coupled
        m is the magnetic quantum number (not coupled so m = m_prime)
        z is the nuclear charge and is 1 for hydrogen ion
        wigner_3j_dict is a dictionary that contans different wigner 3j terms (look up Wigner_3j.py)"""

    Ro = Ro / 2.0

    if abs(m) > l or abs(m) > l_prime:
        if l == l_prime:
            return  0.5*l*(l+1.0)*pow(r,-2.0)
        else:
            return 0.0
            
    if r <= Ro:
        for lamda in range(abs(l-l_prime), l + l_prime + 2, 2): 
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            coef = wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]
            potential_value += pow(r/Ro, lamda)/Ro * coef   
    else:
         for lamda in range(abs(l-l_prime), l + l_prime + 2, 2):
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            coef = wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]
            potential_value += pow(Ro/r,lamda)/r * coef

    potential_value *= -2.0 * z * pow(-1.0, m)* pow((2.0*l+1.0)*(2.0*l_prime+1.0), 0.5) 

    if l == l_prime:
        potential_value +=  0.5*l*(l+1.0)*pow(r,-2.0) 

    return potential_value

def  Oxygen( r, l,  l_prime,  m,  Ro,  z,  wigner_3j_dict):

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
    Ro = Ro / 2.0
    
    
    if abs(m) > l or abs(m) > l_prime:
        if l == l_prime:
            return  0.5*l*(l+1.0)*pow(r,-2.0)
        else:
            return 0.0
            
    if r <= Ro:
        for lamda in range(abs(l-l_prime), l + l_prime + 2, 2): 
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            coef = wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]
            
            coulomb_part = pow(r/Ro, lamda)/Ro
            yukawa_part = 8*(1.0/c)*(2*lamda+1)/(4*pi)*In(lamda, (1.0/c)*r)*Kn(lamda, (1.0/c)*Ro)
            exponential_part_a = In(lamda, (1.0/c)*r)*Kn(lamda, (1.0/c)*Ro)
            exponential_part_b = (1/b1)*(In(lamda, (1.0/b1)*r, True)*Kn(lamda, (1.0/b1)*Ro)*r + In(lamda, (1.0/b1)*r)*Kn(lamda, (1.0/b1)*Ro, True)*Ro)
            exponential_part = 16*(2*lamda+1)/(4*pi)*(exponential_part_a + exponential_part_b)

            potential_value_spherical +=  coef * (-Co*coulomb_part - Zc*yukawa_part -a1*exponential_part)
    else:
         for lamda in range(abs(l-l_prime), l + l_prime + 2, 2):
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            coef = wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]

            coulomb_part = pow(Ro/r, lamda)/r
            yukawa_part = 8*(1.0/c)*(2*lamda+1)/(4*pi)*In(lamda, (1.0/c)*Ro)*Kn(lamda, (1.0/c)*r)
            exponential_part_a = In(lamda, (1.0/c)*Ro)*Kn(lamda, (1.0/c)*r)
            exponential_part_b = (1/b1)*(In(lamda, (1.0/b1)*Ro, True)*Kn(lamda, (1.0/b1)*r)*Ro + In(lamda, (1.0/b1)*Ro)*Kn(lamda, (1.0/b1)*r, True)*r)
            exponential_part = 16*(2*lamda+1)/(4*pi)*(exponential_part_a + exponential_part_b)

            potential_value_spherical +=  coef * (-Co*coulomb_part - Zc*yukawa_part -a1*exponential_part)

    N = 10
    spherical_harmonic_coefficents = Cylindrical_Potential_Spherical_Harmonic_Taylor(N)
    
    for i, coeff_list in enumerate(spherical_harmonic_coefficents):
        ith_order_term = 0
        for pair in coeff_list:
            coeff, lamda = pair[1], pair[0]

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            ith_order_term +=  coeff*sqrt(2*lamda+1)*wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]

        ith_order_term *= pow(-1*cbar*pow(r,2), i)/np.math.factorial(i)
        
        potential_value_cylinder += ith_order_term
            
    potential_value_cylinder *= d*exp(-pow(r/crho, 2))/sqrt(4*pi)

    potential_value = pow(-1.0, m)* pow((2.0*l+1.0)*(2.0*l_prime+1.0), 0.5) *(potential_value_spherical + potential_value_cylinder)

    if l == l_prime:
        potential_value +=  0.5*l*(l+1.0)*pow(r,-2.0) 

    return potential_value

def Hydrogen(r,  l,  l_prime,  m,  Ro,  z, wigner_3j_dict):

    Co = 1
    Zc = 1
    c = 0.316
    co = 13.841
    crho = 18.637
    d = -0.162
    a1 = 1.205
    b1 = 0.497
    cc = (1/co)**2 - (1/crho)**2
    cbar = (4*pi)/3*cc
    Ro = Ro / 2.0
    
    
    if abs(m) > l or abs(m) > l_prime:
        if l == l_prime:
            return  0.5*l*(l+1.0)*pow(r,-2.0)
        else:
            return 0.0
            
    if r <= Ro:
        for lamda in range(abs(l-l_prime), l + l_prime + 2, 2): 
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            coef = wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]
            
            coulomb_part = pow(r/Ro, lamda)/Ro
            yukawa_part = 8*(1.0/c)*(2*lamda+1)/(4*pi)*In(lamda, (1.0/c)*r)*Kn(lamda, (1.0/c)*Ro)
            exponential_part_a = In(lamda, (1.0/c)*r)*Kn(lamda, (1.0/c)*Ro)
            exponential_part_b = (1/b1)*(In(lamda, (1.0/b1)*r, True)*Kn(lamda, (1.0/b1)*Ro)*r + In(lamda, (1.0/b1)*r)*Kn(lamda, (1.0/b1)*Ro, True)*Ro)
            exponential_part = 16*(2*lamda+1)/(4*pi)*(exponential_part_a + exponential_part_b)

            potential_value_spherical +=  coef * (-Co*coulomb_part - Zc*yukawa_part -a1*exponential_part)
    else:
         for lamda in range(abs(l-l_prime), l + l_prime + 2, 2):
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            coef = wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]

            coulomb_part = pow(Ro/r, lamda)/r
            yukawa_part = 8*(1.0/c)*(2*lamda+1)/(4*pi)*In(lamda, (1.0/c)*Ro)*Kn(lamda, (1.0/c)*r)
            exponential_part_a = In(lamda, (1.0/c)*Ro)*Kn(lamda, (1.0/c)*r)
            exponential_part_b = (1/b1)*(In(lamda, (1.0/b1)*Ro, True)*Kn(lamda, (1.0/b1)*r)*Ro + In(lamda, (1.0/b1)*Ro)*Kn(lamda, (1.0/b1)*r, True)*r)
            exponential_part = 16*(2*lamda+1)/(4*pi)*(exponential_part_a + exponential_part_b)

            potential_value_spherical +=  coef * (-Co*coulomb_part - Zc*yukawa_part -a1*exponential_part)

    N = 10
    spherical_harmonic_coefficents = Cylindrical_Potential_Spherical_Harmonic_Taylor(N)
    
    for i, coeff_list in enumerate(spherical_harmonic_coefficents):
        ith_order_term = 0
        for pair in coeff_list:
            coeff, lamda = pair[1], pair[0]

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l,lamda,l_prime,-m,0,m))
            ith_order_term +=  coeff*sqrt(2*lamda+1)*wigner_3j_dict[(l,lamda,l_prime,0,0,0)] * wigner_3j_dict[(l,lamda,l_prime,-m,0,m)]

        ith_order_term *= pow(-1*cbar*pow(r,2), i)/np.math.factorial(i)
        
        potential_value_cylinder += ith_order_term
            
    potential_value_cylinder *= d*exp(-pow(r/crho, 2))/sqrt(4*pi)

    potential_value = pow(-1.0, m)* pow((2.0*l+1.0)*(2.0*l_prime+1.0), 0.5) *(potential_value_spherical + potential_value_cylinder)

    if l == l_prime:
        potential_value +=  0.5*l*(l+1.0)*pow(r,-2.0) 

    return potential_value

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


def Spherical_Harmonic_Product(l1, l2, c1, c2):
    coefficent_array = []
    for L in range(abs(l1-l2), l1+l2+2, 2):
        coeff =0.0#c1*c2*(Wigner3j(l1,l2,L,0,0,0)**2)*sqrt((2*l1+1)*(2*l2+1)*(2*L+1)/(4*pi))
        coefficent_array.append((L, coeff))
        
    return coefficent_array