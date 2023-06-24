import h5py
import sys
import numpy as np
from sympy.physics.wigner import gaunt, wigner_3j

idx =  [i for i, itr in enumerate(sys.path[0]) if itr == "/"]
path_address = sys.path[0][:idx[-1]]
    
sys.path.append(path_address + '/General_Functions')
import Module as Mod



def Inital_State_Wave_Function(input_par):
    """ Returns the intial state wavefuncton """
    file = h5py.File(input_par["Target_File"], 'r')
    inital_state = input_par["inital_state"]
    bound_states = {}
    Bound_State_Map = Map_Bound_State_Index(input_par)

    if input_par["center"] == "single":
        for n_l_m in inital_state["q_numbers"]:
            n, l = n_l_m[0], n_l_m[1]
            bound_states[(n, l)] = file["Psi_" + str(Bound_State_Map[(n, l)][0]) + "_" + str(Bound_State_Map[(n, l)][1])]
            bound_states[(n, l)] = np.array(bound_states[(n, l)][:,0] + 1.0j*bound_states[(n, l)][:,1])
 
    if input_par["center"] == "double":
        for n_m in inital_state["q_numbers"]:
            n, m = n_m[0], n_m[1]
            bound_states[(n, m)] = file["Psi_" + str(abs(m)) + "_" + str(n -1)]
            bound_states[(n, m)] = np.array(bound_states[(n, m)][:,0] + 1.0j*bound_states[(n, m)][:,1])

    return bound_states

def Map_Bound_State_Index(input_par):

    Bound_State_Map = {}

    if input_par["center"] == "single":
        for l in range(input_par["n_max"]):
            n_range = range(l + 1, input_par["n_max"]+1)
            for i, n in enumerate(n_range):
                Bound_State_Map[(n, l)] = (l, i)
                
    if input_par["center"] == "double":
        for m in range(-1*input_par["m_max_bs"], input_par["m_max_bs"] + 1):
            n_quantum_number = 1
            for i in range(input_par["n_max"]):
                Bound_State_Map[(n_quantum_number, m)] = (m, i)
                n_quantum_number += 1
 
    return Bound_State_Map

def Make_Inital_State(input_par, wave_function, grid):
    """ Makes the inital state using the return from Inital_State_Wave_Function and maps it to
        the block structure thats used for the TDSE"""
    inital_state = input_par["inital_state"]
    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    psi = np.zeros(grid.size * len(qn_to_block), dtype=complex)

    if input_par["center"] == "single":
        
        for amp, n_l_m in zip(inital_state["amplitudes"], inital_state["q_numbers"]):
            psi_n_l = amp * wave_function[(n_l_m[0], n_l_m[1])]
            box_idx =  qn_to_block[(n_l_m[1],n_l_m[2])] * grid.size
            psi[box_idx : box_idx + grid.size] += psi_n_l
   
    if input_par["center"] == "double":
        
        for amp, n_m in zip(inital_state["amplitudes"], inital_state["q_numbers"]):
            n, m = n_m[0], n_m[1]
            psi_m_n = amp * wave_function[(n, m)]    
            box_idx =  qn_to_block[(abs(m), m)] * grid.size
            low_idx = grid.size*abs(m)
            high_idx = input_par["l_max_bs"]*grid.size
            psi_m_n = psi_m_n[low_idx:high_idx + grid.size]
            psi[box_idx: box_idx + len(psi_m_n)] += psi_m_n
    
    psi /= np.linalg.norm(psi)   
    return psi  

def Dictionary_Remap(dictA):
    dictB = {}
    for key, value in dictA.items():
        new_key = eval(key)
        dictB[new_key] = value
    
    return dictB

def Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, key):
    if key in wigner_3j_dict:
        return wigner_3j_dict
    else:
        j1, j2, j3, m1, m2, m3 = key[0],key[1],key[2],key[3],key[4],key[5]
        
        wigner_3j_dict[key] = wigner_3j(j1, j2, j3, m1, m2, m3)
        
        wigner_3j_dict[(j3, j2, j1, m3, m2, m1)] = wigner_3j_dict[(j1, j2, j3, m1, m2, m3)]/pow(-1.0, j1+j2+j3)

        return wigner_3j_dict
       