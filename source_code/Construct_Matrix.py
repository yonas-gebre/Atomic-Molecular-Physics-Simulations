import sys
import os
import json
import numpy as np
import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import socket


import TDSE_Module
import Laser_Pulse as LP
import Interaction_Matrix as Int
import DC_Field_Free_Hamiltonian as DCFFH
import SC_Field_Free_Hamiltonian as SCFFH
import Dipole_Acceleration_Matrix as DA

sys.path.append(os.getcwd()[:-6] + "General_Functions")

import Module as Mod

def Necessary_Matrices(input_par):
    
    build_status = {}
    build_status["Int_Mat_X_Stat"], build_status["Int_Mat_Y_Stat"],  build_status["Int_Mat_Z_Stat"] = False, False, False
    build_status["Int_Mat_Right_Stat"], build_status["Int_Mat_Left_Stat"] = False, False
    build_status["Dip_Acc_Mat_X_Stat"], build_status["Dip_Acc_Mat_Y_Stat"], build_status["Dip_Acc_Mat_Z_Stat"]  = False, False, False
    build_status["Dip_Mat_X_Stat"], build_status["Dip_Mat_Y_Stat"], build_status["Dip_Mat_Z_Stat"] = False, False, False

    laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse, free_prop_idx  = LP.Build_Laser_Pulse(input_par)
   
    if elliptical_pulse == True or np.dot(total_polarization, np.array([1,0,0])) != 0 or np.dot(total_polarization, np.array([0,1,0])) != 0:
        if input_par['Int_Mat_Base'] == 'Linear':
            
            if np.dot(total_poynting,np.array([1,0,0])) != 1: 
                build_status["Int_Mat_X_Stat"] = True
            
            if np.dot(total_poynting,np.array([0,1,0])) != 1: 
                build_status["Int_Mat_Y_Stat"] = True

        if input_par['Int_Mat_Base'] == 'Circular':
            
            build_status["Int_Mat_Right_Stat"]  = True
            build_status["Int_Mat_Left_Stat"]  = True

        if input_par["HHG"] == 1:
            if np.dot(total_polarization, np.array([1,0,0])) != 0:
                build_status["Dip_Acc_Mat_X_Stat"] = True
            if np.dot(total_polarization, np.array([0,1,0])) != 0:
                build_status["Dip_Acc_Mat_Y_Stat"] = True

        if input_par["Dipole"] == 1:
            if np.dot(total_polarization, np.array([1,0,0])) != 0:
                build_status["Dip_Mat_X_Stat"] = True
            if np.dot(total_polarization, np.array([0,1,0])) != 0:    
                build_status["Dip_Mat_Y_Stat"] = True

    if np.dot(total_poynting,np.array([0,0,1])) != 1: 
        build_status["Int_Mat_Z_Stat"] = True

        if input_par["HHG"] == 1:
            build_status["Dip_Acc_Mat_Z_Stat"] = True

        if input_par["Dipole"] == 1:
            build_status["Dip_Mat_Z_Stat"] = True

    return build_status

def Make_Matrices(input_par):
    
    FF_Ham, Int_Mat_Left, Int_Mat_Right = None, None, None
    Int_Mat_X, Int_Mat_Y, Int_Mat_Z = None, None, None
    Dip_Acc_Mat_X, Dip_Acc_Mat_Y, Dip_Acc_Mat_Z = None, None, None

    build_status = Necessary_Matrices(input_par)
    grid = Mod.Grid(input_par["grid_spacing"], input_par["grid_size"])

    if "terra" in socket.gethostname() or "node" in socket.gethostname():
        with open('/users/becker/yoge8051/Research/TDSE/Solver/Json_Files/W3J_New.json') as file:
            data = file.read()
    if "photon" in socket.gethostname():
        with open('/mpdata/becker/yoge8051/Research/TDSE/Solver/Json_Files/W3J_New.json') as file:
            data = file.read()
    
    wigner_3j_dict = json.loads(data)
    
    # wigner_3j_dict = {}
    
    if input_par["center"] == "single":
        if rank == 0:
            print("Building single center field free matrix \n")
            
        FF_Ham = eval("SCFFH." + input_par["order"] +"_Order_Hamiltonian(input_par, grid, wigner_3j_dict)")
        FF_Ham.scale(-1.0j * input_par["time_spacing"] * 0.5)

    if input_par["center"] == "double":
        FF_Ham = eval("DCFFH." + input_par["order"] +"_Order_Hamiltonian(input_par, grid, wigner_3j_dict)")
        FF_Ham.scale(-1.0j * input_par["time_spacing"] * 0.5)

        
    if build_status["Int_Mat_X_Stat"] == True:
        if rank == 0:
            print("Building X interaction matrices \n ")
        
        Int_Mat_X = eval("Int." + input_par["gauge"] + "_Gauge_X_Matrix(input_par, grid, wigner_3j_dict)")        
        Int_Mat_X.scale(-1.0j * input_par["time_spacing"] * 0.5)

    if build_status["Int_Mat_Y_Stat"] == True:
        if rank == 0:
            print("Building Y interaction matrices \n ")
            
        Int_Mat_Y = eval("Int." + input_par["gauge"] + "_Gauge_Y_Matrix(input_par, grid, wigner_3j_dict)")        
        Int_Mat_Y.scale(-1.0j * input_par["time_spacing"] * 0.5)
    
    if build_status["Int_Mat_Z_Stat"] == True:
        if rank == 0:
            print("Building Z interaction matrices \n ")
            
        Int_Mat_Z = eval("Int." + input_par["gauge"] + "_Gauge_Z_Matrix(input_par, grid, wigner_3j_dict)")
        Int_Mat_Z.scale(-1.0j * input_par["time_spacing"] * 0.5)
        
    
    if build_status["Int_Mat_Right_Stat"] == True and build_status["Int_Mat_Left_Stat"] == True:
        
        if rank == 0:
            print("Building right/left interaction matrices \n ")
            
        Int_Mat_Right = eval("Int." + input_par["gauge"] + "_Gauge_Right_Circular_Matrix(input_par, grid, wigner_3j_dict)")        
        Int_Mat_Right.scale(-1.0j * input_par["time_spacing"] * 0.5)

        Int_Mat_Left = eval("Int." + input_par["gauge"] + "_Gauge_Left_Circular_Matrix(input_par, grid, wigner_3j_dict)")
        Int_Mat_Left.scale(-1.0j * input_par["time_spacing"] * 0.5)

            
    if build_status["Dip_Acc_Mat_X_Stat"] == True:
        Dip_Acc_Mat_X = eval("DA.Dipole_Acceleration_X_Matrix(input_par, grid, wigner_3j_dict)")
    
    if build_status["Dip_Acc_Mat_Y_Stat"] == True:
        Dip_Acc_Mat_Y = eval("DA.Dipole_Acceleration_Y_Matrix(input_par, grid, wigner_3j_dict)")

        
        if build_status["Dip_Acc_Mat_Z_Stat"] == True:
            Dip_Acc_Mat_Z = eval("DA.Dipole_Acceleration_Z_Matrix(input_par, grid, wigner_3j_dict)")


    return FF_Ham, Int_Mat_X, Int_Mat_Y, Int_Mat_Z, Int_Mat_Left, Int_Mat_Right, Dip_Acc_Mat_X, Dip_Acc_Mat_Y, Dip_Acc_Mat_Z
    
if __name__ == "__main__":
    


    input_par = Mod.Input_File_Reader("input.json")

    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    
    Make_Matrices(input_par)
   
    pr.disable()
    if rank == 0:
        pr.print_stats(sort='time')
        
    # build_status = Necessary_Matrices(input_par)
    # Load_Matrices(input_par, build_status)

