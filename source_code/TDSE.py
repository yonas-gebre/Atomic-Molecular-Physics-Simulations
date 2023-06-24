import sys
import json
import numpy as np
import TISE
import TDSE_Module
import Error_Check
import Propagator
import Construct_Matrix

import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)
import slepc4py 
from slepc4py import SLEPc
petsc4py.init(comm=PETSc.COMM_WORLD)
slepc4py.init(sys.argv)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

idx =  [i for i, itr in enumerate(sys.path[0]) if itr == "/"]
path_address = sys.path[0][:idx[-1]]
    
sys.path.append(path_address + '/General_Functions')
import Module as Mod

def Eigen_State_Solver(input_par):
    
    if input_par["solver"] == "File":
        if rank == 0:
            print("Reading eigenstates from " + input_par["Target_File"] + "\n")
              
        wave_function = TDSE_Module.Inital_State_Wave_Function(input_par)
        return wave_function

    elif input_par["solver"] == "SLEPC":
        if rank == 0:
            print("Calculating the Eigenstates and storing them in " + input_par["Target_File"] + "\n")

        TISE.TISE(input_par)
        wave_function = TDSE_Module.Inital_State_Wave_Function(input_par)
        return wave_function

    else:
        if rank == 0:
            print("\n Argumet for the solver must be 'File' or 'SLEPC' \n")
            exit()
  
    
if __name__=="__main__":

    input_par = Mod.Input_File_Reader("input.json")
    grid = Mod.Grid(input_par["grid_spacing"], input_par["grid_size"])
    wave_function = Eigen_State_Solver(input_par)
    
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    
    print(rank)
    # if input_par["propagate"] == 1: 
    #     if input_par["center"] == "single":
    #         psi_inital = TDSE_Module.Make_Inital_State(input_par, wave_function, grid)
        
    #     if input_par["center"] == "double":
    #         psi_inital = TDSE_Module.Make_Inital_State(input_par, wave_function, grid)
            
    #     if rank == 0:
    #         print("Got inital state \n")
            
    #     build_status = Construct_Matrix.Necessary_Matrices(input_par)
    #     FF_Ham, Int_Mat_X, Int_Mat_Y, Int_Mat_Z, Int_Mat_Left, Int_Mat_Right, Dip_Acc_Mat_X, Dip_Acc_Mat_Y, Dip_Acc_Mat_Z = Construct_Matrix.Make_Matrices(input_par)
    #     Propagator.Crank_Nicolson_Time_Propagator(input_par, psi_inital, FF_Ham, Int_Mat_X, Int_Mat_Y, Int_Mat_Z, Int_Mat_Left, Int_Mat_Right, Dip_Acc_Mat_X, Dip_Acc_Mat_Y, Dip_Acc_Mat_Z, build_status)

    # pr.disable()
    # if rank == 0:
    #     pr.print_stats(sort='time')

