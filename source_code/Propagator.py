import numpy as np
import sys
import time
from math import ceil
import Laser_Pulse as LP

import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def Build_Psi(psi_inital):

    psi = PETSc.Vec().createMPI(np.size(psi_inital), comm=PETSc.COMM_WORLD)
    istart, iend = psi.getOwnershipRange()
    for i  in range(istart, iend):
        psi.setValue(i, psi_inital[i])

    psi.assemblyBegin()
    psi.assemblyEnd()
    return psi

def Crank_Nicolson_Time_Propagator(input_par, psi_inital, FF_Ham, Int_Mat_X, Int_Mat_Y, Int_Mat_Z, Int_Mat_Left, Int_Mat_Right, Dip_Acc_Mat_X, Dip_Acc_Mat_Y, Dip_Acc_Mat_Z, build_status):
    
    laser_pulse, laser_time, total_polarization, total_poynting, elliptical_pulse, free_prop_idx  = LP.Build_Laser_Pulse(input_par)

    Full_Ham = FF_Ham.duplicate()

    if build_status["Int_Mat_X_Stat"] == True:
        Full_Ham.axpy(0.0, Int_Mat_X, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if build_status["Int_Mat_Y_Stat"] == True:
        Full_Ham.axpy(0.0, Int_Mat_Y, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if build_status["Int_Mat_Right_Stat"] == True:
        Full_Ham.axpy(0.0, Int_Mat_Right, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if build_status["Int_Mat_Left_Stat"] == True:
        Full_Ham.axpy(0.0, Int_Mat_Left, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    if build_status["Int_Mat_Z_Stat"] == True:
        Full_Ham.axpy(0.0, Int_Mat_Z, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    
    Full_Ham_Left = Full_Ham.duplicate()

    Psi = Build_Psi(psi_inital)
    Psi_Right = Psi.duplicate()
    Psi_Dipole = Psi.duplicate()  
  
    Dip_Acc_X = np.zeros(len(laser_time), dtype=complex)
    Dip_Acc_Y = np.zeros(len(laser_time), dtype=complex)
    Dip_Acc_Z = np.zeros(len(laser_time), dtype=complex)

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOptionsPrefix("prop_")

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(input_par["TDSE_File"], mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)

    def Build_Time_Dep_Hamiltonian(i, t):
        FF_Ham.copy(Full_Ham, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        
        if build_status["Int_Mat_Z_Stat"]:  
            Full_Ham.axpy(laser_pulse['z'][i], Int_Mat_Z, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
           
            if build_status["Dip_Acc_Mat_Z_Stat"]:
                Dip_Acc_Mat_Z.mult(Psi, Psi_Dipole)
                Dip_Acc_Z[i] = Psi_Dipole.dot(Psi)

        if build_status["Int_Mat_X_Stat"]:
            Full_Ham.axpy(laser_pulse['x'][i], Int_Mat_X, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

        if build_status["Int_Mat_Y_Stat"]:
            Full_Ham.axpy(laser_pulse['y'][i], Int_Mat_Y, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

        if build_status["Int_Mat_Right_Stat"]:
            Full_Ham.axpy(laser_pulse['Right'][i], Int_Mat_Right, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
           
        if build_status["Int_Mat_Left_Stat"]:
            Full_Ham.axpy(laser_pulse['Left'][i], Int_Mat_Left, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            
        if build_status["Dip_Acc_Mat_X_Stat"]:
            Dip_Acc_Mat_X.mult(Psi, Psi_Dipole)
            Dip_Acc_X[i] = Psi_Dipole.dot(Psi)

        if build_status["Dip_Acc_Mat_Y_Stat"]:
            Dip_Acc_Mat_Y.mult(Psi, Psi_Dipole)
            Dip_Acc_Y[i] = Psi_Dipole.dot(Psi)

                
    if free_prop_idx == len(laser_time) - 1:
        save_idx = [free_prop_idx]
    else:
        save_idx = list(range(free_prop_idx, len(laser_time), int((len(laser_time) - free_prop_idx)/5)))
    save_count = 0
    if rank == 0:
            print("Starting time propagation \n")
            start_time = time.time()
            print(save_idx)
            print("\n \n")
    
    if rank == 0:
        print("Saving Psi inital \n")
    Psi.setName("Psi_Inital") 
    ViewHDF5.view(Psi)

    interval = int(len(laser_time)/input_par["psi_write_frequency"])
    psi_save_array = np.arange(interval , interval + len(laser_time), interval)
    save_count = 0

    for i, t in enumerate(laser_time):
        
        Build_Time_Dep_Hamiltonian(i, t)
        Full_Ham.copy(Full_Ham_Left, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        
        Full_Ham_Left.scale(-1.0)
        Full_Ham_Left.shift(1.0)
        Full_Ham.shift(1.0)

        ksp.setOperators(Full_Ham_Left, Full_Ham_Left)
        ksp.setTolerances(input_par["propagate_tolerance"], PETSc.DEFAULT, PETSc.DEFAULT, PETSc.DEFAULT)
        ksp.setFromOptions()

        Full_Ham.mult(Psi, Psi_Right)
        ksp.solve(Psi_Right, Psi)
        
        if rank == 0:
            print(i, len(laser_time)-1)
    
        if i in save_idx and len(save_idx) > 1:
            if rank == 0:
                print("Saving Wavefunction at " + str(i))

            Psi_name = "Psi" + str(save_count)
            save_count += 1
            Psi.setName(Psi_name) 
            ViewHDF5.view(Psi)

        
    Psi.setName("Psi_Final") 
    ViewHDF5.view(Psi)  
    ViewHDF5.destroy()
    
    if rank == 0:
            print("Finished time propagation")
            print(str(round((time.time() - start_time) / 60, 3)) + " minutes" ) 
    
    if build_status["Dip_Acc_Mat_X_Stat"]:
        np.savetxt("Dip_Acc_X.txt", Dip_Acc_X.view(float))
        
    if build_status["Dip_Acc_Mat_Y_Stat"]:
        np.savetxt("Dip_Acc_Y.txt", Dip_Acc_Y.view(float))
        
    if build_status["Dip_Acc_Mat_Z_Stat"]:
        np.savetxt("Dip_Acc_Z.txt", Dip_Acc_Z.view(float))

    np.savetxt("time.txt", laser_time)

    return None