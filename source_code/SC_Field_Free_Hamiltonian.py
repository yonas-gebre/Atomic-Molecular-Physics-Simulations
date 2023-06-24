import sys
from numpy import pi, exp, sqrt, where, zeros
from math import floor
import SC_Potential as SCP
         
idx =  [i for i, itr in enumerate(sys.path[0]) if itr == "/"]
path_address = sys.path[0][:idx[-1]]
    
sys.path.append(path_address + '/General_Functions')
import Module

import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def Fourth_Order_Hamiltonian(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Module.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h2 = grid.h2
    ECS_idx = ECS_IDX(input_par, grid) 
    ECS_Stencil = Fourth_Order_Stencil()

    potential = zeros(shape=(input_par["l_max"] + 1, grid.size))
    for l, r in enumerate(potential):
        potential[l] = eval("SCP." + input_par["potential"] + "(grid.grid, l, input_par['z'])")

    FF_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=5, comm=PETSc.COMM_WORLD)
    istart, iend = FF_Hamiltonian.getOwnershipRange()

    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        grid_idx = i % grid.size
        
      
        if grid_idx < ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + (15.0/ 12.0)/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
            if grid_idx < grid.size - 1:
                FF_Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
            if grid_idx < grid.size - 2:
                FF_Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)
        
        if grid_idx == ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + ECS_Stencil[2]/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, ECS_Stencil[1]/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, ECS_Stencil[0]/h2)
            if grid_idx < grid.size - 1:
                FF_Hamiltonian.setValue(i, i+1, ECS_Stencil[3]/h2)
            if grid_idx < grid.size - 2:
                FF_Hamiltonian.setValue(i, i+2, ECS_Stencil[4]/h2)

        if grid_idx > ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + (15.0/ 12.0)* -1.0j/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, (-2.0/3.0) * -1.0j/h2)
            if grid_idx >= 2:
                FF_Hamiltonian.setValue(i, i-2, (1.0/24.0) * -1.0j/h2)
            if grid_idx < grid.size - 1:
                FF_Hamiltonian.setValue(i, i+1, (-2.0/3.0) * -1.0j/h2)
            if grid_idx < grid.size - 2:
                FF_Hamiltonian.setValue(i, i+2, (1.0/24.0) * -1.0j/h2)

    for i in range(0, matrix_size, grid.size):
        l_block = block_to_qn[floor(i/grid.size)][0]
        
        FF_Hamiltonian.setValue(i, i, potential[l_block][0] + (20.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+1, (-6.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+2, (-4.0/24.0)/h2)
        FF_Hamiltonian.setValue(i, i+3, (1.0/24.0)/h2)

        j = i + (grid.size - 1)
        FF_Hamiltonian.setValue(j,j, potential[l_block][-1] + (20.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 1, (-6.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 2, (-4.0/24.0) * -1.0j/h2)
        FF_Hamiltonian.setValue(j,j - 3, (1.0/24.0) * -1.0j/h2)

    FF_Hamiltonian.assemblyBegin()
    FF_Hamiltonian.assemblyEnd()
    return FF_Hamiltonian  

def Second_Order_Hamiltonian(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Module.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h2 = grid.h2
    ECS_idx = ECS_IDX(input_par, grid) 
    

    potential = zeros(shape=(input_par["l_max"] + 1, grid.size))
    for l, r in enumerate(potential):
        potential[l] = eval("SCP." + input_par["potential"] + "(grid.grid, l, input_par['z'])")
        

    FF_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = FF_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        grid_idx = i % grid.size

        if grid_idx < ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] + 1.0/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, -0.5/h2)
            if grid_idx < grid.size - 1:
                FF_Hamiltonian.setValue(i, i+1, -0.5/h2)
        
        elif grid_idx > ECS_idx:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] - 1.0j/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, 0.5j/h2)
            if grid_idx < grid.size - 1:
                FF_Hamiltonian.setValue(i, i+1, 0.5j/h2)

        else:
            FF_Hamiltonian.setValue(i, i, potential[l_block][grid_idx] +  exp(-1.0j*pi/4.0)/h2)
            if grid_idx >=  1:
                FF_Hamiltonian.setValue(i, i-1, -1/(1 + exp(1.0j*pi/4.0))/h2)
            if grid_idx < grid.size - 1:
                FF_Hamiltonian.setValue(i, i+1, -1*exp(-1.0j*pi/4.0)/ (1+exp(1.0j*pi/4.0))/h2)

 
    FF_Hamiltonian.assemblyBegin()
    FF_Hamiltonian.assemblyEnd()
    return FF_Hamiltonian  

def ECS_IDX(input_par, grid):
    if input_par["ECS_region"] < 1.00 and input_par["ECS_region"] > 0.00:
        ECS_idx = where(grid.grid > grid.grid[-1] * input_par["ECS_region"])[0][0]
        return ECS_idx
    elif input_par["ECS_region"] == 1.00:
        ECS_idx = grid.size
        if rank == 0:
            print("No ECS applied for this run \n")
        return ECS_idx
    else:
        if rank == 0:
            print("ECS region has to be between 0.0 and 1.00\n")
            exit()

def Fourth_Order_Stencil():
    x_2 = 0.25*(2j - 3*exp(3j*pi/4)) / (1 + 2j + 3*exp(1j*pi/4))
    x_1 = (-2j + 6*exp(3j*pi/4)) / (2 + 1j + 3*exp(1j*pi/4))
    x = 0.25*(-2 + 2j - 9*exp(3j*pi/4))
    x__1 = (2 + 2j - 6*sqrt(2)) / (3 + 1j + 3*sqrt(2))
    x__2 = 0.25*(-2 -2j + 3*sqrt(2)) / (3 - 1j + 3*sqrt(2))
    return (x__2, x__1, x, x_1, x_2)

   