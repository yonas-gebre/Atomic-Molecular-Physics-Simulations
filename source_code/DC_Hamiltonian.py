import sys
import json
import TDSE_Module
from math import floor, ceil
import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def Second_Order_Hamiltonian(Grid, m, l_max_bs, Ro, z, potential):

    h2 = Grid.h2
    matrix_size = Grid.size * (l_max_bs + 1)
    nnz = ceil(l_max_bs/ 2 + 1) + 2

    wigner_3j_dict = {}

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()

    for i  in range(istart, iend):
        l_block = floor(i/Grid.size)
        grid_idx = i %Grid.size
        r = Grid.grid[grid_idx]
        Hamiltonian.setValue(i, i, 1.0/h2 + potential(r, l_block, l_block, m, Ro, z, wigner_3j_dict))
        if grid_idx >=  1:
            Hamiltonian.setValue(i, i-1, (-1.0/2.0)/h2)
        if grid_idx < Grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-1.0/2.0)/h2)
    
        if l_block % 2 == 0:
            l_prime_list = list(range(0, l_max_bs + 1, 2))
        else:
            l_prime_list = list(range(1, l_max_bs + 1, 2))

        l_prime_list.remove(l_block)
        for l_prime in l_prime_list:
            col_idx =Grid.size*l_prime + grid_idx
            Hamiltonian.setValue(i, col_idx, potential(r, l_block, l_prime, m, Ro, z, wigner_3j_dict))

    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def Fourth_Order_Hamiltonian(Grid, m, l_max_bs, Ro, z, potential):

    h2 = Grid.h2
    matrix_size = Grid.size * (l_max_bs + 1)
    nnz = ceil(l_max_bs / 2 + 1) + 4

    wigner_3j_dict = {}

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = Hamiltonian.getOwnershipRange()

    for i  in range(istart, iend):
        l_block = floor(i/Grid.size)
        grid_idx = i %Grid.size
        r = Grid.grid[grid_idx]

        Hamiltonian.setValue(i, i, (15.0/ 12.0)/h2 + potential(r, l_block, l_block, m, Ro, z, wigner_3j_dict))
        if grid_idx >=  1:
            Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
        if grid_idx >= 2:
            Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
        if grid_idx < Grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
        if grid_idx < Grid.size - 2:
            Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)


        if l_block % 2 == 0:
            l_prime_list = list(range(0, l_max_bs + 1, 2))
        else:
            l_prime_list = list(range(1, l_max_bs + 1, 2))

        l_prime_list.remove(l_block)
        for l_prime in l_prime_list:
            col_idx =Grid.size*l_prime + grid_idx
            Hamiltonian.setValue(i, col_idx, potential(r, l_block, l_prime, m, Ro, z, wigner_3j_dict))

    for i in range(0, matrix_size,Grid.size):
        l_block = floor(i/Grid.size)

        Hamiltonian.setValue(i, i, (20.0/24.0)/h2 + potential(Grid.grid[0], l_block, l_block, m, Ro, z, wigner_3j_dict)) 
        Hamiltonian.setValue(i, i+1, (-6.0/24.0)/h2)
        Hamiltonian.setValue(i, i+2, (-4.0/24.0)/h2)
        Hamiltonian.setValue(i, i+3, (1.0/24.0)/h2) 

        j = i + (Grid.size - 1)
        Hamiltonian.setValue(j, j, (20.0/24.0)/h2 + potential(Grid.grid[-1], l_block, l_block, m, Ro, z, wigner_3j_dict)) 
        Hamiltonian.setValue(j, j-1, (-6.0/24.0)/h2)
        Hamiltonian.setValue(j, j-2, (-4.0/24.0)/h2)
        Hamiltonian.setValue(j, j-3, (1.0/24.0)/h2)

    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian
