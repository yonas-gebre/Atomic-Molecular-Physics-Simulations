import sys
import json
from math import floor, ceil
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI

petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def second_order_hamiltonian(grid, m, l_max_bs, ro, z, potential):
    """
    Generate the second-order Hamiltonian matrix.

    Args:
        grid: Grid object representing the spatial grid.
        m: Quantum number.
        l_max_bs: Maximum value of l in the block structure.
        ro: Parameter for potential function.
        z: Parameter for potential function.
        potential: Function for evaluating the potential.

    Returns:
        Hamiltonian matrix in PETSc format.
    """
    h2 = grid.h2
    matrix_size = grid.size * (l_max_bs + 1)
    nnz = ceil(l_max_bs / 2 + 1) + 2

    wigner_3j_dict = {}

    hamiltonian = PETSc.Mat().createAIJ(
        [matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = hamiltonian.getOwnershipRange()

    for i in range(istart, iend):
        l_block = floor(i / grid.size)
        grid_idx = i % grid.size
        r = grid.grid[grid_idx]
        hamiltonian.setValue(i, i, 1.0 / h2 + potential(r,
                             l_block, l_block, m, ro, z, wigner_3j_dict))
        if grid_idx >= 1:
            hamiltonian.setValue(i, i - 1, (-1.0 / 2.0) / h2)
        if grid_idx < grid.size - 1:
            hamiltonian.setValue(i, i + 1, (-1.0 / 2.0) / h2)

        if l_block % 2 == 0:
            l_prime_list = list(range(0, l_max_bs + 1, 2))
        else:
            l_prime_list = list(range(1, l_max_bs + 1, 2))

        l_prime_list.remove(l_block)
        for l_prime in l_prime_list:
            col_idx = grid.size * l_prime + grid_idx
            hamiltonian.setValue(i, col_idx, potential(
                r, l_block, l_prime, m, ro, z, wigner_3j_dict))

    hamiltonian.assemblyBegin()
    hamiltonian.assemblyEnd()
    return hamiltonian


def fourth_order_hamiltonian(grid, m, l_max_bs, ro, z, potential):
    """
    Generate the fourth-order Hamiltonian matrix.

    Args:
        grid: Grid object representing the spatial grid.
        m: Quantum number.
        l_max_bs: Maximum value of l in the block structure.
        ro: Parameter for potential function.
        z: Parameter for potential function.
        potential: Function for evaluating the potential.

    Returns:
        Hamiltonian matrix in PETSc format.
    """
    h2 = grid.h2
    matrix_size = grid.size * (l_max_bs + 1)
    nnz = ceil(l_max_bs / 2 + 1) + 4

    wigner_3j_dict = {}

    hamiltonian = PETSc.Mat().createAIJ(
        [matrix_size, matrix_size], nnz=nnz, comm=PETSc.COMM_WORLD)
    istart, iend = hamiltonian.getOwnershipRange()

    for i in range(istart, iend):
        l_block = floor(i / grid.size)
        grid_idx = i % grid.size
        r = grid.grid[grid_idx]

        hamiltonian.setValue(i, i, (15.0 / 12.0) / h2 +
                             potential(r, l_block, l_block, m, ro, z, wigner_3j_dict))
        if grid_idx >= 1:
            hamiltonian.setValue(i, i - 1, (-2.0 / 3.0) / h2)
        if grid_idx >= 2:
            hamiltonian.setValue(i, i - 2, (1.0 / 24.0) / h2)
        if grid_idx < grid.size - 1:
            hamiltonian.setValue(i, i + 1, (-2.0 / 3.0) / h2)
        if grid_idx < grid.size - 2:
            hamiltonian.setValue(i, i + 2, (1.0 / 24.0) / h2)

        if l_block % 2 == 0:
            l_prime_list = list(range(0, l_max_bs + 1, 2))
        else:
            l_prime_list = list(range(1, l_max_bs + 1, 2))

        l_prime_list.remove(l_block)
        for l_prime in l_prime_list:
            col_idx = grid.size * l_prime + grid_idx
            hamiltonian.setValue(i, col_idx, potential(
                r, l_block, l_prime, m, ro, z, wigner_3j_dict))

    for i in range(0, matrix_size, grid.size):
        l_block = floor(i / grid.size)

        hamiltonian.setValue(i, i, (20.0 / 24.0) / h2 + potential(
            grid.grid[0], l_block, l_block, m, ro, z, wigner_3j_dict))
        hamiltonian.setValue(i, i + 1, (-6.0 / 24.0) / h2)
        hamiltonian.setValue(i, i + 2, (-4.0 / 24.0) / h2)
        hamiltonian.setValue(i, i + 3, (1.0 / 24.0) / h2)

        j = i + (grid.size - 1)
        hamiltonian.setValue(j, j, (20.0 / 24.0) / h2 + potential(
            grid.grid[-1], l_block, l_block, m, ro, z, wigner_3j_dict))
        hamiltonian.setValue(j, j - 1, (-6.0 / 24.0) / h2)
        hamiltonian.setValue(j, j - 2, (-4.0 / 24.0) / h2)
        hamiltonian.setValue(j, j - 3, (1.0 / 24.0) / h2)

    hamiltonian.assemblyBegin()
    hamiltonian.assemblyEnd()
    return hamiltonian
