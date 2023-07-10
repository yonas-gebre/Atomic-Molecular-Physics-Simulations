import sys
import petsc4py
from petsc4py import PETSc
from math import sqrt, floor
from numpy import power
from TDSE_Module import Check_Dictionary_Key_For_W3J_Symbol
import Module as Mod

# Initialize petsc4py
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)

# Get the index of the last occurrence of "/"
idx = [i for i, itr in enumerate(sys.path[0]) if itr == "/"]
path_address = sys.path[0][:idx[-1]]

# Append the path for the General_Functions module
sys.path.append(path_address + '/General_Functions')


def Dipole_Acceleration_Z_Matrix(input_par, grid, wigner_3j_dict):
    """
    Compute the dipole acceleration matrix in the Z direction.

    Args:
        input_par: Input parameters.
        grid: Grid information.
        wigner_3j_dict: Dictionary of pre-computed Wigner 3j symbols.

    Returns:
        Dipole_Acceleration_Matrix: The dipole acceleration matrix.
    """
    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    grid2 = power(grid.grid, 2.0)

    # Create the matrix
    Dipole_Acceleration_Matrix = PETSc.Mat().createAIJ(
        [matrix_size, matrix_size], nnz=2, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Acceleration_Matrix.getOwnershipRange()

    for i in range(istart, iend):
        l_block = block_to_qn[floor(i / grid.size)][0]
        m_block = block_to_qn[floor(i / grid.size)][1]
        grid_idx = i % grid.size

        if l_block < input_par["l_max"]:
            columon_idx = grid.size * \
                qn_to_block[(l_block + 1, m_block)] + grid_idx
            matrix_element = (l_block + 1) / sqrt((2 * l_block + 1)
                                                  * (2 * l_block + 3)) / grid2[grid_idx]
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, matrix_element)

        if abs(m_block) < l_block and l_block > 0:
            columon_idx = grid.size * \
                qn_to_block[(l_block - 1, m_block)] + grid_idx
            matrix_element = l_block / \
                sqrt((2 * l_block - 1) * (2 * l_block + 1)) / grid2[grid_idx]
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, matrix_element)

    Dipole_Acceleration_Matrix.assemblyBegin()
    Dipole_Acceleration_Matrix.assemblyEnd()
    return Dipole_Acceleration_Matrix


def Dipole_Acceleration_X_Matrix(input_par, grid, wigner_3j_dict):
    """
    Compute the dipole acceleration matrix in the X direction.

    Args:
        input_par: Input parameters.
        grid: Grid information.
        wigner_3j_dict: Dictionary of pre-computed Wigner 3j symbols.

    Returns:
        Dipole_Acceleration_Matrix: The dipole acceleration matrix.
    """
    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    grid2 = power(grid.grid, 2.0)

    # Create the matrix
    Dipole_Acceleration_Matrix = PETSc.Mat().createAIJ(
        [matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Acceleration_Matrix.getOwnershipRange()

    for i in range(istart, iend):
        l_block = block_to_qn[floor(i / grid.size)][0]
        m_block = block_to_qn[floor(i / grid.size)][1]
        grid_idx = i % grid.size

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(
                wigner_3j_dict, (l_block, 1, l_prime, 0, 0, 0))
            common_factor = pow(-1.0, m_block) * sqrt((2 * l_block + 1) * (2 * l_prime + 1) / 2) * wigner_3j_dict[
                (l_block, 1, l_prime, 0, 0, 0)]

            m_prime = m_block - 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
            matrix_element = common_factor * (
                wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] - wigner_3j_dict[
                    (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
            columon_idx = grid.size * \
                qn_to_block[(l_prime, m_prime)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, matrix_element)

            m_prime = m_block + 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
            matrix_element = common_factor * (
                wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] - wigner_3j_dict[
                    (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
            columon_idx = grid.size * \
                qn_to_block[(l_prime, m_prime)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, matrix_element)

        if l_block > 0:
            l_prime = l_block - 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(
                wigner_3j_dict, (l_block, 1, l_prime, 0, 0, 0))
            common_factor = pow(-1.0, m_block) * l_prime = l_block - 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(
                wigner_3j_dict, (l_block, 1, l_prime, 0, 0, 0))
            common_factor = pow(-1.0, m_block) * sqrt((2 * l_block + 1) * (2 * l_prime + 1) / 2) * wigner_3j_dict[
                (l_block, 1, l_prime, 0, 0, 0)]

            if -1 * m_block < l_block - 1:
                m_prime = m_block - 1
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
                matrix_element = common_factor * (
                    wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] - wigner_3j_dict[
                        (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
                columon_idx = grid.size * \
                    qn_to_block[(l_prime, m_prime)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(
                    i, columon_idx, matrix_element)

            if m_block < l_block - 1:
                m_prime = m_block + 1
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
                matrix_element = common_factor * (
                    wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] - wigner_3j_dict[
                        (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
                columon_idx = grid.size * \
                    qn_to_block[(l_prime, m_prime)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(
                    i, columon_idx, matrix_element)

    Dipole_Acceleration_Matrix.assemblyBegin()
    Dipole_Acceleration_Matrix.assemblyEnd()
    return Dipole_Acceleration_Matrix


def Dipole_Acceleration_Y_Matrix(input_par, grid, wigner_3j_dict):
    """
    Compute the dipole acceleration matrix in the Y direction.

    Args:
        input_par: Input parameters.
        grid: Grid information.
        wigner_3j_dict: Dictionary of pre-computed Wigner 3j symbols.

    Returns:
        Dipole_Acceleration_Matrix: The dipole acceleration matrix.
    """
    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    grid2 = power(grid.grid, 2.0)

    # Create the matrix
    Dipole_Acceleration_Matrix = PETSc.Mat().createAIJ(
        [matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Dipole_Acceleration_Matrix.getOwnershipRange()

    for i in range(istart, iend):
        l_block = block_to_qn[floor(i / grid.size)][0]
        m_block = block_to_qn[floor(i / grid.size)][1]
        grid_idx = i % grid.size

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(
                wigner_3j_dict, (l_block, 1, l_prime, 0, 0, 0))
            common_factor = 1.0j * pow(-1.0, m_block) * sqrt((2 * l_block + 1) * (2 * l_prime + 1) / 2) * wigner_3j_dict[
                (l_block, 1, l_prime, 0, 0, 0)]

            m_prime = m_block - 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
            matrix_element = common_factor * (
                wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] + wigner_3j_dict[
                    (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
            columon_idx = grid.size * \
                qn_to_block[(l_prime, m_prime)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, matrix_element)

            m_prime = m_block + 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                 (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
            matrix_element = common_factor * (
                wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] + wigner_3j_dict[
                    (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
            columon_idx = grid.size * \
                qn_to_block[(l_prime, m_prime)] + grid_idx
            Dipole_Acceleration_Matrix.setValue(i, columon_idx, matrix_element)

        if l_block > 0:
            l_prime = l_block - 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(
                wigner_3j_dict, (l_block, 1, l_prime, 0, 0, 0))
            common_factor = 1.0j * pow(-1.0, m_block) * sqrt((2 * l_block + 1) * (2 * l_prime + 1) / 2) * wigner_3j_dict[
                (l_block, 1, l_prime, 0, 0, 0)]

            if -1 * m_block < l_block - 1:
                m_prime = m_block - 1
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
                matrix_element = factor * (
                    wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] + wigner_3j_dict[
                        (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
                columon_idx = grid.size * \
                    qn_to_block[(l_prime, m_prime)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(
                    i, columon_idx, matrix_element)

            if m_block < l_block - 1:
                m_prime = m_block + 1
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, 1, m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict,
                                                                     (l_block, 1, l_prime, -1 * m_block, -1, m_prime))
                matrix_element = factor * (
                    wigner_3j_dict[(l_block, 1, l_prime, -1 * m_block, -1, m_prime)] + wigner_3j_dict[
                        (l_block, 1, l_prime, -1 * m_block, 1, m_prime)]) / grid2[grid_idx]
                columon_idx = grid.size * \
                    qn_to_block[(l_prime, m_prime)] + grid_idx
                Dipole_Acceleration_Matrix.setValue(
                    i, columon_idx, matrix_element)

    Dipole_Acceleration_Matrix.assemblyBegin()
    Dipole_Acceleration_Matrix.assemblyEnd()
    return Dipole_Acceleration_Matrix
