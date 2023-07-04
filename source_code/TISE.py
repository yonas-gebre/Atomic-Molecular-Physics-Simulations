import slepc4py
import Module as Mod
import SC_Potential as SCP
import DC_Hamiltonian as DCH
import SC_Hamiltonian as SCH
from mpi4py import MPI
from slepc4py import SLEPc
import time
import sys
import os
from scipy.special import spherical_in as In
from scipy.special import spherical_kn as Kn
import DC_Potential as DCP
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)
petsc4py.init(comm=PETSc.COMM_WORLD)
slepc4py.init(sys.argv)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sys.path.append(os.getcwd()[:-6] + "General_Functions")


def single_center_tise(input_par, grid, view_hdf5):
    for l in range(0, input_par["n_max"]):
        if rank == 0:
            print("Calculating the eigenstates for l =", l, "\n")

        potential = eval(
            "SCP." + input_par["potential"] + "(grid.grid, l, input_par['z'])")
        hamiltonian = eval(
            "SCH." + input_par["order"] + "_Order_Hamiltonian(grid, potential)")
        dimension_size = int(PETSc.Mat.getSize(hamiltonian)[0]) * 0.1
        eigen_value_solver(
            hamiltonian, input_par["n_max"] - l, l, input_par["tolerance"], view_hdf5, dimension_size)


def double_center_tise(input_par, grid, view_hdf5):
    l_max_bs = input_par["l_max_bs_for_double_center"]
    z, Ro = input_par["z"], input_par["Ro"]
    potential = input_par['potential']
    for m in range(0, input_par["m_max_bs_for_double_center"] + 1):
        if rank == 0:
            print("Calculating the states for m =", m, "\n")

        hamiltonian = eval(
            "DCH." + input_par["order"] + "_Order_Hamiltonian(grid, m, l_max_bs, Ro, z, DCP." + input_par['potential'] + ")")
        dimension_size = int(PETSc.Mat.getSize(hamiltonian)[0]) * 0.001
        if rank == 0:
            print("Finished Building Hamiltonian" + "\n")

        eigen_value_solver(
            hamiltonian, input_par["n_max"], m, input_par["tolerance"], view_hdf5, dimension_size)
        hamiltonian.destroy()


def eigen_value_solver(hamiltonian, number_of_eigenvalues, qn_number, tolerance, viewer, dimension_size):
    if rank == 0:
        print("Diagonalizing Hamiltonian for calculating bound states\n")

    ev_solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    ev_solver.setOperators(hamiltonian)
    ev_solver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    ev_solver.setTolerances(tolerance, PETSc.DECIDE)
    ev_solver.setWhichEigenpairs(ev_solver.Which.SMALLEST_REAL)
    ev_solver.setDimensions(number_of_eigenvalues,
                            PETSc.DECIDE, dimension_size)
    ev_solver.solve()

    converged = ev_solver.getConverged()
    count = 0
    for i in range(converged):
        eigen_vector = hamiltonian.getVecLeft()
        eigen_state = ev_solver.getEigenpair(i, eigen_vector)

        if eigen_state.real > 0:
            break

        eigen_vector.setName("Psi_" + str(qn_number) + "_" + str(i))
        viewer.view(eigen_vector)

        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        energy.setValue(0, eigen_state)

        if rank == 0:
            print("Eigenvalue for " + str(qn_number) + " " +
                  str(i + 1) + " is " + str(eigen_state))

        energy.setName("Energy_" + str(qn_number) + "_" + str(i))
        energy.assemblyBegin()
        energy.assemblyEnd()
        viewer.view(energy)
        count += 1

    if rank == 0:
        print("\n")
        print("Number of eigenvalues requested and saved")
        print(number_of_eigenvalues, count, "\n")


def TISE(input_par):
    Grid = Mod.Grid(input_par["grid_spacing"], input_par["grid_size"])

    if rank == 0:
        start_time = time.time()
        print("Calculating states for " +
              str(input_par["potential"]) + " potential." "\n")

    # hdf5 petsc viewer, this is a hdf5 file where the bound states will be stored.
    view_hdf5 = PETSc.Viewer()
    view_hdf5.createHDF5(
        input_par["Target_File"], mode=PETSc.Viewer.Mode.WRITE, comm=PETSc.COMM_WORLD)

    if input_par["center"] == "single":
        single_center_tise(input_par, Grid, view_hdf5)
    if input_par["center"] == "double":
        double_center_tise(input_par, Grid, view_hdf5)

    if rank == 0:
        total_time = (time.time() - start_time) / 60
        print("Total time taken for calculating bound states is " +
              str(round(total_time, 1)) + " minutes !!!")
        print("**************************************************************** \n")


if __name__ == "__main__":
    input_par = Mod.Input_File_Reader("input.json")
    TISE(input_par)
