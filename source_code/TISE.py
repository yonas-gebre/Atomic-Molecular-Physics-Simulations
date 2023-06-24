import time
import sys
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD

import petsc4py
from petsc4py import PETSc
# petsc4py.init(sys.argv)
# petsc4py.init(comm=PETSc.COMM_WORLD)
import slepc4py 
from slepc4py import SLEPc
# petsc4py.init(comm=PETSc.COMM_WORLD)
# slepc4py.init(sys.argv)

exit()
rank = comm.Get_rank()

import SC_Hamiltonian as SCH
import DC_Hamiltonian as DCH
import SC_Potential as SCP
# import DC_Potential as DCP

sys.path.append(os.getcwd()[:-6] + "General_Functions")
import Module as Mod

def Single_Center_TISE(input_par, Grid, ViewHDF5):
    
    l_list = [1]
    for l in l_list:#range(input_par["n_max"]):
        if rank == 0:
            print("Calculating the eigenstates for l = " + str(l) + "\n")

        potential = eval("SCP." + input_par["potential"] + "(Grid.grid, l, input_par['z'])") 
        hamiltonian = eval("SCH." + input_par["order"] + "_Order_Hamiltonian(Grid, potential)")
        dimension_size = int(PETSc.Mat.getSize(hamiltonian)[0]) * 0.1
        Eigen_Value_Solver(hamiltonian, input_par["n_max"] - l, l, input_par["tolerance"], ViewHDF5, dimension_size)

def Double_Center_TISE(input_par, Grid, ViewHDF5):

    l_max_bs = input_par["l_max_bs_for_double_center"]
    z, Ro = input_par["z"], input_par["Ro"] 
    potential = input_par['potential']
    for m in range(0, input_par["m_max_bs_for_double_center"] + 1):
        if rank == 0:
            print("Calculating the states for m = " + str(m) + "\n")

        hamiltonian = eval("DCH." + input_par["order"] + "_Order_Hamiltonian(Grid, m, l_max_bs, Ro, z, DCP." + input_par['potential'] + ")")
        dimension_size = int(PETSc.Mat.getSize(hamiltonian)[0]) * 0.001
        if rank == 0:
            print("Finished Building Hamiltonian" + "\n")

        Eigen_Value_Solver(hamiltonian, input_par["n_max"], m, input_par["tolerance"], ViewHDF5, dimension_size)
        hamiltonian.destroy()

def Eigen_Value_Solver(Hamiltonian, number_of_eigenvalues, qn_number, tolerance, Viewer, dimension_size):
        
    if rank == 0:
        print("Diagonalizing hamiltonian for calculating bound states\n ")
    
    EV_Solver = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    EV_Solver.setOperators(Hamiltonian)
    EV_Solver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    EV_Solver.setTolerances(tolerance, PETSc.DECIDE)
    EV_Solver.setWhichEigenpairs(EV_Solver.Which.SMALLEST_REAL)
    EV_Solver.setDimensions(number_of_eigenvalues , PETSc.DECIDE, dimension_size)
    EV_Solver.solve()
    
    
    converged = EV_Solver.getConverged()
    count = 0 
    for i in range(converged):

        eigen_vector = Hamiltonian.getVecLeft()
        eigen_state = EV_Solver.getEigenpair(i, eigen_vector)
        
        if eigen_state.real > 0:
            break
        
        eigen_vector.setName("Psi_" + str(qn_number) + "_"  + str(i))
        Viewer.view(eigen_vector)
        
        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
        energy.setValue(0,eigen_state)
        
        if rank == 0:
            print("Eigenvalue for " + str(qn_number) + " " + str(i + 1) + " is " + str(eigen_state))

        energy.setName("Energy_" + str(qn_number) + "_"  + str(i))
        energy.assemblyBegin()
        energy.assemblyEnd()
        Viewer.view(energy)
        count += 1
        
    if rank == 0:
        print("\n")
        print("Number of eigenvalues requested and saved")
        print(number_of_eigenvalues, count, "\n")
        
def TISE(input_par):
       
    Grid = Mod.Grid(input_par["grid_spacing"], input_par["grid_size"])
    
    if rank == 0:
        start_time = time.time()
        print("Calculating states for " + str(input_par["potential"]) +  " potential." "\n ")

    ViewHDF5 = PETSc.Viewer() ##hdf5 petsc viewer, this is a hdf5 file where the bound states will be stored.
    ViewHDF5.createHDF5(input_par["Target_File"], mode=PETSc.Viewer.Mode.WRITE, comm = PETSc.COMM_WORLD)

    if input_par["center"] == "single":
        Single_Center_TISE(input_par, Grid, ViewHDF5)
    if input_par["center"] == "double":
        Double_Center_TISE(input_par, Grid, ViewHDF5)

    if rank == 0:
        total_time = (time.time() - start_time) / 60
        print("Total time taken for calculating bound states is " + str(round(total_time, 1)) + " minutes !!!")
        print("**************************************************************** \n")

if __name__ == "__main__":
    
    input_par = Mod.Input_File_Reader("input.json")
    TISE(input_par)
        