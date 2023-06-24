import sys
import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)


def Fourth_Order_Hamiltonian(Grid, potential):
    
    """Returns the fourth order hamiltonian matrix for single center system.
       The matrix is a petsc matrix that is sparce. The arouments are a Grid class object and
       an array called potential that has the potential values on the grid."""
    matrix_size = Grid.size ## this is the grid size
    h2 = Grid.h2 ### this is the grid spacing

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=5, comm=PETSc.COMM_WORLD) ### creating sparce matrix and nnz specifies number of non zeros elements
    istart, iend = Hamiltonian.getOwnershipRange() ### for multiple cores this tells them which blocks of the matrix belong to each core
    for i  in range(istart, iend):
        Hamiltonian.setValue(i, i, potential[i] + (15.0/ 12.0)/h2) ### diagonal stencil and the potential value at the grid point
        
        #### Adding fourth order non diagonal stencil elements
        if i >=  1:
            Hamiltonian.setValue(i, i-1, (-2.0/3.0)/h2)
        if i >= 2:
            Hamiltonian.setValue(i, i-2, (1.0/24.0)/h2)
        if i < Grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-2.0/3.0)/h2)
        if i < Grid.size - 2:
            Hamiltonian.setValue(i, i+2, (1.0/24.0)/h2)

   
    Hamiltonian.setValue(0,0, potential[0]  + (20.0/24.0)/h2)
    Hamiltonian.setValue(0,1, (-6.0/24.0)/h2)
    Hamiltonian.setValue(0,2, (-4.0/24.0)/h2)
    Hamiltonian.setValue(0,3, (1.0/24.0)/h2)
    
    j = Grid.size - 1
    Hamiltonian.setValue(j,j, potential[j] + (20.0/24.0)/h2)
    Hamiltonian.setValue(j,j - 1, (-6.0/24.0)/h2)
    Hamiltonian.setValue(j,j - 2, (-4.0/24.0)/h2)
    Hamiltonian.setValue(j,j - 3, (1.0/24.0)/h2)

    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

def Second_Order_Hamiltonian(Grid, potential):
    """Returns the fourth order hamiltonian matrix for single center system.
       The matrix is a petsc matrix that is sparce. The arouments are a Grid class object and
       an array called potential that has the potential values on the grid."""
       
    matrix_size = Grid.size ## this is the grid size
    h2 = Grid.h2 ### this is the grid spacing

    Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD) ### creating sparce matrix and nnz specifies number of non zeros elements
    istart, iend = Hamiltonian.getOwnershipRange() ### for multiple cores this tells them which blocks of the matrix belong to each core
    for i  in range(istart, iend):

        Hamiltonian.setValue(i, i, potential[i] +  1.0/h2)    ### diagonal stencil and the potential value at the grid point
        #### Adding second order non diagonal stencil elements
        if i >=  1:
            Hamiltonian.setValue(i, i-1, (-1.0/2.0)/h2)
        if i < Grid.size - 1:
            Hamiltonian.setValue(i, i+1, (-1.0/2.0)/h2)

    Hamiltonian.assemblyBegin()
    Hamiltonian.assemblyEnd()
    return Hamiltonian

