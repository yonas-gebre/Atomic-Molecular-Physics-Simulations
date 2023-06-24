import sys
import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)
from math import sqrt, floor
from sympy.physics.wigner import gaunt, wigner_3j
from TDSE_Module import Check_Dictionary_Key_For_W3J_Symbol
idx = [i for i, itr in enumerate(sys.path[0]) if itr == "/"]
path_address = sys.path[0][:idx[-1]]
sys.path.append(path_address + '/General_Functions')
import Module as Mod

def Length_Gauge_Z_Matrix(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=2, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,0,m_block))
            factor = pow(-1.0, m_block)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,0,m_block)]

            column_idx = grid.size*qn_to_block[(l_prime, m_block)] + grid_idx
            matrix_element = grid.grid[grid_idx] * sqrt((2.0*l_prime+1.0)*(2.0*l_block+1.0)) * factor
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)
        
        
        if abs(m_block) < l_block and l_block > 0:
            l_prime = l_block - 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,0,m_block))
            factor = pow(-1.0, m_block)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,0,m_block)]

            column_idx = grid.size*qn_to_block[(l_prime, m_block)] + grid_idx
            matrix_element = grid.grid[grid_idx] * sqrt((2.0*l_prime+1.0)*(2.0*l_block+1.0)) * factor
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)
        
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    return Int_Hamiltonian 

def Length_Gauge_X_Matrix(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            m_prime = m_block - 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)

            m_prime = m_block + 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)
        
        if l_block > 0:
            l_prime = l_block - 1
            
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            if -1*m_block < l_prime:
                m_prime = m_block - 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
                Int_Hamiltonian.setValue(i, column_idx, matrix_element)
            
            if m_block < l_prime:
                m_prime = m_block + 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
                Int_Hamiltonian.setValue(i, column_idx, matrix_element)

           
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    return Int_Hamiltonian

def Length_Gauge_Y_Matrix(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*pow((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0,0.5)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            m_prime = m_block - 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)

            m_prime = m_block + 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)
        
        if l_block > 0:
            l_prime = l_block - 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*pow((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0,0.5)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            if -1*m_block < l_prime:
                m_prime = m_block - 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
                Int_Hamiltonian.setValue(i, column_idx, matrix_element)
            
            if m_block < l_prime:
                m_prime = m_block + 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                matrix_element = grid.grid[grid_idx] * common_factor * (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])
                Int_Hamiltonian.setValue(i, column_idx, matrix_element)

           
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    Int_Hamiltonian.scale(1.0j)
    return Int_Hamiltonian

def Length_Gauge_Left_Circular_Matrix(input_par, grid, wigner_3j_dict):
    
    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=4, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 

    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            m_prime = m_block - 1

            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))

            common_factor =  pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]
            matrix_element = grid.grid[grid_idx] * common_factor * -2 * wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)]
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)

        if l_block > 0:
            l_prime = l_block - 1
            
            if -1*m_block < l_prime:
                m_prime = m_block - 1
                
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))

                common_factor =  pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]
                matrix_element = grid.grid[grid_idx] * common_factor * -2 * wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)]
                Int_Hamiltonian.setValue(i, column_idx, matrix_element)
                     
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    return Int_Hamiltonian

def Length_Gauge_Right_Circular_Matrix(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
   
    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=2, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    

    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            m_prime = m_block + 1

            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            common_factor =  pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]
            matrix_element = grid.grid[grid_idx] * common_factor * 2 * wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)]
            Int_Hamiltonian.setValue(i, column_idx, matrix_element)
        

        if l_block > 0:
            l_prime = l_block - 1
            
            if m_block < l_prime:
                m_prime = m_block + 1

                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                common_factor =  pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]
                matrix_element = grid.grid[grid_idx] * common_factor * 2 * wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)]
                Int_Hamiltonian.setValue(i, column_idx, matrix_element)

    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    return Int_Hamiltonian 

def Velocity_Gauge_Z_Matrix(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h = grid.spacing

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=6, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1

            common_factor = pow(-1.0,m_block)*sqrt((2.0*l_prime+1.0)*(2.0*l_block+1.0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,0,m_block))
            common_factor *= wigner_3j_dict[(l_block,1,l_prime,0,0,0)]*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,0,m_block)]
           
            column_idx = grid.size*qn_to_block[(l_prime, m_block)] + grid_idx

            Int_Hamiltonian.setValue(i, column_idx, common_factor * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor/(2.0*h))
        
        
        if abs(m_block) < l_block and l_block > 0:
            l_prime = l_block - 1
            
            common_factor = pow(-1.0,m_block)*sqrt((2.0*l_prime+1.0)*(2.0*l_block+1.0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,0,m_block))
            common_factor *= wigner_3j_dict[(l_block,1,l_prime,0,0,0)]*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,0,m_block)]
            
            column_idx = grid.size*qn_to_block[(l_prime, m_block)] + grid_idx

            Int_Hamiltonian.setValue(i, column_idx, common_factor * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor/(2.0*h))
        
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    Int_Hamiltonian.scale(-1.0j)
    return Int_Hamiltonian 

def Velocity_Gauge_X_Matrix(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h = grid.spacing

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=12, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            m_prime = m_block - 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

            Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))
            

            m_prime = m_block + 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

            Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))

        
        if l_block > 0:
            l_prime = l_block - 1
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            if -1*m_block < l_prime:
                m_prime = m_block - 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

                Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
                if grid_idx < grid.size - 1:
                    Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
                if grid_idx >=  1:
                    Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))
                
            if m_block < l_prime:
                m_prime = m_block + 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] - wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

                Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
                if grid_idx < grid.size - 1:
                    Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
                if grid_idx >=  1:
                    Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))

           
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    Int_Hamiltonian.scale(-1.0j)
    return Int_Hamiltonian

def Velocity_Gauge_Y_Matrix(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h = grid.spacing

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=12, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            m_prime = m_block - 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

            Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))
            

            m_prime = m_block + 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

            Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))

        
        if l_block > 0:
            l_prime = l_block - 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            if -1*m_block < l_prime:
                m_prime = m_block - 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

                Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
                if grid_idx < grid.size - 1:
                    Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
                if grid_idx >=  1:
                    Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))
                
            if m_block < l_prime:
                m_prime = m_block + 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                wigner_term = (wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)] + wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)])

                Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
                if grid_idx < grid.size - 1:
                    Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
                if grid_idx >=  1:
                    Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))

           
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    return Int_Hamiltonian

def Velocity_Gauge_Left_Circular_Matrix(input_par, grid, wigner_3j_dict):
    VGLCM_Top_Half = Velocity_Gauge_Left_Circular_Matrix_Top_Half(input_par, grid, wigner_3j_dict)
    VGLCM_Bottom_Half = Velocity_Gauge_Left_Circular_Matrix_Bottom_Half(input_par, grid, wigner_3j_dict)
    VGLCM_Top_Half.axpy(1.0, VGLCM_Bottom_Half, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    return VGLCM_Top_Half

def Velocity_Gauge_Right_Circular_Matrix(input_par, grid, wigner_3j_dict):
    VGRCM_Top_Half = Velocity_Gauge_Right_Circular_Matrix_Top_Half(input_par, grid, wigner_3j_dict)
    VGRCM_Bottom_Half = Velocity_Gauge_Right_Circular_Matrix_Bottom_Half(input_par, grid, wigner_3j_dict)
    VGRCM_Top_Half.axpy(1.0, VGRCM_Bottom_Half, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    return VGRCM_Top_Half

def Velocity_Gauge_Left_Circular_Matrix_Top_Half(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h = grid.spacing

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1
            
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            m_prime = m_block - 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
            wigner_term = -2.0*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)] 

            Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))
            
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    Int_Hamiltonian.scale(-1.0j)
    return Int_Hamiltonian

def Velocity_Gauge_Left_Circular_Matrix_Bottom_Half(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h = grid.spacing

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block > 0:
            l_prime = l_block - 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            if -1*m_block < l_prime:
                m_prime = m_block - 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,1,m_prime))
                wigner_term = -2.0*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,1,m_prime)] 

                Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
                if grid_idx < grid.size - 1:
                    Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
                if grid_idx >=  1:
                    Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))
                
           
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    Int_Hamiltonian.scale(-1.0j)
    return Int_Hamiltonian

def Velocity_Gauge_Right_Circular_Matrix_Top_Half(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h = grid.spacing

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 

        if l_block < input_par["l_max"]:
            l_prime = l_block + 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]

            m_prime = m_block + 1
            column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
            
            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
            wigner_term = 2.0*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)]

            Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
            if grid_idx < grid.size - 1:
                Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
            if grid_idx >=  1:
                Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))

    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    Int_Hamiltonian.scale(-1.0j)
    return Int_Hamiltonian

def Velocity_Gauge_Right_Circular_Matrix_Bottom_Half(input_par, grid, wigner_3j_dict):

    block_to_qn, qn_to_block = Mod.Index_Map(input_par)
    matrix_size = grid.size * len(block_to_qn)
    h = grid.spacing

    Int_Hamiltonian = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=3, comm=PETSc.COMM_WORLD)
    istart, iend = Int_Hamiltonian.getOwnershipRange() 
    for i in range(istart, iend):
        l_block = block_to_qn[floor(i/grid.size)][0]
        m_block = block_to_qn[floor(i/grid.size)][1]
        grid_idx = i % grid.size 
        
        if l_block > 0:
            l_prime = l_block - 1

            wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,0,0,0))
            common_factor = pow(-1.0, m_block)*sqrt((2.0*l_block+1.0)*(2.0*l_prime+1.0)/2.0)*wigner_3j_dict[(l_block,1,l_prime,0,0,0)]
                
            if m_block < l_prime:
                m_prime = m_block + 1
                column_idx = grid.size*qn_to_block[(l_prime, m_prime)] + grid_idx
                
                wigner_3j_dict = Check_Dictionary_Key_For_W3J_Symbol(wigner_3j_dict, (l_block,1,l_prime,-1*m_block,-1,m_prime))
                wigner_term = 2.0*wigner_3j_dict[(l_block,1,l_prime,-1*m_block,-1,m_prime)]

                Int_Hamiltonian.setValue(i, column_idx, common_factor * wigner_term * (l_prime*(l_prime+1.0) - l_block*(l_block+1.0)) /(2.0*grid.grid[grid_idx]))
                if grid_idx < grid.size - 1:
                    Int_Hamiltonian.setValue(i, column_idx + 1, common_factor * wigner_term/(2.0*h))
                if grid_idx >=  1:
                    Int_Hamiltonian.setValue(i, column_idx - 1, -1.0*common_factor * wigner_term /(2.0*h))

           
    Int_Hamiltonian.assemblyBegin()
    Int_Hamiltonian.assemblyEnd()
    Int_Hamiltonian.scale(-1.0j)
    return Int_Hamiltonian