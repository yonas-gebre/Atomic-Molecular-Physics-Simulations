from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def DCIS_Check(input_par):
    if input_par["l_max"] != input_par["l_max_bs"]:
        if rank == 0:   
            print("l_max and l_max_bound_state have to be the same for double center calculations")
            exit()
            ##Alternative code for this case 
            ##high_idx = min(input_par["l_max"], input_par["l_max_bound_state"])*grid.size