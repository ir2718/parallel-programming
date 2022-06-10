import numpy as np
from time import time
from boundary import boundary_psi 
from jacobi import jacobi_step, delta_sq
from numba import cuda
import warnings
warnings.filterwarnings('ignore')

@cuda.jit
def copy_kernel(psi_tmp, psi, m, n):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    i += 1
    j += 1

    if i > m:
        return

    if i > n:
        return
    
    psi[i * (m + 2) + j] = psi_tmp[i * (m + 2) + j]

def main():
    scale_factor = 64
    num_iter = 100

    ###################################################################

    print_freq = 100
    tolerance = 0.0

    bbase, hbase, wbase, mbase, nbase = 10, 15, 5, 32, 32
    check_err = 0

    if tolerance > 0:
        check_err = 1

    if not check_err:
        print(f'scale factor = {scale_factor}, iterations = {num_iter}\n')
    else:
        print(f'scale factor = {scale_factor}, iterations = {num_iter}, tolerance = {tolerance}')

    print('irrotational flow\n')
    b = bbase * scale_factor
    h = hbase * scale_factor
    w = wbase * scale_factor
    m = mbase * scale_factor
    n = nbase * scale_factor
    
    ##########################################
    threads_per_block = 1024
    blocks_per_grid = (int(m/threads_per_block) + 1, n)
    ##########################################

    print(f'running CFD on {m} x {n} grid\n')

    ##### calculating CFD #####
    psi, psi_tmp = np.zeros((m+2)*(n+2)), np.zeros((m+2)*(n+2))

    boundary_psi(psi, m, n, b, h, w, blocks_per_grid, threads_per_block)

    b_norm = (np.sum(psi**2))**0.5

    print(f'starting main loop . . .\n\n')
    start = time()

    for i in range(1, num_iter+1):
        jacobi_step(psi_tmp, psi, m, n, blocks_per_grid, threads_per_block)

        if check_err or i == num_iter or True:
            error = delta_sq(psi_tmp, psi, m, n, blocks_per_grid, threads_per_block)**0.5/b_norm
            
        if check_err:
            if error < tolerance:
                print(f'converged on iteration {i}\n')
                break

        copy_kernel[blocks_per_grid, threads_per_block](psi_tmp, psi, m, n)

        if i % print_freq == 0:
            if not check_err:
                print(f'completed iteration {i}\n')
            else:
                print(f'completed iteration {i}, error = {error}')
        
        if i > num_iter:
            i = num_iter

    end = time()
    total = end-start
    iter_time = total/i

    print()
    print(f'after {i} iterations, the error is: {error}\n')
    print(f'time for {i} iterations was: {total} seconds\n')
    print(f'each iteration took: {iter_time} seconds\n')

main()