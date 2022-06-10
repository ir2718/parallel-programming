from numba import cuda
import numpy as np

@cuda.jit
def kernel_jacobi(psi_new, psi, m, n):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    i += 1
    j += 1

    if i > m:
        return

    if j > n:
        return
    
    psi_new[i*(m+2) + j] = 0.25 * (
        psi[(i-1) * (m+2) + j] +
        psi[(i+1) * (m+2) + j] +
        psi[i * (m+2) + j - 1] + 
        psi[i * (m+2) + j + 1]
    )

@cuda.jit
def kernel_jacobi2(psi_new, psi, m, n):
    i = cuda.grid(1)
    i += 1

    if i > m:
        return

    print(i)
    for j in range(1, n+1):
        psi_new[i*(m+2) + j] = 0.25 * (
            psi[(i-1) * (m+2) + j] +
            psi[(i+1) * (m+2) + j] +
            psi[i * (m+2) + j - 1] + 
            psi[i * (m+2) + j + 1]
        )

@cuda.jit
def kernel_delta_sq(new_arr, old_arr, m, n, dsq):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    i += 1
    j += 1

    if i > m:
        return

    if j > n:
        return
    
    tmp = new_arr[i * (m+2) + j] - old_arr[i * (m+2) + j]
    cuda.atomic.add(dsq, 0, tmp**2)

    
@cuda.jit
def kernel_delta_sq2(new_arr, old_arr, m, n, dsq):
    i = cuda.grid(1)
    i += 1

    if i > m:
        return

    for j in range(1, n+1):
        tmp = new_arr[i * (m+2) + j] - old_arr[i * (m+2) + j]
        cuda.atomic.add(dsq, 0, tmp**2)

def jacobi_step(psi_new, psi, m, n, blocks_per_grid, threads_per_block):
    kernel_jacobi[blocks_per_grid, threads_per_block](psi_new, psi, m, n)

    # for i in range(1, m+1):
    #     for j in range(1, n+1):
    #         psi_new[i*(m+2) + j] = 0.25 * (
    #             psi[(i-1) * (m+2) + j] +
    #             psi[(i+1) * (m+2) + j] +
    #             psi[i * (m+2) + j - 1] + 
    #             psi[i * (m+2) + j + 1]
    #         )

def delta_sq(new_arr, old_arr, m, n, blocks_per_grid, threads_per_block):
    dsq = np.array([0.0])
    kernel_delta_sq[blocks_per_grid, threads_per_block](new_arr, old_arr, m, n, dsq)
    return dsq[0]

    # final = 0.0
    # for i in range(1, m+1):
    #     for j in range(1, n+1):
    #         tmp = new_arr[i * (m+2) + j] - old_arr[i * (m+2) + j]
    #         final += tmp**2
    # return final