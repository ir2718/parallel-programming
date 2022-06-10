from numba import cuda

@cuda.jit
def kernel1(psi, b, w, m):
    i = cuda.grid(1)

    if i < b + 1 or i > b + w - 1:
        return

    psi[i*(m+2)] = i - b


@cuda.jit
def kernel2(psi, b, w, m):
    i = cuda.grid(1)

    if i < b + w or i > m:
        return
    
    psi[i*(m+2)] = w

@cuda.jit
def kernel3(psi, h, w, m):
    j = cuda.grid(1)

    if j < 1 or j > h:
        return
    
    psi[(m+1)*(m+2)+j] = w

@cuda.jit
def kernel4(psi, h, w, m):
    j = cuda.grid(1)

    if j < h + 1 or j > h + w - 1:
        return
    
    psi[(m+1)*(m+2)+j] = w-j+h

def boundary_psi(psi, m, n, b, h, w, blocks_per_grid, threads_per_block):
    kernel1[blocks_per_grid, threads_per_block](psi, b, w, m)
    kernel2[blocks_per_grid, threads_per_block](psi, b, w, m)
    kernel3[blocks_per_grid, threads_per_block](psi, h, w, m)
    kernel4[blocks_per_grid, threads_per_block](psi, h, w, m)

    # for i in range(b+1, b+w):
    #     psi[i*(m+2)] = i - b
    # for i in range(b+w, m+1):
    #     psi[i*(m+2)] = w
    # for j in range(1, h+1):
    #     psi[(m+1)*(m+2)+j] = w
    # for j in range(h+1, h+w):
    #     psi[(m+1)*(m+2)+j] = w-j+h