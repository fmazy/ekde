# distutils: language = c++

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const void *, const void *)) nogil

from libcpp cimport bool
from libc.stdlib cimport malloc, free
import cython

from libc.math cimport exp
from libc.math cimport pow as cpow

from tqdm import tqdm

import numpy as np
cimport numpy as np

from cpython cimport array
import array     

from time import time

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Deactivate zero division checking.
cdef int compare_1d(const void *a, const void *b):
    cdef Py_ssize_t j
    
    cdef int *ai = (<int**>a)[0]
    cdef int *bi = (<int**>b)[0]
    
    if ai[0] < bi[0]:
        return -1
    if ai[0] > bi[0]:
        return +1
    
    return 0

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void np_quicksort(int[:] a):
    cdef Py_ssize_t i
    
    cdef int n = a.shape[0]
    cdef int **b = <int **> malloc(n * sizeof(int *))
    
    for i in range(n):
        b[i] = <int *> malloc(2 * sizeof(int))
        b[i][0] = a[i]
        b[i][1] = i
    
    cdef double st = time()
    qsort(b, n, sizeof(int*), compare_1d)
    
    print(time()-st)
    
    for i in range(n):
        a[i] = b[i][0]
        free(b[i])
    free(b)
    

# This code is contributed by Mohit Kumra
#This code in improved by https://github.com/anushkrishnav


@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void count_diff_desc(int[:,:] A,
               int[:,:] out):
    cdef Py_ssize_t j, i_asc, i
    
    cdef int n = A.shape[0]
    cdef int d = A.shape[1]
    
    for i_asc in range(n - 1):
        i = n - 2 - i_asc
        for j in range(d):
            if A[i,j] == A[i+1, j]:
                if j == 0:
                    out[i,j] = out[i+1,j] + 1
                elif out[i,j-1] > 1:
                    out[i,j] = out[i+1,j] + 1  

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void count_diff_asc(int[:,:] A,
                          int[:,:] out):
    cdef Py_ssize_t j, i
    
    cdef int n = A.shape[0]
    cdef int d = A.shape[1]
    
    for i in range(1, n):
        for j in range(d):
            if A[i,j] == A[i-1, j]:
                if j == 0:
                    out[i,j] = out[i-1,j] + 1
                elif out[i,j-1] > 1:
                    out[i,j] = out[i-1,j] + 1


@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int *** sparse(int[:,:] U,
                    int[:,:] U_diff_desc,
                    int[:,:] U_diff_one_side,
                    int *S_shape):
    cdef Py_ssize_t i_U, i_S, j, j_asc
    
    cdef int n = U.shape[0]
    cdef int d = U.shape[1]
        
    cdef int ***S = <int ***> malloc(d *sizeof(int **))
    
    for j in range(d):
        S[j] = <int **> malloc(S_shape[j] * sizeof(int *))
    
    for i_U in range(n):
        S[d-1][i_U] = <int *> malloc(2 * sizeof(int))
        S[d-1][i_U][0] = U[i_U, d-1]
        S[d-1][i_U][1] = i_U
    
    for j in range(d-1):
        S[j][0] = <int *> malloc(2 * sizeof(int))
        S[j][0][0] = U[0, j]
        S[j][0][1] = 0
    
    for j in range(d-1):
        i_U = 0
        for i_S in range(1, S_shape[j]):
            S[j][i_S] = <int *> malloc(2 * sizeof(int))
            
            S[j][i_S][1] = S[j][i_S - 1][1] + U_diff_one_side[i_U, j]
            
            i_U = i_U + U_diff_desc[i_U, j]
            
            S[j][i_S][0] = U[i_U, j]                
    
    return(S)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int [:,:] count_one_side(int[:,:] U_diff_desc):
    cdef Py_ssize_t i, j, i_asc
    
    cdef int n = U_diff_desc.shape[0]
    cdef int d = U_diff_desc.shape[1]
    cdef int [:,:] U_diff_one_side = np.zeros((n, d-1), dtype=np.intc)
    
    cdef int cnt
    
    for j in range(d-1):
        cnt = 0
        for i_asc in range(n):
            i = n - 1 - i_asc
            if U_diff_desc[i, j] == 1:
                cnt = 0
                
            if U_diff_desc[i, j+1] == 1:
                cnt = cnt + 1
            
            U_diff_one_side[i, j] = cnt
    
    return(U_diff_one_side)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef bool search_first_U(int ***S, 
                         int *S_shape,
                         int[:,:] U_diff_asc,
                         int * s,
                         int d,
                         int *target,
                         int j_start):
    cdef Py_ssize_t j
    
    cdef int a
    cdef int b
    
    for j in range(j_start, d):
        if j == 0:
            a = 0
            b = S_shape[0]
        
        else:
            a = S[j-1][s[j-1]][1]
            if s[j-1] + 1 < S_shape[j-1]:
                b = S[j-1][s[j-1] + 1][1]
            else:
                b = S_shape[j]
            
        s[j] = binary_search(L = S[j],
                             x = target[j],
                             a = a,
                             b = b)
        
        if s[j] == -1:
            return(next_s(S = S, 
                          S_shape=S_shape,
                          U_diff_asc=U_diff_asc,
                          d = d,
                          s = s,
                          j_max = j-1))
    
    return(True)

# @cython.boundscheck(False)  # Deactivate bounds checking.
# @cython.wraparound(False)   # Deactivate negative indexing.
# cdef bool search_next_U(int ***S, 
#                         int *S_shape,
#                         int[:,:] U_diff_asc,
#                         int * s,
#                         int d,
#                         int *target,
#                         int margin,
#                         int[:,:] Z,
#                         int i_Z,
#                         int j):
    
#     cdef int a
#     cdef int b
    
#     if j == 0:
#         a = 0
#         b = S_shape[0]
    
#     else:
#         a = S[j-1][s[j-1]][1]
#         if s[j-1] + 1 < S_shape[j-1]:
#             b = S[j-1][s[j-1] + 1][1]
#         else:
#             b = S_shape[j]
        
#     s[j] = binary_search(L = S[j],
#                          x = target[j],
#                          a = a,
#                          b = b)
    
#     # if no element found after binary search
#     # increment the column before
#     if s[j] == -1:
#         next_s(S = S, 
#                S_shape=S_shape,
#                U_diff_asc=U_diff_asc,
#                d = d,
#                s = s,
#                j_max = j-1)
#         return()
    
#     if S[j][s[j]][0] - margin > Z[i_Z, j]:
#         # U is too high
#         # the column before is incremented
#         return(next_s(S=S, 
#                       S_shape=S_shape,
#                       U_diff_asc=U_diff_asc,
#                       d=d,
#                       s=s,
#                       j_max=j_max))
    
#     elif S[j][s[j]][0] + margin < Z[i_Z, j]:
#         # U is too low
#         # let's search above
#         # if nothing is found, the column before is incremented
#         pass
    
#     return(True)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int binary_search_left(int **L,
                            int x,
                            int a,
                            int b):
    cdef int m
    
    if x < L[a][0]:
        return(a)
    
    if x > L[b-1][0]:
        return(b-1)
    
    while a < b:
        m = <int> (a + b) / 2
        
        if L[m][0] < x:
            a = m + 1
        else:
            b = m
    
    return(a)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int binary_search_right(int **L,
                             int x,
                             int a,
                             int b):
    cdef int m
    
    if x < L[a][0]:
        return(a)
    
    if x > L[b-1][0]:
        return(b-1)
    
    while a < b:
        m = <int> (a + b) / 2
        
        if L[m][0] > x:
            b = m
        else:
            a = m +1
    
    return(b - 1)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int binary_search(int **L,
                       int x,
                       int a,
                       int b):
    cdef int m
    
    if x < L[a][0]:
        return(a)
    
    if x > L[b-1][0]:
        return(-1)
    
    while a < b:
        m = <int> (a + b) / 2
        
        if L[m][0] < x:
            a = m + 1
        else:
            b = m
    
    return(a)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef bool next_s(int ***S, 
                 int *S_shape,
                 int [:,:] U_diff_asc,
                 int d,
                 int *s,
                 int j_max):
    cdef Py_ssize_t j    
    cdef int i_U = s[j_max] + 1
    
    if j_max < 0:
        return(False)
    
    if i_U >= S_shape[j_max]:
        s[j_max] = -1
        return(False)
        
    for j in range(j_max, d):
        i_U = S[j][i_U][1]
    
    
    for j in range(j_max + 1):
        if U_diff_asc[i_U, j] == 1:
            s[j] = s[j] + 1
    
    for j in range(j_max + 1, d):
        s[j] = S[j-1][s[j-1]][1]
    
    return(True)
        
@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int * get_S_shape(int[:,:] U_diff_desc, 
                       int n_U,
                       int d):
    cdef Py_ssize_t j
    
    cdef int *S_shape = <int *> malloc(d * sizeof(int))
    
    for j in range(d):
        S_shape[j] = 0
        
        for i_U in range(n_U):
            if U_diff_desc[i_U, j] == 1:
                S_shape[j] = S_shape[j] + 1
    return(S_shape)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double [:] set_estimation(int[:,:] Z_diff_asc,
                                int[:] Z_indices,
                                double[:] g):
    cdef Py_ssize_t i_Z, i_g
    cdef int n_Z = Z_diff_asc.shape[0]
    cdef int d = Z_diff_asc.shape[1]
    
    cdef double [:] f = np.zeros(n_Z, dtype=np.double)
    
    i_g = 0
    for i_Z in range(n_Z):
        if Z_diff_asc[i_Z, d-1] == 1:
            f[Z_indices[i_Z]] = g[i_g]
            i_g = i_g + 1
        else:
            f[Z_indices[i_Z]] = g[i_g]
    
    return(f)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division 
@cython.cdivision(True) # modulo operator (%)
cpdef double [:] merge2(int[:, :] U,
                        double[:] nu,
                        int[:,:] Z,
                        int margin,
                        int kernel_id,
                        double dx,
                        int verbose=0):
    cdef Py_ssize_t j, i, i_repeat, k, q
    cdef int n_U = U.shape[0]
    cdef int d = U.shape[1]
    cdef int n_Z = Z.shape[0]
    
    cdef int ***A = sort_columns(U)
    cdef int ***B = sort_columns(Z)
    
    cdef int **R = count_diff_desc_in_sorted_columns(B, n_Z, d)
    
    cdef double [:] f = np.zeros(n_Z, dtype=np.double)
    
    
    
    cdef int low, high
    
    cdef int max_R = 0
    
    cdef int ***low_high = <int ***> malloc(d * sizeof(int **))
    for j in range(d):
        low_high[j] = <int **> malloc(n_Z * sizeof(int *))
        
        i = 0
        while i < n_Z:
        # for i in range(n_Z):
            low = binary_search_left(A[j], B[j][i][0] - margin, 0, n_U)
            high = binary_search_right(A[j], B[j][i][0] + margin, low, n_U)
            
            if R[j][i] > max_R:
                max_R = R[j][i]
            
            for i_repeat in range(R[j][i]):
                low_high[j][i + i_repeat] = <int *> malloc(2 * sizeof(int))
                low_high[j][i + i_repeat][0] = low
                low_high[j][i + i_repeat][1] = high
            
            i = i + R[j][i]
    
    cdef int ***inv_B = <int ***> malloc(d * sizeof(int **))
    for j in range(d):
        inv_B[j] = <int **> malloc(n_Z * sizeof(int *))
        for i in range(n_Z):
            inv_B[j][i] = <int *> malloc(2 * sizeof(int))
            inv_B[j][i][0] = B[j][i][1]
            inv_B[j][i][1] = i
        
        qsort(inv_B[j], n_Z, sizeof(int*), compare_1d)
    
    # cdef int **I = <int **> malloc(d * n_U * sizeof(int *))
    # for i in range(d * n_Z):
    #     I[i] = <int *> malloc(2 * sizeof(int))
        
    # cdef int s
    # for i in range(n_Z):
    #     s = 0
    #     q = 0
    #     for j in range(d):
    #         low = low_high[j][inv_B[j][i][1]][0]
    #         high = low_high[j][inv_B[j][i][1]][1]
            
    #         s = s + high - low
            
    #         for k in range(low, high + 1):
    #             I[q][0] = A[j][k][1]
    #             I[q][1] = q
                
    #             q = q + 1
        
        # qsort(I, s, sizeof(int*), compare_1d)
    
    for j in range(d):
        for i in range(n_U):
            free(A[j][i])
        free(A[j])
        
        for i in range(n_Z):
            free(B[j][i])
            free(inv_B[j][i])
            free(low_high[j][i])
        free(B[j])
        free(inv_B[j])
        free(R[j])
        free(low_high[j])
    free(A)
    free(B)
    free(inv_B)
    free(R)
    free(low_high)
    
    # for i in range(d * n_U):
        # free(I[i])
    # free(I)
    
    return(f)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int*** sort_columns(int[:,:] X):
    cdef int ***A
    
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    
    cdef Py_ssize_t j, i
    
    A = <int ***> malloc(d * sizeof(int **))
    
    for j in range(d):
        A[j] = <int **> malloc(n * sizeof(int *))
        
        for i in range(n):
            A[j][i] = <int *> malloc(2 * sizeof(int))
            A[j][i][0] = X[i, j]
            A[j][i][1] = i
        
        qsort(A[j], n, sizeof(int*), compare_1d)
        
        
    
    return(A)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int** count_diff_desc_in_sorted_columns(int*** A,
                                             int n,
                                             int d):
    cdef int **R
    
    cdef Py_ssize_t j, i, i_asc
    
    R = <int **> malloc(d * sizeof(int *))
    for j in range(d):
        R[j] = <int *> malloc(n * sizeof(int))
        
        R[j][n-1] = 1
        
        for i_asc in range(n - 1):
            i = n - 2 - i_asc
            if A[j][i][0] == A[j][i+1][0]:
                R[j][i] = R[j][i+1] + 1
            else:
                R[j][i] = 1
    return(R)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division 
@cython.cdivision(True) # modulo operator (%)
cpdef double [:] merge(int[:, :] U, 
                       int[:, :] U_diff_asc,
                       int[:, :] U_diff_desc,
                       double[:] nu, 
                       int[:, :] Z,
                       int q,
                       double h,
                       int kernel_id,
                       double dx,
                       int verbose=0):
    cdef Py_ssize_t i_U, i_Z, j, k, i_T
    
    cdef int n_U = U.shape[0]
    cdef int d = U.shape[1]
    cdef int n_Z = Z.shape[0]
    
    cdef int margin = (q - 1) / 2
    
    cdef double [:] f = np.zeros(n_Z, dtype=np.double)
                
    cdef int *S_shape = get_S_shape(U_diff_desc,
                                    n_U,
                                    d)
    
    cdef int [:,:] U_diff_one_side = count_one_side(U_diff_desc=U_diff_desc)
    
    cdef int ***S = sparse(U=U,
                           U_diff_desc=U_diff_desc,
                           U_diff_one_side = U_diff_one_side,
                           S_shape=S_shape)
    
    if verbose > 0:
        pbar = tqdm(total=n_Z)
    
    cdef int[:,:] Z_diff_desc = np.ones((n_Z, d), dtype=np.intc)
    count_diff_desc(Z, Z_diff_desc)
    
    cdef int *T_shape = get_S_shape(Z_diff_desc,
                                    n_Z,
                                    d)
    
    cdef int [:,:] Z_diff_one_side = count_one_side(U_diff_desc=Z_diff_desc)
    
    cdef int ***T = sparse(U=Z,
                           U_diff_desc=Z_diff_desc,
                           U_diff_one_side = Z_diff_one_side,
                           S_shape=T_shape)
    
    cdef int low_S, high_S
    
    low_S = 0
    high_S = S_shape[0]
    for i_T in range(0, T_shape[0]):
        low_S = explore2(S=S,
                         S_shape=S_shape,
                         nu = nu,
                         d=d,
                         T = T,
                         T_shape=T_shape,
                         f=f,
                         margin=margin,
                         j=0,
                         low_S=0,
                         high_S=high_S,
                         i_T=i_T)
    
    # for i_Z in range(0, n_Z):
    #     if verbose > 0:
    #         if i_Z % (n_Z / 100) == 0:
    #             pbar.update(<int> n_Z / 100)
                
    #     f[i_Z] = estimate4(S, 
    #                         S_shape,
    #                         nu=nu,
    #                         Z=Z, 
    #                         i_Z=i_Z, 
    #                         margin=margin,
    #                         d=d,
    #                         h=h,
    #                         dx=dx,
    #                         kernel_id=kernel_id)
    
    for j in range(d):
        for i_U in range(S_shape[j]):
            free(S[j][i_U])
        free(S[j])
    free(S)
    
    free(S_shape)
    
    if verbose > 0:
        pbar.close()
    
    return(f)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division 
cdef int explore2(int ***S, 
                   int *S_shape, 
                   double[:] nu,
                   int d,
                   int ***T,
                   int *T_shape,
                   double[:] f,
                   int margin,
                   int j,
                   int low_S,
                   int high_S,
                   int i_T):
    cdef Py_ssize_t i_S
    
    cdef int z = T[j][i_T][0]
    cdef int low_T = T[j][i_T][1]
    cdef int high_T = get_high(T[j], T_shape, d, j, i_T)
    cdef int i_Z
    cdef int a, b
    
    a = binary_search_left(L = S[j],
                           x = z - margin,
                           a = low_S,
                           b = high_S)
    b = binary_search_right(L = S[j],
                            x = z + margin,
                            a = a,
                            b = high_S)
    
    cdef int up_low_S = a
    
    if j < d-1:
        for i_S in range(a, b):
            low_S = S[j][i_S][1]
            high_S = get_high(S[j], S_shape, d, j, i_S)
            for i_T in range(low_T, high_T):
                low_S = explore2(S=S,
                                 S_shape=S_shape,
                                 nu = nu,
                                 d=d,
                                 T=T,
                                 T_shape=T_shape,
                                 f=f,
                                 margin=margin,
                                 j=j+1,
                                 low_S=low_S,
                                 high_S=high_S,
                                 i_T = i_T)
    else:
        low_S = S[j][a][1]
        high_S = get_high(S[j], S_shape, d, j, b)
        
        i_Z = T[j][i_T][1]
        
        for i_S in range(low_S, high_S):
            f[i_Z] = f[i_Z] + nu[i_S]
    
    return(up_low_S)
        

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division 
cdef double explore(int ***S, 
                    int *S_shape, 
                    double[:] nu,
                    int d,
                    int[:,:] Z, 
                    int i_Z, 
                    int margin,
                    int j,
                    int low,
                    int high,
                    double est):
    
    cdef Py_ssize_t i
    cdef int a, b
    
    a = binary_search_left(L = S[j],
                           x = Z[i_Z, j] - margin,
                           a = low,
                           b = high)
    b = binary_search_right(L = S[j],
                            x = Z[i_Z, j] + margin,
                            a = a,
                            b = high)
    
    if j == d-1:
        low = S[j][a][1]
        high = get_high(S[j], S_shape, d, j, b)
        # if b == S_shape[j] - 1:
        #     high = S_shape[j+1]
        # else:
        #     high = S[j][b+1][1]
        
        for i in range(low, high):
            est = est + nu[i]
            
        return(est)
    else:
        for i in range(a, b):
            low = S[j][i][1]
            high = get_high(S[j], S_shape, d, j, i)
            # if i == S_shape[j] - 1:
                # high = S_shape[j+1]
            # else:
                # high = S[j][i+1][1]
            
            est = explore(S=S,
                          S_shape=S_shape,
                          nu = nu,
                          d=d,
                          Z=Z,
                          i_Z=i_Z,
                          margin=margin,
                          j=j+1,
                          low=low,
                          high=high,
                          est = est)
    
    return(est)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef get_high(int** S_j,
              int* S_shape,
              int d,
              int j,
              int i):
    
    if i == S_shape[j] - 1:
        if j == d - 1:
            return(S_j[i][1] + 1)
        else:
            return(S_shape[j+1])
    else:
        return(S_j[i+1][1])

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double estimate4(int ***S, 
              int *S_shape,
              double[:] nu,
              int[:,:] Z, 
              int i_Z, 
              int margin,
              int d,
              double h,
              double dx,
              int kernel_id):
    
    cdef Py_ssize_t j
    cdef double est
    
    # if Z[i_Z, 0] == 150 and Z[i_Z, 1] == 361:
    #     print('---')
    est = explore(S=S,
                  S_shape=S_shape,
                  nu = nu,
                  d=d,
                  Z=Z,
                  i_Z=i_Z,
                  margin=margin,
                  j=0,
                  low=0,
                  high=S_shape[0],
                  est = 0.0)
    
    return(est)

# @cython.boundscheck(False)  # Deactivate bounds checking.
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.cdivision(True)  # non check division 
# cdef double estimate3(
#               int[:,:] U, 
#               double[:] nu,
#               int[:,:] Z, 
#               int i_Z, 
#               int margin,
#               int d,
#               double h,
#               double dx,
#               int kernel_id):
    
#     cdef int a, b
#     cdef int j = 0
#     cdef int n_u = U.shape[0]
    
#     cdef Py_ssize_t i
#     for i in range(d):
#         a = binary_search_left(U, Z[i_Z, j] - margin, 0, U.shape[0], j)
#         b = binary_search_right(U, Z[i_Z, j] + margin, a, U.shape[0], j)
    
#     return(0.0)


# @cython.boundscheck(False)  # Deactivate bounds checking.
# @cython.wraparound(False)   # Deactivate negative indexing.
# @cython.cdivision(True)  # non check division 
# cdef double estimate2(int ***S, 
#               int *S_shape, 
#               int[:,:] U_diff_asc, 
#               double[:] nu,
#               int[:,:] Z, 
#               int i_Z, 
#               int margin,
#               int d,
#               double h,
#               double dx,
#               int kernel_id):
#     cdef Py_ssize_t j
#     cdef double dist_sq, est, contrib
    
#     cdef bool trigger_search_U
#     cdef int *target = <int *> malloc(d * sizeof(int))
#     cdef int *s = <int *> malloc(d * sizeof(int))
    
#     est = 0.0
    
#     for j in range(d):
#         target[j] = Z[i_Z, j] - margin
    
#     trigger_search_U =  search_first_U(S = S,
#                                        S_shape = S_shape,
#                                        U_diff_asc = U_diff_asc,
#                                        s = s,
#                                        d = d,
#                                        target = target,
#                                        j_start=0)
    
#     while search_next_U(S=S,
#                         S_shape = S_shape,
#                         U_diff_asc = U_diff_asc,
#                         s = s,
#                         d = d,
#                         target = target,
#                         margin = margin,
#                         Z = Z,
#                         i_Z = i_Z):
#         # U is good for Z !
#         # here it is possible to set another type of kernel
#         contrib = nu[S[d-1][s[d-1]][1]]
        
#         if kernel_id == 0:
#             pass
            
#         elif kernel_id == 1:
            
#             dist_sq = 0
#             for j in range(d):
#                 dist_sq = dist_sq + cpow(Z[i_Z, j] - S[j][s[j]][0], 2.0)
#             dist_sq = dist_sq * cpow(dx, 2.0)
            
#             contrib = contrib * exp(-dist_sq / cpow(h,2.0) / 2)
        
#         est = est + contrib
    
#     free(s)
#     free(target)
#     return(est)

@cython.boundscheck(False)  # Deactivate bounds checking.
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # non check division 
cdef double estimate(int ***S, 
              int *S_shape, 
              int[:,:] U_diff_asc, 
              double[:] nu,
              int[:,:] Z, 
              int i_Z, 
              int margin,
              int d,
              double h,
              double dx,
              int kernel_id):
    cdef Py_ssize_t j
    cdef double dist_sq, est, contrib
    
    cdef bool trigger_search_U
    cdef int *target = <int *> malloc(d * sizeof(int))
    cdef int *s = <int *> malloc(d * sizeof(int))
    
    est = 0.0
    
    for j in range(d):
        target[j] = Z[i_Z, j] - margin
    
    trigger_search_U =  search_first_U(S = S,
                                       S_shape = S_shape,
                                       U_diff_asc = U_diff_asc,
                                       s = s,
                                       d = d,
                                       target = target,
                                       j_start=0)
    
    while trigger_search_U:
        for j in range(d):
            if S[j][s[j]][0] - margin > Z[i_Z, j]:
                # U is too high
                # the column before is incremented
                trigger_search_U = next_s(S = S, 
                                          S_shape=S_shape,
                                          U_diff_asc=U_diff_asc,
                                          d = d,
                                          s = s,
                                          j_max = j-1)
                break
            
            elif S[j][s[j]][0] + margin < Z[i_Z, j]:
                # U is too low
                # let's search above
                # if nothing is found, the column before is incremented
                trigger_search_U = search_first_U(S = S,
                                                   S_shape = S_shape,
                                                   U_diff_asc = U_diff_asc,
                                                   s = s,
                                                   d = d,
                                                   target = target,
                                                   j_start=j)
                break
        else:
            # U is good for Z !
            # here it is possible to set another type of kernel
            contrib = nu[S[d-1][s[d-1]][1]]
            
            if kernel_id == 0:
                pass
                
            elif kernel_id == 1:
                
                dist_sq = 0
                for j in range(d):
                    dist_sq = dist_sq + cpow(Z[i_Z, j] - S[j][s[j]][0], 2.0)
                dist_sq = dist_sq * cpow(dx, 2.0)
                
                contrib = contrib * exp(-dist_sq / cpow(h,2.0) / 2)
            
            est = est + contrib
                            
            # then, next U
            trigger_search_U = next_s(S = S, 
                                      S_shape=S_shape,
                                      U_diff_asc=U_diff_asc,
                                      d = d,
                                      s = s,
                                      j_max = d-1)
    
    free(s)
    free(target)
    return(est)