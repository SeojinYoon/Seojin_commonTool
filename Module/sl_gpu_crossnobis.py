
import numpy as np
from numba import cuda, jit
import cupy as cp
import itertools
from tqdm import trange

class ShrinkageMethod:
    shrinkage_eye = "shrinkage_eye"
    shrinkage_diag = "shrinkage_diag"
    
def convert_1d_to_symmertic(a_1d, size, k = 0):
    """
    Convert 1d array to symmetric matrix
    
    :param a_1d(1d array): 
    :param size: matrix size
    :param k(int): offset 
    
    return (np.array)
    """

    # put it back into a 2D symmetric array
    if k == 0:
        X = np.zeros((size,size))
        X[np.triu_indices(size, k = 0)] = a_1d
        X = X + X.T - np.diag(np.diag(X))
    else:
        X = np.zeros((size_X,size_X))
        X[np.triu_indices(size, k = 1)] = a_1d
        X = X + X.T

    return X

@jit(nopython=True)
def upper_tri_1d_index(i, j, n_col, k):
    """
    Get upper triangle 1d index
    
    if k = 1)
    
    (0,1), (0,2), (0,3), (0,4) -> 0, 1, 2, 3
           (1,2), (1,3), (1,4) -> 4, 5, 6
                  (2,3), (2,4) -> 7, 8
                         (3,3) -> 9
                         
    :param i: row index
    :param j: column index
    :param n_col: column number
    :param k: #padding
    """
    if i > j:
        return None
    else:
        sum_val = 0
        for loop_row_i in range(0, i):
            sum_val += (n_col - k) # maximum filled count of row.
            sum_val += (-1) * loop_row_i # non-filled element is increased as row value is increased.
        return sum_val + (j - i - k)
    
@jit(nopython=True)
def lower_tri_1d_index(i, j):
    """
    Get lower triangle 1d index
    
    :param i: row index
    :param j: column index
    """
    
    if i < j:
        return None
    else:        
        total_fill = 0
        for pr_row_i in range(1, i + 1):
            total_fill += (pr_row_i - 1)
        return total_fill + j
    
@cuda.jit
def set_mask(neighbors, brain_1d_indexes, out):
    """
    Set neighbor mask(iterate over all neighbors)

    :param neighbors(np.array): list of neighbor , shape: (#center, #neighbor)
    :param brain_1d_indexes(np.array): , shape: #channel
    :param out: masked_residual, output device memory , shape: (#center, #channel)
    """
    i = cuda.grid(1)

    if i < len(neighbors):
        neighbor_positions = neighbors[i]

        for neighbor_pos in neighbor_positions:
            for brain_i, brain_pos in enumerate(brain_1d_indexes):
                if brain_pos == neighbor_pos:
                    out[i][brain_i] = 1

@cuda.jit
def outer_sum(matrices, out):
    i = cuda.grid(1)

    if i < len(matrices):
        matrix = matrices[i]

        for m_line in matrix:
            for j, e1 in enumerate(m_line):
                for k, e2 in enumerate(m_line):
                    out[i][j][k] += e1 * e2

@cuda.jit
def outer_sum_square(matrices, out):
    i = cuda.grid(1)

    if i < len(matrices):
        matrix = matrices[i]

        for m_line in matrix:
            for j, e1 in enumerate(m_line):
                for k, e2 in enumerate(m_line):
                    out[i][j][k] += (e1 * e2) ** 2

@cuda.jit
def diag(matrices, out):
    i = cuda.grid(1)

    if i < len(matrices):
        matrix = matrices[i]

        n_row = len(matrix)
        for j in range(n_row):
            out[i][j] = matrix[j][j]

@cuda.jit
def eyes(out):
    i = cuda.grid(1)

    nd = out.shape[0]
    nr = out.shape[1]
    nc = out.shape[2]

    if i < len(out):
        for j in range(nr):
            out[i][j][j] = 1

@cuda.jit
def scaling(out, lambs):
    i = cuda.grid(1)
    lamb = lambs[i]

    nd = out.shape[0]
    nr = out.shape[1]
    nc = out.shape[2]

    if i < len(out):
        for j in range(nr):
            for k in range(nc):
                if j != k:
                    out[i][j][k] = (1 - lamb)


def mean_fold_variance(variances, fold_info):
    """
    Calculate fold variacne from fold info
    
    :param variances: variances (#data, #cov.shape)
    :param fold_info(2d array): fold information - [[fold1, fold2], ...]
    
    return (np.array) - (#data * fold_len, cov.shape)
    """
    n_d = len(variances)
    
    result_variances = []
    for i in range(n_d):       
        for fold1_i, fold2_i in fold_info:            
            cov1 = variances[i][fold1_i]
            cov2 = variances[i][fold2_i]
            
            result_variances.append((cov1 + cov2) / 2)
    
    return np.array(result_variances)

@cuda.jit(device=True, inline=True)
def matmul(a,b, out):
    """
    Matrix multiplication a @ b
    
    :param a(np.array): 2d matrix
    :param b(np.array): 2d matrix
    :param out(device array): output
    """
    ar,ac = a.shape 
    br,bc = b.shape 
    
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                out[i,j] += a[i,k] * b[k,j]
    return out

@cuda.jit
def rdm_from_kernel(kernels, div, out):
    """
    Calculate rdm matrix
    
    :param kernels(Device array): kernel, shape: (n_data, n_fold, n_cond, n_cond))
    :param div(int): div value
    :param out(Device array): rdm output, shape: (n_data, n_fold, n_dissim)
    """
    n_data = kernels.shape[0]
    n_validation = kernels.shape[1]
    n_cond = kernels.shape[-1]
    
    i, j = cuda.grid(2)
    
    if i < n_data:
        if j < n_validation:
            kernel = kernels[i][j]
            
            for row_i in range(n_cond):
                for column_i in range(n_cond):
                    if row_i < column_i:
                        dissim_i = int(upper_tri_1d_index(row_i, column_i, n_cond, 1))

                        # Assign dissim value
                        v1 = kernel[row_i][row_i] + kernel[column_i][column_i]
                        v2 = kernel[row_i][column_i] + kernel[column_i][row_i]
                        out[i][j][dissim_i] = (v1 - v2) / div
                    

@cuda.jit
def calc_kernel(measurments, precisions, fold_info, out1, out2):
    """
    Calculate rdm kernel for calculating crossnobis
    
    (2048, 4, 8, 93)
    
    :param measurments(Device array): , shape: (n_data, n_run, n_cond, n_neighbor)
    :param precisions(Device array): , shape: (n_data, n_fold, n_neighbor, n_neighbor)
    :param fold_info(Device array): fold information - [[fold1, fold2], ...]
    :param out1(Device array): intermediate matmul output , shape: (n_data, n_fold, n_cond, n_neighbor) 
    :param out2(Device array): kernel output , shape: (n_data, n_fold, n_cond, n_cond))
    """
    n_data = out1.shape[0]
    n_validation = out1.shape[1]

    i, j = cuda.grid(2)
    if i < n_data:
        if j < n_validation:
            data1_i, data2_i = fold_info[j]
            
            # measurements1 @ noise @ measurements2.T
            matmul(measurments[i][data1_i], precisions[i][j], out1[i][j])
            matmul(out1[i][j], measurments[i][data2_i].T, out2[i][j])

def _covariance_eye(residuals, threads_per_block = 1024):
    """
    Computes an optimal shrinkage estimate of a sample covariance matrix as described by the following publication:
    **matrix should be demeaned before!
    
    Ledoit and Wolfe (2004): "A well-conditioned estimator for large-dimensional covariance matrices"
    
    :param residuals(np.ndarray): , shape: (#data, #n_point, #n_channel)
    """
    print("shrinakge method:", ShrinkageMethod.shrinkage_eye)
    
    # Constant
    data_len = len(residuals)
    n_point = residuals.shape[1]
    n_channel = residuals.shape[2]
    
    n_block = int(np.ceil(data_len / threads_per_block))
    
    # sum
    out_sum_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))

    # sum square
    out_sum_square_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))
    
    # Calc sum, sum square
    outer_sum[n_block, threads_per_block](residuals, out_sum_device)
    outer_sum_square[n_block, threads_per_block](residuals, out_sum_square_device)

    # b2
    s = out_sum_device.copy_to_host() / n_point
    s2 = out_sum_square_device.copy_to_host() / n_point
    b2 = np.sum(s2 - s * s, axis = (1, 2)) / n_point

    # calculate the scalar estimators to find the optimal shrinkage:
    # m, d^2, b^2 as in Ledoit & Wolfe paper
    # m - shape: (data_len)
    # d2 - shape: (data_len)
    # b2 - shape: (data_len)
    repeat_eyes = np.repeat(np.eye(n_channel)[:, :, np.newaxis], data_len, axis = 2).T
    
    diag_s = np.diagonal(s, axis1 = 1, axis2 = 2)
    m = (np.sum(diag_s, axis = 1) / n_channel)
    d2 = np.sum((s - m[:, None, None] * repeat_eyes) ** 2, axis = (1, 2))
    
    b2 = np.minimum(d2, b2)
    
    # shrink covariance matrix
    s_shrink = (b2 / d2 * m)[:, None, None] * repeat_eyes + ((d2-b2) / d2)[:, None, None] * s
    
    # correction for degrees of freedom
    dof = n_point - 1
    s_shrink = s_shrink * n_point / dof
    
    return s_shrink

def _covariance_diag(residuals, threads_per_block = 1024):
    """
    Calculate covariance 
    **matrix should be demeaned before!
    
    Schäfer, J., & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale
    Covariance Matrix Estimation and Implications for Functional Genomics.
    
    :param residuals(np.ndarray): , shape: (#data, #n_point, #n_channel)
    """
    print("shrinakge method:", ShrinkageMethod.shrinkage_diag)
    
    # Constant
    data_len = len(residuals)
    n_point = residuals.shape[1]
    n_channel = residuals.shape[2]
    
    n_block = int(np.ceil(data_len / threads_per_block))

    # sum
    out_sum_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))

    # sum square
    out_sum_square_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))

    # Calc sum, sum square
    outer_sum[n_block, threads_per_block](residuals, out_sum_device)
    outer_sum_square[n_block, threads_per_block](residuals, out_sum_square_device)

    # s
    dof = n_point - 1
    s = out_sum_device.copy_to_host() / dof

    # var
    stack_var_device = cuda.to_device(np.zeros((data_len, n_channel)))
    diag[n_block, threads_per_block](s, stack_var_device)

    # std
    stack_std = np.sqrt(stack_var_device)

    # sum mean
    stack_s_mean = out_sum_device / np.expand_dims(stack_std, 1) / np.expand_dims(stack_std, 2) / (n_point - 1)

    # s2 mean
    stack_s2_mean = out_sum_square_device / np.expand_dims(stack_var_device, 1) / np.expand_dims(stack_var_device, 2) / (n_point - 1)

    # var_hat
    stack_var_hat = n_point / dof ** 2 * (stack_s2_mean - stack_s_mean ** 2)

    # mask
    mask = ~np.eye(n_channel, dtype=bool)

    # lamb
    stack_lamb_device = np.sum(stack_var_hat[:, mask], axis = 1) / np.sum(stack_s_mean[:, mask] ** 2, axis = 1)
    stack_lamb_device = cp.maximum(cp.minimum(cp.array(stack_lamb_device), 1), 0)

    # Scaling
    stack_scaling_mats_device = cuda.to_device(np.zeros((data_len, n_channel, n_channel)))
    eyes[n_block, threads_per_block](stack_scaling_mats_device)

    scaling[n_block, threads_per_block](stack_scaling_mats_device, stack_lamb_device)
    stack_s_shrink = s * stack_scaling_mats_device
    
    return stack_s_shrink            

def calc_sl_precision(residuals, 
                      neighbors, 
                      n_split_data, 
                      masking_indexes, 
                      n_thread_per_block = 1024,
                      shrinkage_method = "shrinkage_diag"):
    """
    Calculate precision
    
    :param residuals(np.ndarray):  , shape: (#run, #point, #channel)
    :param neighbors(np.ndarray): , shape: (#center, #neighbor)
    :param n_split_data(int): how many datas to process at once
    :param masking_indexes(np.array):  , shape: (#channel) / index of masking brain
    :param n_thread_per_block(int): block per thread
    
    return (np.ndarray), shape: (#channel, #run, #neighbor, #neighbor)
    """
    
    n_run = residuals.shape[0]
    n_p = residuals.shape[1]
    n_channel = residuals.shape[-1]
    
    n_center = len(neighbors)
    n_block = int(np.ceil(n_split_data / n_thread_per_block))
    n_neighbor = neighbors.shape[-1]
    r, c = np.triu_indices(n_neighbor, k = 0)
    
    mempool = cp.get_default_memory_pool()
    
    chunk_precisions = []
    for i in trange(0, n_center, n_split_data):
        # select neighbors
        target_neighbors = neighbors[i:i + n_split_data, :]
        len_target = len(target_neighbors)
        
        # output_1d
        mask_out = cuda.to_device(np.zeros((len_target, n_channel)))

        # Make mask - neighbor
        set_mask[n_block, n_thread_per_block](target_neighbors, masking_indexes, mask_out)
        
        # sync
        cuda.synchronize()

        # Apply mask
        cpu_mask = mask_out.copy_to_host()
        masked_residuals = []
        for j in range(len(target_neighbors)):
            masked_residuals.append(residuals[:, :, cpu_mask[j] == 1])
        masked_residuals = np.array(masked_residuals)

        del mask_out
        cuda.defer_cleanup()

        # Calculate demean
        target_residuals = masked_residuals.reshape(-1, n_p, n_neighbor)
        mean_residuals = np.mean(target_residuals, axis = 1, keepdims=1)
        target_residuals = (target_residuals - mean_residuals)

        # Calculate covariance
        if shrinkage_method == ShrinkageMethod.shrinkage_diag:
            covariances = _covariance_diag(target_residuals)
        elif shrinkage_method == ShrinkageMethod.shrinkage_eye:
            covariances = _covariance_eye(target_residuals)

        # Calculate precision matrix
        stack_precisions = cp.linalg.inv(cp.asarray(covariances)).get()
        
        # sync
        cuda.synchronize()
        
        # concat
        stack_precisions = stack_precisions.reshape(len_target, n_run, n_neighbor, n_neighbor)
        stack_precisions = stack_precisions[:, :, r, c]
    
        # add chunk
        chunk_precisions.append(stack_precisions)
        
        # Clean data
        cuda.defer_cleanup()
        mempool.free_all_blocks()
        
    return chunk_precisions

def calc_sl_rdm_crossnobis(n_split_data, 
                           centers, 
                           neighbors, 
                           precs,
                           measurements,
                           masking_indexes,
                           conds, 
                           sessions, 
                           n_thread_per_block = 1000):
    """
    Calculate searchlight crossnobis rdm
    
    :param n_split_data(int): how many datas to process at once
    :param centers(np.array): centers, shape: (#center)
    :param neighbors(np.array): neighbors , shape: (#center, #neighbor)
    :param precs(np.array): precisions , shape: (#channel, #run, #precision_mat_element)
    :param measurements(np.array): measurment values , shape: (#cond, #channel)
    :param masking_indexes: (np.array) , shape: (#channel) , index of masking brain
    :param conds: conds(np.array - 1d)
    :param sessions(np.array - 1d): session corressponding to conds
    :param n_thread_per_block(int): , block per thread
    
    """
    # Data configuration
    n_run = len(np.unique(sessions))
    n_cond = len(np.unique(conds))
    n_dissim = int((n_cond * n_cond - n_cond) / 2)
    n_neighbor = neighbors.shape[-1]
    uq_conds = np.unique(conds)
    n_channel = measurements.shape[-1]
    uq_sessions = np.unique(sessions)
    
    assert n_channel == masking_indexes.shape[0], "n_channel should be same"
    
    # Fold
    fold_info = cuda.to_device(list(itertools.combinations(np.arange(len(uq_sessions)), 2)))
    n_fold = len(fold_info)
    total_calculation = n_split_data * n_fold
    
    # GPU Configuration
    n_block = int(np.ceil(n_split_data / n_thread_per_block))
    n_thread_per_block_2d = int(np.ceil(np.sqrt(n_thread_per_block)))
    block_2ds = (total_calculation // n_thread_per_block_2d, total_calculation // n_thread_per_block_2d)
    thread_2ds = (n_thread_per_block_2d, n_thread_per_block_2d)
    
    # Memory pool
    mempool = cp.get_default_memory_pool()
    
    # Calculation
    rdm_outs = []
    for i in trange(0, len(centers), n_split_data):
        # select neighbors
        target_centers = centers[i:i + n_split_data]
        target_neighbors = neighbors[i:i + n_split_data, :]

        n_target_centers  = len(target_centers)

        # output_1d
        mask_out = cuda.to_device(np.zeros((n_target_centers, n_channel)))

        # Make mask - neighbor
        set_mask[n_block, n_thread_per_block](target_neighbors, masking_indexes, mask_out)
        cuda.synchronize()

        # Apply mask
        cpu_mask = mask_out.copy_to_host()
        masked_measurements = []
        for j in range(n_target_centers):
            masked_measurements.append(measurements[:, cpu_mask[j] == 1])
        masked_measurements = np.array(masked_measurements)
        masked_measurements = cp.asarray(masked_measurements)

        del mask_out
        cuda.defer_cleanup()

        # precision
        prec_mat_shape = int((n_neighbor * n_neighbor - n_neighbor) / 2) + n_neighbor
        target_precs = precs[i:i+n_target_centers].reshape(-1, prec_mat_shape)
        target_precs = np.array([convert_1d_to_symmertic(pre, size = n_neighbor) for pre in target_precs])
        variances = cp.linalg.inv(cp.asarray(target_precs))
        variances = variances.reshape(n_target_centers, n_run, n_neighbor, n_neighbor).get()
        fold_preicions = cp.linalg.inv(cp.asarray(mean_fold_variance(variances, fold_info.copy_to_host())))
        fold_preicions = cuda.to_device(fold_preicions.reshape(n_target_centers, len(fold_info), n_neighbor, n_neighbor).get())
        mempool.free_all_blocks()

        # Avg conds per session
        avg_measurements = []
        avg_conds = []
        for session in uq_sessions:
            filtering_session = sessions == session
            sess_cond = conds[filtering_session]
            sess_measurements = cp.compress(filtering_session, masked_measurements, axis = 1)

            mean_measurments = []
            for cond in uq_conds:
                filtering_cond = sess_cond == cond
                cond_measurments = cp.compress(filtering_cond, sess_measurements, axis = 1)
                mean_cond_measurement = cp.mean(cond_measurments, axis = 1)
                mean_measurments.append(cp.expand_dims(mean_cond_measurement, axis = 1))

                avg_conds.append(cond)

            avg_measurements.append(cp.expand_dims(cp.concatenate(mean_measurments, axis = 1), axis = 1))
        avg_measurements = cp.concatenate(avg_measurements, axis = 1).get()

        avg_conds = np.array(avg_conds)

        mempool.free_all_blocks()

        # make kernel
        avg_measurements = cuda.to_device(avg_measurements)

        matmul1_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_cond, n_neighbor)))
        kernel_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_cond, n_cond)))
        calc_kernel[block_2ds, thread_2ds](avg_measurements, fold_preicions, fold_info, matmul1_out, kernel_out)

        cuda.synchronize()
        del matmul1_out
        cuda.defer_cleanup()

        rdm_out = cuda.to_device(np.zeros((n_target_centers, n_fold, n_dissim)))
        rdm_from_kernel[block_2ds, thread_2ds](kernel_out, n_neighbor, rdm_out)

        cuda.synchronize()

        mean_rdms = cp.mean(rdm_out.copy_to_host(), axis = 1)
        rdm_outs.append(mean_rdms)

        del kernel_out
        del rdm_out
        cuda.defer_cleanup()
        
    return rdm_outs, uq_conds

