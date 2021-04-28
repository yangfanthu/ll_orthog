import math
import torch
import numpy as np

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def generate_projection_matrix(num_tasks, feature_dim=256, share_dims=0, qr=True):
    """
    Project features (v) to subspace parametrized by A:

    Returns: A.(A^T.A)^-1.A^T
    """
    rank = int((feature_dim - share_dims)/ num_tasks)
    assert num_tasks*rank <= (feature_dim - share_dims), "Feature dimension should be less than num_tasks * rank"
    
    # Generate ONBs
    if qr:
        print('Generating ONBs from QR decomposition')
        rand_matrix = np.random.uniform(size=(feature_dim, feature_dim))
        q, r = np.linalg.qr(rand_matrix, mode='complete')
    else:
        print('Generating ONBs from Identity matrix')
        q = np.identity(feature_dim)
    projections = []
    
    for tt in range(num_tasks):
        offset = tt*rank
        A = np.concatenate((q[:, offset:offset+rank], q[:, feature_dim-share_dims:]), axis=1)
        proj = np.matmul(A, np.transpose(A)) 
        projections.append(proj)
        
    return projections


def unit_test_projection_matrices(projection_matrices):
    """
    Unit test for projection matrices
    """
    num_matrices = len(projection_matrices)
    feature_dim = projection_matrices[0].shape[0]
    rand_vetcor = np.random.rand(1, feature_dim)
    projections = []
    for tt in range(num_matrices):
        print('Task:{}, Projection Dims: {}, Projection Rank: {}'.format(tt, projection_matrices[tt].shape, np.linalg.matrix_rank(projection_matrices[tt])))
        projections.append((np.squeeze(np.matmul(rand_vetcor, projection_matrices[tt]))))
    print('\n\n ******\n Sanity testing projections \n********')
    for i in range(num_matrices):
        for j in range(num_matrices):
            print('P{}.P{}={}'.format(i, j, np.dot(projections[i], projections[j])))