import torch
import numpy as np


def rotation_transform(dataset, arbitrary=True, random_seed=None, sampling='uniform'):
    if random_seed:
        np.random.seed(random_seed)
        
    for i in range(dataset.gt_data.shape[0]):
        if not arbitrary:
            ang = np.random.rand() * 2 * np.pi
            rotation_matrix = np.array(((np.cos(ang), 0, np.sin(ang)),
                                        (0, 1, 0),
                                        (-np.sin(ang), 0, np.cos(ang))))
        else:
            if sampling == 'uniform':
                azi = np.random.rand() * 2 * np.pi
                R = np.array(((np.cos(azi), np.sin(azi), 0),
                              (-np.sin(azi), np.cos(azi), 0),
                              (0, 0, 1)))
                x = np.random.rand(2)
                v = np.array((np.cos(2*np.pi*x[0])*np.sqrt(x[1]),
                              np.sin(2*np.pi*x[0])*np.sqrt(x[1]),
                              np.sqrt(1-x[1])))
                H = np.eye(3) - 2 * np.outer(v, v)
                rotation_matrix = -H @ R
            elif sampling == 'euler':
                angs = np.random.rand(3) * 2 * np.pi
                rot_z = np.array(((np.cos(angs[0]), -np.sin(angs[0]), 0),
                                  (np.sin(angs[0]), np.cos(angs[0]), 0),
                                  (0, 0, 1)))
                rot_y = np.array(((np.cos(angs[1]), 0, np.sin(angs[1])),
                                  (0, 1, 0),
                                  (-np.sin(angs[1]), 0, np.cos(angs[1]))))
                rot_x = np.array(((1, 0, 0),
                                  (0, np.cos(angs[2]), -np.sin(angs[2])),
                                  (0, np.sin(angs[2]), np.cos(angs[2]))))
                rotation_matrix = rot_z @ rot_y @ rot_x
        dataset.gt_data[i] = dataset.gt_data[i] @ rotation_matrix
        dataset.input_data[26*i:26*i+25] = dataset.input_data[26*i:26*i+25] @ rotation_matrix
        
        
def point_projection_feature(points, pca=False, axis1=None, axis2=None, axis3=None):
    # Convert 3D position feature to 4D point projection feature
    # Input shape: [batch_size, num_point, 3]
    # Return: [batch_size, num_point, 4]
    batch_size = points.size()[0]
    num_point = points.size()[1]
    
    vector_norm = torch.sqrt(torch.sum(points * points, 2, keepdim=False))
    
    if not pca and axis1 is None:
        # Calculate 3 axises
    
        # Axis 1 is the vector with the maximum norm
        _, ids1 = torch.max(vector_norm, 1, keepdim=False)
        batch_indices = range(batch_size)
        axis1 = torch.cat([points[batch_indice, id1, :].unsqueeze(0) for (batch_indice, id1) in zip(batch_indices, ids1)], 0)
        axis1 = axis1 / (torch.norm(axis1, 2, 1, keepdim=True) + 1e-7)
    
        # Axis 2 is the vector with the minimum norm
        _, ids2 = torch.min(vector_norm, 1, keepdim=False)
        axis2 = torch.cat([points[batch_indice, id2, :].unsqueeze(0) for (batch_indice, id2) in zip(batch_indices, ids2)], 0)
        axis2 = axis2 / (torch.norm(axis2, 2, 1, keepdim=True) + 1e-7)
        
        # Axis 3 is the cross result of axis 1 and axis 2
        axis3 = torch.cross(axis1, axis2, dim=1)
        axis3 = axis3 / (torch.norm(axis3, 2, 1, keepdim=True) + 1e-7)
        
    elif pca:
        # Using PCA to define 3 axises
        _, _, V = torch.pca_lowrank(points)
        axis1, axis2, axis3 = V.chunk(3, dim=2)
        axis1 = axis1.squeeze()
        axis2 = axis2.squeeze()
        axis3 = axis3.squeeze()
        axis1 = axis1 / (torch.norm(axis1, 2, 1, keepdim=True) + 1e-7)
        axis2 = axis2 / (torch.norm(axis2, 2, 1, keepdim=True) + 1e-7)
        axis3 = axis3 / (torch.norm(axis3, 2, 1, keepdim=True) + 1e-7)
        
    else:
        axis1, axis2, axis3 = axis1, axis2, axis3
        
    #den = torch.norm(points, 2, 2, keepdim=True) + 1e-7
    #c1 = torch.sum(points * axis1.unsqueeze(1), 2, keepdim=True) / den
    #c2 = torch.sum(points * axis2.unsqueeze(1), 2, keepdim=True) / den
    #c3 = torch.sum(points * axis3.unsqueeze(1), 2, keepdim=True) / den
    #c4 = vector_norm.unsqueeze(2)
        
    #new_c = torch.cat([c1, c2, c3, c4], 2)
    #assert (new_c.size() == torch.Size([batch_size, num_point, 4]))
    c1 = torch.sum(points * axis1.unsqueeze(1), 2, keepdim=True)
    c2 = torch.sum(points * axis2.unsqueeze(1), 2, keepdim=True)
    c3 = torch.sum(points * axis3.unsqueeze(1), 2, keepdim=True)
    new_c = torch.cat([c1, c2, c3], 2)
    return axis1, axis2, axis3, new_c


def inverse_point_projection_feature(axis1, axis2, axis3, points):
    # Revert 4D point project feature to 3D position feature
    # Axis shape: [batch_size, 3]
    # Input shape: [batch_size, num_point, 4]
    # Return: [batch_size, num_point, 3]
    if len(list(points.size())) == 2 and len(list(axis1.size())) == 1:
        axis1 = axis1.unsqueeze(0)
        axis2 = axis2.unsqueeze(0)
        axis3 = axis3.unsqueeze(0)
        points = points.unsqueeze(0)

    batch_size = points.size()[0]
    num_point = points.size()[1]
    
    '''
    Note: the SVD solution proposed by original paper is deprecated, as SVD does not give unique solution
    
    # Compute inner product of axises
    #m_12 = torch.sum(axis1 * axis2, 1, keepdim=False).unsqueeze(1)
    #m_23 = torch.sum(axis2 * axis3, 1, keepdim=False).unsqueeze(1)
    #m_31 = torch.sum(axis3 * axis1, 1, keepdim=False).unsqueeze(1)
    
    # Construct M matrices
    M = torch.zeros(batch_size, num_point, 3, 3)
    M[:, :, 0, 1:3] = points[:, :, 0:2]
    M[:, :, 1, 2] = m_12
    M = M + torch.transpose(M, 2, 3)
    diag = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, num_point, 1, 1)
    M = M + diag
    
    # Compute singular value decomposition of M
    U, S, Vh = torch.linalg.svd(M)
    S = torch.sqrt(S)
    C = U @ torch.diag_embed(S) @ Vh
    '''
    
    A = torch.cat([axis1.unsqueeze(2), axis2.unsqueeze(2), axis3.unsqueeze(2)], dim=2)
    A = torch.transpose(A, 1, 2).unsqueeze(1).repeat(1, num_point, 1, 1)
    B = points[:, :, 0:3]
    #X = torch.linalg.solve(A, B)
    X = torch.matmul(torch.linalg.pinv(A), B.unsqueeze(3)).squeeze()
    X = X / (torch.norm(X, 2, 2, keepdim=True) + 1e-7) * points[:, :, 3].unsqueeze(2)
    
    return X.squeeze()


def point_ortho_feature(points, axis1=None, axis2=None, axis3=None, pca=True):
    # Input shape: [*, num_point, 3]
    # Return: [*, num_point, 3]
    batch_size = points.size()[0]
    num_point = points.size()[-2]
            
    if axis1 is not None:
        axis1, axis2, axis3 = axis1, axis2, axis3
    else:
        # Using PCA to define 3 axises
        _, _, V = torch.pca_lowrank(points)
        axis1, axis2, axis3 = V.chunk(3, dim=-1)
        axis1 = axis1.squeeze()
        axis2 = axis2.squeeze()
        axis3 = axis3.squeeze()
        axis1 = axis1 / (torch.norm(axis1, 2, -1, keepdim=True) + 1e-7)
        axis2 = axis2 / (torch.norm(axis2, 2, -1, keepdim=True) + 1e-7)
        axis3 = axis3 / (torch.norm(axis3, 2, -1, keepdim=True) + 1e-7)
        
    c1 = torch.sum(points * axis1.unsqueeze(-2), 2, keepdim=True)
    c2 = torch.sum(points * axis2.unsqueeze(-2), 2, keepdim=True)
    c3 = torch.sum(points * axis3.unsqueeze(-2), 2, keepdim=True)
        
    new_c = torch.cat([c1, c2, c3], 2)
    assert (new_c.size() == torch.Size([batch_size, num_point, 3]))
    return axis1, axis2, axis3, new_c


def inverse_point_ortho_feature(axis1, axis2, axis3, points):
    # Axis shape: [batch_size, 3]
    # Input shape: [batch_size, num_point, 3]
    # Return: [batch_size, num_point, 3]
    if len(list(points.size())) == 2 and len(list(axis1.size())) == 1:
        axis1 = axis1[None,:]
        axis2 = axis2[None,:]
        axis3 = axis3[None,:]
        points = points[None,:]

    batch_size = points.size()[0]
    num_point = points.size()[1]
    
    A = torch.cat([axis1.unsqueeze(2), axis2.unsqueeze(2), axis3.unsqueeze(2)], dim=2)
    A = torch.transpose(A, 1, 2).unsqueeze(1).repeat(1, num_point, 1, 1)
    X = torch.matmul(torch.linalg.pinv(A), points.unsqueeze(3)).squeeze()
    
    return X.squeeze()