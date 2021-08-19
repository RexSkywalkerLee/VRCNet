import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from utils.model_utils import *


class point_projection_feature(nn.Module):
    def __init__(self):
        super(point_project_feature, self).__init__()
        
    def forward(self, points):
        # Convert 3D position feature to 4D point projection feature
        # Input shape: [batch_size, 3, num_point]
        # Return: [batch_size, 4, num_point]
        batch_size = points.size()[0]
        num_point = points.size()[2]
        
        # Calculate 3 axises
        
        # Axis 1 is the vector with the maximum norm
        vector_norm = torch.sqrt(torch.sum(points * points, 1, keepdim=False))
        _, ids1 = torch.max(vector_norm, 1, keepdim=True)
        batch_indices = torch.reshape(torch.range(batch_size), (-1, 1))
	    axis1 = torch.cat([points[batch_indice, :, id1].unsqueeze(0) for (batch_indice, id1) in zip(batch_indices, ids1)], 0)
	    axis1 = axis1 / (tf.norm(axis1, 2, None, keepdim=True) + 1e-7)
        
        # Axis 2 is the vector with the minimum norm
        _, ids2 = torch.min(vector_norm, 1, keepdim=True)
	    axis2 = torch.cat([points[batch_indice, :, id2].unsqueeze(0) for (batch_indice, id2) in zip(batch_indices, ids2)], 0)
	    axis2 = axis2 / (tf.norm(axis2, 2, None, keepdim=True) + 1e-7)
        
        # Axis 3 is the cross result of axis 1 and axis 2
        axis3 = torch.cross(axis1, axis2, dim=1)
        axis3 = axis3 / (tf.norm(axis3, 2, None, keepdim=True) + 1e-7)
        
        den = torch.norm(points, 2, 1, keepdim=True) + 1e-7
        c1 = torch.sum(points * axis1.unsqueeze(2), 1, keepdim=True) / den
        c2 = torch.sum(points * axis2.unsqueeze(2), 1, keepdim=True) / den
        c3 = torch.sum(points * axis3.unsqueeze(2), 1, keepdim=True) / den
        c4 = vector_norm.unsqueeze(1)
        
        new_c = torch.cat([c1, c2, c3, c4], 1)
        assert (new_c.size() == torch.Size([batch_size, 4, num_point]))
        return axis1, axis2, axis3, new_c
    

class inverse_point_projection_feature(nn.Module):
    def __init__(self):
        super(inverse_point_project_feature, self).__init__()
        
    def forward(self, axis1, axis2, axis3, features):
        