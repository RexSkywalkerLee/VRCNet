from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from utils.model_utils import *
from utils.ri_utils import *
from models.vrcnet import Linear_ResBlock

class local_decoder(nn.Module):
    def __init__(self, knn=32, dilation=2, mlp=[64, 128, 128], mlp_merge=[512, 1024, 1024], input_channel=3):
        super(local_decoder, self).__init__()
        self.knn = knn
        self.dilation = dilation
        
        self.mlp1 = nn.Sequential(nn.Conv2d(input_channel, mlp[0], kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv2d(mlp[0], mlp[1], kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv2d(mlp[1], mlp[2], kernel_size=1))
        self.mlp2 = nn.Sequential(nn.Conv2d(input_channel, mlp[0], kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv2d(mlp[0], mlp[1], kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv2d(mlp[1], mlp[2], kernel_size=1))
        self.mlp_merge = nn.Sequential(nn.Conv1d(2*mlp[2], mlp_merge[0], kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv1d(mlp_merge[0], mlp_merge[1], kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv1d(mlp_merge[1], mlp_merge[2], kernel_size=1))

    def forward(self, points, features=None):
        batch_size, _, num_points = points.size()
        
        if features is None:
            points_knn = get_edge_features(points, knn(points, self.knn))
            points_knn = points_knn - points.unsqueeze(2).repeat(1,1,self.knn,1)
        
            points_knn_d = get_edge_features(points, dilated_knn(points, self.knn, self.dilation))
            points_knn_d = points_knn_d - points.unsqueeze(2).repeat(1,1,self.knn,1)
        
            a1, a2, a3, points_knn_d = acenn_rir_feature(points_knn_d.transpose(1,3).contiguous(),
                                                         points.transpose(1,2).contiguous())
            _, _, _, points_knn = point_projection_feature(points_knn.transpose(1,3).contiguous(), a1, a2, a3)
        
            points_knn = points_knn.transpose(1,3).contiguous()
            points_knn_d = points_knn_d.transpose(1,3).contiguous()
        else:
            points_knn = get_edge_features(features, knn(points, self.knn))
            points_knn_d = get_edge_features(features, dilated_knn(points, self.knn, self.dilation))
            
        points_knn = self.mlp1(points_knn)
        points_knn_d = self.mlp2(points_knn_d)
        
        local_feature_knn, _ = torch.max(points_knn, 2)
        local_feature_knn_d, _ = torch.max(points_knn_d, 2)

        local_feature = torch.cat((local_feature_knn, local_feature_knn_d), 1)
        local_feature = self.mlp_merge(local_feature)
        return local_feature
    
    
class RIEncoder(nn.Module):
    def __init__(self, input_size=3, d_model=1024, num_coarse=1024, n_head=1, dropout=0.0):
        super(RIEncoder, self).__init__()
        self.num_coarse = num_coarse
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.conv1 = nn.Conv1d(input_size, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, d_model, 1)
        
        self.fc_qkv = nn.Linear(d_model, d_model * 3)
        self.attn = nn.MultiheadAttention(self.d_model, self.n_head, self.dropout)
        self.ln = nn.LayerNorm(d_model)
        self.fcs1 = nn.Linear(d_model, d_model)
        self.fcs2 = nn.Linear(d_model, 1)
        
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, self.num_coarse * 3)

    def forward(self, points):
        batch_size, _, num_points = points.size()
        div_batch_size = batch_size // 8
        x = F.relu(self.conv1(points))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        global_feature = global_feature.view(batch_size, -1)
        global_feature_tup = global_feature.chunk(8, dim=0)
        global_feature = torch.stack(global_feature_tup)
        
        qkv = self.fc_qkv(global_feature)
        q, k, v = torch.split(qkv, self.d_model, dim=2)
        multi_attn, _ = self.attn(q, k, v)
        sel = self.ln(multi_attn + global_feature)
        sel = F.softmax(self.fcs2(self.fcs1(sel)))
        global_feature = (sel * global_feature).sum(dim=0, keepdim=False)
        _, inx = sel.squeeze().max(dim=0, keepdim=False)
        
        coarse = F.relu(self.fc1(global_feature))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)
        
        return global_feature, coarse, inx


class RIDecoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, global_feature_size, local_feature_size, k=32, d=2):
        super(RIDecoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.local_feature_size = local_feature_size
        self.cat_feature_num = 2 + 3 + global_feature_size + local_feature_size

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(self.cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        
        self.decoder = local_decoder(k, d)

    def forward(self, global_feat, sampled_points):
        batch_size = sampled_points.size()[0]
        local_feat = self.decoder(sampled_points)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()
        
        local_feat = torch.cat((sampled_points, local_feat), 1)

        point_feat = ((local_feat.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine, 3+self.local_feature_size)).transpose(1, 2).contiguous()

        global_feat = global_feat.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((sampled_points.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine, 3)).transpose(1, 2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return fine
    

class Model(nn.Module):
    def __init__(self, args, global_feature_size=1024, local_feature_size=1024, k=32, d=2):
        super(Model, self).__init__()

        self.input_size = args.input_size
        self.num_coarse = args.num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.scale = self.num_points // self.num_coarse

        self.encoder = RIEncoder(3, global_feature_size, self.num_coarse)
        self.decoder = RIDecoder(self.num_coarse, self.num_points, self.scale, global_feature_size, local_feature_size, k, d)

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        batch_size = x.size()[0]
        a1, a2, a3, x = point_projection_feature(x.transpose(1, 2).contiguous(), method='pca')
        x = x.transpose(1, 2).contiguous()
        org_point_input = x
        
        x0, x1, x2 = x.chunk(3, dim=1)
        cloud_1 = x
        #cloud_1 = torch.cat([x0, x1, x2], dim=1)
        cloud_2 = torch.cat([x0, x1, -x2], dim=1)
        cloud_3 = torch.cat([x0, -x1, x2], dim=1)
        cloud_4 = torch.cat([x0, -x1, -x2], dim=1)
        cloud_5 = torch.cat([-x0, x1, x2], dim=1)
        cloud_6 = torch.cat([-x0, x1, -x2], dim=1)
        cloud_7 = torch.cat([-x0, -x1, x2], dim=1)
        cloud_8 = torch.cat([-x0, -x1, -x2], dim=1)
        points = torch.cat([cloud_1, cloud_2, cloud_3, cloud_4,
                            cloud_5, cloud_6, cloud_7, cloud_8], dim=0)
        
        feat, out1, inx = self.encoder(points)
        
        points = torch.cat((out1, org_point_input), 2)
        _, sampled_points = furthest_point_sampling(points.transpose(1,2).contiguous(), 
                                                    self.num_coarse)
        sampled_points = sampled_points.transpose(1,2).contiguous()
        
        out2 = self.decoder(feat, sampled_points)
        
        for i in range(batch_size):
            if inx[i] >= 4:
                a1[i] *= -1
            if inx[i] % 4 >= 2:
                a2[i] *= -1
            if inx[i] % 2 == 1:
                a3[i] *= -1
        
        out1 = inverse_point_projection_feature(a1, a2, a3, out1.transpose(1,2).contiguous())
        out2 = inverse_point_projection_feature(a1, a2, a3, out2.transpose(1,2).contiguous())

        if is_training:
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1, _ = calc_cd(out1, gt)
                loss2, _ = calc_cd(out2, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss = loss1.mean() + loss2.mean() * alpha
            return out2, loss2, total_train_loss
        else:
            #emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            return {'out1': out1, 'out2': out2, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
        
        