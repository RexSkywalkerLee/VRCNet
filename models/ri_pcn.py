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


class PCN_encoder(nn.Module):
    def __init__(self, input_size=3, output_size=1024):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, global_feature_size, output_size=3):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.output_size = output_size
        self.cat_feature_num = 2 + 3 + global_feature_size
        self.fc1 = nn.Linear(global_feature_size, global_feature_size)
        self.fc2 = nn.Linear(global_feature_size, global_feature_size)
        self.fc3 = nn.Linear(global_feature_size, num_coarse * output_size)

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(self.cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, self.output_size, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, self.output_size, self.num_coarse)
        
        #id0 = torch.zeros(coarse.shape[0], 1, coarse.shape[2]).cuda().contiguous()
        #coarse_input = torch.cat((coarse, id0), 1)
        #id1 = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).cuda().contiguous()
        #org_points_input = torch.cat((org_points_input, id1), 1)

        #points = torch.cat((coarse_input, org_points_input), 2)
        
        #coarse = inverse_point_ortho_feature(a1, a2, a3, coarse.transpose(1, 2).contiguous())
        #full_points = torch.cat((org.transpose(1, 2).contiguous(), coarse), dim=1)
        #a1, a2, a3, _ = point_ortho_feature(full_points, pca=True)
        #_, _, _, coarse = point_ortho_feature(coarse, False, a1.detach(), a2.detach(), a3.detach())
        #coarse = coarse.transpose(1, 2).contiguous()

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine, self.output_size)).transpose(1, 2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine, self.output_size)).transpose(1, 2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine
    
'''
class Model(nn.Module):
    def __init__(self, args, global_feature_size=1024, feature_append=9):
        super(Model, self).__init__()

        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_coarse = args.num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.scale = self.num_points // self.num_coarse

        self.encoder = PCN_encoder(output_size=global_feature_size)
        self.decoder = PCN_decoder(self.num_coarse, self.num_points, self.scale, global_feature_size+feature_append)
        
        #self.axis_inference = PCN_encoder(input_size=3, output_size=9)

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        #axis = self.axis_inference(x)
        #a1, a2, a3 = axis.chunk(3, dim=1)
        #a1 = a1.squeeze() / (torch.norm(a1, 2, 1, keepdim=True) + 1e-7)
        #a2 = a2.squeeze() / (torch.norm(a2, 2, 1, keepdim=True) + 1e-7)
        #a3 = a3.squeeze() / (torch.norm(a3, 2, 1, keepdim=True) + 1e-7)
        a1, a2, a3, x = point_ortho_feature(x.transpose(1, 2).contiguous())
        #_, _, _, x = point_projection_feature(x.transpose(1, 2).contiguous(), False, a1, a2, a3)
        x = x.transpose(1, 2).contiguous()
        feat = self.encoder(x)
        feat = torch.cat([feat, a1, a2, a3], dim=1)
        out1, out2 = self.decoder(feat)
        out1 = inverse_point_ortho_feature(a1, a2, a3, out1.transpose(1, 2).contiguous())
        out2 = inverse_point_ortho_feature(a1, a2, a3, out2.transpose(1, 2).contiguous())

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
'''


class Model(nn.Module):
    def __init__(self, args, size_z=128, global_feature_size=1024, feature_append=9):
        super(Model, self).__init__()
        
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_coarse = args.num_coarse
        self.num_points = args.num_points
        self.scale = self.num_points // self.num_coarse
        self.size_z = size_z
        
        self.encoder = PCN_encoder(output_size=global_feature_size)
        self.posterior_infer1 = Linear_ResBlock(input_size=global_feature_size+feature_append, output_size=global_feature_size+feature_append)
        self.posterior_infer2 = Linear_ResBlock(input_size=global_feature_size+feature_append, output_size=size_z * 2)
        self.prior_infer = Linear_ResBlock(input_size=global_feature_size+feature_append, output_size=size_z * 2)
        self.generator = Linear_ResBlock(input_size=size_z, output_size=global_feature_size+feature_append)
        self.decoder = PCN_decoder(self.num_coarse, self.num_points, self.scale, global_feature_size+feature_append)

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        num_input = x.size()[2]

        if is_training:
            y = pn2.gather_operation(gt.transpose(1, 2).contiguous(), pn2.furthest_point_sample(gt, num_input))
            gt = torch.cat([gt, gt], dim=0)
            points = torch.cat([x, y], dim=0)
            x = torch.cat([x, x], dim=0)
        else:
            points = x
        
        a1, a2, a3, points = point_ortho_feature(points.transpose(1, 2).contiguous(), pca=True)
        feat = self.encoder(points.transpose(1, 2).contiguous())
        feat = torch.cat([feat, a1, a2, a3], dim=1)

        if is_training:
            feat_x, feat_y = feat.chunk(2)
            o_x = self.posterior_infer2(self.posterior_infer1(feat_x))
            q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
            o_y = self.prior_infer(feat_y)
            p_mu, p_std = torch.split(o_y, self.size_z, dim=1)
            q_std = F.softplus(q_std)
            p_std = F.softplus(p_std)
            q_distribution = torch.distributions.Normal(q_mu, q_std)
            p_distribution = torch.distributions.Normal(p_mu, p_std)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_std.detach())
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_std))
            z_q = q_distribution.rsample()
            z_p = p_distribution.rsample()
            z = torch.cat([z_q, z_p], dim=0)
            feat = torch.cat([feat_x, feat_x], dim=0)
        else:
            o_x = self.posterior_infer2(self.posterior_infer1(feat))
            q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
            q_std = F.softplus(q_std)
            q_distribution = torch.distributions.Normal(q_mu, q_std)
            p_distribution = q_distribution
            p_distribution_fix = p_distribution
            m_distribution = p_distribution
            z = q_distribution.rsample()

        feat += self.generator(z)

        coarse, fine = self.decoder(feat)
        coarse = inverse_point_ortho_feature(a1, a2, a3, coarse.transpose(1, 2).contiguous())
        fine = inverse_point_ortho_feature(a1, a2, a3, fine.transpose(1, 2).contiguous())
        
        if is_training:
            dl_rec = torch.distributions.kl_divergence(m_distribution, p_distribution)
            dl_g = torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
            dl_g_ = torch.distributions.kl_divergence(m_distribution, q_distribution)

            loss2, _ = calc_cd(coarse, gt)
            loss1, _ = calc_cd(fine, gt)

            total_train_loss = loss1.mean() + loss2.mean() * alpha
            total_train_loss += (dl_rec.mean() + dl_g.mean() + dl_g_.mean()) * 10
            return fine, loss2, total_train_loss
        else:
            #emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(fine, gt, calc_f1=True)
            return {'out1': coarse, 'out2': fine, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
