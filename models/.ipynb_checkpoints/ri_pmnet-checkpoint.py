import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from utils.model_utils import *
from utils.ri_utils import *
from models.pcn import PCN_encoder

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU(inplace=False)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)


class PMNet_decoder(nn.Module):
    def __init__(self, num_fine):
        super(PMNet_decoder, self).__init__()
        self.num_fine = num_fine

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_fine * 4)

        self.af = nn.ReLU(inplace=False)

    def forward(self, global_feat, a1, a2, a3):
        batch_size = global_feat.size()[0]
        
        fine = self.fc3(self.af(self.fc2(self.af(self.fc1(global_feat))))).view(batch_size, 4, self.num_fine)
        fine = inverse_point_projection_feature(a1, a2, a3, fine.transpose(1, 2).contiguous())

        return fine
    

class Model(nn.Module):
    def __init__(self, args, size_z=128, global_feature_size=1024):
        super(Model, self).__init__()

        layers = [int(i) for i in args.layers.split(',')]
        knn_list = [int(i) for i in args.knn_list.split(',')]

        self.size_z = size_z
        self.distribution_loss = args.distribution_loss
        self.train_loss = args.loss
        self.encoder = PCN_encoder(input_size=4, output_size=global_feature_size)
        self.posterior_infer1 = Linear_ResBlock(input_size=global_feature_size, output_size=global_feature_size)
        self.posterior_infer2 = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        self.prior_infer = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        self.generator = Linear_ResBlock(input_size=size_z, output_size=global_feature_size)
        self.decoder = PMNet_decoder(num_fine=args.num_points)

    def compute_kernel(self, x, y):
        x_size = x.size()[0]
        y_size = y.size()[0]
        dim = x.size()[1]

        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / float(dim))

    def mmd_loss(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):
        num_input = x.size()[2]

        if is_training:
            y = pn2.gather_operation(gt.transpose(1, 2).contiguous(), pn2.furthest_point_sample(gt, num_input))
            gt = torch.cat([gt, gt], dim=0)
            points = torch.cat([x, y], dim=0)
            '''
            points_copy = points.transpose(1, 2).contiguous()
            for i in range(points_copy.size()[0]):
                azi = np.random.rand() * 2 * np.pi
                R = np.array(((np.cos(azi), np.sin(azi), 0),
                              (-np.sin(azi), np.cos(azi), 0),
                              (0, 0, 1)))
                xs = np.random.rand(2)
                v = np.array((np.cos(2*np.pi*xs[0])*np.sqrt(xs[1]),
                              np.sin(2*np.pi*xs[0])*np.sqrt(xs[1]),
                              np.sqrt(1-xs[1])))
                H = np.eye(3) - 2 * np.outer(v, v)
                rotation_matrix = torch.from_numpy(-H @ R).float().cuda()
                points_copy[i] = points_copy[i] @ rotation_matrix
                gt[i] = gt[i] @ rotation_matrix
            points = points_copy.transpose(1, 2).contiguous()
            '''
            x = torch.cat([x, x], dim=0)
        else:
            points = x
            
        a1, a2, a3, points = point_projection_feature(points.transpose(1, 2).contiguous())
        points = points.transpose(1, 2).contiguous()
            
        feat = self.encoder(points)

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

        fine = self.decoder(feat, a1, a2, a3)
        #fine = fine.transpose(1, 2).contiguous()
        
        if is_training:
            if self.distribution_loss == 'MMD':
                z_m = m_distribution.rsample()
                z_q = q_distribution.rsample()
                z_p = p_distribution.rsample()
                z_p_fix = p_distribution_fix.rsample()
                dl_rec = self.mmd_loss(z_m, z_p)
                dl_g = self.mmd_loss2(z_q, z_p_fix)
            elif self.distribution_loss == 'KLD':
                dl_rec = torch.distributions.kl_divergence(m_distribution, p_distribution)
                dl_g = torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
                dl_g_ = torch.distributions.kl_divergence(m_distribution, q_distribution)
            else:
                raise NotImplementedError('Distribution loss is either MMD or KLD')

            if self.train_loss == 'cd':
                loss4, _ = calc_cd(fine, gt)
            else:
                raise NotImplementedError('Only CD is supported')

            total_train_loss = loss4.mean() * 10
            total_train_loss += (dl_rec.mean() + dl_g.mean() + dl_g_.mean()) * 20
            return fine, loss4, total_train_loss
        else:
            #emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(fine, gt, calc_f1=True)
            return {'out2': fine, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
