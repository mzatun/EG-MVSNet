import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *



class UniNetDS(nn.Module):
    def __init__(self):
        super(UniNetDS, self).__init__()
        self.base_filter = 8
        self.conv0_0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv0_1 = ConvBnReLU(8, 8, 3, 1, 1)
        self.conv1_0 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv1_1 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)

    def forward(self, x): # x [2, 3, 384, 768]
        x = self.conv0_1(self.conv0_0(x))
        x = self.conv1_1(self.conv1_0(x))
        x = self.conv1_2(x) # [2, 16, 192, 384]

        return x

# fixme: Feature Extraction Network
class FENet(nn.Module):
    def __init__(self, base_channels, num_stage=3,):
        super(FENet, self).__init__()
        base_filter = base_channels
        self.efe_module = EFEModule(base_filter)
        # self.efe_module = EFEModule_V1(base_filter)
        self.sfe_module = SFEModule(base_filter)

    def forward(self, x):
        ef = self.efe_module(x) # [1, 32, 128, 160]
        sf = self.sfe_module(x) # [1, 32, 128, 160]

        feature = {
            'edge_feature': ef,
            'surface_feature': sf
        }

        return feature

class RED_Regularization(nn.Module):
    def __init__(self, base_channels = 8):
        super(RED_Regularization, self).__init__()
        self.base_channels = base_channels
        self.conv_gru1 = ConvGRUCell2(16, 8, 3)
        self.conv_gru2 = ConvGRUCell2(16, 16, 3)
        self.conv_gru3 = ConvGRUCell2(32, 32, 3)
        self.conv_gru4 = ConvGRUCell2(64, 64, 3)
        self.conv1 = ConvReLU(16, 16, 3, 2, 1)
        self.conv2 = ConvReLU(16, 32, 3, 2, 1)
        self.conv3 = ConvReLU(32, 64, 3, 2, 1)
        self.upconv3 = ConvTransReLU(64, 32, 3, 2, 1, 1)
        self.upconv2 = ConvTransReLU(32, 16, 3, 2, 1, 1)
        self.upconv1 = ConvTransReLU(16, 8, 3, 2, 1, 1)
        self.upconv2d = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, volume_variance):
        depth_costs = []
        b_num, f_num, d_num, img_h, img_w = volume_variance.shape
        state1 = torch.zeros((b_num, 8, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 16, int(img_h / 2), int(img_w / 2))).cuda()
        state3 = torch.zeros((b_num, 32, int(img_h / 4), int(img_w / 4))).cuda()
        state4 = torch.zeros((b_num, 64, int(img_h / 8), int(img_w / 8))).cuda()

        cost_list = volume_variance.chunk(d_num, dim=2)
        cost_list = [cost.squeeze(2) for cost in cost_list]

        for cost in cost_list:
            # Recurrent Regularization
            conv_cost1 = self.conv1(-cost)
            conv_cost2 = self.conv2(conv_cost1)
            conv_cost3 = self.conv3(conv_cost2)
            reg_cost4, state4 = self.conv_gru4(conv_cost3, state4)
            up_cost3 = self.upconv3(reg_cost4)
            reg_cost3, state3 = self.conv_gru3(conv_cost2, state3)
            up_cost33 = torch.add(up_cost3, reg_cost3)
            up_cost2 = self.upconv2(up_cost33)
            reg_cost2, state2 = self.conv_gru2(conv_cost1, state2)
            up_cost22 = torch.add(up_cost2, reg_cost2)
            up_cost1 = self.upconv1(up_cost22)
            reg_cost1, state1 = self.conv_gru1(-cost, state1)
            up_cost11 = torch.add(up_cost1, reg_cost1)
            reg_cost = self.upconv2d(up_cost11)
            depth_costs.append(reg_cost)

        prob_volume = torch.stack(depth_costs, dim=1)
        prob_volume = prob_volume.squeeze(2)

        return prob_volume

class slice_RED_Regularization(nn.Module):
    def __init__(self, base_channels = 8):
        super(slice_RED_Regularization, self).__init__()
        self.base_channels = base_channels
        self.conv_gru1 = ConvGRUCell2(16, 8, 3)
        self.conv_gru2 = ConvGRUCell2(16, 16, 3)
        self.conv_gru3 = ConvGRUCell2(32, 32, 3)
        self.conv_gru4 = ConvGRUCell2(64, 64, 3)
        self.conv1 = ConvReLU(16, 16, 3, 2, 1)
        self.conv2 = ConvReLU(16, 32, 3, 2, 1)
        self.conv3 = ConvReLU(32, 64, 3, 2, 1)
        self.upconv3 = ConvTransReLU(64, 32, 3, 2, 1, 1)
        self.upconv2 = ConvTransReLU(32, 16, 3, 2, 1, 1)
        self.upconv1 = ConvTransReLU(16, 8, 3, 2, 1, 1)
        self.upconv2d = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, cost, state1, state2, state3, state4):

        # Recurrent Regularization
        conv_cost1 = self.conv1(-cost)
        conv_cost2 = self.conv2(conv_cost1)
        conv_cost3 = self.conv3(conv_cost2)
        reg_cost4, state4 = self.conv_gru4(conv_cost3, state4)
        up_cost3 = self.upconv3(reg_cost4)
        reg_cost3, state3 = self.conv_gru3(conv_cost2, state3)
        up_cost33 = torch.add(up_cost3, reg_cost3)
        up_cost2 = self.upconv2(up_cost33)
        reg_cost2, state2 = self.conv_gru2(conv_cost1, state2)
        up_cost22 = torch.add(up_cost2, reg_cost2)
        up_cost1 = self.upconv1(up_cost22)
        reg_cost1, state1 = self.conv_gru1(-cost, state1)
        up_cost11 = torch.add(up_cost1, reg_cost1)
        reg_cost = self.upconv2d(up_cost11)

        return reg_cost, state1, state2, state3, state4



def mvsnet_loss(depth_est, coarse_est, edge_est, depth_gt, mask):
    mask = mask > 0.5
    loss = None
    loss1 = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
    loss2 = F.smooth_l1_loss(coarse_est[mask], depth_gt[mask], size_average=True)
    depth_gt_l = depth_gt.unsqueeze(1)
    depths_X = F.pad(
        torch.abs(2 * depth_gt_l[:, :, :, 1:-1] - depth_gt_l[:, :, :, :-2] - depth_gt_l[:, :, :, 2:]) > 5,
        (1, 1, 0, 0, 0, 0, 0, 0))
    depths_Y = F.pad(
        torch.abs(2 * depth_gt_l[:, :, 1:-1, :] - depth_gt_l[:, :, :-2, :] - depth_gt_l[:, :, 2:, :]) > 5,
        (0, 0, 1, 1, 0, 0, 0, 0))

    depths_GRAD = torch.logical_or(depths_Y, depths_X).float().squeeze(1)
    edge_loss = F.smooth_l1_loss(depths_GRAD, edge_est, reduction="mean")  # 5 12 oct
    loss = loss1 + loss2 + edge_loss
    return loss

class EGNet(nn.Module):
    def __init__(self):
        super(EGNet, self).__init__()
        # self.feature = UniNetDS()
        self.feature = FENet(base_channels=8)
        self.cost_regularization = RED_Regularization(base_channels=8)
        self.volume_fusion = VolumeFusionModule(in_channels=16)
        self.edge_net = EedgeRegNet(in_channels=16)

    def compute_depth(self, prob_volume, depth_values=None):
        '''
        prob_volume: 1 x D x H x W
        '''
        B, M, H, W = prob_volume.shape[0], prob_volume.shape[1], prob_volume.shape[2], prob_volume.shape[3]
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        # depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        # depth_range = depth_values.to(prob_volume.device)
        depths = torch.index_select(depth_values, 1, indices.flatten())
        depth_image = depths.view(B, H, W)
        prob_image = probs.view(B, H, W)

        return depth_image, prob_image

    def compute_volume(self, ref_feature, src_features, src_projs, ref_proj, num_depth, num_views, depth_values):
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        return volume_variance

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        sf_features = []  # 表面特征
        ef_features = []  # 边缘特征
        for img in imgs:
            features = self.feature(img)
            sf = features['surface_feature']
            ef = features['edge_feature']
            sf_features.append(sf)
            ef_features.append(ef)
        # features = [self.feature(img) for img in imgs]
        # ref_feature, src_features = features[0], features[1:]
        sf_ref_feat, sf_src_feats = sf_features[0], sf_features[1:]
        ef_ref_feat, ef_src_feats = ef_features[0], ef_features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        sf_volume_variance = self.compute_volume(sf_ref_feat, sf_src_feats, src_projs, ref_proj, num_depth, num_views, depth_values)
        ef_volume_variance = self.compute_volume(ef_ref_feat, ef_src_feats, src_projs, ref_proj, num_depth, num_views, depth_values)

        # step 3. cost volume regularization
        coarse_prob_volume_pre = self.cost_regularization(sf_volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        coarse_prob_volume = F.softmax(coarse_prob_volume_pre, dim=1)

        # regression
        coarse_depth = depth_regression(coarse_prob_volume, depth_values=depth_values)
        # coarse_photometric_confidence, indices = coarse_prob_volume.max(1)

        batch, img_height, img_width = coarse_depth.shape[0], coarse_depth.shape[1], coarse_depth.shape[2]
        ef_ref_feature_return = F.interpolate(ef_ref_feat, [img_height, img_width], mode='bilinear')
        # edge map
        edge_map = self.edge_net(ef_ref_feat, coarse_depth).squeeze(1)

        # refine map
        refine_volume_variance = self.volume_fusion(sf_volume_variance, ef_volume_variance)  # Volume Fusion

        refine_prob_volume_pre = self.cost_regularization(refine_volume_variance).squeeze(1)  # 代价体正则化阶段  正则化的cost volume [B, D, W/(4/2/1), H/(4/2/1)]
        refine_prob_volume = F.softmax(refine_prob_volume_pre, dim=1)  # 计算softmax得到概率体 [B, D, W/(4/2/1), H/(4/2/1)]
        refine_depth = depth_regression(refine_prob_volume, depth_values=depth_values)  # 回归得到深度图 [B, W/(4/2/1), H/(4/2/1)]
        photometric_confidence, indices = refine_prob_volume.max(1)
        """
        # max
        depth, photometric_confidence = self.compute_depth(prob_volume, depth_values=depth_values)
        """
        """
        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume,
                                           depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                     dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        """

        return {"coarse_depth": coarse_depth, "edge_map": edge_map, "depth": refine_depth, "photometric_confidence": photometric_confidence, "ef_ref_feat":ef_ref_feature_return}

class InferenceREDNet(nn.Module):
    def __init__(self):
        super(InferenceREDNet, self).__init__()
        self.feature = UniNetDS()
        self.cost_regularization = slice_RED_Regularization(base_channels = 8)

    def compute_depth(self, prob_volume, depth_values=None):
        '''
        prob_volume: 1 x D x H x W
        '''
        B, M, H, W = prob_volume.shape[0], prob_volume.shape[1], prob_volume.shape[2], prob_volume.shape[3]
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        # depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        # depth_range = depth_values.to(prob_volume.device)
        depths = torch.index_select(depth_values, 1, indices.flatten())
        depth_image = depths.view(B, H, W)
        prob_image = probs.view(B, H, W)

        return depth_image, prob_image

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        b_num, f_num, img_h, img_w = ref_feature.shape
        state1 = torch.zeros((b_num, 8, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 16, int(img_h / 2), int(img_w / 2))).cuda()
        state3 = torch.zeros((b_num, 32, int(img_h / 4), int(img_w / 4))).cuda()
        state4 = torch.zeros((b_num, 64, int(img_h / 8), int(img_w / 8))).cuda()

        # initialize variables
        exp_sum = torch.zeros((b_num, 1, img_h*2, img_w*2)).cuda()
        depth_image = torch.zeros((b_num, 1, img_h*2, img_w*2)).cuda()
        max_prob_image = torch.zeros((b_num, 1, img_h*2, img_w*2)).cuda()

        for d in range(num_depth):
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
            depth_value = depth_values[:, d:d + 1]
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
            del ref_volume
            for src_fea, src_proj in zip(src_features, src_projs):
                # warpped features
                warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_value)
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                del warped_volume

            # aggregate multiple feature volumes by variance
            volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            volume_variance = volume_variance.squeeze(2)

            # step 3. cost volume regularization
            reg_cost, state1, state2, state3, state4 = self.cost_regularization(volume_variance, state1, state2, state3, state4)
            prob = reg_cost.exp()

            update_flag_image = (max_prob_image < prob).float()
            new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
            # update the best
            # new_depth_image = update_flag_image * depth_value + (1 - update_flag_image) * depth_image
            # update the sum_avg
            new_depth_image = depth_value * prob + depth_image

            max_prob_image = new_max_prob_image
            depth_image = new_depth_image
            exp_sum = exp_sum + prob

        # get output
        # update the best
        #forward_prob_map = (max_prob_image/(exp_sum + 1e-7)).squeeze(1)
        #forward_depth_map = depth_image.squeeze(1)

        # update the sum_avg
        forward_exp_sum = exp_sum + 1e-10
        forward_depth_map = (depth_image / forward_exp_sum).squeeze(1)
        forward_prob_map = (max_prob_image/ forward_exp_sum).squeeze(1)

        return {"depth": forward_depth_map, "photometric_confidence": forward_prob_map}

