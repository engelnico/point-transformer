import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction


#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020 
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)

def create_rFF(channel_list, input_dim):
    rFF = nn.ModuleList([nn.Conv2d(in_channels=channel_list[i], 
                                   out_channels=channel_list[i+1],
                                   kernel_size=(1,1)) for i in range(len(channel_list) - 1)])
    rFF.insert(0, nn.Conv2d(in_channels=1, 
                            out_channels=channel_list[0], 
                            kernel_size=(input_dim,1)))

    return rFF

def create_rFF3d(channel_list, num_points, dim):
    rFF = nn.ModuleList([nn.Conv3d(in_channels=channel_list[i], 
                                   out_channels=channel_list[i+1],
                                   kernel_size=(1,1,1)) for i in range(len(channel_list) - 1)])
    rFF.insert(0, nn.Conv3d(in_channels=1, 
                            out_channels=channel_list[0], 
                            kernel_size=(1, num_points, dim)))

    return rFF

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss


 
class Point_Transformer(nn.Module):
    def __init__(self, config):
        super(Point_Transformer, self).__init__()
        
        # Parameters
        self.actv_fn = Mish()

        self.p_dropout = config['dropout']
        self.norm_channel = config['use_normals']
        self.input_dim = 6 if config['use_normals'] else 3
        self.num_sort_nets = config['M']
        self.top_k = config['K']
        self.d_model = config['d_m']
 

        self.radius_max_points = 16
        self.radius = 0.1


        ## Create rFF to project input points to latent feature space
        ## Local Feature Generation --> rFF
        self.sort_ch = [64, 128]
        self.sort_cnn = create_rFF(self.sort_ch, self.input_dim)
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList([nn.BatchNorm2d(num_features=self.sort_ch[i]) for i in range(len(self.sort_ch))])
        
        ## Create Self-Attention layer
        ##  Local Feature Generation --> A^self
        self.input_selfattention_layer = nn.TransformerEncoderLayer(self.sort_ch[-1], nhead=8)

        self.sortnets = nn.ModuleList([SortNet(self.sort_ch[-1],
                                                  self.input_dim,
                                                  self.actv_fn,
                                                  top_k = self.top_k) for _ in range(self.num_sort_nets)])
     

        ## Create ball query search + feature aggregation of SortNet
        ## ball query + feat. agg
        ## Note: We put the ball query search and feature aggregation outside the SortNet implementation as it greatly decreased computational time
        ## This however, does not change the method in any way
        self.radius_ch = [128, 256, self.d_model-1-self.input_dim]
        self.radius_cnn = create_rFF3d(self.radius_ch, self.radius_max_points+1, self.input_dim)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList([nn.BatchNorm3d(num_features=self.radius_ch[i]) for i in range(len(self.radius_ch))])

        ## Create set abstraction (MSG)
        ##  Global Feature Generation --> Set Abstraction (MSG)
        out_points = 128
        in_channel = 3 if self.norm_channel else 0 

        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.2, 0.4], [16, 32, 64], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(out_points, [0.2, 0.4, 0.6], [32, 64, 128], 320,[[32, 64, 128], [64, 64, 128], [64, 128, 253]])

        ## Create Local-Global Attention
        ##  A^LG
        out_dim = 64
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=8)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=8, last_dim=out_dim)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 1, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=8, dim_feedforward=512, num_encoder_layers=1, num_decoder_layers=1, custom_decoder=self.custom_decoder)
        self.transformer_model.apply(init_weights)

        # Create Classification Head

        dim_flatten = out_dim * self.num_sort_nets * self.top_k
        self.flatten_linear_ch = [dim_flatten, 512, 128, 40]
        self.flatten_linear = nn.ModuleList([nn.Linear(in_features=self.flatten_linear_ch[i], 
                                                   out_features=self.flatten_linear_ch[i+1]) for i in range(len(self.flatten_linear_ch) - 1)])
        self.flatten_linear.apply(init_weights)
        self.flatten_bn = nn.ModuleList([nn.BatchNorm1d(num_features=self.flatten_linear_ch[i+1]) for i in range(len(self.flatten_linear_ch) - 1)])

        ## Create Dropout layers for classification heads
        self.dropout1 = nn.Dropout(p=self.p_dropout)
        self.dropout2 = nn.Dropout(p=self.p_dropout)
        self.dropout3 = nn.Dropout(p=self.p_dropout)
        self.dropout4 = nn.Dropout(p=self.p_dropout)


    def forward(self, input):
  
        #############################################
        ## Global Features 
        #############################################
        xyz = input

        B, _, _ = xyz.shape
        
        if self.norm_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        ## Set Abstraction with MSG
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        global_feat = torch.cat([l2_xyz, l2_points], dim=1)

        #############################################
        ## Local Features
        #############################################
        
        x_local = input.unsqueeze(dim=1)

        # Project to latent feature dim
        for i, sort_conv in enumerate(self.sort_cnn):
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))
        x_local = x_local.transpose(2,1)

        # Perform Self Attention
        x_local = x_local.squeeze(dim=1)
        x_local = x_local.permute(2,0,1)
        x_local = self.input_selfattention_layer(x_local)
        x_local = x_local.permute(1,2,0)
        x_local = x_local.unsqueeze(dim=1)
        # Concatenate outputs of SortNet
        x_local_sorted = torch.cat([sortnet(x_local, input)[0] for sortnet in self.sortnets], dim=-1)

        # this corresponds to s^j_i
        x_local_scores = x_local_sorted[: ,6:, :].permute(0,2,1)
        # this corresponds to p^j_i
        x_local_sorted = x_local_sorted[:, :6, :].permute(0,2,1)

        # Perform ball query search with feature aggregation
        all_points = input.squeeze(dim=1).permute(0,2,1)
        query_points = x_local_sorted
        radius_indices = query_ball_point(self.radius, self.radius_max_points,all_points[:,:,:3], query_points[:,:,:3])
        
        radius_points = index_points(all_points, radius_indices) 
        radius_centroids = query_points.unsqueeze(dim=-2)

        # This corresponds to g^j
        radius_grouped = torch.cat([radius_centroids, radius_points], dim=-2).unsqueeze(dim=1)

        for i, radius_conv in enumerate(self.radius_cnn):
            bn = self.radius_bn[i]
            radius_grouped = self.actv_fn(bn(radius_conv(radius_grouped)))

        radius_grouped = radius_grouped.squeeze()
        # This corresponds to f^j_i
        radius_grouped = torch.cat([x_local_sorted.transpose(2,1), radius_grouped, x_local_scores.transpose(2,1)], dim=1)

        #############################################
        ## Point Transformer
        #############################################

        source = global_feat.permute(2,0,1)
        target = radius_grouped.permute(2,0,1)

        embedding = self.transformer_model(source, target)
        embedding = embedding.permute(1, 2, 0)

        #############################################
        ## Classification
        #############################################
        output = torch.flatten(embedding, start_dim=1)

        for i, linear in enumerate(self.flatten_linear):
            bn = self.flatten_bn[i]
            # Use activation function and batch norm for every layer except last
            if i < len(self.flatten_linear) - 1:
                output = self.actv_fn(bn(linear(output)))
                if i == 0:
                    output = self.dropout1(output)
                elif i == 1:
                    output = self.dropout2(output)
                elif i == 2:
                    output = self.dropout3(output)
                elif i == 3:
                    output = self.dropout4(output)
            else:
                output = linear(output)

        output = F.log_softmax(output, -1)
   
        return output


class SortNet(nn.Module):
    def __init__(self, num_feat, input_dims, actv_fn=F.relu, top_k = 5):
        super(SortNet, self).__init__()

        self.num_feat = num_feat
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = top_k

        self.feat_channels =  [64, 16, 1]
        self.feat_generator = create_rFF(self.feat_channels, num_feat)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList([nn.BatchNorm2d(num_features=self.feat_channels[i]) for i in range(len(self.feat_channels))])

    def forward(self, sortvec, input):
        
        top_k = self.top_k
        batch_size = input.shape[0]
        feat_dim = input.shape[1]

        for i, conv in enumerate(self.feat_generator):
            bn = self.feat_bn[i]
            sortvec = self.actv_fn(bn(conv(sortvec)))
        sortvec = sortvec.squeeze(dim=1)


        topk = torch.topk(sortvec, k=top_k, dim=-1)
        indices = topk.indices.squeeze()

        sorted_input = index_points(input.permute(0,2,1), indices).permute(0,2,1)
        sorted_score = index_points(sortvec.permute(0,2,1), indices).permute(0,2,1)

        feat = torch.cat([sorted_input, sorted_score], dim=1)      

        return feat, indices

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, last_dim=64, dropout=0.1, activation=F.relu):
        super(PTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, last_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        return tgt


class PTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, last_layer, norm=None):
        super(PTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.last_layer = last_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)
            
        output = self.last_layer(output, memory)

        return output

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm?
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
