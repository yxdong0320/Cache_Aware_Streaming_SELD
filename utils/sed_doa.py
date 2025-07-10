import numpy as np
import torch
import torch.nn as nn
import os
import random
import torch.nn.functional as F

def process_foa_input_sed_doa_labels(feat, label):
    mel_bins = 64
    #nb_ch = 1
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label

def process_foa_input_sed_doa(feat):
    mel_bins = 64
    #nb_ch = 1
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat

def process_foa_input_128d_sed_doa_labels(feat, label):
    mel_bins = 128
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label

def process_foa_input_ssast_data_labels(feat, label):
    mel_bins = 128
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    feat = feat[0, :, :]
    return feat, label


def process_foa_input_sed_labels(feat, label):
    nb_classes = 13
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat, label[:,:nb_classes]

class SedDoaResult():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
    def add_item(self, wav_name, sed_pred, doa_pred):
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        for frame_cnt in range(sed_pred.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt]>0.5:
                # if sed_pred[frame_cnt][class_cnt] > self.class_thre[class_cnt]:    
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])
    
    def add_items(self, wav_names, net_output):
        sed = net_output[:,:,:13]
        # doa = net_output[:,:,13:]
        doa = net_output[:,:,13:52]
        if isinstance(sed, torch.Tensor):
            sed = sed.detach().cpu().numpy()
        if isinstance(doa, torch.Tensor):
            doa = doa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, sed[b], doa[b])

    def get_result(self):
        return self.output_dict

class SedDoaResult_Streaming_Inf():
    def __init__(self) -> None:
        self.output_dict = {}
        # self.class_thre = [0.7, 0.7, 0.7, 0.45, 0.6, 0.3, 0.65, 0.55, 0.65, 0.7, 0.3, 0.7, 0.7]
        # self.class_thre = [0.6, 0.45, 0.6, 0.45, 0.4, 0.5, 0.25, 0.5, 0.4, 0.4, 0.2, 0.55, 0.55]
    def add_item(self, csv_name, sed_pred, doa_pred):
        # pdb.set_trace()
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        for frame_cnt in range(sed_pred.shape[0]): # [500,13]
            output_dict_frame_cnt = frame_cnt
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt]>0.5:
                # if sed_pred[frame_cnt][class_cnt] > self.class_thre[class_cnt]:    
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])
    
    def add_items(self, wav_name, net_output):
        sed = net_output[:,:,:13]
        doa = net_output[:,:,13:52]
        sed = sed.squeeze(0)
        doa = doa.squeeze(0)
        if isinstance(sed, torch.Tensor):
            sed = sed.detach().cpu().numpy()
        if isinstance(doa, torch.Tensor):
            doa = doa.detach().cpu().numpy()
        self.add_item(wav_name, sed, doa)

    def get_result(self):
        return self.output_dict

class SedDoaResult_Class_Thre():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
        self.class_thre = [0.7, 0.7, 0.7, 0.45, 0.6, 0.3, 0.65, 0.55, 0.65, 0.7, 0.3, 0.7, 0.7]
    def add_item(self, wav_name, sed_pred, doa_pred):
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        for frame_cnt in range(sed_pred.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred.shape[1]):
                # if sed_pred[frame_cnt][class_cnt]>0.5:
                if sed_pred[frame_cnt][class_cnt] > self.class_thre[class_cnt]:    
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])
    
    def add_items(self, wav_names, net_output):
        sed = net_output[:,:,:13]
        # doa = net_output[:,:,13:]
        doa = net_output[:,:,13:52]
        if isinstance(sed, torch.Tensor):
            sed = sed.detach().cpu().numpy()
        if isinstance(doa, torch.Tensor):
            doa = doa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, sed[b], doa[b])

    def get_result(self):
        return self.output_dict
    
class SedDoaResult_Streaming_Inf_Class_Thre():
    def __init__(self) -> None:
        self.output_dict = {}
        self.class_thre = [0.7, 0.7, 0.7, 0.45, 0.6, 0.3, 0.65, 0.55, 0.65, 0.7, 0.3, 0.7, 0.7]
    def add_item(self, csv_name, sed_pred, doa_pred):
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        for frame_cnt in range(sed_pred.shape[0]): # [500,13]
            output_dict_frame_cnt = frame_cnt
            for class_cnt in range(sed_pred.shape[1]):
                # if sed_pred[frame_cnt][class_cnt]>0.5:
                if sed_pred[frame_cnt][class_cnt] > self.class_thre[class_cnt]:    
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])
    
    def add_items(self, wav_name, net_output):
        sed = net_output[:,:,:13]
        doa = net_output[:,:,13:52]
        sed = sed.squeeze(0)
        doa = doa.squeeze(0)
        if isinstance(sed, torch.Tensor):
            sed = sed.detach().cpu().numpy()
        if isinstance(doa, torch.Tensor):
            doa = doa.detach().cpu().numpy()
        self.add_item(wav_name, sed, doa)

    def get_result(self):
        return self.output_dict

class SedDoaLoss(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss
    
class SedDoaLoss_SedClass(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0], class_weights=None):
        super().__init__()
        self.criterion_sed = nn.BCELoss(reduction='none')  # 不进行自动求平均，保留所有样本和类别的损失
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
        self.class_weights = class_weights if class_weights is not None else torch.ones(13)  # 默认每个类别的权重为1.0
    
    def forward(self, output, target):
        sed_out = output[:, :, :13]
        doa_out = output[:, :, 13:]
        sed_label = target[:, :, :13]
        doa_label = target[:, :, 13:52]

        # 计算每个类别的损失
        loss_sed_per_class = self.criterion_sed(sed_out, sed_label)
        
        # 将类别权重应用到每个类别的损失上
        weighted_loss_sed = loss_sed_per_class * self.class_weights.to(loss_sed_per_class.device)
        
        # 对所有类别的损失进行求平均
        loss_sed = weighted_loss_sed.mean()

        sed_label_repeat = sed_label.repeat(1, 1, 3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)

        # 组合最终损失
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss

class SedDoaKLLoss(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:]
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        loss_kl_sub1 = (sed_label*torch.log(1e-7+sed_label/(1e-7+sed_out))).mean()
        loss_kl_sub2 = ((1-sed_label)*torch.log(1e-7+(1-sed_label)/(1e-7+1-sed_out))).mean()
        loss_sed = loss_kl_sub1 + loss_kl_sub2
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss

class SedDoaKLLoss_2(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:]
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label * sed_label_repeat)
        loss_kl_sub1 = (sed_label*torch.log(1e-7+sed_label/(1e-7+sed_out))).mean()
        loss_kl_sub2 = ((1-sed_label)*torch.log(1e-7+(1-sed_label)/(1e-7+1-sed_out))).mean()
        loss_sed = loss_kl_sub1 + loss_kl_sub2
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss
    
class SemanticRepresentationDistillationLoss_KLLoss_2(nn.Module):
    """
    语义表示蒸馏损失 - 基于SedDoaKLLoss_2的形式
    """
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: 学生适配特征通过教师分类头的输出 [B, T, 39]
            teacher_output: 教师特征通过教师分类头的输出 [B, T, 39]
        """
        # 分离SED和DOA部分
        student_sed = student_output[:,:,:13]
        student_doa = student_output[:,:,13:]
        teacher_sed = teacher_output[:,:,:13]
        teacher_doa = teacher_output[:,:,13:]
        
        # DOA损失计算
        teacher_sed_repeat = teacher_sed.repeat(1,1,3)  # 适配DOA维度
        loss_doa = self.criterion_doa(student_doa * teacher_sed_repeat, teacher_doa * teacher_sed_repeat)
        
        # SED损失计算
        loss_kl_sub1 = (teacher_sed*torch.log(1e-7+teacher_sed/(1e-7+student_sed))).mean()
        loss_kl_sub2 = ((1-teacher_sed)*torch.log(1e-7+(1-teacher_sed)/(1e-7+1-student_sed))).mean()
        loss_sed = loss_kl_sub1 + loss_kl_sub2
        
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss
    
class SemanticRepresentationDistillationLoss(nn.Module):
    """
    语义表示蒸馏损失 (Semantic Representational Distillation Loss)
    使用教师模型的分类器作为语义批评者来评估学生表示
    """
    def __init__(self, loss_weight=0.1, distance_type='mse'):
        super().__init__()
        self.loss_weight = loss_weight
        self.distance_type = distance_type
        
    def forward(self, student_features, teacher_features, teacher_classifier):
        """
        计算SRD损失
        
        Args:
            student_features: 学生模型的特征表示 [B, T, D]
            teacher_features: 教师模型的特征表示 [B, T, D] 
            teacher_classifier: 教师模型的分类器
            
        Returns:
            srd_loss: 语义表示蒸馏损失
        """
        # 获取教师模型的logits
        with torch.no_grad():
            teacher_logits = teacher_classifier(teacher_features)
            
        # 通过教师分类器获取学生特征的cross-network logits
        student_cross_logits = teacher_classifier(student_features)
        
        # 计算损失
        if self.distance_type == 'mse':
            # MSE损失 - 直接在logit空间对齐
            srd_loss = F.mse_loss(student_cross_logits, teacher_logits.detach())
        elif self.distance_type == 'kl':
            # KL散度损失 - 在概率空间对齐
            teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
            student_probs = F.log_softmax(student_cross_logits, dim=-1)
            srd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")
            
        return srd_loss * self.loss_weight

class SedDoaKLLoss_3(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:]
        sed_label = (target[:,:,:13] > 0.5) * 1.0
        doa_label = target[:,:,13:]
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label * sed_label_repeat)
        loss_kl_sub1 = (sed_label*torch.log(1e-7+sed_label/(1e-7+sed_out))).mean()
        loss_kl_sub2 = ((1-sed_label)*torch.log(1e-7+(1-sed_label)/(1e-7+1-sed_out))).mean()
        loss_sed = loss_kl_sub1 + loss_kl_sub2
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss
    
class HiddenStateMSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, teacher_hidden_states, student_hidden_states):
        """
        计算教师模型和学生模型隐藏层状态之间的MSE损失
        
        Args:
            teacher_hidden_states: 教师模型各层的隐藏状态列表
            student_hidden_states: 学生模型各层的隐藏状态列表
            
        Returns:
            loss: 加权平均的MSE损失
        """
        total_loss = 0
        num_layers = len(teacher_hidden_states)
        
        for i in range(num_layers):
            layer_loss = self.mse_loss(teacher_hidden_states[i], student_hidden_states[i])
            total_loss += layer_loss
            
        # 计算所有层的平均损失并应用权重
        return (total_loss / num_layers) * self.loss_weight
    
class HiddenStateMSELoss_weighted(nn.Module):
    def __init__(self, loss_weight=1.0, layer_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse_loss = nn.MSELoss()
        self.layer_weights = layer_weights
        
    def forward(self, teacher_hidden_states, student_hidden_states):
        """
        计算教师模型和学生模型隐藏层状态之间的MSE损失
        
        Args:
            teacher_hidden_states: 教师模型各层的隐藏状态列表
            student_hidden_states: 学生模型各层的隐藏状态列表
            
        Returns:
            loss: 加权平均的MSE损失
        """
        # layer_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 最后一层权重减半

        total_loss = 0
        num_layers = len(teacher_hidden_states)
        
        for i in range(num_layers):
            layer_loss = self.mse_loss(teacher_hidden_states[i], student_hidden_states[i])
            total_loss += layer_loss * self.layer_weights[i]
            
        # 计算所有层的平均损失并应用权重
        return (total_loss / sum(self.layer_weights)) * self.loss_weight

class AttentionMapMSELoss(nn.Module):
    def __init__(self, loss_weight=0.05):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, teacher_attn_maps, student_attn_maps):
        """
        计算教师模型和学生模型注意力图之间的MSE损失
        
        Args:
            teacher_attn_maps: 教师模型各层的注意力图列表
            student_attn_maps: 学生模型各层的注意力图列表
            
        Returns:
            loss: 加权平均的MSE损失
        """
        total_loss = 0
        num_layers = len(teacher_attn_maps)
        
        for i in range(num_layers):
            # 注意：这里假设注意力图有相同的维度
            # 如果维度不同，需要进行截断或填充处理
            layer_loss = self.mse_loss(teacher_attn_maps[i], student_attn_maps[i])
            total_loss += layer_loss
            
        # 计算所有层的平均损失并应用权重
        return (total_loss / num_layers) * self.loss_weight
    
class HiddenStateMSELoss_norm(nn.Module):
    def __init__(self, loss_weight=0.01, normalize=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        
    def forward(self, teacher_hidden_states, student_hidden_states):
        total_loss = 0
        num_layers = len(teacher_hidden_states)
        
        # 可选：为不同层设置不同权重
        layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 最后一层权重减半
        
        for i in range(num_layers):
            t_hidden = teacher_hidden_states[i]
            s_hidden = student_hidden_states[i]
            
            if self.normalize:
                # 对特征进行归一化
                t_norm = torch.norm(t_hidden, p=2)
                s_norm = torch.norm(s_hidden, p=2)
                t_hidden = t_hidden / (t_norm + 1e-6)
                s_hidden = s_hidden / (s_norm + 1e-6)
            
            # 计算层级损失
            layer_loss = F.mse_loss(t_hidden, s_hidden)
            # 应用层级权重
            total_loss += layer_loss * layer_weights[i]
            
        # 计算加权平均损失
        return (total_loss / sum(layer_weights)) * self.loss_weight
    
class SimpleAttentionDivergenceLoss(nn.Module):
    def __init__(self, loss_weight=5.0):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, teacher_attn_maps, student_attn_maps):
        total_loss = 0
        num_layers = len(teacher_attn_maps)
        
        for i in range(num_layers):
            t_attn = teacher_attn_maps[i]
            s_attn = student_attn_maps[i]
            
            # 对注意力图进行归一化处理
            t_norm = torch.norm(t_attn, p=2)
            s_norm = torch.norm(s_attn, p=2)
            t_attn_norm = t_attn / (t_norm + 1e-6)
            s_attn_norm = s_attn / (s_norm + 1e-6)
            
            # 计算归一化后的MSE损失
            layer_loss = F.mse_loss(t_attn_norm, s_attn_norm)
            total_loss += layer_loss
            
        return (total_loss / num_layers) * self.loss_weight
    
class HiddenStateCosineLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, teacher_hidden_states, student_hidden_states):
        total_loss = 0
        num_layers = len(teacher_hidden_states)
        
        layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
        
        for i in range(num_layers):
            t_hidden = teacher_hidden_states[i]
            s_hidden = student_hidden_states[i]
            
            # 计算余弦距离 (1 - 余弦相似度)
            # 归一化已经内置在cosine_similarity中
            cos_sim = F.cosine_similarity(t_hidden.view(t_hidden.size(0), -1), 
                                          s_hidden.view(s_hidden.size(0), -1), 
                                          dim=1)
            
            # 余弦距离的范围是0-2，通常值会更大
            cos_dist = (1 - cos_sim).mean()
            total_loss += cos_dist * layer_weights[i]
            
        return (total_loss / sum(layer_weights)) * self.loss_weight