import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import argparse
from torch.utils.data import DataLoader
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入你的模块
from lmdb_data_loader_A import LmdbDataset
from models.cache_resnet_conformer_TS import ResnetConformer_sed_doa_nopool_TS_hidstate_att_CMAPC
from models.resnet_conformer_audio_T import ResnetConformer_sed_doa_nopool_original
from utils.sed_doa import process_foa_input_sed_doa_labels

class KnowledgeDistillationVisualizer:
    def __init__(self, config_path, student_model_path, teacher_model_path, student_finetuned_path):
        """
        初始化可视化器
        Args:
            config_path: 配置文件路径
            student_model_path: 预训练学生模型路径
            teacher_model_path: 教师模型路径  
            student_finetuned_path: 微调后学生模型路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.args = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.teacher_model = self._load_teacher_model(teacher_model_path)
        self.student_model = self._load_student_model(student_model_path)
        self.student_finetuned = self._load_student_model(student_finetuned_path)
        
        # 加载数据
        self.test_dataloader = self._load_test_data()
        
        print("所有模型和数据加载完成！")
        
    def _load_teacher_model(self, model_path):
        """加载教师模型"""
        teacher_model = ResnetConformer_sed_doa_nopool_original(
            in_channel=7, in_dim=64, out_dim=39
        ).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        teacher_model.load_state_dict(state_dict, strict=False)
        teacher_model.eval()
        
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        print(f"教师模型加载完成: {model_path}")
        return teacher_model
        
    def _load_student_model(self, model_path):
        """加载学生模型"""
        student_model = ResnetConformer_sed_doa_nopool_TS_hidstate_att_CMAPC(
            in_channel=self.args['model']['in_channel'],
            in_dim=self.args['model']['in_dim'],
            out_dim=self.args['model']['out_dim'],
            att_context_size=self.args['model']['att_context_size'],
            num_conformer_layer=self.args['model']['num_conformer_layer'],
            encoder_dim=self.args['model']['encoder_dim'],
            use_hidden_distill=True,
            use_attn_distill=True,
            use_CMAPC_distill=True,
            CMAPC_future_steps=self.args['model']['apc_future_steps']
        ).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        student_model.load_state_dict(state_dict, strict=False)
        student_model.eval()
        
        for param in student_model.parameters():
            param.requires_grad = False
            
        print(f"学生模型加载完成: {model_path}")
        return student_model
        
    def _load_test_data(self):
        """加载测试数据"""
        test_split = [4]
        test_dataset = LmdbDataset(
            self.args['data']['test_lmdb_dir'], 
            test_split,
            normalized_features_wts_file=self.args['data']['norm_file'],
            ignore=self.args['data']['test_ignore'],
            segment_len=self.args['data']['segment_len'],
            data_process_fn=process_foa_input_sed_doa_labels
        )
        
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,  # 设置为1便于分析
            shuffle=False,
            num_workers=0,
            collate_fn=test_dataset.collater
        )
        
        print(f"测试数据加载完成，共 {len(test_dataset)} 个样本")
        return test_dataloader
        
    def extract_features(self, model, input_data, model_type='student'):
        """提取模型的hidden states和attention maps"""
        with torch.no_grad():
            if model_type == 'teacher':
                # 教师模型特征提取
                hidden_states = []
                attention_maps = []
                
                conv_outputs = model.resnet(input_data)
                N, C, T, W = conv_outputs.shape
                conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
                conformer_outputs = model.input_projection(conv_outputs)
                
                for layer in model.conformer_layers:
                    conformer_outputs, attn_weights = layer(conformer_outputs, return_attention=True)
                    hidden_states.append(conformer_outputs.cpu())
                    attention_maps.append(attn_weights.cpu())
                
                # 最终输出
                outputs = conformer_outputs.permute(0, 2, 1)
                outputs = model.t_pooling(outputs)
                outputs = outputs.permute(0, 2, 1)
                
                sed = model.sed_out_layer(outputs)
                doa = model.out_layer(outputs)
                final_output = torch.cat((sed, doa), dim=-1)
                
            else:  # student or student_finetuned
                # 学生模型特征提取
                model_outputs = model(input_data)
                final_output = model_outputs[0]
                
                # 提取hidden states和attention maps
                if len(model_outputs) >= 4:
                    teacher_hidden, student_hidden = model_outputs[2]
                    teacher_attns, student_attns = model_outputs[3]
                    hidden_states = [h.cpu() for h in student_hidden]
                    attention_maps = [a.cpu() for a in student_attns]
                else:
                    hidden_states = []
                    attention_maps = []
                    
        return {
            'hidden_states': hidden_states,
            'attention_maps': attention_maps,
            'final_output': final_output.cpu()
        }
        
    def plot_hidden_state_similarity(self, teacher_features, student_features, student_ft_features, sample_idx):
        """绘制Hidden State相似性分析图"""
        plt.figure(figsize=(15, 10))
        
        # 1. 逐层相似性热力图
        plt.subplot(2, 3, 1)
        similarities_before = []
        similarities_after = []
        
        for i, (t_h, s_h, sf_h) in enumerate(zip(teacher_features['hidden_states'], 
                                                 student_features['hidden_states'],
                                                 student_ft_features['hidden_states'])):
            # 计算平均池化后的特征相似度
            t_h_pooled = F.adaptive_avg_pool1d(t_h.permute(0,2,1), 1).squeeze(-1)
            s_h_pooled = F.adaptive_avg_pool1d(s_h.permute(0,2,1), 1).squeeze(-1)
            sf_h_pooled = F.adaptive_avg_pool1d(sf_h.permute(0,2,1), 1).squeeze(-1)
            
            sim_before = F.cosine_similarity(t_h_pooled, s_h_pooled, dim=1).mean().item()
            sim_after = F.cosine_similarity(t_h_pooled, sf_h_pooled, dim=1).mean().item()
            
            similarities_before.append(sim_before)
            similarities_after.append(sim_after)
        
        layers = list(range(1, len(similarities_before) + 1))
        plt.plot(layers, similarities_before, 'o-', label='蒸馏前', linewidth=2, markersize=6)
        plt.plot(layers, similarities_after, 's-', label='蒸馏后', linewidth=2, markersize=6)
        plt.xlabel('Conformer层')
        plt.ylabel('余弦相似度')
        plt.title('逐层Hidden State相似度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. t-SNE降维可视化
        plt.subplot(2, 3, 2)
        # 选择最后一层进行降维
        if teacher_features['hidden_states'] and student_features['hidden_states']:
            t_last = teacher_features['hidden_states'][-1][0].flatten().unsqueeze(0)  # [1, D]
            s_last = student_features['hidden_states'][-1][0].flatten().unsqueeze(0)
            sf_last = student_ft_features['hidden_states'][-1][0].flatten().unsqueeze(0)
            
            # 合并特征进行降维
            combined_features = torch.cat([t_last, s_last, sf_last], dim=0).numpy()

            if combined_features.shape[1] > 50:  # 256 > 50，会进入这个分支
                n_components = min(30, combined_features.shape[0] - 1)  # 3-1=2，所以n_components=2
                pca = PCA(n_components=n_components)
                combined_features = pca.fit_transform(combined_features)
            
            # if combined_features.shape[1] > 50:  # 如果维度太高，先用PCA降维
            #     pca = PCA(n_components=50)
            #     combined_features = pca.fit_transform(combined_features)
                
            tsne = TSNE(n_components=2, random_state=42, perplexity=1)
            embedded = tsne.fit_transform(combined_features)
            
            colors = ['red', 'blue', 'green']
            labels = ['教师模型', '学生模型(蒸馏前)', '学生模型(蒸馏后)']
            
            for i, (color, label) in enumerate(zip(colors, labels)):
                plt.scatter(embedded[i, 0], embedded[i, 1], c=color, s=100, label=label, alpha=0.7)
                
            plt.xlabel('t-SNE维度1')
            plt.ylabel('t-SNE维度2')
            plt.title('Hidden State t-SNE可视化')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. 相似度矩阵热力图
        plt.subplot(2, 3, 3)
        if len(similarities_before) > 0:
            sim_matrix = np.array([similarities_before, similarities_after])
            sns.heatmap(sim_matrix, 
                       xticklabels=[f'Layer{i+1}' for i in range(len(similarities_before))],
                       yticklabels=['蒸馏前', '蒸馏后'],
                       annot=True, fmt='.3f', cmap='viridis')
            plt.title('相似度矩阵热力图')
        
        # 4. Hidden State分布对比
        plt.subplot(2, 3, 4)
        if teacher_features['hidden_states'] and student_features['hidden_states']:
            # 选择中间层进行分布对比
            mid_layer = len(teacher_features['hidden_states']) // 2
            
            t_values = teacher_features['hidden_states'][mid_layer][0].flatten().numpy()
            s_values = student_features['hidden_states'][mid_layer][0].flatten().numpy()
            sf_values = student_ft_features['hidden_states'][mid_layer][0].flatten().numpy()
            
            plt.hist(t_values, bins=50, alpha=0.5, label='教师模型', density=True)
            plt.hist(s_values, bins=50, alpha=0.5, label='学生模型(蒸馏前)', density=True)
            plt.hist(sf_values, bins=50, alpha=0.5, label='学生模型(蒸馏后)', density=True)
            
            plt.xlabel('激活值')
            plt.ylabel('密度')
            plt.title(f'第{mid_layer+1}层激活值分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. 层间差异变化
        plt.subplot(2, 3, 5)
        if len(similarities_before) > 1:
            diff_before = np.diff(similarities_before)
            diff_after = np.diff(similarities_after)
            
            layer_transitions = [f'{i}->{i+1}' for i in range(1, len(similarities_before))]
            
            x = np.arange(len(layer_transitions))
            width = 0.35
            
            plt.bar(x - width/2, diff_before, width, label='蒸馏前', alpha=0.7)
            plt.bar(x + width/2, diff_after, width, label='蒸馏后', alpha=0.7)
            
            plt.xlabel('层间转换')
            plt.ylabel('相似度变化')
            plt.title('层间相似度变化')
            plt.xticks(x, layer_transitions, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. 整体改进指标
        plt.subplot(2, 3, 6)
        if similarities_before and similarities_after:
            metrics = ['平均相似度', '最大相似度', '最小相似度', '相似度方差']
            before_metrics = [
                np.mean(similarities_before),
                np.max(similarities_before),
                np.min(similarities_before),
                np.var(similarities_before)
            ]
            after_metrics = [
                np.mean(similarities_after),
                np.max(similarities_after),
                np.min(similarities_after),
                np.var(similarities_after)
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, before_metrics, width, label='蒸馏前', alpha=0.7)
            plt.bar(x + width/2, after_metrics, width, label='蒸馏后', alpha=0.7)
            
            plt.xlabel('指标')
            plt.ylabel('数值')
            plt.title('整体相似度指标对比')
            plt.xticks(x, metrics, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f't_s_finetuned_models_visualize/hidden_state_analysis_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_attention_analysis(self, teacher_features, student_features, student_ft_features, sample_idx):
        """绘制注意力机制对比分析图"""
        plt.figure(figsize=(15, 12))
        
        if not teacher_features['attention_maps'] or not student_features['attention_maps']:
            print("注意力图数据不足，跳过注意力分析")
            return
            
        # 1. 注意力权重相似度热力图
        plt.subplot(3, 3, 1)
        attn_similarities_before = []
        attn_similarities_after = []
        
        for i, (t_attn, s_attn, sf_attn) in enumerate(zip(teacher_features['attention_maps'],
                                                          student_features['attention_maps'],
                                                          student_ft_features['attention_maps'])):
            # 计算注意力图的平均相似度
            t_attn_flat = t_attn.flatten()
            s_attn_flat = s_attn.flatten()
            sf_attn_flat = sf_attn.flatten()
            
            sim_before = F.cosine_similarity(t_attn_flat.unsqueeze(0), s_attn_flat.unsqueeze(0)).item()
            sim_after = F.cosine_similarity(t_attn_flat.unsqueeze(0), sf_attn_flat.unsqueeze(0)).item()
            
            attn_similarities_before.append(sim_before)
            attn_similarities_after.append(sim_after)
        
        layers = list(range(1, len(attn_similarities_before) + 1))
        plt.plot(layers, attn_similarities_before, 'o-', label='蒸馏前', linewidth=2)
        plt.plot(layers, attn_similarities_after, 's-', label='蒸馏后', linewidth=2)
        plt.xlabel('Conformer层')
        plt.ylabel('注意力相似度')
        plt.title('逐层注意力相似度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-4. 展示特定层的注意力图对比
        layers_to_show = [0, len(teacher_features['attention_maps'])//2, -1]  # 第一层、中间层、最后一层
        layer_names = ['第1层', f'第{len(teacher_features["attention_maps"])//2 + 1}层', '最后一层']
        
        for idx, (layer_idx, layer_name) in enumerate(zip(layers_to_show, layer_names)):
            # 教师模型注意力图
            plt.subplot(3, 3, 2 + idx*3)
            t_attn = teacher_features['attention_maps'][layer_idx][0, 0].numpy()  # [seq_len, seq_len]
            sns.heatmap(t_attn, cmap='Blues', cbar=True)
            plt.title(f'教师模型 - {layer_name}')
            plt.xlabel('Key位置')
            plt.ylabel('Query位置')
            
            # 学生模型(蒸馏前)注意力图
            plt.subplot(3, 3, 3 + idx*3)
            s_attn = student_features['attention_maps'][layer_idx][0, 0].numpy()
            sns.heatmap(s_attn, cmap='Greens', cbar=True)
            plt.title(f'学生模型(蒸馏前) - {layer_name}')
            plt.xlabel('Key位置')
            plt.ylabel('Query位置')
            
            # 学生模型(蒸馏后)注意力图
            plt.subplot(3, 3, 4 + idx*3)
            sf_attn = student_ft_features['attention_maps'][layer_idx][0, 0].numpy()
            sns.heatmap(sf_attn, cmap='Reds', cbar=True)
            plt.title(f'学生模型(蒸馏后) - {layer_name}')
            plt.xlabel('Key位置')
            plt.ylabel('Query位置')
        
        plt.tight_layout()
        plt.savefig(f't_s_finetuned_models_visualize/attention_analysis_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 单独绘制注意力差异分析图
        plt.figure(figsize=(12, 8))
        
        # 1. 注意力模式差异
        plt.subplot(2, 2, 1)
        if attn_similarities_before and attn_similarities_after:
            improvement = np.array(attn_similarities_after) - np.array(attn_similarities_before)
            layers = list(range(1, len(improvement) + 1))
            
            colors = ['red' if x < 0 else 'green' for x in improvement]
            plt.bar(layers, improvement, color=colors, alpha=0.7)
            plt.xlabel('Conformer层')
            plt.ylabel('相似度改进')
            plt.title('各层注意力相似度改进')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
        
        # 2. 注意力熵分析
        plt.subplot(2, 2, 2)
        teacher_entropy = []
        student_entropy = []
        student_ft_entropy = []
        
        for t_attn, s_attn, sf_attn in zip(teacher_features['attention_maps'],
                                          student_features['attention_maps'],
                                          student_ft_features['attention_maps']):
            # 计算注意力分布的熵
            t_entropy = -torch.sum(t_attn * torch.log(t_attn + 1e-8), dim=-1).mean().item()
            s_entropy = -torch.sum(s_attn * torch.log(s_attn + 1e-8), dim=-1).mean().item()
            sf_entropy = -torch.sum(sf_attn * torch.log(sf_attn + 1e-8), dim=-1).mean().item()
            
            teacher_entropy.append(t_entropy)
            student_entropy.append(s_entropy)
            student_ft_entropy.append(sf_entropy)
        
        layers = list(range(1, len(teacher_entropy) + 1))
        plt.plot(layers, teacher_entropy, 'o-', label='教师模型', linewidth=2)
        plt.plot(layers, student_entropy, 's-', label='学生模型(蒸馏前)', linewidth=2)
        plt.plot(layers, student_ft_entropy, '^-', label='学生模型(蒸馏后)', linewidth=2)
        plt.xlabel('Conformer层')
        plt.ylabel('注意力熵')
        plt.title('注意力分布熵对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 注意力集中度分析
        plt.subplot(2, 2, 3)
        teacher_concentration = []
        student_concentration = []
        student_ft_concentration = []
        
        for t_attn, s_attn, sf_attn in zip(teacher_features['attention_maps'],
                                          student_features['attention_maps'],
                                          student_ft_features['attention_maps']):
            # 计算最大注意力权重(注意力集中度)
            t_conc = torch.max(t_attn, dim=-1)[0].mean().item()
            s_conc = torch.max(s_attn, dim=-1)[0].mean().item()
            sf_conc = torch.max(sf_attn, dim=-1)[0].mean().item()
            
            teacher_concentration.append(t_conc)
            student_concentration.append(s_conc)
            student_ft_concentration.append(sf_conc)
        
        layers = list(range(1, len(teacher_concentration) + 1))
        plt.plot(layers, teacher_concentration, 'o-', label='教师模型', linewidth=2)
        plt.plot(layers, student_concentration, 's-', label='学生模型(蒸馏前)', linewidth=2)
        plt.plot(layers, student_ft_concentration, '^-', label='学生模型(蒸馏后)', linewidth=2)
        plt.xlabel('Conformer层')
        plt.ylabel('注意力集中度')
        plt.title('注意力集中度对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 整体注意力指标对比
        plt.subplot(2, 2, 4)
        metrics = ['平均相似度', '相似度标准差', '平均熵', '平均集中度']
        before_values = [
            np.mean(attn_similarities_before),
            np.std(attn_similarities_before),
            np.mean(student_entropy),
            np.mean(student_concentration)
        ]
        after_values = [
            np.mean(attn_similarities_after),
            np.std(attn_similarities_after),
            np.mean(student_ft_entropy),
            np.mean(student_ft_concentration)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, before_values, width, label='蒸馏前', alpha=0.7)
        plt.bar(x + width/2, after_values, width, label='蒸馏后', alpha=0.7)
        
        plt.xlabel('指标')
        plt.ylabel('数值')
        plt.title('整体注意力指标对比')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f't_s_finetuned_models_visualize/attention_difference_analysis_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_representation_analysis(self, teacher_features, student_features, student_ft_features, sample_idx):
        """绘制特征表示分析图"""
        plt.figure(figsize=(15, 10))
        
        # 1. 最终输出特征分布对比
        plt.subplot(2, 3, 1)
        t_output = teacher_features['final_output'][0].flatten().numpy()
        s_output = student_features['final_output'][0].flatten().numpy()
        sf_output = student_ft_features['final_output'][0].flatten().numpy()
        
        plt.hist(t_output, bins=50, alpha=0.5, label='教师模型', density=True)
        plt.hist(s_output, bins=50, alpha=0.5, label='学生模型(蒸馏前)', density=True)
        plt.hist(sf_output, bins=50, alpha=0.5, label='学生模型(蒸馏后)', density=True)
        
        plt.xlabel('输出值')
        plt.ylabel('密度')
        plt.title('最终输出分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 输出相似度对比
        plt.subplot(2, 3, 2)
        sim_before = F.cosine_similarity(
            torch.from_numpy(t_output).unsqueeze(0),
            torch.from_numpy(s_output).unsqueeze(0)
        ).item()
        sim_after = F.cosine_similarity(
            torch.from_numpy(t_output).unsqueeze(0),
            torch.from_numpy(sf_output).unsqueeze(0)
        ).item()
        
        categories = ['蒸馏前', '蒸馏后']
        similarities = [sim_before, sim_after]
        colors = ['lightcoral', 'lightgreen']
        
        bars = plt.bar(categories, similarities, color=colors, alpha=0.7)
        plt.ylabel('余弦相似度')
        plt.title('输出与教师模型相似度')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, sim in zip(bars, similarities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sim:.3f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
        
        # 3. 特征空间可视化(PCA)
        plt.subplot(2, 3, 3)
        if teacher_features['hidden_states'] and student_features['hidden_states']:
            # 收集所有层的特征进行PCA
            all_features = []
            labels = []
            
            for i, (t_h, s_h, sf_h) in enumerate(zip(teacher_features['hidden_states'][:3],  # 只取前3层
                                                    student_features['hidden_states'][:3],
                                                    student_ft_features['hidden_states'][:3])):
                # 平均池化
                t_pooled = F.adaptive_avg_pool1d(t_h.permute(0,2,1), 1).squeeze().numpy()
                s_pooled = F.adaptive_avg_pool1d(s_h.permute(0,2,1), 1).squeeze().numpy()
                sf_pooled = F.adaptive_avg_pool1d(sf_h.permute(0,2,1), 1).squeeze().numpy()
                
                all_features.extend([t_pooled, s_pooled, sf_pooled])
                labels.extend([f'T-L{i+1}', f'S-L{i+1}', f'SF-L{i+1}'])
            
            all_features = np.array(all_features)
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(all_features)
            
            colors_map = {'T': 'red', 'S': 'blue', 'SF': 'green'}
            for i, (feature, label) in enumerate(zip(pca_features, labels)):
                model_type = label.split('-')[0]
                plt.scatter(feature[0], feature[1], 
                           c=colors_map[model_type], 
                           label=model_type if i % 3 == 0 else "", 
                           alpha=0.7, s=60)
                plt.annotate(label, (feature[0], feature[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.title('特征空间PCA可视化')
            
            # 手动添加图例
            import matplotlib.patches as mpatches
            teacher_patch = mpatches.Patch(color='red', label='教师模型')
            student_patch = mpatches.Patch(color='blue', label='学生模型(蒸馏前)')
            student_ft_patch = mpatches.Patch(color='green', label='学生模型(蒸馏后)')
            plt.legend(handles=[teacher_patch, student_patch, student_ft_patch])
            plt.grid(True, alpha=0.3)
        
        # 4. 层间特征变化分析
        plt.subplot(2, 3, 4)
        if len(teacher_features['hidden_states']) > 1:
            teacher_changes = []
            student_changes = []
            student_ft_changes = []
            
            for i in range(len(teacher_features['hidden_states']) - 1):
                # 计算连续层之间的余弦相似度变化
                t_curr = F.adaptive_avg_pool1d(teacher_features['hidden_states'][i].permute(0,2,1), 1).squeeze()
                t_next = F.adaptive_avg_pool1d(teacher_features['hidden_states'][i+1].permute(0,2,1), 1).squeeze()
                t_change = F.cosine_similarity(t_curr.unsqueeze(0), t_next.unsqueeze(0)).item()
                
                s_curr = F.adaptive_avg_pool1d(student_features['hidden_states'][i].permute(0,2,1), 1).squeeze()
                s_next = F.adaptive_avg_pool1d(student_features['hidden_states'][i+1].permute(0,2,1), 1).squeeze()
                s_change = F.cosine_similarity(s_curr.unsqueeze(0), s_next.unsqueeze(0)).item()
                
                sf_curr = F.adaptive_avg_pool1d(student_ft_features['hidden_states'][i].permute(0,2,1), 1).squeeze()
                sf_next = F.adaptive_avg_pool1d(student_ft_features['hidden_states'][i+1].permute(0,2,1), 1).squeeze()
                sf_change = F.cosine_similarity(sf_curr.unsqueeze(0), sf_next.unsqueeze(0)).item()
                
                teacher_changes.append(t_change)
                student_changes.append(s_change)
                student_ft_changes.append(sf_change)
            
            transitions = [f'{i+1}->{i+2}' for i in range(len(teacher_changes))]
            
            plt.plot(transitions, teacher_changes, 'o-', label='教师模型', linewidth=2)
            plt.plot(transitions, student_changes, 's-', label='学生模型(蒸馏前)', linewidth=2)
            plt.plot(transitions, student_ft_changes, '^-', label='学生模型(蒸馏后)', linewidth=2)
            
            plt.xlabel('层间转换')
            plt.ylabel('层间相似度')
            plt.title('层间特征变化模式')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. 特征方差分析
        plt.subplot(2, 3, 5)
        if teacher_features['hidden_states'] and student_features['hidden_states']:
            teacher_vars = []
            student_vars = []
            student_ft_vars = []
            
            for t_h, s_h, sf_h in zip(teacher_features['hidden_states'],
                                     student_features['hidden_states'],
                                     student_ft_features['hidden_states']):
                teacher_vars.append(torch.var(t_h).item())
                student_vars.append(torch.var(s_h).item())
                student_ft_vars.append(torch.var(sf_h).item())
            
            layers = list(range(1, len(teacher_vars) + 1))
            plt.plot(layers, teacher_vars, 'o-', label='教师模型', linewidth=2)
            plt.plot(layers, student_vars, 's-', label='学生模型(蒸馏前)', linewidth=2)
            plt.plot(layers, student_ft_vars, '^-', label='学生模型(蒸馏后)', linewidth=2)
            
            plt.xlabel('Conformer层')
            plt.ylabel('特征方差')
            plt.title('各层特征方差对比')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. 知识传递效果总结
        plt.subplot(2, 3, 6)
        # 计算多个指标的改进情况
        metrics = []
        improvements = []
        
        # 输出相似度改进
        metrics.append('输出相似度')
        improvements.append(sim_after - sim_before)
        
        # 平均hidden state相似度改进
        if teacher_features['hidden_states'] and student_features['hidden_states']:
            avg_sim_before = np.mean([
                F.cosine_similarity(
                    F.adaptive_avg_pool1d(t_h.permute(0,2,1), 1).squeeze().unsqueeze(0),
                    F.adaptive_avg_pool1d(s_h.permute(0,2,1), 1).squeeze().unsqueeze(0)
                ).item()
                for t_h, s_h in zip(teacher_features['hidden_states'], student_features['hidden_states'])
            ])
            avg_sim_after = np.mean([
                F.cosine_similarity(
                    F.adaptive_avg_pool1d(t_h.permute(0,2,1), 1).squeeze().unsqueeze(0),
                    F.adaptive_avg_pool1d(sf_h.permute(0,2,1), 1).squeeze().unsqueeze(0)
                ).item()
                for t_h, sf_h in zip(teacher_features['hidden_states'], student_ft_features['hidden_states'])
            ])
            metrics.append('Hidden相似度')
            improvements.append(avg_sim_after - avg_sim_before)
        
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = plt.bar(metrics, improvements, color=colors, alpha=0.7)
        
        plt.ylabel('改进程度')
        plt.title('知识蒸馏效果总结')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.001 if imp > 0 else -0.005),
                    f'{imp:.3f}', ha='center', 
                    va='bottom' if imp > 0 else 'top')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f't_s_finetuned_models_visualize/feature_representation_analysis_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_analysis(self, num_samples=3):
        """运行完整的知识蒸馏分析"""
        print("开始知识蒸馏可视化分析...")
        
        sample_count = 0
        for batch_idx, data in enumerate(self.test_dataloader):
            if sample_count >= num_samples:
                break
                
            input_data = data['input'].to(self.device)
            print(f"\n分析样本 {sample_count + 1}/{num_samples} (文件: {data['wav_names'][0]})")
            
            # 提取特征
            print("提取教师模型特征...")
            teacher_features = self.extract_features(self.teacher_model, input_data, 'teacher')
            
            print("提取学生模型(蒸馏前)特征...")
            student_features = self.extract_features(self.student_model, input_data, 'student')
            
            print("提取学生模型(蒸馏后)特征...")
            student_ft_features = self.extract_features(self.student_finetuned, input_data, 'student')
            
            # 生成可视化
            print("生成Hidden State相似性分析图...")
            self.plot_hidden_state_similarity(teacher_features, student_features, student_ft_features, sample_count)
            
            print("生成注意力机制对比分析图...")
            self.plot_attention_analysis(teacher_features, student_features, student_ft_features, sample_count)
            
            print("生成特征表示分析图...")
            self.plot_feature_representation_analysis(teacher_features, student_features, student_ft_features, sample_count)
            
            sample_count += 1
            print(f"样本 {sample_count} 分析完成！")
        
        print(f"\n所有 {num_samples} 个样本的可视化分析完成！")
        print("生成的图片文件:")
        for i in range(num_samples):
            print(f"  - hidden_state_analysis_sample_{i}.png")
            print(f"  - attention_analysis_sample_{i}.png") 
            print(f"  - attention_difference_analysis_sample_{i}.png")
            print(f"  - feature_representation_analysis_sample_{i}.png")

def main():
    parser = argparse.ArgumentParser('知识蒸馏可视化分析')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', 
                       help='配置文件名称')
    parser.add_argument('--teacher_model', type=str, required=True,
                       help='教师模型路径')
    parser.add_argument('--student_model', type=str, required=True,
                       help='预训练学生模型路径')
    parser.add_argument('--student_finetuned', type=str, required=True,
                       help='微调后学生模型路径')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='分析的样本数量')
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_path = os.path.join('config', f'{args.config_name}.yaml')
    
    # 创建可视化器
    visualizer = KnowledgeDistillationVisualizer(
        config_path=config_path,
        student_model_path=args.student_model,
        teacher_model_path=args.teacher_model,
        student_finetuned_path=args.student_finetuned
    )
    
    # 运行分析
    visualizer.run_analysis(num_samples=args.num_samples)

if __name__ == "__main__":
    main()

# python visualize_knowledge_distillation.py \
#     -c your_config_name \
#     --teacher_model path/to/teacher/model.pth \
#     --student_model path/to/student/pretrained.pth \
#     --student_finetuned path/to/student/finetuned.pth \
#     --num_samples 3