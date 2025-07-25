import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import pairwise_distances
import os
import glob
import pdb
from pathlib import Path

class AlignmentHandler:
    """专门处理两组预测不对齐的情况"""
    
    def __init__(self, pred1, pred2, alignment_strategy='union'):
        self.pred1 = pred1
        self.pred2 = pred2
        self.alignment_strategy = alignment_strategy
        
    def create_aligned_comparison_matrix(self):
        """创建对齐的比较矩阵"""
        
        # 获取所有可能的(frame, event_class)组合
        pred1_keys = set(zip(self.pred1['frame'], self.pred1['event_class']))
        pred2_keys = set(zip(self.pred2['frame'], self.pred2['event_class']))
        
        if self.alignment_strategy == 'union':
            all_keys = pred1_keys | pred2_keys
        elif self.alignment_strategy == 'intersection':
            all_keys = pred1_keys & pred2_keys
        elif self.alignment_strategy == 'pred1_only':
            all_keys = pred1_keys
        elif self.alignment_strategy == 'pred2_only':
            all_keys = pred2_keys
        
        # 创建对齐的数据结构
        aligned_data = []
        
        for frame, event_class in sorted(all_keys):
            # 查找pred1中的对应预测
            pred1_match = self.pred1[
                (self.pred1['frame'] == frame) & 
                (self.pred1['event_class'] == event_class)
            ]
            
            # 查找pred2中的对应预测
            pred2_match = self.pred2[
                (self.pred2['frame'] == frame) & 
                (self.pred2['event_class'] == event_class)
            ]
            
            # 构建对齐的记录
            aligned_record = {
                'frame': frame,
                'event_class': event_class,
                'pred1_exists': len(pred1_match) > 0,
                'pred2_exists': len(pred2_match) > 0,
                'agreement_type': self.classify_agreement(len(pred1_match) > 0, len(pred2_match) > 0)
            }
            
            # 添加pred1的信息
            if len(pred1_match) > 0:
                pred1_row = pred1_match.iloc[0]
                aligned_record.update({
                    'pred1_x': pred1_row['x'], 'pred1_y': pred1_row['y'], 'pred1_z': pred1_row['z'],
                    'pred1_distance': pred1_row['distance'],
                    'pred1_azimuth': pred1_row['azimuth'],
                    'pred1_elevation': pred1_row['elevation'],
                    'pred1_count': pred1_row.get('prediction_count', 1)
                })
            else:
                aligned_record.update({
                    'pred1_x': np.nan, 'pred1_y': np.nan, 'pred1_z': np.nan,
                    'pred1_distance': np.nan, 'pred1_azimuth': np.nan, 'pred1_elevation': np.nan,
                    'pred1_count': 0
                })
            
            # 添加pred2的信息
            if len(pred2_match) > 0:
                pred2_row = pred2_match.iloc[0]
                aligned_record.update({
                    'pred2_x': pred2_row['x'], 'pred2_y': pred2_row['y'], 'pred2_z': pred2_row['z'],
                    'pred2_distance': pred2_row['distance'],
                    'pred2_azimuth': pred2_row['azimuth'],
                    'pred2_elevation': pred2_row['elevation'],
                    'pred2_count': pred2_row.get('prediction_count', 1)
                })
            else:
                aligned_record.update({
                    'pred2_x': np.nan, 'pred2_y': np.nan, 'pred2_z': np.nan,
                    'pred2_distance': np.nan, 'pred2_azimuth': np.nan, 'pred2_elevation': np.nan,
                    'pred2_count': 0
                })
            
            aligned_data.append(aligned_record)
        
        return pd.DataFrame(aligned_data)
    
    def classify_agreement(self, pred1_exists, pred2_exists):
        """分类预测一致性类型"""
        if pred1_exists and pred2_exists:
            return 'both_predict'
        elif pred1_exists and not pred2_exists:
            return 'only_pred1'
        elif not pred1_exists and pred2_exists:
            return 'only_pred2'
        else:
            return 'neither_predict'  # 这种情况在union策略下不会出现
    
    def compute_alignment_statistics(self):
        """计算对齐统计信息"""
        aligned_df = self.create_aligned_comparison_matrix()
        
        total_comparisons = len(aligned_df)
        agreement_counts = aligned_df['agreement_type'].value_counts()
        
        stats = {
            'total_frame_class_combinations': total_comparisons,
            'both_predict': agreement_counts.get('both_predict', 0),
            'only_pred1': agreement_counts.get('only_pred1', 0),
            'only_pred2': agreement_counts.get('only_pred2', 0),
            'neither_predict': agreement_counts.get('neither_predict', 0),
            'agreement_ratio': agreement_counts.get('both_predict', 0) / total_comparisons if total_comparisons > 0 else 0,
            'pred1_coverage': (agreement_counts.get('both_predict', 0) + agreement_counts.get('only_pred1', 0)) / total_comparisons if total_comparisons > 0 else 0,
            'pred2_coverage': (agreement_counts.get('both_predict', 0) + agreement_counts.get('only_pred2', 0)) / total_comparisons if total_comparisons > 0 else 0
        }
        
        return stats, aligned_df

class PredictionAnalyzer:
    def __init__(self, group1_path, group2_path, group1_name="Teacher", group2_name="Student"):
        self.group1_path = group1_path
        self.group2_path = group2_path
        self.group1_name = group1_name
        self.group2_name = group2_name

    def load_predictions(self, folder_path):
        """加载文件夹下所有CSV预测文件"""
        predictions = {}
        p = Path(folder_path)
        csv_files = list(p.glob("*.csv"))
        csv_files = [str(f) for f in csv_files]  # 转换为字符串路径

        # pdb.set_trace()
        
        for file_path in csv_files:
            filename = os.path.basename(file_path).replace('.csv', '')
            df = pd.read_csv(file_path, header=None, 
                           names=['frame', 'event_class', 'unused', 'x', 'y', 'z'])
            predictions[filename] = df
        
        return predictions
        
    def preprocess_predictions(self, predictions):
        """预处理预测数据，处理不对齐情况"""
        processed = {}
        
        for filename, df in predictions.items():
            # 修正：只要出现在CSV中就是活跃事件
            df['active'] = 1  # 所有行都是活跃的
            
            # 计算球坐标
            df['distance'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
            df['azimuth'] = np.arctan2(df['y'], df['x']) * 180 / np.pi
            df['elevation'] = np.arcsin(df['z'] / (df['distance'] + 1e-8)) * 180 / np.pi
            
            # 为了后续分析，创建规范化的数据结构
            df_normalized = self.normalize_prediction_format(df)
            processed[filename] = df_normalized
            
        return processed
    
    def normalize_prediction_format(self, df):
        """将预测数据规范化为便于对比的格式"""
        # 按时间帧和事件类别分组，处理同一帧多事件的情况
        normalized_data = []
        
        for (frame, event_class), group in df.groupby(['frame', 'event_class']):
            # 如果同一帧同一事件类别有多个预测，取平均值或最新值
            if len(group) > 1:
                # 策略1：取平均坐标
                avg_coords = group[['x', 'y', 'z']].mean()
                normalized_data.append({
                    'frame': frame,
                    'event_class': event_class,
                    'x': avg_coords['x'], 'y': avg_coords['y'], 'z': avg_coords['z'],
                    'distance': np.sqrt(avg_coords['x']**2 + avg_coords['y']**2 + avg_coords['z']**2),
                    'azimuth': np.arctan2(avg_coords['y'], avg_coords['x']) * 180 / np.pi,
                    'elevation': np.arcsin(avg_coords['z'] / (np.sqrt(avg_coords['x']**2 + avg_coords['y']**2 + avg_coords['z']**2) + 1e-8)) * 180 / np.pi,
                    'active': 1,
                    'prediction_count': len(group)  # 记录同一帧同一事件的预测数量
                })
            else:
                row = group.iloc[0]
                normalized_data.append({
                    'frame': row['frame'],
                    'event_class': row['event_class'],
                    'x': row['x'], 'y': row['y'], 'z': row['z'],
                    'distance': row['distance'],
                    'azimuth': row['azimuth'],
                    'elevation': row['elevation'],
                    'active': 1,
                    'prediction_count': 1
                })
        
        return pd.DataFrame(normalized_data)
    
    def analyze_spatial_distribution_aligned(self, aligned_df, filename):
        """基于对齐数据分析空间分布差异"""
        
        # 筛选出两个模型都有预测的数据
        both_predict = aligned_df[aligned_df['agreement_type'] == 'both_predict'].copy()
        
        spatial_analysis = {
            'alignment_stats': {
                'total_comparisons': len(aligned_df),
                'both_predict_count': len(both_predict),
                'both_predict_ratio': len(both_predict) / len(aligned_df) if len(aligned_df) > 0 else 0
            }
        }
        
        if len(both_predict) > 0:
            # 计算空间差异
            both_predict['spatial_distance'] = np.sqrt(
                (both_predict['pred1_x'] - both_predict['pred2_x'])**2 +
                (both_predict['pred1_y'] - both_predict['pred2_y'])**2 +
                (both_predict['pred1_z'] - both_predict['pred2_z'])**2
            )
            
            both_predict['distance_diff'] = both_predict['pred1_distance'] - both_predict['pred2_distance']
            both_predict['azimuth_diff'] = self.angle_difference(both_predict['pred1_azimuth'], both_predict['pred2_azimuth'])
            both_predict['elevation_diff'] = both_predict['pred1_elevation'] - both_predict['pred2_elevation']
            
            spatial_analysis['spatial_differences'] = {
                'mean_spatial_distance': both_predict['spatial_distance'].mean(),
                'std_spatial_distance': both_predict['spatial_distance'].std(),
                'median_spatial_distance': both_predict['spatial_distance'].median(),
                'mean_distance_diff': both_predict['distance_diff'].mean(),
                'std_distance_diff': both_predict['distance_diff'].std(),
                'mean_azimuth_diff': both_predict['azimuth_diff'].mean(),
                'std_azimuth_diff': both_predict['azimuth_diff'].std(),
                'mean_elevation_diff': both_predict['elevation_diff'].mean(),
                'std_elevation_diff': both_predict['elevation_diff'].std()
            }
            
            # 分析不同距离范围的预测差异
            spatial_analysis['distance_range_analysis'] = self.analyze_by_distance_range(both_predict)
        
        # 分析不对齐的情况
        only_pred1 = aligned_df[aligned_df['agreement_type'] == 'only_pred1']
        only_pred2 = aligned_df[aligned_df['agreement_type'] == 'only_pred2']
        
        spatial_analysis['disagreement_analysis'] = {
            'only_pred1_count': len(only_pred1),
            'only_pred2_count': len(only_pred2),
            'only_pred1_avg_distance': only_pred1['pred1_distance'].mean() if len(only_pred1) > 0 else None,
            'only_pred2_avg_distance': only_pred2['pred2_distance'].mean() if len(only_pred2) > 0 else None,
            'disagreement_by_class': self.analyze_disagreement_by_class(only_pred1, only_pred2)
        }
        
        return spatial_analysis, both_predict

    def angle_difference(self, angle1, angle2):
        """计算角度差异（考虑周期性）"""
        diff = angle1 - angle2
        diff = ((diff + 180) % 360) - 180  # 归一化到[-180, 180]
        return np.abs(diff)

    def analyze_by_distance_range(self, both_predict):
        """按距离范围分析预测差异"""
        distance_ranges = [(0, 1), (1, 2), (2, 3), (3, float('inf'))]
        range_analysis = {}
        
        for min_dist, max_dist in distance_ranges:
            if max_dist == float('inf'):
                mask = both_predict['pred1_distance'] >= min_dist
                range_name = f'{min_dist}+'
            else:
                mask = (both_predict['pred1_distance'] >= min_dist) & (both_predict['pred1_distance'] < max_dist)
                range_name = f'{min_dist}-{max_dist}'
            
            range_data = both_predict[mask]
            if len(range_data) > 0:
                range_analysis[range_name] = {
                    'count': len(range_data),
                    'mean_spatial_error': range_data['spatial_distance'].mean(),
                    'mean_distance_error': np.abs(range_data['distance_diff']).mean(),
                    'mean_angular_error': range_data['azimuth_diff'].mean()
                }
        
        return range_analysis

    def analyze_disagreement_by_class(self, only_pred1, only_pred2):
        """分析不同事件类别的预测分歧"""
        disagreement_by_class = {}
        
        # 分析只有pred1预测的情况
        if len(only_pred1) > 0:
            pred1_class_counts = only_pred1['event_class'].value_counts()
            disagreement_by_class['only_pred1_by_class'] = pred1_class_counts.to_dict()
        
        # 分析只有pred2预测的情况
        if len(only_pred2) > 0:
            pred2_class_counts = only_pred2['event_class'].value_counts()
            disagreement_by_class['only_pred2_by_class'] = pred2_class_counts.to_dict()
        
        return disagreement_by_class

    # def analyze_spatial_distribution(self, pred1, pred2, filename):
    #     """分析3D空间分布差异"""
        
    #     # 提取活跃预测
    #     active1 = pred1[pred1['active'] == 1]
    #     active2 = pred2[pred2['active'] == 1]
        
    #     spatial_analysis = {}
        
    #     # 1. 距离分布对比
    #     spatial_analysis['distance_stats'] = {
    #         f'{self.group1_name}_mean_dist': active1['distance'].mean(),
    #         f'{self.group2_name}_mean_dist': active2['distance'].mean(),
    #         f'{self.group1_name}_std_dist': active1['distance'].std(),
    #         f'{self.group2_name}_std_dist': active2['distance'].std(),
    #         'distance_ks_test': stats.ks_2samp(active1['distance'], active2['distance'])
    #     }
        
    #     # 2. 角度分布对比
    #     spatial_analysis['angle_stats'] = {
    #         f'{self.group1_name}_azimuth_circular_std': self.circular_std(active1['azimuth']),
    #         f'{self.group2_name}_azimuth_circular_std': self.circular_std(active2['azimuth']),
    #         'azimuth_watson_test': self.watson_u2_test(active1['azimuth'], active2['azimuth'])
    #     }
        
    #     # 3. 3D空间聚类对比
    #     spatial_analysis['clustering'] = self.compare_spatial_clustering(active1, active2)
        
    #     return spatial_analysis

    def visualize_alignment_analysis(self, aligned_df, filename, save_path=None):
        """可视化对齐分析结果"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Prediction Alignment Analysis - {filename}', fontsize=16)
        
        # 1. 预测一致性饼图
        agreement_counts = aligned_df['agreement_type'].value_counts()
        axes[0,0].pie(agreement_counts.values, labels=agreement_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Prediction Agreement Distribution')
        
        # 2. 空间误差分布（仅对都有预测的情况）
        both_predict = aligned_df[aligned_df['agreement_type'] == 'both_predict'].copy()
        if len(both_predict) > 0:
            both_predict['spatial_distance'] = np.sqrt(
                (both_predict['pred1_x'] - both_predict['pred2_x'])**2 +
                (both_predict['pred1_y'] - both_predict['pred2_y'])**2 +
                (both_predict['pred1_z'] - both_predict['pred2_z'])**2
            )
            axes[0,1].hist(both_predict['spatial_distance'], bins=30, alpha=0.7, color='green')
            axes[0,1].set_xlabel('Spatial Distance Error')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Spatial Error Distribution (Both Predict)')
        
        # 3. 按事件类别的预测覆盖率
        class_stats = []
        for event_class in aligned_df['event_class'].unique():
            class_data = aligned_df[aligned_df['event_class'] == event_class]
            both_count = len(class_data[class_data['agreement_type'] == 'both_predict'])
            total_count = len(class_data)
            coverage = both_count / total_count if total_count > 0 else 0
            class_stats.append({'class': event_class, 'coverage': coverage, 'total': total_count})
        
        class_df = pd.DataFrame(class_stats)
        if len(class_df) > 0:
            axes[0,2].bar(range(len(class_df)), class_df['coverage'])
            axes[0,2].set_xlabel('Event Class')
            axes[0,2].set_ylabel('Agreement Coverage')
            axes[0,2].set_title('Agreement Coverage by Event Class')
            axes[0,2].set_xticks(range(len(class_df)))
            axes[0,2].set_xticklabels(class_df['class'], rotation=45)
        
        # 4. 时序上的预测一致性
        frame_agreement = aligned_df.groupby('frame')['agreement_type'].apply(
            lambda x: (x == 'both_predict').sum() / len(x)
        )
        axes[1,0].plot(frame_agreement.index, frame_agreement.values)
        axes[1,0].set_xlabel('Frame')
        axes[1,0].set_ylabel('Agreement Ratio')
        axes[1,0].set_title('Temporal Agreement Pattern')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. 距离预测对比（仅对都有预测的情况）
        if len(both_predict) > 0:
            axes[1,1].scatter(both_predict['pred1_distance'], both_predict['pred2_distance'], alpha=0.6)
            max_dist = max(both_predict['pred1_distance'].max(), both_predict['pred2_distance'].max())
            axes[1,1].plot([0, max_dist], [0, max_dist], 'r--', alpha=0.5)
            axes[1,1].set_xlabel('Pred1 Distance')
            axes[1,1].set_ylabel('Pred2 Distance')
            axes[1,1].set_title('Distance Prediction Correlation')
            axes[1,1].grid(True, alpha=0.3)
            
            # 添加相关系数
            corr = np.corrcoef(both_predict['pred1_distance'], both_predict['pred2_distance'])[0,1]
            axes[1,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1,1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. 角度预测对比
        if len(both_predict) > 0:
            axes[1,2].scatter(both_predict['pred1_azimuth'], both_predict['pred2_azimuth'], alpha=0.6)
            axes[1,2].plot([-180, 180], [-180, 180], 'r--', alpha=0.5)
            axes[1,2].set_xlabel('Pred1 Azimuth')
            axes[1,2].set_ylabel('Pred2 Azimuth')
            axes[1,2].set_title('Azimuth Prediction Correlation')
            axes[1,2].grid(True, alpha=0.3)
            
            # 计算角度相关性
            azimuth_corr = np.corrcoef(both_predict['pred1_azimuth'], both_predict['pred2_azimuth'])[0,1]
            axes[1,2].text(0.05, 0.95, f'Correlation: {azimuth_corr:.3f}', transform=axes[1,2].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{filename}_alignment_analysis.png", dpi=300, bbox_inches='tight')
        
        return fig

    def create_disagreement_summary(self, all_aligned_results, save_path=None):
        """创建所有文件的分歧情况总结"""
        
        summary_data = []
        
        for filename, aligned_df in all_aligned_results.items():
            agreement_counts = aligned_df['agreement_type'].value_counts()
            total = len(aligned_df)
            
            summary_data.append({
                'filename': filename,
                'total_predictions': total,
                'both_predict': agreement_counts.get('both_predict', 0),
                'only_pred1': agreement_counts.get('only_pred1', 0),
                'only_pred2': agreement_counts.get('only_pred2', 0),
                'agreement_ratio': agreement_counts.get('both_predict', 0) / total if total > 0 else 0,
                'pred1_exclusive_ratio': agreement_counts.get('only_pred1', 0) / total if total > 0 else 0,
                'pred2_exclusive_ratio': agreement_counts.get('only_pred2', 0) / total if total > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 可视化总结
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Disagreement Summary Across All Files', fontsize=16)
        
        # 一致性比例
        axes[0,0].bar(range(len(summary_df)), summary_df['agreement_ratio'])
        axes[0,0].set_xlabel('File Index')
        axes[0,0].set_ylabel('Agreement Ratio')
        axes[0,0].set_title('Agreement Ratio by File')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 独有预测比例对比
        x = np.arange(len(summary_df))
        width = 0.35
        axes[0,1].bar(x - width/2, summary_df['pred1_exclusive_ratio'], width, label='Pred1 Exclusive')
        axes[0,1].bar(x + width/2, summary_df['pred2_exclusive_ratio'], width, label='Pred2 Exclusive')
        axes[0,1].set_xlabel('File Index')
        axes[0,1].set_ylabel('Exclusive Prediction Ratio')
        axes[0,1].set_title('Exclusive Predictions by File')
        axes[0,1].legend()
        
        # 总体统计
        overall_stats = {
            'Mean Agreement Ratio': summary_df['agreement_ratio'].mean(),
            'Std Agreement Ratio': summary_df['agreement_ratio'].std(),
            'Mean Pred1 Exclusive': summary_df['pred1_exclusive_ratio'].mean(),
            'Mean Pred2 Exclusive': summary_df['pred2_exclusive_ratio'].mean()
        }
        
        axes[1,0].bar(overall_stats.keys(), overall_stats.values())
        axes[1,0].set_ylabel('Ratio')
        axes[1,0].set_title('Overall Statistics')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 分布直方图
        axes[1,1].hist(summary_df['agreement_ratio'], bins=10, alpha=0.7, label='Agreement Ratio')
        axes[1,1].set_xlabel('Agreement Ratio')
        axes[1,1].set_ylabel('Number of Files')
        axes[1,1].set_title('Distribution of Agreement Ratios')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/disagreement_summary.png", dpi=300, bbox_inches='tight')
            summary_df.to_csv(f"{save_path}/disagreement_summary.csv", index=False)
        
        return fig, summary_df

    def circular_std(self, angles):
        """计算角度的圆形标准差"""
        angles_rad = np.radians(angles)
        mean_vec = np.mean(np.exp(1j * angles_rad))
        return np.sqrt(-2 * np.log(np.abs(mean_vec))) * 180 / np.pi

    def compare_spatial_clustering(self, pred1, pred2):
        """比较空间聚类模式"""
        from sklearn.cluster import DBSCAN
        
        # 对两组预测进行聚类
        coords1 = pred1[['x', 'y', 'z']].values
        coords2 = pred2[['x', 'y', 'z']].values
        
        clustering1 = DBSCAN(eps=0.5, min_samples=3).fit(coords1)
        clustering2 = DBSCAN(eps=0.5, min_samples=3).fit(coords2)
        
        return {
            f'{self.group1_name}_n_clusters': len(set(clustering1.labels_)) - (1 if -1 in clustering1.labels_ else 0),
            f'{self.group2_name}_n_clusters': len(set(clustering2.labels_)) - (1 if -1 in clustering2.labels_ else 0),
            f'{self.group1_name}_noise_ratio': (clustering1.labels_ == -1).sum() / len(clustering1.labels_),
            f'{self.group2_name}_noise_ratio': (clustering2.labels_ == -1).sum() / len(clustering2.labels_)
        }

    def analyze_temporal_patterns(self, pred1, pred2, filename):
        """分析时序模式差异"""
        
        temporal_analysis = {}
        
        # 1. 活跃度时序对比
        frame_activity1 = pred1.groupby('frame')['active'].sum()
        frame_activity2 = pred2.groupby('frame')['active'].sum()
        
        temporal_analysis['activity_patterns'] = {
            'correlation': np.corrcoef(frame_activity1, frame_activity2)[0,1],
            'cross_correlation': self.compute_cross_correlation(frame_activity1, frame_activity2),
            'onset_offset_agreement': self.compute_onset_offset_agreement(pred1, pred2)
        }
        
        # 2. 事件持续时间分布
        duration1 = self.compute_event_durations(pred1)
        duration2 = self.compute_event_durations(pred2)
        
        temporal_analysis['duration_stats'] = {
            f'{self.group1_name}_mean_duration': np.mean(duration1),
            f'{self.group2_name}_mean_duration': np.mean(duration2),
            'duration_ks_test': stats.ks_2samp(duration1, duration2) if len(duration1) > 0 and len(duration2) > 0 else None
        }
        
        return temporal_analysis

    def compute_event_durations(self, pred):
        """计算事件持续时间"""
        durations = []
        for event_class in pred['event_class'].unique():
            class_data = pred[pred['event_class'] == event_class]
            active_segments = self.find_continuous_segments(class_data['active'].values)
            durations.extend([seg[1] - seg[0] + 1 for seg in active_segments])
        return durations

    def find_continuous_segments(self, binary_array):
        """找到连续的1段"""
        segments = []
        start = None
        for i, val in enumerate(binary_array):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                segments.append((start, i-1))
                start = None
        if start is not None:
            segments.append((start, len(binary_array)-1))
        return segments
    
    def analyze_detection_consistency(self, pred1, pred2, filename):
        """分析事件检测的一致性"""
        
        consistency_analysis = {}
        
        # 1. 按事件类别分析
        for event_class in set(pred1['event_class'].unique()) | set(pred2['event_class'].unique()):
            class1 = pred1[pred1['event_class'] == event_class]
            class2 = pred2[pred2['event_class'] == event_class]
            
            if len(class1) == 0 or len(class2) == 0:
                continue
                
            # 计算检测一致性
            consistency_analysis[f'class_{event_class}'] = {
                'detection_correlation': self.compute_detection_correlation(class1, class2),
                'spatial_agreement': self.compute_spatial_agreement(class1, class2),
                'temporal_overlap': self.compute_temporal_overlap(class1, class2)
            }
        
        return consistency_analysis

    def compute_detection_correlation(self, class1, class2):
        """计算检测结果的相关性"""
        # 对齐时间帧
        frames = set(class1['frame']) | set(class2['frame'])
        
        activity1 = []
        activity2 = []
        
        for frame in sorted(frames):
            act1 = 1 if frame in class1[class1['active']==1]['frame'].values else 0
            act2 = 1 if frame in class2[class2['active']==1]['frame'].values else 0
            activity1.append(act1)
            activity2.append(act2)
        
        if len(set(activity1)) > 1 and len(set(activity2)) > 1:
            return np.corrcoef(activity1, activity2)[0,1]
        else:
            return 0.0

    def compute_spatial_agreement(self, class1, class2):
        """计算空间预测的一致性"""
        active1 = class1[class1['active'] == 1]
        active2 = class2[class2['active'] == 1]
        
        if len(active1) == 0 or len(active2) == 0:
            return 0.0
        
        # 计算平均空间距离
        coords1 = active1[['x', 'y', 'z']].values
        coords2 = active2[['x', 'y', 'z']].values
        
        # 使用最近邻匹配
        from scipy.spatial.distance import cdist
        distances = cdist(coords1, coords2)
        min_distances = np.min(distances, axis=1)
        
        return np.mean(min_distances)
    
    def visualize_3d_spatial_distribution(self, pred1, pred2, filename, save_path=None):
        """3D空间分布可视化"""
        
        active1 = pred1[pred1['active'] == 1]
        active2 = pred2[pred2['active'] == 1]
        
        fig = go.Figure()
        
        # 添加第一组预测
        fig.add_trace(go.Scatter3d(
            x=active1['x'], y=active1['y'], z=active1['z'],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.7),
            name=self.group1_name,
            text=[f'Frame: {f}, Class: {c}' for f, c in zip(active1['frame'], active1['event_class'])]
        ))
        
        # 添加第二组预测
        fig.add_trace(go.Scatter3d(
            x=active2['x'], y=active2['y'], z=active2['z'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.7),
            name=self.group2_name,
            text=[f'Frame: {f}, Class: {c}' for f, c in zip(active2['frame'], active2['event_class'])]
        ))
        
        fig.update_layout(
            title=f'3D Spatial Distribution Comparison - {filename}',
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate'
            ),
            width=900, height=700
        )
        
        if save_path:
            fig.write_html(f"{save_path}/{filename}_3d_spatial.html")
        
        return fig

    def visualize_distance_distribution(self, pred1, pred2, filename, save_path=None):
        """距离分布对比可视化"""
        
        active1 = pred1[pred1['active'] == 1]
        active2 = pred2[pred2['active'] == 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Distance Distribution Comparison - {filename}', fontsize=16)
        
        # 距离直方图
        axes[0,0].hist(active1['distance'], bins=30, alpha=0.7, label=self.group1_name, color='red')
        axes[0,0].hist(active2['distance'], bins=30, alpha=0.7, label=self.group2_name, color='blue')
        axes[0,0].set_xlabel('Distance')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distance Distribution')
        axes[0,0].legend()
        
        # 距离箱线图
        distance_data = [active1['distance'], active2['distance']]
        axes[0,1].boxplot(distance_data, labels=[self.group1_name, self.group2_name])
        axes[0,1].set_ylabel('Distance')
        axes[0,1].set_title('Distance Box Plot')
        
        # 方位角分布
        axes[1,0].hist(active1['azimuth'], bins=36, alpha=0.7, label=self.group1_name, color='red')
        axes[1,0].hist(active2['azimuth'], bins=36, alpha=0.7, label=self.group2_name, color='blue')
        axes[1,0].set_xlabel('Azimuth (degrees)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Azimuth Distribution')
        axes[1,0].legend()
        
        # 仰角分布
        axes[1,1].hist(active1['elevation'], bins=18, alpha=0.7, label=self.group1_name, color='red')
        axes[1,1].hist(active2['elevation'], bins=18, alpha=0.7, label=self.group2_name, color='blue')
        axes[1,1].set_xlabel('Elevation (degrees)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Elevation Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{filename}_distance_distribution.png", dpi=300, bbox_inches='tight')
        
        return fig

    def visualize_temporal_patterns(self, pred1, pred2, filename, save_path=None):
        """时序模式可视化"""
        
        # 计算每帧的活跃度
        frame_activity1 = pred1.groupby('frame')['active'].sum()
        frame_activity2 = pred2.groupby('frame')['active'].sum()
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Temporal Pattern Comparison - {filename}', fontsize=16)
        
        # 活跃度时序图
        axes[0].plot(frame_activity1.index, frame_activity1.values, 'r-', label=self.group1_name, alpha=0.7)
        axes[0].plot(frame_activity2.index, frame_activity2.values, 'b-', label=self.group2_name, alpha=0.7)
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Number of Active Events')
        axes[0].set_title('Temporal Activity Pattern')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 活跃度差异
        common_frames = set(frame_activity1.index) & set(frame_activity2.index)
        if common_frames:
            common_frames = sorted(common_frames)
            diff = [frame_activity1.get(f, 0) - frame_activity2.get(f, 0) for f in common_frames]
            axes[1].plot(common_frames, diff, 'g-', linewidth=2)
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Frame')
            axes[1].set_ylabel(f'{self.group1_name} - {self.group2_name}')
            axes[1].set_title('Activity Difference')
            axes[1].grid(True, alpha=0.3)
        
        # 活跃度散点图
        if common_frames:
            activity1_common = [frame_activity1.get(f, 0) for f in common_frames]
            activity2_common = [frame_activity2.get(f, 0) for f in common_frames]
            axes[2].scatter(activity1_common, activity2_common, alpha=0.6)
            axes[2].plot([0, max(max(activity1_common), max(activity2_common))], 
                        [0, max(max(activity1_common), max(activity2_common))], 'r--', alpha=0.5)
            axes[2].set_xlabel(f'{self.group1_name} Activity')
            axes[2].set_ylabel(f'{self.group2_name} Activity')
            axes[2].set_title('Activity Correlation')
            axes[2].grid(True, alpha=0.3)
            
            # 添加相关系数
            corr = np.corrcoef(activity1_common, activity2_common)[0,1]
            axes[2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[2].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{filename}_temporal_patterns.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparison_heatmap(self, all_results, save_path=None):
        """创建所有文件的对比分析热图"""
        
        # 整理数据为矩阵形式
        metrics = []
        filenames = []
        
        for filename, results in all_results.items():
            filenames.append(filename)
            file_metrics = []
            
            # 提取关键指标
            spatial = results.get('spatial', {})
            temporal = results.get('temporal', {})
            detection = results.get('detection', {})
            
            file_metrics.extend([
                spatial.get('distance_stats', {}).get('distance_ks_test', (0, 1))[1],  # p-value
                temporal.get('activity_patterns', {}).get('correlation', 0),
                np.mean([detection.get(k, {}).get('spatial_agreement', 0) 
                        for k in detection.keys() if 'class_' in k])
            ])
            
            metrics.append(file_metrics)
        
        metrics_df = pd.DataFrame(metrics, 
                                index=filenames,
                                columns=['Distance KS p-value', 'Temporal Correlation', 'Avg Spatial Agreement'])
        plt.figure(figsize=(10, len(filenames) * 0.5))
        sns.heatmap(metrics_df, annot=True, cmap='RdYlBu', center=0.5, fmt='.3f')
        plt.title('Prediction Comparison Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/comparison_heatmap.png", dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    

    def run_complete_analysis_with_alignment(self):
        """运行包含对齐处理的完整分析"""
        
        # 加载数据
        pred1_data = self.load_predictions(self.group1_path)
        pred2_data = self.load_predictions(self.group2_path)
        
        # 预处理
        pred1_processed = self.preprocess_predictions(pred1_data)
        pred2_processed = self.preprocess_predictions(pred2_data)
        
        # 找到共同文件
        common_files = set(pred1_processed.keys()) & set(pred2_processed.keys())
        
        all_results = {}
        all_aligned_results = {}
        
        for filename in common_files:
            print(f"Analyzing {filename}...")
            
            pred1 = pred1_processed[filename]
            pred2 = pred2_processed[filename]
            
            # 创建对齐处理器
            aligner = AlignmentHandler(pred1, pred2, alignment_strategy='union')
            alignment_stats, aligned_df = aligner.compute_alignment_statistics()
            
            # 基于对齐数据进行分析
            spatial_analysis, both_predict_df = self.analyze_spatial_distribution_aligned(aligned_df, filename)
            
            all_results[filename] = {
                'alignment_stats': alignment_stats,
                'spatial_analysis': spatial_analysis
            }
            
            all_aligned_results[filename] = aligned_df
            
            # 生成可视化
            self.visualize_alignment_analysis(aligned_df, filename, save_path="./analysis_results")
        
        # 生成总结报告
        self.create_disagreement_summary(all_aligned_results, save_path="./analysis_results")
        
        return all_results, all_aligned_results

    def run_complete_analysis(self):
        """运行完整的预测分布对比分析"""
    
        
        # 加载数据
        pred1_data = self.load_predictions(self.group1_path)
        pred2_data = self.load_predictions(self.group2_path)
        
        # 预处理
        pred1_processed = self.preprocess_predictions(pred1_data)
        pred2_processed = self.preprocess_predictions(pred2_data)

        # pdb.set_trace()
        
        # 找到共同文件
        common_files = set(pred1_processed.keys()) & set(pred2_processed.keys())
        
        all_results = {}
        
        for filename in common_files:
            print(f"Analyzing {filename}...")
            
            pred1 = pred1_processed[filename]
            pred2 = pred2_processed[filename]
            
            # 各类分析
            spatial_analysis = self.analyze_spatial_distribution(pred1, pred2, filename)
            temporal_analysis = self.analyze_temporal_patterns(pred1, pred2, filename)
            detection_analysis = self.analyze_detection_consistency(pred1, pred2, filename)
            
            all_results[filename] = {
                'spatial': spatial_analysis,
                'temporal': temporal_analysis,
                'detection': detection_analysis
            }
            
            # 生成可视化
            self.visualize_3d_spatial_distribution(pred1, pred2, filename, save_path="./analysis_results")
            self.visualize_distance_distribution(pred1, pred2, filename, save_path="./analysis_results")
            self.visualize_temporal_patterns(pred1, pred2, filename, save_path="./analysis_results")
        
        # 生成总结报告
        self.create_comparison_heatmap(all_results, save_path="./analysis_results")
        
        return all_results
    
if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("analysis_results", exist_ok=True)

    # 使用示例
    analyzer = PredictionAnalyzer(
        group1_path="results/Test_RC_24h_Offline_8ConformerLayer_Buffered_P8F1/results",
        group2_path="results/Test_Streamingly_Cache_RC_24h_Chunk[100,49]_8ConformerLayer_2/results",
        group1_name="Offline_Buffered_StreamingP8F1",
        group2_name="Streaming_[100,49]"
    )

    # results = analyzer.run_complete_analysis()
    results, aligned_results = analyzer.run_complete_analysis_with_alignment()