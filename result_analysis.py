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

# 数据加载和预处理
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
        """预处理预测数据"""
        processed = {}
        
        for filename, df in predictions.items():
            # 计算球坐标
            df['distance'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
            df['azimuth'] = np.arctan2(df['y'], df['x']) * 180 / np.pi
            df['elevation'] = np.arcsin(df['z'] / (df['distance'] + 1e-8)) * 180 / np.pi
            
            # 活跃检测（距离大于阈值认为有事件）
            df['active'] = 1
            
            processed[filename] = df
            
        return processed
    
    def analyze_spatial_distribution(self, pred1, pred2, filename):
        """分析3D空间分布差异"""
        
        # 提取活跃预测
        active1 = pred1[pred1['active'] == 1]
        active2 = pred2[pred2['active'] == 1]
        
        spatial_analysis = {}
        
        # 1. 距离分布对比
        spatial_analysis['distance_stats'] = {
            f'{self.group1_name}_mean_dist': active1['distance'].mean(),
            f'{self.group2_name}_mean_dist': active2['distance'].mean(),
            f'{self.group1_name}_std_dist': active1['distance'].std(),
            f'{self.group2_name}_std_dist': active2['distance'].std(),
            'distance_ks_test': stats.ks_2samp(active1['distance'], active2['distance'])
        }
        
        # 2. 角度分布对比
        spatial_analysis['angle_stats'] = {
            f'{self.group1_name}_azimuth_circular_std': self.circular_std(active1['azimuth']),
            f'{self.group2_name}_azimuth_circular_std': self.circular_std(active2['azimuth']),
            'azimuth_watson_test': self.watson_u2_test(active1['azimuth'], active2['azimuth'])
        }
        
        # 3. 3D空间聚类对比
        spatial_analysis['clustering'] = self.compare_spatial_clustering(active1, active2)
        
        return spatial_analysis

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

    results = analyzer.run_complete_analysis()