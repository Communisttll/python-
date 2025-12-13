"""
三维概率体可视化和等值面分析
实现三维空间中分类概率的体绘制、等值面提取和交互式可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class Iris3DProbabilityVisualizer:
    """鸢尾花三维概率体可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        self.models = {}
        self.results = {}
        
        # 定义要分析的分类器
        self.classifiers = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNeighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        # 颜色映射
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        self.markers = ['o', 's', '^']
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("正在加载鸢尾花数据集...")
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # 选择前三个特征用于3D可视化
        self.X_3d = self.X[:, :3]
        
        # 数据划分
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_3d, self.y, test_size=0.3, random_state=42, stratify=self.y)
        
        # 数据标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"数据准备完成:")
        print(f"  训练集大小: {self.X_train.shape}")
        print(f"  测试集大小: {self.X_test.shape}")
        print(f"  使用特征: {', '.join(self.feature_names[:3])}")
        
    def train_all_models(self):
        """训练所有模型"""
        print("\n正在训练所有分类器...")
        
        for name, model in self.classifiers.items():
            print(f"  训练 {name}...")
            
            try:
                # 训练模型
                model.fit(self.X_train_scaled, self.y_train)
                self.models[name] = model
                
                # 计算性能指标
                y_pred = model.predict(self.X_test_scaled)
                accuracy = (y_pred == self.y_test).mean()
                
                # 存储结果
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                
                print(f"    准确率: {accuracy:.4f}")
                
            except Exception as e:
                print(f"    训练失败: {str(e)}")
                continue
                
    def create_3d_probability_volume(self, classifier_name, threshold=0.1):
        """创建3D概率体可视化"""
        if classifier_name not in self.models:
            print(f"分类器 {classifier_name} 未找到")
            return
            
        print(f"\n正在为 {classifier_name} 创建3D概率体...")
        
        model = self.models[classifier_name]
        
        # 创建3D网格
        resolution = 0.1
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        z_min, z_max = self.X_train_scaled[:, 2].min() - 1, self.X_train_scaled[:, 2].max() + 1
        
        x_range = np.arange(x_min, x_max, resolution)
        y_range = np.arange(y_min, y_max, resolution)
        z_range = np.arange(z_min, z_max, resolution)
        
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # 扁平化网格点
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        print(f"  正在预测 {len(grid_points)} 个网格点...")
        
        # 获取预测概率
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(grid_points)
        else:
            decision = model.decision_function(grid_points)
            probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
        
        # 获取最大概率（置信度）
        max_probas = np.max(probas, axis=1)
        predicted_classes = np.argmax(probas, axis=1)
        
        # 创建图形
        fig = plt.figure(figsize=(20, 15))
        
        # 子图1：概率体可视化（散点图）
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # 只显示概率大于阈值的部分（降采样）
        high_prob_mask = max_probas > threshold
        sample_indices = np.where(high_prob_mask)[0]
        
        if len(sample_indices) > 5000:  # 限制点数以提高性能
            sample_indices = np.random.choice(sample_indices, size=5000, replace=False)
        
        # 绘制概率体
        for class_idx in range(len(self.target_names)):
            class_mask = predicted_classes[sample_indices] == class_idx
            if np.any(class_mask):
                indices = sample_indices[class_mask]
                # 使用单一颜色值而不是数组
                ax1.scatter(grid_points[indices, 0],
                           grid_points[indices, 1],
                           grid_points[indices, 2],
                           c=self.colors[class_idx],
                           marker='.',
                           s=max_probas[indices] * 50,  # 大小表示概率
                           alpha=0.6,  # 固定透明度
                           label=f'{self.target_names[class_idx]} (p>{threshold})')
        
        # 绘制训练数据点
        for i, target_name in enumerate(self.target_names):
            mask = self.y_train == i
            ax1.scatter(self.X_train_scaled[mask, 0], 
                       self.X_train_scaled[mask, 1], 
                       self.X_train_scaled[mask, 2],
                       c=[self.colors[i]], 
                       marker=self.markers[i],
                       s=80, edgecolors='black', linewidth=1.5,
                       label=f'{target_name} (训练)')
        
        ax1.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
        ax1.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
        ax1.set_zlabel(f'标准化 {self.feature_names[2]}', fontfamily='SimHei')
        ax1.set_title(f'{classifier_name} - 3D概率体 (p>{threshold})', 
                     fontsize=12, fontweight='bold', fontfamily='SimHei')
        ax1.legend(fontsize=8, prop={'family': 'SimHei'}, title_fontsize=10)
        ax1.view_init(elev=20, azim=45)
        
        # 子图2：概率分布切片（XY平面，固定Z）
        ax2 = fig.add_subplot(2, 3, 2)
        z_fixed = self.X_train_scaled[:, 2].mean()
        z_mask = np.abs(grid_points[:, 2] - z_fixed) < resolution
        
        if np.any(z_mask):
            slice_points = grid_points[z_mask]
            slice_probas = max_probas[z_mask]
            slice_classes = predicted_classes[z_mask]
            
            # 创建2D网格
            x_unique = np.unique(slice_points[:, 0])
            y_unique = np.unique(slice_points[:, 1])
            xx_2d, yy_2d = np.meshgrid(x_unique, y_unique)
            
            # 重新组织数据
            prob_grid = np.full((len(y_unique), len(x_unique)), np.nan)
            class_grid = np.full((len(y_unique), len(x_unique)), np.nan)
            
            for i, x in enumerate(x_unique):
                for j, y in enumerate(y_unique):
                    point_mask = (slice_points[:, 0] == x) & (slice_points[:, 1] == y)
                    if np.any(point_mask):
                        prob_grid[j, i] = slice_probas[point_mask].mean()
                        values, counts = np.unique(slice_classes[point_mask], return_counts=True)
                        class_grid[j, i] = values[np.argmax(counts)]
            
            # 绘制概率热图
            im = ax2.imshow(prob_grid, extent=[x_unique.min(), x_unique.max(), 
                                              y_unique.min(), y_unique.max()],
                           origin='lower', cmap='hot', alpha=0.7)
            
            # 添加等值线
            ax2.contour(xx_2d, yy_2d, prob_grid, levels=[0.5, 0.7, 0.9], 
                       colors='white', linewidths=1, linestyles='--')
            
            # 绘制数据点（在切片附近）
            tolerance = 0.2
            for i, target_name in enumerate(self.target_names):
                mask = (self.y_train == i) & (np.abs(self.X_train_scaled[:, 2] - z_fixed) < tolerance)
                if np.any(mask):
                    ax2.scatter(self.X_train_scaled[mask, 0], 
                               self.X_train_scaled[mask, 1],
                               c=[self.colors[i]], marker=self.markers[i],
                               s=60, edgecolors='black', linewidth=1,
                               label=target_name)
            
            ax2.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
            ax2.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
            ax2.set_title(f'XY切片 (z={z_fixed:.2f})', fontsize=12, fontweight='bold', fontfamily='SimHei')
            ax2.legend(fontsize=8, prop={'family': 'SimHei'}, title_fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax2, label='最大概率').set_label('最大概率', fontfamily='SimHei')
        
        # 子图3：概率分布切片（XZ平面，固定Y）
        ax3 = fig.add_subplot(2, 3, 3)
        y_fixed = self.X_train_scaled[:, 1].mean()
        y_mask = np.abs(grid_points[:, 1] - y_fixed) < resolution
        
        if np.any(y_mask):
            slice_points = grid_points[y_mask]
            slice_probas = max_probas[y_mask]
            slice_classes = predicted_classes[y_mask]
            
            # 创建2D网格
            x_unique = np.unique(slice_points[:, 0])
            z_unique = np.unique(slice_points[:, 2])
            xx_2d, zz_2d = np.meshgrid(x_unique, z_unique)
            
            # 重新组织数据
            prob_grid = np.full((len(z_unique), len(x_unique)), np.nan)
            
            for i, x in enumerate(x_unique):
                for j, z in enumerate(z_unique):
                    point_mask = (slice_points[:, 0] == x) & (slice_points[:, 2] == z)
                    if np.any(point_mask):
                        prob_grid[j, i] = slice_probas[point_mask].mean()
            
            # 绘制概率热图
            im = ax3.imshow(prob_grid, extent=[x_unique.min(), x_unique.max(), 
                                              z_unique.min(), z_unique.max()],
                           origin='lower', cmap='hot', alpha=0.7)
            
            # 绘制数据点
            tolerance = 0.2
            for i, target_name in enumerate(self.target_names):
                mask = (self.y_train == i) & (np.abs(self.X_train_scaled[:, 1] - y_fixed) < tolerance)
                if np.any(mask):
                    ax3.scatter(self.X_train_scaled[mask, 0], 
                               self.X_train_scaled[mask, 2],
                               c=[self.colors[i]], marker=self.markers[i],
                               s=60, edgecolors='black', linewidth=1,
                               label=target_name)
            
            ax3.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
            ax3.set_ylabel(f'标准化 {self.feature_names[2]}', fontfamily='SimHei')
            ax3.set_title(f'XZ切片 (y={y_fixed:.2f})', fontsize=12, fontweight='bold', fontfamily='SimHei')
            ax3.legend(fontsize=8, prop={'family': 'SimHei'}, title_fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            plt.colorbar(im, ax=ax3, label='最大概率').set_label('最大概率', fontfamily='SimHei')
        
        # 子图4：概率统计
        ax4 = fig.add_subplot(2, 3, 4)
        
        # 计算概率分布
        prob_bins = np.linspace(0, 1, 21)
        hist, bins = np.histogram(max_probas, bins=prob_bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax4.bar(bin_centers, hist, width=0.04, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('最大概率', fontfamily='SimHei')
        ax4.set_ylabel('密度', fontfamily='SimHei')
        ax4.set_title('概率分布直方图', fontsize=12, fontweight='bold', fontfamily='SimHei')
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_prob = np.mean(max_probas)
        std_prob = np.std(max_probas)
        ax4.axvline(mean_prob, color='red', linestyle='--', 
                   label=f'均值: {mean_prob:.3f}')
        ax4.axvline(mean_prob + std_prob, color='orange', linestyle=':', 
                   label=f'+1σ: {mean_prob + std_prob:.3f}')
        ax4.axvline(mean_prob - std_prob, color='orange', linestyle=':', 
                   label=f'-1σ: {mean_prob - std_prob:.3f}')
        ax4.legend(prop={'family': 'SimHei'}, title_fontsize=10)
        
        # 子图5：类别概率对比
        ax5 = fig.add_subplot(2, 3, 5)
        
        # 计算每个类别的平均概率
        class_probs = []
        class_names = []
        for class_idx in range(len(self.target_names)):
            class_mask = predicted_classes == class_idx
            if np.any(class_mask):
                class_probs.append(max_probas[class_mask].mean())
                class_names.append(self.target_names[class_idx])
        
        bars = ax5.bar(class_names, class_probs, color=self.colors[:len(class_names)], 
                      alpha=0.7, edgecolor='black')
        ax5.set_xlabel('类别', fontfamily='SimHei')
        ax5.set_ylabel('平均概率', fontfamily='SimHei')
        ax5.set_title('各类别平均概率', fontsize=12, fontweight='bold', fontfamily='SimHei')
        ax5.set_xticklabels(class_names, fontfamily='SimHei')
        ax5.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, prob in zip(bars, class_probs):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold', fontfamily='SimHei') 
        
        # 子图6：概率熵分析
        ax6 = fig.add_subplot(2, 3, 6)
        
        # 计算熵
        def entropy(p):
            p = np.clip(p, 1e-10, 1.0)  # 避免log(0)
            return -np.sum(p * np.log2(p))
        
        # 计算每个点的熵
        entropies = []
        for i in range(0, len(probas), 100):  # 采样以提高性能
            entropies.append(entropy(probas[i]))
        
        ax6.hist(entropies, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax6.set_xlabel('熵', fontfamily='SimHei')
        ax6.set_ylabel('频次', fontfamily='SimHei')
        ax6.set_title('预测熵分布', fontsize=12, fontweight='bold', fontfamily='SimHei')
        ax6.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_entropy = np.mean(entropies)
        ax6.axvline(mean_entropy, color='red', linestyle='--', 
                   label=f'平均熵: {mean_entropy:.3f}')
        ax6.legend(prop={'family': 'SimHei'}, title_fontsize=10)
        
        plt.suptitle(f'{classifier_name} - 3D概率体综合分析', fontsize=16, fontweight='bold', fontfamily='SimHei')
        plt.tight_layout()
        plt.savefig(f'3d_probability_volume_{classifier_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_3d_isosurfaces(self, classifier_name, isovalues=[0.3, 0.5, 0.7]):
        """创建3D等值面"""
        if classifier_name not in self.models:
            print(f"分类器 {classifier_name} 未找到")
            return
            
        print(f"\n正在为 {classifier_name} 创建3D等值面...")
        
        model = self.models[classifier_name]
        
        # 创建3D网格
        resolution = 0.08
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        z_min, z_max = self.X_train_scaled[:, 2].min() - 1, self.X_train_scaled[:, 2].max() + 1
        
        x_range = np.arange(x_min, x_max, resolution)
        y_range = np.arange(y_min, y_max, resolution)
        z_range = np.arange(z_min, z_max, resolution)
        
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # 扁平化网格点
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        print(f"  正在预测 {len(grid_points)} 个网格点...")
        
        # 获取预测概率
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(grid_points)
        else:
            decision = model.decision_function(grid_points)
            probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
        
        # 创建图形
        fig = plt.figure(figsize=(20, 15))
        
        # 为每个类别和每个等值面创建子图
        for class_idx, class_name in enumerate(self.target_names):
            prob_volume = probas[:, class_idx].reshape(xx.shape)
            
            for iso_idx, isovalue in enumerate(isovalues):
                ax = fig.add_subplot(len(self.target_names), len(isovalues), 
                                   class_idx * len(isovalues) + iso_idx + 1, projection='3d')
                
                # 找到等值面点（简化版）
                vertices = []
                for i in range(xx.shape[0]-1):
                    for j in range(xx.shape[1]-1):
                        for k in range(xx.shape[2]-1):
                            # 检查立方体是否跨越等值面
                            cube_values = [
                                prob_volume[i, j, k],
                                prob_volume[i+1, j, k],
                                prob_volume[i, j+1, k],
                                prob_volume[i+1, j+1, k],
                                prob_volume[i, j, k+1],
                                prob_volume[i+1, j, k+1],
                                prob_volume[i, j+1, k+1],
                                prob_volume[i+1, j+1, k+1]
                            ]
                            
                            min_val, max_val = min(cube_values), max(cube_values)
                            if min_val <= isovalue <= max_val:
                                # 添加立方体中心的点作为等值面点
                                center_x = (xx[i, j, k] + xx[i+1, j, k]) / 2
                                center_y = (yy[i, j, k] + yy[i, j+1, k]) / 2
                                center_z = (zz[i, j, k] + zz[i, j, k+1]) / 2
                                vertices.append([center_x, center_y, center_z])
                
                if vertices:
                    vertices = np.array(vertices)
                    # 确保颜色是单一颜色而不是数组
                    color = self.colors[class_idx]
                    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                              c=color, marker='.',
                              s=10, alpha=0.6,
                              label=f'{class_name} 等值面: {isovalue}')
                
                # 绘制训练数据点
                for i, target_name in enumerate(self.target_names):
                    mask = self.y_train == i
                    # 确保颜色是单一颜色而不是数组
                    color = self.colors[i]
                    ax.scatter(self.X_train_scaled[mask, 0], 
                              self.X_train_scaled[mask, 1], 
                              self.X_train_scaled[mask, 2],
                              c=color, 
                              marker=self.markers[i],
                              s=60, edgecolors='black', linewidth=1,
                              label=f'{target_name}')
                
                ax.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
                ax.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
                ax.set_zlabel(f'标准化 {self.feature_names[2]}', fontfamily='SimHei')
                ax.set_title(f'{class_name} - 等值面: {isovalue}', 
                            fontsize=10, fontweight='bold', fontfamily='SimHei')
                ax.legend(fontsize=6, prop={'family': 'SimHei'})
                ax.view_init(elev=20, azim=45)
        
        plt.suptitle(f'{classifier_name} - 3D概率等值面分析', fontsize=16, fontweight='bold', fontfamily='SimHei')
        plt.tight_layout()
        plt.savefig(f'3d_isosurfaces_{classifier_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_3d_probability(self, classifier_name):
        """创建交互式3D概率可视化（需要Plotly）"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            if classifier_name not in self.models:
                print(f"分类器 {classifier_name} 未找到")
                return
                
            print(f"\n正在为 {classifier_name} 创建交互式3D概率可视化...")
            
            model = self.models[classifier_name]
            
            # 创建3D网格
            resolution = 0.15
            x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
            y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
            z_min, z_max = self.X_train_scaled[:, 2].min() - 1, self.X_train_scaled[:, 2].max() + 1
            
            x_range = np.arange(x_min, x_max, resolution)
            y_range = np.arange(y_min, y_max, resolution)
            z_range = np.arange(z_min, z_max, resolution)
            
            xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            
            # 扁平化网格点
            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            
            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(grid_points)
            else:
                decision = model.decision_function(grid_points)
                probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            
            # 创建图形
            fig = go.Figure()
            
            # 为每个类别创建等值面
            isovalues = [0.3, 0.5, 0.7]
            
            for class_idx, class_name in enumerate(self.target_names):
                prob_volume = probas[:, class_idx].reshape(xx.shape)
                
                for iso_idx, isovalue in enumerate(isovalues):
                    opacity = 0.3 - iso_idx * 0.1  # 外层更透明
                    
                    fig.add_trace(go.Isosurface(
                        x=xx.ravel(),
                        y=yy.ravel(),
                        z=zz.ravel(),
                        value=prob_volume.ravel(),
                        isomin=isovalue,
                        isomax=isovalue,
                        surface=dict(count=1),
                        opacity=opacity,
                        colorscale=[[0, self.colors[class_idx]], [1, self.colors[class_idx]]],
                        name=f'{class_name} 等值面: {isovalue}',
                        showscale=False
                    ))
            
            # 添加数据点
            for i, target_name in enumerate(self.target_names):
                mask = self.y_train == i
                fig.add_trace(go.Scatter3d(
                    x=self.X_train_scaled[mask, 0],
                    y=self.X_train_scaled[mask, 1],
                    z=self.X_train_scaled[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.colors[i],
                        line=dict(width=2, color='black')
                    ),
                    name=f'{target_name} (训练数据)'
                ))
            
            fig.update_layout(
                title=dict(
                    text=f'{classifier_name} - 交互式3D概率等值面',
                    font=dict(family='SimHei')
                ),
                scene=dict(
                    xaxis_title=f'标准化 {self.feature_names[0]}',
                    yaxis_title=f'标准化 {self.feature_names[1]}',
                    zaxis_title=f'标准化 {self.feature_names[2]}',
                    xaxis=dict(tickfont=dict(family='SimHei')),
                    yaxis=dict(tickfont=dict(family='SimHei')),
                    zaxis=dict(tickfont=dict(family='SimHei'))
                ),
                width=1200, height=900
            )
            
            # 保存为PNG格式（已删除HTML生成功能）
            fig.write_image(f'interactive_3d_probability_{classifier_name}.png', width=1200, height=800)
            print(f"交互式3D概率可视化已保存为 interactive_3d_probability_{classifier_name}.png")
            
        except ImportError:
            print("Plotly未安装。")
            
    def create_probability_comparison_across_classifiers(self):
        """创建多个分类器的概率对比"""
        
        # 创建3D网格
        resolution = 0.12
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        z_min, z_max = self.X_train_scaled[:, 2].min() - 1, self.X_train_scaled[:, 2].max() + 1
        
        x_range = np.arange(x_min, x_max, resolution)
        y_range = np.arange(y_min, y_max, resolution)
        z_range = np.arange(z_min, z_max, resolution)
        
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # 扁平化网格点
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        # 选择几个分类器
        selected_classifiers = ['LogisticRegression', 'SVM_RBF', 'RandomForest']
        available_classifiers = [name for name in selected_classifiers if name in self.models]
        
        # 创建图形
        fig = plt.figure(figsize=(20, 15))
        
        # 为每个分类器和类别创建子图
        for classifier_idx, classifier_name in enumerate(available_classifiers):
            model = self.models[classifier_name]
            
            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(grid_points)
            else:
                decision = model.decision_function(grid_points)
                probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            
            for class_idx, class_name in enumerate(self.target_names):
                ax = fig.add_subplot(len(available_classifiers), len(self.target_names), 
                                   classifier_idx * len(self.target_names) + class_idx + 1, 
                                   projection='3d')
                
                prob_volume = probas[:, class_idx].reshape(xx.shape)
                
                # 找到高概率区域（简化版等值面）
                high_prob_mask = prob_volume > 0.5
                
                if np.any(high_prob_mask):
                    # 获取高概率点的坐标
                    high_prob_indices = np.where(high_prob_mask)
                    
                    # 降采样以提高性能
                    if len(high_prob_indices[0]) > 1000:
                        sample_indices = np.random.choice(len(high_prob_indices[0]), size=1000, replace=False)
                    else:
                        sample_indices = np.arange(len(high_prob_indices[0]))
                    
                    x_coords = xx[high_prob_indices[0][sample_indices], 
                                high_prob_indices[1][sample_indices], 
                                high_prob_indices[2][sample_indices]]
                    y_coords = yy[high_prob_indices[0][sample_indices], 
                                high_prob_indices[1][sample_indices], 
                                high_prob_indices[2][sample_indices]]
                    z_coords = zz[high_prob_indices[0][sample_indices], 
                                high_prob_indices[1][sample_indices], 
                                high_prob_indices[2][sample_indices]]
                    
                    # 绘制高概率区域
                    ax.scatter(x_coords, y_coords, z_coords,
                              c=self.colors[class_idx], marker='.',
                              alpha=0.6, label=f'{class_name} (p>0.5)')
                
                # 绘制训练数据点
                for i, target_name in enumerate(self.target_names):
                    mask = self.y_train == i
                    # 确保颜色是单一颜色而不是数组
                    color = self.colors[i]
                    ax.scatter(self.X_train_scaled[mask, 0], 
                              self.X_train_scaled[mask, 1], 
                              self.X_train_scaled[mask, 2],
                              c=[color], 
                              marker=self.markers[i],
                              s=60, edgecolors='black', linewidth=1,
                              label=f'{target_name}')
                
                ax.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
                ax.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
                ax.set_zlabel(f'标准化 {self.feature_names[2]}', fontfamily='SimHei')
                ax.set_title(f'{classifier_name}\n{class_name} 概率分布', 
                            fontsize=10, fontweight='bold', fontfamily='SimHei')
                ax.legend(fontsize=6, prop={'family': 'SimHei'})
                ax.view_init(elev=20, azim=45)
        
        plt.suptitle('多分类器3D概率分布对比', fontsize=16, fontweight='bold', fontfamily='SimHei')
        plt.tight_layout()
        plt.savefig('3d_probability_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_3d_probability_report(self):
        """生成3D概率体可视化分析报告"""
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 训练所有模型
        self.train_all_models()
        
        # 为每个分类器创建3D概率体可视化
        for classifier_name in ['LogisticRegression', 'SVM_RBF', 'RandomForest']:
            if classifier_name in self.models:
                print(f"\n正在分析 {classifier_name}...")
                self.create_3d_probability_volume(classifier_name)
                self.create_3d_isosurfaces(classifier_name)
                self.create_interactive_3d_probability(classifier_name)
        
        # 创建多分类器概率对比
        self.create_probability_comparison_across_classifiers()
        
        
        return {
            'models': self.models,
            'results': self.results,
            'generated_files': [
                '3d_probability_volume_*.png',
                '3d_isosurfaces_*.png',
                'interactive_3d_probability_*.png',
                '3d_probability_comparison.png'
            ]
        }


def main():
    """主函数"""
    
    # 创建3D概率可视化器
    visualizer = Iris3DProbabilityVisualizer()
    
    # 生成3D概率体可视化分析报告
    results = visualizer.generate_3d_probability_report()

if __name__ == "__main__":
    main()