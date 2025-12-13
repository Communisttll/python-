"""
三维特征空间决策边界可视化
实现多个分类器在三维特征空间中的决策边界和分类曲面
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class Iris3DVisualizer:
    """鸢尾花三维特征空间可视化器"""
    
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
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'GaussianNB': GaussianNB(),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis()
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
                
    def create_3d_scatter_plot(self):
        """创建3D散点图"""
        print("\n正在创建3D散点图...")
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制每个类别的数据点
        for i, target_name in enumerate(self.target_names):
            mask = self.y_train == i
            ax.scatter(self.X_train_scaled[mask, 0], 
                      self.X_train_scaled[mask, 1], 
                      self.X_train_scaled[mask, 2],
                      c=self.colors[i], 
                      marker=self.markers[i],
                      s=100, 
                      label=f'{target_name} (训练集)',
                      alpha=0.8)
        
        # 绘制测试集点（较小且半透明）
        for i, target_name in enumerate(self.target_names):
            mask = self.y_test == i
            ax.scatter(self.X_test_scaled[mask, 0], 
                      self.X_test_scaled[mask, 1], 
                      self.X_test_scaled[mask, 2],
                      c=self.colors[i], 
                      marker=self.markers[i],
                      s=50, 
                      alpha=0.4)
        
        ax.set_xlabel(f'标准化 {self.feature_names[0]}', fontsize=12, fontfamily='SimHei')
        ax.set_ylabel(f'标准化 {self.feature_names[1]}', fontsize=12, fontfamily='SimHei')
        ax.set_zlabel(f'标准化 {self.feature_names[2]}', fontsize=12, fontfamily='SimHei')
        ax.set_title('鸢尾花三维特征空间分布', fontsize=16, fontweight='bold', fontfamily='SimHei')
        ax.legend(fontsize=10, prop={'family': 'SimHei'})
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig('3d_scatter_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_3d_decision_boundary_single(self, classifier_name, resolution=0.1):
        """创建单个分类器的3D决策边界"""
        if classifier_name not in self.models:
            print(f"分类器 {classifier_name} 未找到")
            return
            
        print(f"\n正在为 {classifier_name} 创建3D决策边界...")
        
        model = self.models[classifier_name]
        
        # 创建3D网格点
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        z_min, z_max = self.X_train_scaled[:, 2].min() - 1, self.X_train_scaled[:, 2].max() + 1
        
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, resolution),
                                 np.arange(y_min, y_max, resolution),
                                 np.arange(z_min, z_max, resolution))
        
        # 扁平化网格点
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        # 预测
        print(f"  正在预测 {len(grid_points)} 个网格点...")
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        # 主图：3D决策边界
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 绘制决策边界（使用散点表示不同类别）
        for class_idx in range(len(self.target_names)):
            mask = Z == class_idx
            if np.any(mask):
                ax1.scatter(grid_points[mask.ravel(), 0][::50],  # 降采样以提高性能
                           grid_points[mask.ravel(), 1][::50],
                           grid_points[mask.ravel(), 2][::50],
                           c=self.colors[class_idx], 
                           marker='.',
                           s=1, alpha=0.3,
                           label=f'决策区域: {self.target_names[class_idx]}')
        
        # 绘制训练数据点
        for i, target_name in enumerate(self.target_names):
            mask = self.y_train == i
            ax1.scatter(self.X_train_scaled[mask, 0], 
                       self.X_train_scaled[mask, 1], 
                       self.X_train_scaled[mask, 2],
                       c=self.colors[i], 
                       marker=self.markers[i],
                       s=60, edgecolors='black', linewidth=1.5,
                       label=f'{target_name} (训练数据)')
        
        ax1.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
        ax1.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
        ax1.set_zlabel(f'标准化 {self.feature_names[2]}', fontfamily='SimHei')
        ax1.set_title(f'{classifier_name} - 3D决策边界\n准确率: {self.results[classifier_name]["accuracy"]:.3f}', 
                     fontsize=14, fontweight='bold', fontfamily='SimHei')
        ax1.legend(fontsize=8, prop={'family': 'SimHei'})
        ax1.view_init(elev=20, azim=45)
        
        # 2D投影图 - XY平面
        ax2 = fig.add_subplot(222)
        self._plot_2d_projection(ax2, model, 0, 1, f'{classifier_name} - XY平面投影')
        
        # 2D投影图 - XZ平面
        ax3 = fig.add_subplot(223)
        self._plot_2d_projection(ax3, model, 0, 2, f'{classifier_name} - XZ平面投影')
        
        # 2D投影图 - YZ平面
        ax4 = fig.add_subplot(224)
        self._plot_2d_projection(ax4, model, 1, 2, f'{classifier_name} - YZ平面投影')
        
        plt.tight_layout()
        plt.savefig(f'3d_decision_boundary_{classifier_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_2d_projection(self, ax, model, feature1_idx, feature2_idx, title):
        """绘制2D投影"""
        # 创建网格点
        h = 0.02
        x_min, x_max = self.X_train_scaled[:, feature1_idx].min() - 1, self.X_train_scaled[:, feature1_idx].max() + 1
        y_min, y_max = self.X_train_scaled[:, feature2_idx].min() - 1, self.X_train_scaled[:, feature2_idx].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 创建3D网格点（固定第三个特征为均值）
        grid_points = np.zeros((xx.ravel().shape[0], 3))
        grid_points[:, feature1_idx] = xx.ravel()
        grid_points[:, feature2_idx] = yy.ravel()
        grid_points[:, 3-feature1_idx-feature2_idx] = self.X_train_scaled[:, 3-feature1_idx-feature2_idx].mean()
        
        # 预测
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        
        # 绘制数据点
        for i, target_name in enumerate(self.target_names):
            ax.scatter(self.X_train_scaled[self.y_train == i, feature1_idx], 
                      self.X_train_scaled[self.y_train == i, feature2_idx],
                      c=self.colors[i], marker=self.markers[i],
                      s=60, edgecolors='black', linewidth=1,
                      label=target_name)
        
        ax.set_xlabel(f'标准化 {self.feature_names[feature1_idx]}', fontfamily='SimHei')
        ax.set_ylabel(f'标准化 {self.feature_names[feature2_idx]}', fontfamily='SimHei')
        ax.set_title(title, fontsize=12, fontweight='bold', fontfamily='SimHei')
        ax.legend(fontsize=8, prop={'family': 'SimHei'})
        ax.grid(True, alpha=0.3)
        
    def create_3d_decision_boundary_comparison(self):
        """创建多个分类器的3D决策边界对比"""
        print("\n正在创建3D决策边界对比...")
        
        # 选择几个代表性的分类器
        selected_classifiers = ['LogisticRegression', 'SVM_RBF', 'RandomForest', 'GaussianNB']
        available_classifiers = [name for name in selected_classifiers if name in self.models]
        
        n_classifiers = len(available_classifiers)
        n_cols = 2
        n_rows = (n_classifiers + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(16, 8*n_rows))
        
        for i, classifier_name in enumerate(available_classifiers):
            model = self.models[classifier_name]
            
            # 创建3D网格点（降低分辨率以提高性能）
            resolution = 0.2
            x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
            y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
            z_min, z_max = self.X_train_scaled[:, 2].min() - 1, self.X_train_scaled[:, 2].max() + 1
            
            xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, resolution),
                                     np.arange(y_min, y_max, resolution),
                                     np.arange(z_min, z_max, resolution))
            
            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            
            # 预测（只预测部分点以提高性能）
            sample_indices = np.random.choice(len(grid_points), size=min(5000, len(grid_points)), replace=False)
            sample_points = grid_points[sample_indices]
            
            Z_sample = model.predict(sample_points)
            
            # 创建子图
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
            
            # 绘制决策边界点
            for class_idx in range(len(self.target_names)):
                mask = Z_sample == class_idx
                if np.any(mask):
                    ax.scatter(sample_points[mask, 0],
                              sample_points[mask, 1],
                              sample_points[mask, 2],
                              c=self.colors[class_idx], 
                              marker='.',
                              s=2, alpha=0.6,
                              label=f'决策区域: {self.target_names[class_idx]}')
            
            # 绘制训练数据点
            for j, target_name in enumerate(self.target_names):
                mask = self.y_train == j
                ax.scatter(self.X_train_scaled[mask, 0], 
                          self.X_train_scaled[mask, 1], 
                          self.X_train_scaled[mask, 2],
                          c=self.colors[j], 
                          marker=self.markers[j],
                          s=80, edgecolors='black', linewidth=1.5,
                          label=f'{target_name} (训练)')
            
            ax.set_xlabel(f'{self.feature_names[0]}', fontfamily='SimHei')
            ax.set_ylabel(f'{self.feature_names[1]}', fontfamily='SimHei')
            ax.set_zlabel(f'{self.feature_names[2]}', fontfamily='SimHei')
            ax.set_title(f'{classifier_name}\n准确率: {self.results[classifier_name]["accuracy"]:.3f}', 
                        fontsize=12, fontweight='bold', fontfamily='SimHei')
            ax.legend(fontsize=6, prop={'family': 'SimHei'})
            ax.view_init(elev=20, azim=45)
            
        plt.tight_layout()
        plt.savefig('3d_decision_boundary_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_3d_probability_surface(self, classifier_name, feature3_value=None):
        """创建3D概率曲面"""
        if classifier_name not in self.models:
            print(f"分类器 {classifier_name} 未找到")
            return
            
        print(f"\n正在为 {classifier_name} 创建3D概率曲面...")
        
        model = self.models[classifier_name]
        
        # 如果未指定第三个特征值，使用训练数据的均值
        if feature3_value is None:
            feature3_value = self.X_train_scaled[:, 2].mean()
        
        # 创建2D网格
        h = 0.05
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 创建3D网格点（固定第三个特征）
        grid_points = np.zeros((xx.ravel().shape[0], 3))
        grid_points[:, 0] = xx.ravel()
        grid_points[:, 1] = yy.ravel()
        grid_points[:, 2] = feature3_value
        
        # 获取预测概率
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(grid_points)
        else:
            decision = model.decision_function(grid_points)
            probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
        
        # 创建图形
        fig = plt.figure(figsize=(20, 15))
        
        # 为每个类别创建概率曲面
        for i, target_name in enumerate(self.target_names):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            prob_surface = probas[:, i].reshape(xx.shape)
            
            # 创建曲面图
            surf = ax.plot_surface(xx, yy, prob_surface, 
                                 cmap=plt.cm.RdYlBu,
                                 alpha=0.8, 
                                 rstride=1, cstride=1,
                                 linewidth=0, antialiased=True)
            
            # 添加等高线投影
            ax.contourf(xx, yy, prob_surface, zdir='z', offset=0, 
                       cmap=plt.cm.RdYlBu, alpha=0.3)
            
            # 绘制数据点（在固定特征值附近）
            tolerance = 0.2
            mask = np.abs(self.X_train_scaled[:, 2] - feature3_value) < tolerance
            
            for j, class_name in enumerate(self.target_names):
                class_mask = mask & (self.y_train == j)
                if np.any(class_mask):
                    ax.scatter(self.X_train_scaled[class_mask, 0],
                             self.X_train_scaled[class_mask, 1],
                             np.full(np.sum(class_mask), 0),
                             c=self.colors[j], 
                             marker=self.markers[j],
                             s=80, edgecolors='black', linewidth=1.5,
                             label=f'{class_name}')
            
            ax.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
            ax.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
            ax.set_zlabel('概率', fontfamily='SimHei')
            ax.set_title(f'{classifier_name} - {target_name} 概率曲面\n({self.feature_names[2]} ≈ {feature3_value:.2f})', 
                        fontsize=12, fontweight='bold', fontfamily='SimHei')
            ax.set_zlim(0, 1)
            
            # 添加颜色条
            cbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
            cbar.set_label('概率', fontfamily='SimHei')
            
            ax.view_init(elev=25, azim=45)
        
        # 添加总体标题
        fig.suptitle(f'{classifier_name} - 3D概率曲面分析', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        plt.tight_layout()
        plt.savefig(f'3d_probability_surface_{classifier_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_3d_animation_views(self, classifier_name):
        """创建多角度3D视图"""
        if classifier_name not in self.models:
            print(f"分类器 {classifier_name} 未找到")
            return
            
        print(f"\n正在为 {classifier_name} 创建多角度3D视图...")
        
        model = self.models[classifier_name]
        
        # 创建3D网格点（低分辨率）
        resolution = 0.15
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        z_min, z_max = self.X_train_scaled[:, 2].min() - 1, self.X_train_scaled[:, 2].max() + 1
        
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, resolution),
                                 np.arange(y_min, y_max, resolution),
                                 np.arange(z_min, z_max, resolution))
        
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        # 预测（采样以提高性能）
        sample_indices = np.random.choice(len(grid_points), size=min(3000, len(grid_points)), replace=False)
        sample_points = grid_points[sample_indices]
        
        Z_sample = model.predict(sample_points)
        
        # 不同角度的视图
        view_angles = [(20, 0), (20, 45), (20, 90), (20, 135), (20, 180), (20, 225), (20, 270), (20, 315)]
        
        fig = plt.figure(figsize=(20, 20))
        
        for i, (elev, azim) in enumerate(view_angles):
            ax = fig.add_subplot(2, 4, i+1, projection='3d')
            
            # 绘制决策边界点
            for class_idx in range(len(self.target_names)):
                mask = Z_sample == class_idx
                if np.any(mask):
                    ax.scatter(sample_points[mask, 0],
                              sample_points[mask, 1],
                              sample_points[mask, 2],
                              c=self.colors[class_idx], 
                              marker='.',
                              s=1, alpha=0.5,
                              label=f'决策区域: {self.target_names[class_idx]}')
            
            # 绘制训练数据点
            for j, target_name in enumerate(self.target_names):
                mask = self.y_train == j
                ax.scatter(self.X_train_scaled[mask, 0], 
                          self.X_train_scaled[mask, 1], 
                          self.X_train_scaled[mask, 2],
                          c=self.colors[j], 
                          marker=self.markers[j],
                          s=60, edgecolors='black', linewidth=1,
                          label=f'{target_name}')
            
            ax.set_xlabel(f'{self.feature_names[0]}', fontfamily='SimHei')
            ax.set_ylabel(f'{self.feature_names[1]}', fontfamily='SimHei')
            ax.set_zlabel(f'{self.feature_names[2]}', fontfamily='SimHei')
            ax.set_title(f'{classifier_name} - 视角: {elev}°, {azim}°', 
                        fontsize=10, fontweight='bold', fontfamily='SimHei')
            ax.view_init(elev=elev, azim=azim)
            
            # 只在第一个子图显示图例
            if i == 0:
                ax.legend(fontsize=6, prop={'family': 'SimHei'})
        
        plt.suptitle(f'{classifier_name} - 多角度3D决策边界视图', fontsize=16, fontweight='bold', fontfamily='SimHei')
        plt.tight_layout()
        plt.savefig(f'3d_animation_views_{classifier_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_3d_isosurface_visualization(self, classifier_name):
        """创建3D等值面可视化（需要Plotly）"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            if classifier_name not in self.models:
                print(f"分类器 {classifier_name} 未找到")
                return
                
            print(f"\n正在为 {classifier_name} 创建3D等值面可视化...")
            
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
            
            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            
            # 预测概率
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(grid_points)
            else:
                decision = model.decision_function(grid_points)
                probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            
            # 创建图形
            fig = go.Figure()
            
            # 为每个类别创建等值面
            for i, target_name in enumerate(self.target_names):
                prob_class = probas[:, i].reshape(xx.shape)
                
                # 创建等值面（概率=0.5）
                fig.add_trace(go.Isosurface(
                    x=xx.ravel(),
                    y=yy.ravel(),
                    z=zz.ravel(),
                    value=prob_class.ravel(),
                    isomin=0.5,
                    isomax=0.5,
                    surface=dict(count=1),
                    opacity=0.3,
                    colorscale=[[0, self.colors[i]], [1, self.colors[i]]],
                    name=f'{target_name} 决策边界',
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
                        size=6,
                        color=self.colors[i],
                        line=dict(width=2, color='black')
                    ),
                    name=f'{target_name} (训练数据)'
                ))
            
            fig.update_layout(
                title=dict(
                    text=f'{classifier_name} - 3D概率等值面',
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
                width=1000, height=800
            )
            
            # 保存为PNG格式（已删除HTML生成功能）
            fig.write_image(f'3d_isosurface_{classifier_name}.png', width=1200, height=800)
            print(f"3D等值面可视化已保存为 3d_isosurface_{classifier_name}.png")
            
        except ImportError:
            print("Plotly未安装")
            
    def generate_3d_visualization_report(self):
        """生成3D可视化分析报告"""
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 训练所有模型
        self.train_all_models()
        
        # 创建3D散点图
        self.create_3d_scatter_plot()
        
        # 为每个分类器创建3D决策边界
        for classifier_name in ['LogisticRegression', 'SVM_RBF', 'RandomForest']:
            if classifier_name in self.models:
                self.create_3d_decision_boundary_single(classifier_name)
                self.create_3d_probability_surface(classifier_name)
                self.create_3d_animation_views(classifier_name)
                self.create_3d_isosurface_visualization(classifier_name)
        
        # 创建决策边界对比
        self.create_3d_decision_boundary_comparison()
        
        return {
            'models': self.models,
            'results': self.results,
            'generated_files': [
                '3d_scatter_plot.png',
                '3d_decision_boundary_*.png',
                '3d_probability_surface_*.png',
                '3d_animation_views_*.png',
                '3d_isosurface_*.png',
                '3d_decision_boundary_comparison.png'
            ]
        }


def main():
    """主函数"""
    
    # 创建3D可视化器
    visualizer = Iris3DVisualizer()
    
    # 生成3D可视化分析报告
    results = visualizer.generate_3d_visualization_report()


if __name__ == "__main__":
    main()