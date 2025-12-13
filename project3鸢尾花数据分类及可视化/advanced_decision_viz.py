"""
高级决策边界与概率可视化分析
包含概率热图、不确定性量化、多分类器决策边界对比等高级可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class AdvancedDecisionBoundaryVisualizer:
    """高级决策边界可视化类"""
    
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
            'LDA': LinearDiscriminantAnalysis()
        }
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("正在加载鸢尾花数据集...")
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # 选择前两个特征用于2D可视化
        self.X_2d = self.X[:, :2]
        
        # 数据划分
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_2d, self.y, test_size=0.3, random_state=42, stratify=self.y)
        
        # 数据标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"数据准备完成:")
        print(f"  训练集大小: {self.X_train.shape}")
        print(f"  测试集大小: {self.X_test.shape}")
        print(f"  使用特征: {self.feature_names[0]} 和 {self.feature_names[1]}")
        
    def train_all_models(self):
        """训练所有模型"""
        print("\n正在训练所有分类器...")
        
        for name, model in self.classifiers.items():
            print(f"  训练 {name}...")
            
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
            
    def calculate_prediction_entropy(self, probabilities):
        """计算预测熵（不确定性）"""
        # 避免log(0)
        probabilities = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
        return entropy
        
    def create_decision_boundary_comparison(self):
        """创建决策边界对比"""
        print("\n正在创建决策边界对比...")
        
        # 创建网格点
        h = 0.02  # 网格步长
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 创建子图
        n_classifiers = len(self.classifiers)
        n_cols = 3
        n_rows = (n_classifiers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, (name, model) in enumerate(self.models.items()):
            # 预测网格点
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(grid_points)
            Z = Z.reshape(xx.shape)
            
            # 绘制决策边界
            axes[i].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
            
            # 绘制训练数据点
            scatter = axes[i].scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                    c=self.y_train, cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
            
            # 设置标题和标签
            axes[i].set_title(f'{name}\n准确率: {self.results[name]["accuracy"]:.3f}', 
                            fontsize=12, fontweight='bold', fontfamily='SimHei')
            axes[i].set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
            axes[i].set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
            axes[i].grid(True, alpha=0.3)
            
        # 隐藏多余的子图
        for i in range(n_classifiers, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig('decision_boundaries_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_probability_heatmaps(self):
        """创建概率热图"""
        print("\n正在创建概率热图...")
        
        # 创建网格点
        h = 0.02
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 选择几个代表性的分类器
        selected_classifiers = ['LogisticRegression', 'SVM_RBF', 'RandomForest', 'GaussianNB']
        
        for name in selected_classifiers:
            if name not in self.models:
                continue
                
            model = self.models[name]
            
            # 预测概率
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(grid_points)
            else:
                # 对于没有predict_proba的模型，使用决策函数
                decision = model.decision_function(grid_points)
                # 将决策值转换为概率（使用softmax）
                probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            
            # 创建子图显示每个类别的概率
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制每个类别的概率热图
            for i in range(len(self.target_names)):
                ax = axes[i//2, i%2]
                prob_class = probas[:, i].reshape(xx.shape)
                
                # 绘制概率热图
                contour = ax.contourf(xx, yy, prob_class, levels=20, alpha=0.7, cmap='RdYlBu')
                
                # 绘制训练数据点
                for j, target_name in enumerate(self.target_names):
                    mask = self.y_train == j
                    ax.scatter(self.X_train_scaled[mask, 0], self.X_train_scaled[mask, 1], 
                             label=target_name, edgecolors='black', s=50, alpha=0.8)
                
                # 添加概率等值线
                ax.contour(xx, yy, prob_class, levels=[0.5, 0.7, 0.9], 
                          colors='black', linewidths=1, linestyles='--')
                
                ax.set_title(f'{name} - {self.target_names[i]} 类别概率', 
                           fontsize=12, fontweight='bold', fontfamily='SimHei')
                ax.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
                ax.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
                ax.legend(prop={'family': 'SimHei'})
                ax.grid(True, alpha=0.3)
                
                # 添加颜色条
                cbar = plt.colorbar(contour, ax=ax, label='概率')
                cbar.ax.set_ylabel('概率', fontfamily='SimHei')
            
            plt.suptitle(f'{name} 分类概率热图', fontsize=16, fontweight='bold', fontfamily='SimHei')
            plt.tight_layout()
            plt.savefig(f'probability_heatmap_{name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def create_uncertainty_visualization(self):
        """创建不确定性可视化"""
        print("\n正在创建不确定性可视化...")
        
        # 创建网格点
        h = 0.02
        x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
        y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 选择几个分类器进行不确定性分析
        selected_classifiers = ['LogisticRegression', 'SVM_RBF', 'RandomForest']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, name in enumerate(selected_classifiers):
            if name not in self.models:
                continue
                
            model = self.models[name]
            
            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(grid_points)
            else:
                decision = model.decision_function(grid_points)
                probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            
            # 计算熵（不确定性）
            entropy = self.calculate_prediction_entropy(probas)
            entropy_grid = entropy.reshape(xx.shape)
            
            # 绘制不确定性热图
            ax1 = axes[0, i]
            contour1 = ax1.contourf(xx, yy, entropy_grid, levels=20, alpha=0.7, cmap='hot')
            
            # 绘制数据点
            scatter = ax1.scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                c=self.y_train, cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
            
            ax1.set_title(f'{name} - 预测不确定性 (熵)', fontsize=12, fontweight='bold', fontfamily='SimHei')
            ax1.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
            ax1.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
            ax1.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(contour1, ax=ax1, label='熵 (不确定性)')
            cbar1.ax.set_ylabel('熵 (不确定性)', fontfamily='SimHei')
            
            # 绘制置信度热图
            ax2 = axes[1, i]
            max_proba = np.max(probas, axis=1)
            confidence_grid = max_proba.reshape(xx.shape)
            
            contour2 = ax2.contourf(xx, yy, confidence_grid, levels=20, alpha=0.7, cmap='coolwarm')
            
            # 绘制置信度等值线
            ax2.contour(xx, yy, confidence_grid, levels=[0.5, 0.7, 0.9], 
                       colors='black', linewidths=1, linestyles='--')
            
            # 绘制数据点
            scatter2 = ax2.scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], 
                                 c=self.y_train, cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
            
            ax2.set_title(f'{name} - 预测置信度', fontsize=12, fontweight='bold', fontfamily='SimHei')
            ax2.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
            ax2.set_ylabel(f'标准化 {self.feature_names[1]}', fontfamily='SimHei')
            ax2.grid(True, alpha=0.3)
            cbar2 = plt.colorbar(contour2, ax=ax2, label='最大概率')
            cbar2.ax.set_ylabel('最大概率', fontfamily='SimHei')
            
        plt.tight_layout()
        plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_prediction_confidence_analysis(self):
        """创建预测置信度分析"""
        print("\n正在创建预测置信度分析...")
        
        # 对每个分类器分析预测置信度
        selected_classifiers = ['LogisticRegression', 'SVM_RBF', 'RandomForest']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, name in enumerate(selected_classifiers):
            if name not in self.models:
                continue
                
            model = self.models[name]
            
            # 获取测试集的预测概率
            if hasattr(model, 'predict_proba'):
                test_probas = model.predict_proba(self.X_test_scaled)
            else:
                decision = model.decision_function(self.X_test_scaled)
                test_probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
            
            # 计算最大概率（置信度）
            max_probas = np.max(test_probas, axis=1)
            
            # 计算熵
            entropy = self.calculate_prediction_entropy(test_probas)
            
            if i == 0:
                # 第一个分类器：置信度分布直方图
                axes[0, 0].hist(max_probas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 0].set_title(f'{name} - 预测置信度分布', fontsize=12, fontweight='bold', fontfamily='SimHei')
                axes[0, 0].set_xlabel('最大概率', fontfamily='SimHei')
                axes[0, 0].set_ylabel('频次', fontfamily='SimHei')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_conf = np.mean(max_probas)
                axes[0, 0].axvline(mean_conf, color='red', linestyle='--', 
                                  label=f'平均置信度: {mean_conf:.3f}')
                axes[0, 0].legend(prop={'family': 'SimHei'})
                
            elif i == 1:
                # 第二个分类器：置信度 vs 正确性
                correct = (self.results[name]['predictions'] == self.y_test)
                
                correct_conf = max_probas[correct]
                incorrect_conf = max_probas[~correct]
                
                axes[0, 1].hist(correct_conf, bins=15, alpha=0.7, color='green', 
                               label='正确预测', edgecolor='black')
                axes[0, 1].hist(incorrect_conf, bins=15, alpha=0.7, color='red', 
                               label='错误预测', edgecolor='black')
                axes[0, 1].set_title(f'{name} - 置信度 vs 预测正确性', fontsize=12, fontweight='bold', fontfamily='SimHei')
                axes[0, 1].set_xlabel('最大概率', fontfamily='SimHei')
                axes[0, 1].set_ylabel('频次', fontfamily='SimHei')
                axes[0, 1].legend(prop={'family': 'SimHei'})
                axes[0, 1].grid(True, alpha=0.3)
                
            elif i == 2:
                # 第三个分类器：熵 vs 置信度散点图
                axes[1, 0].scatter(max_probas, entropy, alpha=0.6, s=50)
                axes[1, 0].set_title(f'{name} - 置信度 vs 不确定性', fontsize=12, fontweight='bold', fontfamily='SimHei')
                axes[1, 0].set_xlabel('最大概率 (置信度)', fontfamily='SimHei')
                axes[1, 0].set_ylabel('熵 (不确定性)', fontfamily='SimHei')
                axes[1, 0].grid(True, alpha=0.3)
                
                # 添加相关系数
            correlation = np.corrcoef(max_probas, entropy)[0, 1]
            axes[1, 0].text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                           transform=axes[1, 0].transAxes, fontsize=10, fontfamily='SimHei',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            axes[1, 0].set_title(f'{name} - 置信度 vs 不确定性', fontsize=12, fontweight='bold', fontfamily='SimHei')
            axes[1, 0].set_xlabel('最大概率 (置信度)', fontfamily='SimHei')
            axes[1, 0].set_ylabel('熵 (不确定性)', fontfamily='SimHei')
        
        # 第四个图：所有分类器的平均置信度对比
        classifier_names = []
        mean_confidences = []
        
        for name in selected_classifiers:
            if name in self.models:
                model = self.models[name]
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(self.X_test_scaled)
                else:
                    decision = model.decision_function(self.X_test_scaled)
                    probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
                
                max_probas = np.max(probas, axis=1)
                classifier_names.append(name)
                mean_confidences.append(np.mean(max_probas))
        
        axes[1, 1].bar(classifier_names, mean_confidences, color='lightblue', alpha=0.7)
        axes[1, 1].set_title('分类器平均置信度对比', fontsize=12, fontweight='bold', fontfamily='SimHei')
        axes[1, 1].set_xlabel('分类器', fontfamily='SimHei')
        axes[1, 1].set_ylabel('平均置信度', fontfamily='SimHei')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_probability_visualization(self):
        """创建交互式概率可视化"""
        print("\n正在创建交互式概率可视化...")
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # 创建网格点
            h = 0.05
            x_min, x_max = self.X_train_scaled[:, 0].min() - 1, self.X_train_scaled[:, 0].max() + 1
            y_min, y_max = self.X_train_scaled[:, 1].min() - 1, self.X_train_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            # 选择几个分类器
            selected_classifiers = ['LogisticRegression', 'SVM_RBF', 'RandomForest']
            
            for name in selected_classifiers:
                if name not in self.models:
                    continue
                    
                model = self.models[name]
                
                # 获取预测概率
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(grid_points)
                else:
                    decision = model.decision_function(grid_points)
                    probas = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
                
                # 创建子图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[f'{self.target_names[i]} 概率' for i in range(len(self.target_names))],
                    specs=[[{"type": "scatter"}, {"type": "scatter"}],
                           [{"type": "scatter"}, {"type": "scatter"}]]
                )
                
                # 为每个类别创建热图
                colors = ['Blues', 'Reds', 'Greens']
                
                for i in range(len(self.target_names)):
                    prob_class = probas[:, i].reshape(xx.shape)
                    
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    # 添加等高线图
                    fig.add_trace(
                        go.Contour(
                            x=xx[0], y=yy[:, 0], z=prob_class,
                            colorscale=colors[i], showscale=True,
                            contours=dict(start=0, end=1, size=0.05),
                            name=f'{self.target_names[i]} 概率'
                        ),
                        row=row, col=col
                    )
                    
                    # 添加数据点
                    for j, target_name in enumerate(self.target_names):
                        mask = self.y_train == j
                        fig.add_trace(
                            go.Scatter(
                                x=self.X_train_scaled[mask, 0],
                                y=self.X_train_scaled[mask, 1],
                                mode='markers',
                                name=target_name,
                                marker=dict(size=8, color=['blue', 'red', 'green'][j])
                            ),
                            row=row, col=col
                        )
                
                fig.update_layout(
                    title=f'{name} 分类概率交互式可视化',
                    title_font=dict(family='SimHei', size=20),
                    font=dict(family='SimHei', size=12),
                    height=800, showlegend=True,
                    xaxis=dict(title_font=dict(family='SimHei')),
                    yaxis=dict(title_font=dict(family='SimHei'))
                )

                # 保存为PNG格式（已删除HTML生成功能）
                fig.write_image(f'interactive_probability_{name}.png', width=1200, height=800)
                print(f"交互式概率可视化已保存为 interactive_probability_{name}.png")
                
        except ImportError:
            print("Plotly未安装。")
            
    def generate_advanced_visualization_report(self):
        """生成高级可视化分析报告"""
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 训练所有模型
        self.train_all_models()
        
        # 创建决策边界对比
        self.create_decision_boundary_comparison()
        
        # 创建概率热图
        self.create_probability_heatmaps()
        
        # 创建不确定性可视化
        self.create_uncertainty_visualization()
        
        # 创建预测置信度分析
        self.create_prediction_confidence_analysis()
        
        # 创建交互式可视化
        self.create_interactive_probability_visualization()

        return {
            'models': self.models,
            'results': self.results,
            'generated_files': [
                'decision_boundaries_comparison.png',
                'uncertainty_analysis.png',
                'prediction_confidence_analysis.png',
                'probability_heatmap_*.png',
                'interactive_probability_*.png'
            ]
        }


def main():
    """主函数"""
    print("开始执行高级决策边界与概率可视化分析...")
    
    # 创建高级可视化器
    visualizer = AdvancedDecisionBoundaryVisualizer()
    
    # 生成高级可视化分析报告
    results = visualizer.generate_advanced_visualization_report()


if __name__ == "__main__":
    main()