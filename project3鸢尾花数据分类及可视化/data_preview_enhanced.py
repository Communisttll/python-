"""
增强版鸢尾花数据探索与可视化分析
包含特征统计、相关性分析、数据降维、异常值检测等高级功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置字体大小和清晰度
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class EnhancedIrisDataExplorer:
    """增强版鸢尾花数据探索类"""
    
    def __init__(self):
        """初始化数据探索器"""
        self.iris = None
        self.df = None
        self.feature_names = None
        self.target_names = None
        self.X = None
        self.y = None
        
    def load_and_prepare_data(self):
        """加载和准备鸢尾花数据"""
        print("正在加载鸢尾花数据集...")
        from sklearn.datasets import load_iris
        
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
        
        # 创建DataFrame便于分析
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['species'] = pd.Categorical.from_codes(self.y, self.target_names)
        
        print(f"数据集加载完成，包含 {len(self.df)} 个样本，{len(self.feature_names)} 个特征")
        print(f"类别分布：{self.df['species'].value_counts().to_dict()}")
        
    def comprehensive_statistical_analysis(self):
        """全面的统计分析"""
        print("\n=== 1. 综合统计分析 ===")
        
        # 基本统计描述
        print("\n1.1 基本统计描述")
        basic_stats = self.df[self.feature_names].describe()
        print(basic_stats.round(3))
        
        # 按类别分组的统计
        print("\n1.2 按类别分组的统计描述")
        grouped_stats = self.df.groupby('species')[self.feature_names].agg([
            'count', 'mean', 'std', 'min', 'max', 
            lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
        ]).round(3)
        print(grouped_stats)
        
        # 正态性检验
        print("\n1.3 正态性检验 (Shapiro-Wilk test p-values)")
        for feature in self.feature_names:
            stat, p_value = stats.shapiro(self.df[feature])
            print(f"{feature}: p-value = {p_value:.4f} {'(正态分布)' if p_value > 0.05 else '(非正态分布)'}")
            
        # 方差齐性检验
        print("\n1.4 方差齐性检验 (Levene test)")
        for feature in self.feature_names:
            groups = [self.df[self.df['species'] == species][feature] for species in self.target_names]
            stat, p_value = stats.levene(*groups)
            print(f"{feature}: p-value = {p_value:.4f} {'(方差齐性)' if p_value > 0.05 else '(方差不齐)'}")
            
        return basic_stats, grouped_stats
        
    def correlation_analysis(self):
        """相关性分析"""
        print("\n=== 2. 特征相关性分析 ===")
        
        # 计算相关系数矩阵
        pearson_corr = self.df[self.feature_names].corr(method='pearson')
        spearman_corr = self.df[self.feature_names].corr(method='spearman')
        
        print("\n2.1 皮尔逊相关系数矩阵")
        print(pearson_corr.round(3))
        
        print("\n2.2 斯皮尔曼等级相关系数矩阵")
        print(spearman_corr.round(3))
        
        # 创建相关性热图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 皮尔逊相关热图
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[0],
                   annot_kws={'size': 10, 'fontfamily': 'SimHei'})
        axes[0].set_title('皮尔逊相关系数热图', fontsize=14, fontweight='bold', fontfamily='SimHei')
        
        # 斯皮尔曼相关热图
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[1],
                   annot_kws={'size': 10, 'fontfamily': 'SimHei'})
        axes[1].set_title('斯皮尔曼等级相关系数热图', fontsize=14, fontweight='bold', fontfamily='SimHei')
        
        plt.tight_layout()
        plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pearson_corr, spearman_corr
        
    def dimensionality_reduction_visualization(self):
        """数据降维可视化"""
        print("\n=== 3. 数据降维可视化 ===")
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # PCA降维
        print("\n3.1 主成分分析 (PCA)")
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        
        print(f"PCA 2D 解释方差比: {pca_2d.explained_variance_ratio_}")
        print(f"PCA 2D 累计解释方差比: {np.cumsum(pca_2d.explained_variance_ratio_)}")
        print(f"PCA 3D 解释方差比: {pca_3d.explained_variance_ratio_}")
        print(f"PCA 3D 累计解释方差比: {np.cumsum(pca_3d.explained_variance_ratio_)}")
        
        # LDA降维
        print("\n3.2 线性判别分析 (LDA)")
        lda_2d = LDA(n_components=2)
        X_lda_2d = lda_2d.fit_transform(X_scaled, self.y)
        
        lda_3d = LDA(n_components=2)
        X_lda_3d = lda_3d.fit_transform(X_scaled, self.y)
        
        # t-SNE降维
        print("\n3.3 t-SNE非线性降维")
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne_2d = tsne_2d.fit_transform(X_scaled)
        
        # 创建2D降维可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # PCA 2D
        for i, species in enumerate(self.target_names):
            mask = self.y == i
            axes[0, 0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                              label=species, alpha=0.7, s=50)
        axes[0, 0].set_title('PCA 2D 可视化', fontsize=12, fontweight='bold', fontfamily='SimHei')
        axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} 方差)', fontfamily='SimHei')
        axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} 方差)', fontfamily='SimHei')
        axes[0, 0].legend(prop={'family': 'SimHei'})
        axes[0, 0].grid(True, alpha=0.3)
        
        # LDA 2D
        for i, species in enumerate(self.target_names):
            mask = self.y == i
            axes[0, 1].scatter(X_lda_2d[mask, 0], X_lda_2d[mask, 1], 
                              label=species, alpha=0.7, s=50)
        axes[0, 1].set_title('LDA 2D 可视化', fontsize=12, fontweight='bold', fontfamily='SimHei')
        axes[0, 1].set_xlabel('LD1', fontfamily='SimHei')
        axes[0, 1].set_ylabel('LD2', fontfamily='SimHei')
        axes[0, 1].legend(prop={'family': 'SimHei'})
        axes[0, 1].grid(True, alpha=0.3)
        
        # t-SNE 2D
        for i, species in enumerate(self.target_names):
            mask = self.y == i
            axes[0, 2].scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1], 
                              label=species, alpha=0.7, s=50)
        axes[0, 2].set_title('t-SNE 2D 可视化', fontsize=12, fontweight='bold', fontfamily='SimHei')
        axes[0, 2].set_xlabel('t-SNE 1', fontfamily='SimHei')
        axes[0, 2].set_ylabel('t-SNE 2', fontfamily='SimHei')
        axes[0, 2].legend(prop={'family': 'SimHei'})
        axes[0, 2].grid(True, alpha=0.3)
        
        # PCA 3D (投影到前两个主成分)
        for i, species in enumerate(self.target_names):
            mask = self.y == i
            axes[1, 0].scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], 
                              label=species, alpha=0.7, s=50)
        axes[1, 0].set_title('PCA 3D (PC1-PC2投影)', fontsize=12, fontweight='bold', fontfamily='SimHei')
        axes[1, 0].set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%} 方差)', fontfamily='SimHei')
        axes[1, 0].set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%} 方差)', fontfamily='SimHei')
        axes[1, 0].legend(prop={'family': 'SimHei'})
        axes[1, 0].grid(True, alpha=0.3)
        
        # LDA 3D (投影到前两个判别向量)
        for i, species in enumerate(self.target_names):
            mask = self.y == i
            axes[1, 1].scatter(X_lda_3d[mask, 0], X_lda_3d[mask, 1], 
                              label=species, alpha=0.7, s=50)
        axes[1, 1].set_title('LDA 3D (LD1-LD2投影)', fontsize=12, fontweight='bold', fontfamily='SimHei')
        axes[1, 1].set_xlabel('LD1', fontfamily='SimHei')
        axes[1, 1].set_ylabel('LD2', fontfamily='SimHei')
        axes[1, 1].legend(prop={'family': 'SimHei'})
        axes[1, 1].grid(True, alpha=0.3)
        
        # 特征贡献分析
        feature_contrib = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=self.feature_names
        )
        sns.heatmap(feature_contrib, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('特征在主成分上的贡献', fontsize=12, fontweight='bold', fontfamily='SimHei')
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'pca_2d': (X_pca_2d, pca_2d),
            'pca_3d': (X_pca_3d, pca_3d),
            'lda_2d': (X_lda_2d, lda_2d),
            'lda_3d': (X_lda_3d, lda_3d),
            'tsne_2d': (X_tsne_2d, tsne_2d)
        }
        
    def outlier_detection_and_analysis(self):
        """异常值检测与分析"""
        print("\n=== 4. 异常值检测与分析 ===")
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # 统计方法检测异常值
        print("\n4.1 统计方法异常值检测 (3σ原则)")
        outliers_stat = []
        for i, feature in enumerate(self.feature_names):
            mean = self.df[feature].mean()
            std = self.df[feature].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outlier_mask = (self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)
            outlier_indices = self.df[outlier_mask].index.tolist()
            outliers_stat.extend([(idx, feature) for idx in outlier_indices])
            
            print(f"{feature}: 检测到 {len(outlier_indices)} 个异常值")
            if outlier_indices:
                print(f"  异常值索引: {outlier_indices}")
                print(f"  正常值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # 孤立森林检测异常值
        print("\n4.2 孤立森林异常值检测")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_scaled)
        outliers_iso = np.where(outlier_labels == -1)[0]
        
        print(f"孤立森林检测到 {len(outliers_iso)} 个异常值")
        print(f"异常值索引: {outliers_iso.tolist()}")
        
        # 可视化异常值
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 统计方法异常值可视化
        for i, feature in enumerate(self.feature_names[:4]):  # 只显示前4个特征
            ax = axes[i//2, i%2]
            
            # 箱线图
            box_data = [self.df[self.df['species'] == species][feature] for species in self.target_names]
            bp = ax.boxplot(box_data, labels=self.target_names, patch_artist=True)
            
            # 为不同类别设置不同颜色
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{feature} - 箱线图与异常值检测', fontsize=12, fontweight='bold', fontfamily='SimHei')
            ax.set_ylabel(feature, fontfamily='SimHei')
            ax.grid(True, alpha=0.3)
            
            # 标记统计异常值
            outlier_indices = [idx for idx, feat in outliers_stat if feat == feature]
            if outlier_indices:
                outlier_values = self.df.loc[outlier_indices, feature]
                ax.scatter([1, 2, 3], [outlier_values.mean()] * 3, 
                          color='red', marker='*', s=100, label='统计异常值')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('outlier_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 异常值对分类的影响分析
        print("\n4.3 异常值对分类性能的影响分析")
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # 原始数据分类性能
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, test_size=0.3, random_state=42, stratify=self.y)
        
        lr_original = LogisticRegression(random_state=42, max_iter=1000)
        lr_original.fit(X_train, y_train)
        y_pred_original = lr_original.predict(X_test)
        accuracy_original = accuracy_score(y_test, y_pred_original)
        
        # 移除孤立森林检测的异常值后的分类性能
        mask = ~np.isin(np.arange(len(self.y)), outliers_iso)
        X_clean = X_scaled[mask]
        y_clean = self.y[mask]
        
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean)
        
        lr_clean = LogisticRegression(random_state=42, max_iter=1000)
        lr_clean.fit(X_train_clean, y_train_clean)
        y_pred_clean = lr_clean.predict(X_test_clean)
        accuracy_clean = accuracy_score(y_test_clean, y_pred_clean)
        
        print(f"原始数据分类准确率: {accuracy_original:.4f}")
        print(f"移除异常值后分类准确率: {accuracy_clean:.4f}")
        print(f"异常值移除带来的性能提升: {accuracy_clean - accuracy_original:.4f}")
        
        return {
            'statistical_outliers': outliers_stat,
            'isolation_forest_outliers': outliers_iso,
            'accuracy_comparison': {
                'original': accuracy_original,
                'clean': accuracy_clean,
                'improvement': accuracy_clean - accuracy_original
            }
        }
        
    def class_separability_analysis(self):
        """类别可分离性分析"""
        print("\n=== 5. 类别可分离性分析 ===")
        
        # 计算类间距离和类内距离
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # 计算每个类别的均值向量
        class_means = []
        for i in range(len(self.target_names)):
            class_mean = np.mean(X_scaled[self.y == i], axis=0)
            class_means.append(class_mean)
        
        class_means = np.array(class_means)
        
        # 计算类间距离矩阵
        between_class_distances = np.zeros((len(self.target_names), len(self.target_names)))
        for i in range(len(self.target_names)):
            for j in range(i+1, len(self.target_names)):
                distance = np.linalg.norm(class_means[i] - class_means[j])
                between_class_distances[i, j] = distance
                between_class_distances[j, i] = distance
        
        print("\n5.1 类间距离矩阵 (欧氏距离)")
        distance_df = pd.DataFrame(between_class_distances, 
                                  index=self.target_names, 
                                  columns=self.target_names)
        print(distance_df.round(3))
        
        # 计算类内距离（每个类别内部样本到均值的平均距离）
        within_class_distances = []
        for i in range(len(self.target_names)):
            class_samples = X_scaled[self.y == i]
            class_mean = class_means[i]
            distances = np.linalg.norm(class_samples - class_mean, axis=1)
            within_class_distances.append(np.mean(distances))
        
        print("\n5.2 类内平均距离")
        for i, species in enumerate(self.target_names):
            print(f"{species}: {within_class_distances[i]:.3f}")
        
        # 计算类间/类内距离比值（可分离性指标）
        print("\n5.3 类间/类内距离比值 (可分离性指标)")
        separability_ratios = np.zeros((len(self.target_names), len(self.target_names)))
        for i in range(len(self.target_names)):
            for j in range(i+1, len(self.target_names)):
                ratio = between_class_distances[i, j] / (within_class_distances[i] + within_class_distances[j])
                separability_ratios[i, j] = ratio
                separability_ratios[j, i] = ratio
        
        ratio_df = pd.DataFrame(separability_ratios, 
                               index=self.target_names, 
                               columns=self.target_names)
        print(ratio_df.round(3))
        
        # 可视化可分离性
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 类间距离热图
        sns.heatmap(distance_df, annot=True, cmap='viridis', ax=axes[0],
                   annot_kws={'size': 10, 'fontfamily': 'SimHei'})
        axes[0].set_title('类间距离矩阵', fontsize=12, fontweight='bold', fontfamily='SimHei')
        
        # 可分离性比率热图
        sns.heatmap(ratio_df, annot=True, cmap='plasma', ax=axes[1],
                   annot_kws={'size': 10, 'fontfamily': 'SimHei'})
        axes[1].set_title('类间/类内距离比值 (可分离性)', fontsize=12, fontweight='bold', fontfamily='SimHei')
        
        plt.tight_layout()
        plt.savefig('class_separability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 基于LDA的可分离性评估
        print("\n5.4 基于LDA的类别可分离性")
        lda = LDA()
        X_lda = lda.fit_transform(X_scaled, self.y)
        
        # 计算LDA后的类间距离
        lda_class_means = []
        for i in range(len(self.target_names)):
            lda_class_mean = np.mean(X_lda[self.y == i], axis=0)
            lda_class_means.append(lda_class_mean)
        
        lda_class_means = np.array(lda_class_means)
        
        # 计算LDA判别比
        lda_separability = []
        for i in range(X_lda.shape[1]):
            between_class_var = np.var(lda_class_means[:, i]) * len(self.target_names)
            within_class_var = np.var(X_lda[:, i])
            ratio = between_class_var / within_class_var if within_class_var > 0 else 0
            lda_separability.append(ratio)
        
        print(f"LDA判别比: {lda_separability}")
        print(f"平均判别比: {np.mean(lda_separability):.3f}")
        
        return {
            'between_class_distances': distance_df,
            'within_class_distances': dict(zip(self.target_names, within_class_distances)),
            'separability_ratios': ratio_df,
            'lda_separability': lda_separability
        }
        
    def create_interactive_visualizations(self):
        """创建交互式可视化"""
        print("\n=== 6. 交互式可视化 ===")
        
        # 散点图矩阵
        print("\n6.1 散点图矩阵 (Pair Plot)")
        fig = px.scatter_matrix(self.df, dimensions=self.feature_names, 
                               color='species', title='鸢尾花数据散点图矩阵',
                               symbol='species', height=800)
        fig.update_traces(diagonal_visible=False)
        # 保存为PNG格式（已删除HTML生成功能）
        fig.write_image("interactive_scatter_matrix.png", width=1200, height=800)
        print("交互式散点图矩阵已保存为 interactive_scatter_matrix.png")
        
        # 3D散点图
        print("\n6.2 3D散点图")
        fig_3d = px.scatter_3d(self.df, x=self.feature_names[0], y=self.feature_names[1], 
                              z=self.feature_names[2], color='species',
                              size=self.feature_names[3], title='鸢尾花数据3D散点图',
                              symbol='species', height=900)
        # 更新布局以增加字体大小和更好的显示效果
        fig_3d.update_layout(
            title_font_size=16,
            scene=dict(
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                zaxis_title_font_size=14,
                aspectmode='cube'
            ),
            legend=dict(font_size=12)
        )
        fig_3d.write_image("interactive_3d_scatter.png", width=1400, height=1000, scale=2)
        print("交互式3D散点图已保存为 interactive_3d_scatter.png")
        
        # 平行坐标图
        print("\n6.3 平行坐标图")
        # 将类别名称转换为数值以便着色
        species_numeric = pd.Categorical(self.df['species']).codes
        fig_parallel = px.parallel_coordinates(self.df, color=species_numeric,
                                               dimensions=self.feature_names,
                                               title='鸢尾花数据平行坐标图',
                                               labels={i: self.feature_names[i] for i in range(len(self.feature_names))})
        fig_parallel.write_image("interactive_parallel_coordinates.png", width=1200, height=800)
        print("交互式平行坐标图已保存为 interactive_parallel_coordinates.png")
        
        # 箱线图交互式版本
        print("\n6.4 交互式箱线图")
        df_melted = pd.melt(self.df, id_vars=['species'], value_vars=self.feature_names,
                           var_name='feature', value_name='value')
        
        fig_box = px.box(df_melted, x='species', y='value', color='species',
                        facet_col='feature', title='鸢尾花数据交互式箱线图',
                        height=600, category_orders={"feature": self.feature_names})
        fig_box.update_xaxes(tickangle=45)
        fig_box.write_image("interactive_boxplots.png", width=1200, height=800)
        print("交互式箱线图已保存为 interactive_boxplots.png")
        
        return {
            'scatter_matrix': 'interactive_scatter_matrix.png',
            '3d_scatter': 'interactive_3d_scatter.png',
            'parallel_coordinates': 'interactive_parallel_coordinates.png',
            'boxplots': 'interactive_boxplots.png'
        }
        
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n=== 7. 综合分析报告 ===")
        
        # 执行所有分析
        self.load_and_prepare_data()
        
        print("\n" + "="*60)
        print("鸢尾花数据增强版探索性分析报告")
        print("="*60)
        
        # 1. 统计分析
        basic_stats, grouped_stats = self.comprehensive_statistical_analysis()
        
        # 2. 相关性分析
        pearson_corr, spearman_corr = self.correlation_analysis()
        
        # 3. 降维分析
        dim_results = self.dimensionality_reduction_visualization()
        
        # 4. 异常值分析
        outlier_results = self.outlier_detection_and_analysis()
        
        # 5. 可分离性分析
        separability_results = self.class_separability_analysis()
        
        # 6. 交互式可视化
        interactive_results = self.create_interactive_visualizations()
        
        # 生成总结
        print("\n" + "="*60)
        print("主要发现总结:")
        print("="*60)
        
        print(f"\n1. 数据集概况:")
        print(f"   - 样本数量: {len(self.df)}")
        print(f"   - 特征数量: {len(self.feature_names)}")
        print(f"   - 类别数量: {len(self.target_names)}")
        print(f"   - 特征名称: {', '.join(self.feature_names)}")
        
        print(f"\n2. 数据分布特征:")
        for feature in self.feature_names:
            feature_data = self.df[feature]
            print(f"   - {feature}: 均值={feature_data.mean():.2f}, 标准差={feature_data.std():.2f}")
        
        print(f"\n3. 相关性分析:")
        high_corr = np.where(np.abs(pearson_corr) > 0.8)
        high_corr_pairs = [(pearson_corr.index[x], pearson_corr.columns[y], pearson_corr.iloc[x, y]) 
                          for x, y in zip(*high_corr) if x != y and x < y]
        if high_corr_pairs:
            print("   - 高度相关的特征对:")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"     {feat1} vs {feat2}: r = {corr:.3f}")
        else:
            print("   - 未发现高度相关的特征对 (|r| > 0.8)")
            
        print(f"\n4. 降维分析:")
        pca_2d_obj = dim_results['pca_2d'][1]
        print(f"   - PCA前两个主成分解释方差: {pca_2d_obj.explained_variance_ratio_[0]:.1%} 和 {pca_2d_obj.explained_variance_ratio_[1]:.1%}")
        print(f"   - PCA累计解释方差: {np.sum(pca_2d_obj.explained_variance_ratio_):.1%}")
        
        print(f"\n5. 异常值检测:")
        print(f"   - 统计方法检测到异常值: {len(outlier_results['statistical_outliers'])} 个")
        print(f"   - 孤立森林检测到异常值: {len(outlier_results['isolation_forest_outliers'])} 个")
        print(f"   - 移除异常值后分类准确率提升: {outlier_results['accuracy_comparison']['improvement']:.4f}")
        
        print(f"\n6. 类别可分离性:")
        min_separability = separability_results['separability_ratios'].values
        min_separability = min_separability[min_separability > 0].min()
        print(f"   - 最小类间/类内距离比值: {min_separability:.3f}")
        print("   - 可分离性评价: 比值越大，类别越容易区分")
        
        print(f"\n7. 生成交互式可视化:")
        for name, file in interactive_results.items():
            print(f"   - {name}: {file}")
        
        return {
            'basic_stats': basic_stats,
            'grouped_stats': grouped_stats,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'dimensionality_reduction': dim_results,
            'outlier_analysis': outlier_results,
            'separability_analysis': separability_results,
            'interactive_visualizations': interactive_results
        }


def main():
    """主函数"""
    
    # 创建数据探索器实例
    explorer = EnhancedIrisDataExplorer()
    
    # 生成综合分析报告
    results = explorer.generate_comprehensive_report()
    


if __name__ == "__main__":
    main()