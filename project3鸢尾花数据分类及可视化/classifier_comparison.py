"""
多分类器系统比较分析
包含6+种不同类型分类器的系统比较、超参数优化、性能评估和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class ClassifierComparisonSystem:
    """多分类器系统比较类"""
    
    def __init__(self):
        """初始化分类器比较系统"""
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        self.best_models = {}
        self.feature_names = None
        self.target_names = None
        
        # 定义分类器配置
        self.classifiers_config = {
            # 线性模型
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'type': 'Linear Model'
            },
            'LinearDiscriminantAnalysis': {
                'model': LinearDiscriminantAnalysis(),
                'params': {
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
                },
                'type': 'Linear Model'
            },
            'QuadraticDiscriminantAnalysis': {
                'model': QuadraticDiscriminantAnalysis(),
                'params': {
                    'reg_param': [0.0, 0.1, 0.5, 1.0]
                },
                'type': 'Non-linear Model'
            },
            
            # 支持向量机
            'SVM_Linear': {
                'model': SVC(kernel='linear', random_state=42, probability=True),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100]
                },
                'type': 'Support Vector Machine'
            },
            'SVM_RBF': {
                'model': SVC(kernel='rbf', random_state=42, probability=True),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                },
                'type': 'Support Vector Machine'
            },
            'SVM_Poly': {
                'model': SVC(kernel='poly', random_state=42, probability=True),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto']
                },
                'type': 'Support Vector Machine'
            },
            
            # 基于距离的模型
            'KNeighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'type': 'Distance-based'
            },
            
            # 贝叶斯模型
            'GaussianNB': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                },
                'type': 'Bayesian'
            },
            
            # 决策树模型
            'DecisionTree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'type': 'Tree-based'
            },
            
            # 集成模型
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'type': 'Ensemble'
            },
            'ExtraTrees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'type': 'Ensemble'
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7]
                },
                'type': 'Ensemble'
            },
            
            # 神经网络
            'MLP': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'type': 'Neural Network'
            }
        }
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("正在加载鸢尾花数据集...")
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # 数据划分
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y)
        
        # 数据标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"数据准备完成:")
        print(f"  训练集大小: {self.X_train.shape}")
        print(f"  测试集大小: {self.X_test.shape}")
        print(f"  特征数量: {len(self.feature_names)}")
        print(f"  类别数量: {len(self.target_names)}")
        
    def optimize_single_classifier(self, name, config):
        """优化单个分类器"""
        print(f"\n正在优化 {name}...")
        
        start_time = time.time()
        
        # 网格搜索
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        # 获取最佳模型
        best_model = grid_search.best_estimator_
        
        # 在测试集上评估
        start_time = time.time()
        y_pred = best_model.predict(self.X_test_scaled)
        prediction_time = time.time() - start_time
        
        # 计算各种指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # 交叉验证得分
        cv_scores = cross_val_score(best_model, self.X_train_scaled, self.y_train, cv=5)
        
        result = {
            'name': name,
            'type': config['type'],
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        print(f"  最佳参数: {grid_search.best_params_}")
        print(f"  交叉验证得分: {grid_search.best_score_:.4f} (±{cv_scores.std():.4f})")
        print(f"  测试集准确率: {accuracy:.4f}")
        print(f"  训练时间: {training_time:.2f}秒")
        
        return result, best_model
        
    def train_all_classifiers(self):
        """训练所有分类器"""
        print("\n" + "="*60)
        print("开始训练所有分类器...")
        print("="*60)
        
        for name, config in self.classifiers_config.items():
            try:
                result, best_model = self.optimize_single_classifier(name, config)
                self.results[name] = result
                self.best_models[name] = best_model
            except Exception as e:
                print(f"  训练 {name} 时出错: {str(e)}")
                continue
                
        print(f"\n所有分类器训练完成，共训练了 {len(self.results)} 个模型")
        
    def create_performance_comparison_table(self):
        """创建性能对比表"""
        print("\n正在创建性能对比表...")
        
        # 创建性能对比DataFrame
        performance_data = []
        for name, result in self.results.items():
            performance_data.append({
                '分类器': result['name'],
                '类型': result['type'],
                '交叉验证得分': f"{result['cv_scores_mean']:.4f} ± {result['cv_scores_std']:.4f}",
                '测试准确率': f"{result['test_accuracy']:.4f}",
                '精确率': f"{result['test_precision']:.4f}",
                '召回率': f"{result['test_recall']:.4f}",
                'F1分数': f"{result['test_f1']:.4f}",
                '训练时间(秒)': f"{result['training_time']:.2f}",
                '预测时间(秒)': f"{result['prediction_time']:.4f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # 按测试准确率排序
        performance_df['test_accuracy_num'] = performance_df['测试准确率'].astype(float)
        performance_df = performance_df.sort_values('test_accuracy_num', ascending=False)
        performance_df = performance_df.drop('test_accuracy_num', axis=1)
        
        print("\n分类器性能对比表:")
        print(performance_df.to_string(index=False))
        
        # 保存到CSV文件
        performance_df.to_csv('classifier_performance_comparison.csv', index=False, encoding='utf-8')
        print("\n性能对比表已保存为 classifier_performance_comparison.csv")
        
        return performance_df
        
    def visualize_performance_comparison(self):
        """可视化性能对比"""
        print("\n正在创建性能对比可视化...")
        
        # 准备数据
        classifiers = list(self.results.keys())
        accuracies = [self.results[name]['test_accuracy'] for name in classifiers]
        precisions = [self.results[name]['test_precision'] for name in classifiers]
        recalls = [self.results[name]['test_recall'] for name in classifiers]
        f1_scores = [self.results[name]['test_f1'] for name in classifiers]
        training_times = [self.results[name]['training_time'] for name in classifiers]
        prediction_times = [self.results[name]['prediction_time'] for name in classifiers]
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 准确率对比柱状图
        bars1 = axes[0, 0].bar(classifiers, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('测试准确率对比', fontsize=14, fontweight='bold', fontfamily='SimHei')
        axes[0, 0].set_ylabel('准确率', fontfamily='SimHei')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        # 调整x轴标签对齐和间距
        axes[0, 0].set_xticklabels(classifiers, rotation=45, ha='right', fontfamily='SimHei')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), fontsize=9)
        
        # 2. 多指标对比
        x = np.arange(len(classifiers))
        width = 0.2
        axes[0, 1].bar(x - width, accuracies, width, label='准确率', alpha=0.8)
        axes[0, 1].bar(x, precisions, width, label='精确率', alpha=0.8)
        axes[0, 1].bar(x + width, recalls, width, label='召回率', alpha=0.8)
        axes[0, 1].bar(x + 2*width, f1_scores, width, label='F1分数', alpha=0.8)
        axes[0, 1].set_title('多指标性能对比', fontsize=14, fontweight='bold', fontfamily='SimHei')
        axes[0, 1].set_ylabel('分数', fontfamily='SimHei')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classifiers, rotation=45, ha='right', fontfamily='SimHei')
        axes[0, 1].legend(prop={'family': 'SimHei'})
        axes[0, 1].grid(True, alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), fontsize=9)
        
        # 3. 训练时间对比
        bars3 = axes[0, 2].bar(classifiers, training_times, color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('训练时间对比', fontsize=14, fontweight='bold', fontfamily='SimHei')
        axes[0, 2].set_ylabel('训练时间 (秒)', fontfamily='SimHei')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xticklabels(classifiers, rotation=45, ha='right', fontfamily='SimHei')
        plt.setp(axes[0, 2].xaxis.get_majorticklabels(), fontsize=9)
        
        # 4. 预测时间对比
        bars4 = axes[1, 0].bar(classifiers, prediction_times, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('预测时间对比', fontsize=14, fontweight='bold', fontfamily='SimHei')
        axes[1, 0].set_ylabel('预测时间 (秒)', fontfamily='SimHei')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticklabels(classifiers, rotation=45, ha='right', fontfamily='SimHei')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), fontsize=9)
        
        # 5. 准确率 vs 训练时间散点图
        scatter = axes[1, 1].scatter(training_times, accuracies, s=100, alpha=0.7, c=range(len(classifiers)), cmap='viridis')
        axes[1, 1].set_title('准确率 vs 训练时间', fontsize=14, fontweight='bold', fontfamily='SimHei')
        axes[1, 1].set_xlabel('训练时间 (秒)', fontfamily='SimHei')
        axes[1, 1].set_ylabel('准确率', fontfamily='SimHei')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加分类器标签 - 改进对齐和避免重叠
        for i, classifier in enumerate(classifiers):
            # 添加轻微偏移避免重叠
            offset_x = 5 if i % 2 == 0 else -35
            offset_y = 5 if i % 3 == 0 else -15
            axes[1, 1].annotate(classifier, (training_times[i], accuracies[i]), 
                               xytext=(offset_x, offset_y), textcoords='offset points', 
                               fontsize=8, fontfamily='SimHei', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 6. 性能雷达图
        # 选择前5个最好的分类器进行雷达图对比
        top_5_indices = np.argsort(accuracies)[-5:]
        top_5_classifiers = [classifiers[i] for i in top_5_indices]
        
        # 标准化指标到0-1范围
        metrics = ['准确率', '精确率', '召回率', 'F1分数']
        values = np.array([
            [accuracies[i] for i in top_5_indices],
            [precisions[i] for i in top_5_indices],
            [recalls[i] for i in top_5_indices],
            [f1_scores[i] for i in top_5_indices]
        ])
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax_radar = plt.subplot(2, 3, 6, projection='polar')
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, classifier in enumerate(top_5_classifiers):
            values_plot = values[:, i].tolist()
            values_plot += values_plot[:1]  # 闭合数据
            ax_radar.plot(angles, values_plot, 'o-', linewidth=2, label=classifier, color=colors[i])
            ax_radar.fill(angles, values_plot, alpha=0.25, color=colors[i])
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics, fontfamily='SimHei')
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Top 5 分类器性能雷达图', fontsize=14, fontweight='bold', pad=20, fontfamily='SimHei')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop={'family': 'SimHei'})
        ax_radar.grid(True)
        
        # 调整子图间距以避免标签重叠
        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
        plt.savefig('classifier_performance_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_confusion_matrix_comparison(self):
        """创建混淆矩阵对比"""
        print("\n正在创建混淆矩阵对比...")
        
        # 选择性能最好的6个分类器
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        top_6_classifiers = sorted_results[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(top_6_classifiers):
            cm = result['confusion_matrix']
            
            # 绘制混淆矩阵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name}\n准确率: {result["test_accuracy"]:.3f}', 
                            fontsize=12, fontweight='bold', fontfamily='SimHei')
            axes[i].set_xlabel('预测类别', fontfamily='SimHei')
            axes[i].set_ylabel('真实类别', fontfamily='SimHei')
            
            # 设置类别标签 - 改进对齐
            axes[i].set_xticklabels(self.target_names, fontfamily='SimHei', rotation=0)
            axes[i].set_yticklabels(self.target_names, fontfamily='SimHei', rotation=90)
            # 调整标签位置避免重叠
            axes[i].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.5)
        plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_ensemble_models(self):
        """创建集成模型"""
        print("\n正在创建集成模型...")
        
        # 选择性能最好的3个分类器作为基分类器
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        top_3_classifiers = sorted_results[:3]
        
        base_estimators = [(name, self.best_models[name]) for name, _ in top_3_classifiers]
        
        # 1. 硬投票集成
        voting_hard = VotingClassifier(estimators=base_estimators, voting='hard')
        voting_hard.fit(self.X_train_scaled, self.y_train)
        y_pred_hard = voting_hard.predict(self.X_test_scaled)
        accuracy_hard = accuracy_score(self.y_test, y_pred_hard)
        
        # 2. 软投票集成
        voting_soft = VotingClassifier(estimators=base_estimators, voting='soft')
        voting_soft.fit(self.X_train_scaled, self.y_train)
        y_pred_soft = voting_soft.predict(self.X_test_scaled)
        accuracy_soft = accuracy_score(self.y_test, y_pred_soft)
        
        # 3. 堆叠集成
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stacking.fit(self.X_train_scaled, self.y_train)
        y_pred_stacking = stacking.predict(self.X_test_scaled)
        accuracy_stacking = accuracy_score(self.y_test, y_pred_stacking)
        
        print(f"\n集成模型性能:")
        print(f"  硬投票集成: {accuracy_hard:.4f}")
        print(f"  软投票集成: {accuracy_soft:.4f}")
        print(f"  堆叠集成: {accuracy_stacking:.4f}")
        
        # 对比基分类器性能
        print(f"\n基分类器性能:")
        for name, result in top_3_classifiers:
            print(f"  {name}: {result['test_accuracy']:.4f}")
        
        return {
            'voting_hard': {'model': voting_hard, 'accuracy': accuracy_hard, 'predictions': y_pred_hard},
            'voting_soft': {'model': voting_soft, 'accuracy': accuracy_soft, 'predictions': y_pred_soft},
            'stacking': {'model': stacking, 'accuracy': accuracy_stacking, 'predictions': y_pred_stacking}
        }
        
    def learning_curve_analysis(self):
        """学习曲线分析"""
        print("\n正在创建学习曲线分析...")
        
        # 选择几个代表性的分类器
        selected_classifiers = ['LogisticRegression', 'SVM_RBF', 'RandomForest', 'MLP']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, clf_name in enumerate(selected_classifiers):
            if clf_name not in self.best_models:
                continue
                
            model = self.best_models[clf_name]
            
            # 不同训练集大小的性能
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            test_scores = []
            
            for train_size in train_sizes:
                # 根据训练集比例重新划分数据
                n_samples = int(len(self.X_train) * train_size)
                indices = np.random.choice(len(self.X_train), n_samples, replace=False)
                
                X_subset = self.X_train_scaled[indices]
                y_subset = self.y_train[indices]
                
                # 训练模型
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_subset, y_subset)
                
                # 计算训练集和测试集得分
                train_score = model_copy.score(X_subset, y_subset)
                test_score = model_copy.score(self.X_test_scaled, self.y_test)
                
                train_scores.append(train_score)
                test_scores.append(test_score)
            
            # 绘制学习曲线
            axes[i].plot(train_sizes * 100, train_scores, 'o-', label='训练集得分', color='blue', linewidth=2)
            axes[i].plot(train_sizes * 100, test_scores, 'o-', label='测试集得分', color='red', linewidth=2)
            axes[i].set_title(f'{clf_name} 学习曲线', fontsize=12, fontweight='bold', fontfamily='SimHei')
            axes[i].set_xlabel('训练集大小 (%)', fontfamily='SimHei')
            axes[i].set_ylabel('准确率', fontfamily='SimHei')
            axes[i].legend(prop={'family': 'SimHei'})
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_curves_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 训练所有分类器
        self.train_all_classifiers()
        
        # 创建性能对比表
        performance_df = self.create_performance_comparison_table()
        
        # 性能可视化
        self.visualize_performance_comparison()
        
        # 混淆矩阵对比
        self.create_confusion_matrix_comparison()
        
        # 集成模型
        ensemble_results = self.create_ensemble_models()
        
        # 学习曲线分析
        self.learning_curve_analysis()
        
        # 生成最终报告
        print("\n" + "="*80)
        print("最终分析结果总结")
        print("="*80)
        
        # 最佳分类器
        best_classifier = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n1. 最佳分类器: {best_classifier[0]}")
        print(f"   测试准确率: {best_classifier[1]['test_accuracy']:.4f}")
        print(f"   交叉验证得分: {best_classifier[1]['cv_scores_mean']:.4f} ± {best_classifier[1]['cv_scores_std']:.4f}")
        
        # 最快分类器
        fastest_classifier = min(self.results.items(), key=lambda x: x[1]['training_time'])
        print(f"\n2. 最快分类器: {fastest_classifier[0]}")
        print(f"   训练时间: {fastest_classifier[1]['training_time']:.2f}秒")
        
        # 最稳定分类器（最低标准差）
        most_stable_classifier = min(self.results.items(), key=lambda x: x[1]['cv_scores_std'])
        print(f"\n3. 最稳定分类器: {most_stable_classifier[0]}")
        print(f"   交叉验证标准差: {most_stable_classifier[1]['cv_scores_std']:.4f}")
        
        # 集成模型对比
        print(f"\n4. 集成模型性能:")
        for ensemble_name, ensemble_result in ensemble_results.items():
            print(f"   {ensemble_name}: {ensemble_result['accuracy']:.4f}")
        
        # 分类器类型总结
        print(f"\n5. 分类器类型总结:")
        type_performance = {}
        for name, result in self.results.items():
            clf_type = result['type']
            if clf_type not in type_performance:
                type_performance[clf_type] = []
            type_performance[clf_type].append(result['test_accuracy'])
        
        for clf_type, accuracies in type_performance.items():
            avg_accuracy = np.mean(accuracies)
            print(f"   {clf_type}: 平均准确率 {avg_accuracy:.4f}")
        
        print(f"\n6. 生成的文件:")
        print("   - classifier_performance_comparison.csv: 性能对比表")
        print("   - classifier_performance_visualization.png: 性能对比可视化")
        print("   - confusion_matrices_comparison.png: 混淆矩阵对比")
        print("   - learning_curves_analysis.png: 学习曲线分析")
        
        return {
            'performance_table': performance_df,
            'best_classifier': best_classifier,
            'ensemble_results': ensemble_results,
            'all_results': self.results
        }


def main():
    """主函数"""
    print("开始执行多分类器系统比较分析...")
    
    # 创建分类器比较系统
    comparison_system = ClassifierComparisonSystem()
    
    # 生成综合分析报告
    results = comparison_system.generate_comprehensive_report()


if __name__ == "__main__":
    main()