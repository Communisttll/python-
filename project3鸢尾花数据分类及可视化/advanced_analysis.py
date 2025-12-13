"""
高级分析与创新扩展模块
实现特征重要性分析、集成学习方法、模型解释性分析、分类难度区域分析等高级功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedAnalysis:
    """高级分析器"""
    
    def __init__(self):
        """初始化分析器"""
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
        
        # 颜色映射
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
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
        print(f"  特征数量: {self.X.shape[1]}")
        print(f"  样本数量: {self.X.shape[0]}")
        print(f"  类别数量: {len(self.target_names)}")
        print(f"  特征名称: {', '.join(self.feature_names)}")
        
    def feature_importance_analysis(self):
        """特征重要性分析"""
        print("\n" + "="*60)
        print("特征重要性分析")
        print("="*60)
        
        # 1. 基于随机森林的特征重要性
        print("\n1. 随机森林特征重要性分析...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        
        # 获取特征重要性
        rf_importance = rf_model.feature_importances_
        rf_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_importance,
            'importance_std': np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
        })
        rf_importance_df = rf_importance_df.sort_values('importance', ascending=False)
        
        print("随机森林特征重要性:")
        for _, row in rf_importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f} ± {row['importance_std']:.4f}")
        
        # 2. 基于排列的特征重要性
        print("\n2. 排列特征重要性分析...")
        perm_importance = permutation_importance(rf_model, self.X_test_scaled, self.y_test, 
                                                  n_repeats=10, random_state=42)
        
        perm_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        perm_importance_df = perm_importance_df.sort_values('importance_mean', ascending=False)
        
        print("排列特征重要性:")
        for _, row in perm_importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        # 3. 基于统计检验的特征重要性
        print("\n3. 统计检验特征重要性分析...")
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(self.X_train_scaled, self.y_train)
        
        f_scores = selector.scores_
        p_values = selector.pvalues_
        
        statistical_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'f_score': f_scores,
            'p_value': p_values
        })
        statistical_importance_df = statistical_importance_df.sort_values('f_score', ascending=False)
        
        print("统计检验特征重要性 (F-score):")
        for _, row in statistical_importance_df.iterrows():
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"  {row['feature']}: F={row['f_score']:.2f}, p={row['p_value']:.4f} {significance}")
        
        # 4. 递归特征消除 (RFE)
        print("\n4. 递归特征消除分析...")
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
                 n_features_to_select=2)
        rfe.fit(self.X_train_scaled, self.y_train)
        
        rfe_ranking_df = pd.DataFrame({
            'feature': self.feature_names,
            'ranking': rfe.ranking_,
            'selected': rfe.support_
        })
        rfe_ranking_df = rfe_ranking_df.sort_values('ranking')
        
        print("递归特征消除排名:")
        for _, row in rfe_ranking_df.iterrows():
            selected = "✓" if row['selected'] else "✗"
            print(f"  {row['feature']}: 排名 {row['ranking']} {selected}")
        
        # 可视化特征重要性
        self.visualize_feature_importance(rf_importance_df, perm_importance_df, 
                                        statistical_importance_df, rfe_ranking_df)
        
        return {
            'rf_importance': rf_importance_df,
            'perm_importance': perm_importance_df,
            'statistical_importance': statistical_importance_df,
            'rfe_ranking': rfe_ranking_df
        }
        
    def visualize_feature_importance(self, rf_df, perm_df, stat_df, rfe_df):
        """可视化特征重要性"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('特征重要性分析对比', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        # 1. 随机森林特征重要性
        ax1 = axes[0, 0]
        bars1 = ax1.barh(rf_df['feature'], rf_df['importance'], color='skyblue', alpha=0.8)
        ax1.errorbar(rf_df['importance'], range(len(rf_df)), 
                      xerr=rf_df['importance_std'], fmt='none', color='black', alpha=0.6)
        ax1.set_xlabel('重要性', fontfamily='SimHei')
        ax1.set_title('随机森林特征重要性', fontweight='bold', fontfamily='SimHei')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars1, rf_df['importance'])):
            ax1.text(importance + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', va='center', fontsize=9, fontfamily='SimHei')
        
        # 2. 排列特征重要性
        ax2 = axes[0, 1]
        bars2 = ax2.barh(perm_df['feature'], perm_df['importance_mean'], color='lightcoral', alpha=0.8)
        ax2.errorbar(perm_df['importance_mean'], range(len(perm_df)), 
                      xerr=perm_df['importance_std'], fmt='none', color='black', alpha=0.6)
        ax2.set_xlabel('重要性 (准确率下降)', fontfamily='SimHei')
        ax2.set_title('排列特征重要性', fontweight='bold', fontfamily='SimHei')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars2, perm_df['importance_mean'])):
            ax2.text(importance + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', va='center', fontsize=10, fontfamily='SimHei')
        
        # 3. 统计检验特征重要性
        ax3 = axes[1, 0]
        bars3 = ax3.barh(stat_df['feature'], stat_df['f_score'], color='lightgreen', alpha=0.8)
        ax3.set_xlabel('F-score', fontfamily='SimHei')
        ax3.set_title('统计检验特征重要性 (F-score)', fontweight='bold', fontfamily='SimHei')
        ax3.grid(True, alpha=0.3)
        
        # 添加显著性标记
        for i, (bar, p_val) in enumerate(zip(bars3, stat_df['p_value'])):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            if significance:
                ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       significance, va='center', fontsize=12, fontweight='bold', color='red', fontfamily='SimHei')
        
        # 4. RFE特征排名
        ax4 = axes[1, 1]
        bars4 = ax4.barh(rfe_df['feature'], rfe_df['ranking'], color='gold', alpha=0.8)
        ax4.set_xlabel('排名 (1=最重要)', fontfamily='SimHei')
        ax4.set_title('递归特征消除排名', fontweight='bold', fontfamily='SimHei')
        ax4.grid(True, alpha=0.3)
        ax4.invert_xaxis()  # 排名越小越重要
        
        # 标记选中的特征
        for i, (bar, selected) in enumerate(zip(bars4, rfe_df['selected'])):
            if selected:
                ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                       '✓', va='center', fontsize=12, fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def ensemble_learning_analysis(self):
        """集成学习分析"""
        print("\n" + "="*60)
        print("集成学习分析")
        print("="*60)
        
        # 定义基分类器
        base_classifiers = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('svm', SVC(random_state=42, probability=True)),  # 启用概率预测
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        
        # 1. 硬投票集成
        print("\n1. 硬投票集成分析...")
        voting_hard = VotingClassifier(estimators=base_classifiers, voting='hard')
        voting_hard.fit(self.X_train_scaled, self.y_train)
        
        # 2. 软投票集成
        print("\n2. 软投票集成分析...")
        voting_soft = VotingClassifier(estimators=base_classifiers, voting='soft')
        voting_soft.fit(self.X_train_scaled, self.y_train)
        
        # 3. 堆叠集成
        print("\n3. 堆叠集成分析...")
        stacking = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5
        )
        stacking.fit(self.X_train_scaled, self.y_train)
        
        # 评估所有集成方法
        ensemble_results = {}
        
        for name, model in [('硬投票', voting_hard), ('软投票', voting_soft), ('堆叠', stacking)]:
            train_score = model.score(self.X_train_scaled, self.y_train)
            test_score = model.score(self.X_test_scaled, self.y_test)
            
            # 交叉验证
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            ensemble_results[name] = {
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"\n{name} 集成:")
            print(f"  训练集准确率: {train_score:.4f}")
            print(f"  测试集准确率: {test_score:.4f}")
            print(f"  交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 4. 随机森林集成（作为对比）
        print("\n4. 随机森林集成分析...")
        rf_ensemble = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_ensemble.fit(self.X_train_scaled, self.y_train)
        
        rf_train_score = rf_ensemble.score(self.X_train_scaled, self.y_train)
        rf_test_score = rf_ensemble.score(self.X_test_scaled, self.y_test)
        rf_cv_scores = cross_val_score(rf_ensemble, self.X_train_scaled, self.y_train, cv=5)
        
        ensemble_results['随机森林'] = {
            'train_score': rf_train_score,
            'test_score': rf_test_score,
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std(),
            'model': rf_ensemble
        }
        
        print(f"\n随机森林:")
        print(f"  训练集准确率: {rf_train_score:.4f}")
        print(f"  测试集准确率: {rf_test_score:.4f}")
        print(f"  交叉验证: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
        
        # 可视化集成学习结果
        self.visualize_ensemble_results(ensemble_results)
        
        return ensemble_results
        
    def visualize_ensemble_results(self, ensemble_results):
        """可视化集成学习结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('集成学习方法性能对比', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        names = list(ensemble_results.keys())
        train_scores = [ensemble_results[name]['train_score'] for name in names]
        test_scores = [ensemble_results[name]['test_score'] for name in names]
        cv_means = [ensemble_results[name]['cv_mean'] for name in names]
        cv_stds = [ensemble_results[name]['cv_std'] for name in names]
        
        # 1. 训练集 vs 测试集准确率
        ax1 = axes[0, 0]
        x_pos = np.arange(len(names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, train_scores, width, label='训练集', color='lightblue', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, test_scores, width, label='测试集', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('集成方法', fontfamily='SimHei')
        ax1.set_ylabel('准确率', fontfamily='SimHei')
        ax1.set_title('训练集 vs 测试集准确率', fontweight='bold', fontfamily='SimHei')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend(prop={'family': 'SimHei'})
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontfamily='SimHei')
        
        # 2. 交叉验证结果
        ax2 = axes[0, 1]
        bars3 = ax2.bar(names, cv_means, yerr=cv_stds, capsize=5, color='lightgreen', alpha=0.8)
        ax2.set_xlabel('集成方法', fontfamily='SimHei')
        ax2.set_ylabel('交叉验证准确率', fontfamily='SimHei')
        ax2.set_title('交叉验证性能 (5折)', fontweight='bold', fontfamily='SimHei')
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, mean, std in zip(bars3, cv_means, cv_stds):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9, fontfamily='SimHei')
        
        # 3. 过拟合分析
        ax3 = axes[1, 0]
        overfitting = [train - test for train, test in zip(train_scores, test_scores)]
        bars4 = ax3.bar(names, overfitting, color='gold', alpha=0.8)
        ax3.set_xlabel('集成方法', fontfamily='SimHei')
        ax3.set_ylabel('过拟合程度 (训练-测试)', fontfamily='SimHei')
        ax3.set_title('过拟合程度分析', fontweight='bold', fontfamily='SimHei')
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar, overfit in zip(bars4, overfitting):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{overfit:.3f}', ha='center', va='bottom', fontsize=9, fontfamily='SimHei')
        
        # 4. 集成方法贡献分析
        ax4 = axes[1, 1]
        
        # 获取基分类器的性能（用于对比）
        base_scores = {}
        for name, model in [('逻辑回归', LogisticRegression(random_state=42)),
                           ('SVM', SVC(random_state=42, probability=True)),  # 启用概率预测
                           ('随机森林', RandomForestClassifier(n_estimators=50, random_state=42)),
                           ('K近邻', KNeighborsClassifier()),
                           ('决策树', DecisionTreeClassifier(random_state=42))]:
            model.fit(self.X_train_scaled, self.y_train)
            base_scores[name] = model.score(self.X_test_scaled, self.y_test)
        
        # 绘制对比图
        all_names = names + list(base_scores.keys())
        all_scores = test_scores + list(base_scores.values())
        
        # 分类显示
        ensemble_indices = list(range(len(names)))
        base_indices = list(range(len(names), len(all_names)))
        
        ax4.bar([all_names[i] for i in ensemble_indices], [all_scores[i] for i in ensemble_indices],
               color='lightcoral', alpha=0.8, label='集成方法')
        ax4.bar([all_names[i] for i in base_indices], [all_scores[i] for i in base_indices],
               color='lightblue', alpha=0.8, label='基分类器')
        
        ax4.set_xlabel('分类器')
        ax4.set_ylabel('测试集准确率')
        ax4.set_title('集成方法 vs 基分类器', fontweight='bold')
        ax4.set_xticklabels(all_names, rotation=45, ha='right')
        ax4.legend(prop={'family': 'SimHei'})
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_learning_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def model_interpretability_analysis(self):
        """模型解释性分析"""
        print("\n" + "="*60)
        print("模型解释性分析")
        print("="*60)
        
        try:
            import shap
            print("\n1. SHAP值分析...")
            
            # 训练随机森林模型用于SHAP分析
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(self.X_train_scaled, self.y_train)
            
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(rf_model)
            
            # 计算SHAP值（使用测试集的一部分以提高性能）
            sample_size = min(50, len(self.X_test_scaled))
            sample_indices = np.random.choice(len(self.X_test_scaled), size=sample_size, replace=False)
            X_sample = self.X_test_scaled[sample_indices]
            y_sample = self.y_test[sample_indices]
            
            shap_values = explainer.shap_values(X_sample)
            
            # 可视化SHAP分析结果
            self.visualize_shap_analysis(shap_values, X_sample, rf_model)
            
        except ImportError:
            print("SHAP库未安装，使用替代方法进行模型解释性分析...")
            self.alternative_model_interpretability()
        
        # 2. 特征贡献分析
        print("\n2. 特征贡献分析...")
        self.feature_contribution_analysis()
        
        # 3. 决策路径分析
        print("\n3. 决策路径分析...")
        self.decision_path_analysis()
        
    def visualize_shap_analysis(self, shap_values, X_sample, model):
        """可视化SHAP分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SHAP模型解释性分析', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        # 1. 全局特征重要性（平均绝对SHAP值）
        ax1 = axes[0, 0]
        
        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.mean([np.abs(shap_values[i]) for i in range(len(self.target_names))], axis=0)
        feature_importance = np.mean(mean_abs_shap, axis=0)
        
        bars = ax1.barh(self.feature_names, feature_importance, color='skyblue', alpha=0.8)
        ax1.set_xlabel('平均绝对SHAP值', fontfamily='SimHei')
        ax1.set_title('全局特征重要性 (SHAP)', fontweight='bold', fontfamily='SimHei')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, importance in zip(bars, feature_importance):
            ax1.text(importance + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', va='center', fontsize=9, fontfamily='SimHei')
        
        # 2. 特征依赖图（选择最重要的特征）
        ax2 = axes[0, 1]
        
        # 找到最重要的特征
        most_important_feature_idx = np.argmax(feature_importance)
        most_important_feature = self.feature_names[most_important_feature_idx]
        
        # 绘制依赖图（简化版）
        feature_values = X_sample[:, most_important_feature_idx]
        shap_values_feature = np.mean([shap_values[i][:, most_important_feature_idx] for i in range(len(self.target_names))], axis=0)
        
        # 获取预测概率或决策函数值用于颜色映射
        if hasattr(model, 'predict_proba'):
            # 模型支持概率预测
            color_values = np.mean([model.predict_proba(X_sample)[:, i] for i in range(len(self.target_names))], axis=0)
        elif hasattr(model, 'decision_function'):
            # 模型支持决策函数（如SVM）
            decision_values = model.decision_function(X_sample)
            if len(self.target_names) == 2:
                # 二分类：使用决策函数值
                color_values = decision_values
            else:
                # 多分类：使用平均决策函数值
                color_values = np.mean(decision_values, axis=1) if decision_values.ndim > 1 else decision_values
        else:
            # 模型不支持概率预测或决策函数，使用预测类别
            predictions = model.predict(X_sample)
            color_values = predictions
        
        scatter = ax2.scatter(feature_values, shap_values_feature, 
                            c=color_values,
                            cmap='coolwarm', alpha=0.7, s=50)
        ax2.set_xlabel(f'{most_important_feature}', fontfamily='SimHei')
        ax2.set_ylabel('SHAP值', fontfamily='SimHei')
        ax2.set_title(f'特征依赖图: {most_important_feature}', fontweight='bold', fontfamily='SimHei')
        ax2.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('平均预测概率', fontfamily='SimHei')
        
        # 3. 单个样本解释
        ax3 = axes[1, 0]
        
        # 选择一个样本进行详细解释
        sample_idx = 0
        sample_shap = np.array([shap_values[i][sample_idx] for i in range(len(self.target_names))])
        
        # 绘制瀑布图（简化版）
        y_pos = np.arange(len(self.feature_names))
        
        # 使用第一个类别的SHAP值
        shap_values_sample = sample_shap[0]
        
        # 分离正负值
        positive_mask = shap_values_sample > 0
        negative_mask = shap_values_sample < 0
        
        ax3.barh(y_pos[positive_mask], shap_values_sample[positive_mask], 
                color='red', alpha=0.7, label='正向贡献')
        ax3.barh(y_pos[negative_mask], shap_values_sample[negative_mask], 
                color='blue', alpha=0.7, label='负向贡献')
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(self.feature_names, fontfamily='SimHei')
        ax3.set_xlabel('SHAP值', fontfamily='SimHei')
        ax3.set_title(f'单个样本解释 (样本 {sample_idx})', fontweight='bold', fontfamily='SimHei')
        ax3.legend(prop={'family': 'SimHei'})
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. 类别特定的SHAP值
        ax4 = axes[1, 1]
        
        # 计算每个类别的平均SHAP值
        mean_shap_per_class = np.array([np.mean(np.abs(shap_values[i]), axis=0) for i in range(len(self.target_names))])
        
        # 创建热力图
        im = ax4.imshow(mean_shap_per_class, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(self.feature_names)))
        ax4.set_xticklabels(self.feature_names, rotation=45, ha='right', fontfamily='SimHei')
        ax4.set_yticks(range(len(self.target_names)))
        ax4.set_yticklabels(self.target_names, fontfamily='SimHei')
        ax4.set_xlabel('特征', fontfamily='SimHei')
        ax4.set_ylabel('类别', fontfamily='SimHei')
        ax4.set_title('类别特定的特征重要性', fontweight='bold', fontfamily='SimHei')
        
        # 添加数值
        for i in range(len(self.target_names)):
            for j in range(len(self.feature_names)):
                ax4.text(j, i, f'{mean_shap_per_class[i, j]:.3f}', 
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold', fontfamily='SimHei')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('平均绝对SHAP值', fontfamily='SimHei')
        
        plt.tight_layout()
        plt.savefig('shap_model_interpretability.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def alternative_model_interpretability(self):
        """替代的模型解释性分析（当SHAP不可用时）"""
        print("使用替代方法进行模型解释性分析...")
        
        # 使用排列重要性和部分依赖图的概念
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型解释性分析 (替代方法)', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        # 1. 特征重要性（基于排列）
        ax1 = axes[0, 0]
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        
        perm_importance = permutation_importance(rf_model, self.X_test_scaled, self.y_test, 
                                                  n_repeats=10, random_state=42)
        
        bars = ax1.barh(self.feature_names, perm_importance.importances_mean, 
                       color='skyblue', alpha=0.8)
        ax1.errorbar(perm_importance.importances_mean, range(len(self.feature_names)), 
                    xerr=perm_importance.importances_std, fmt='none', color='black', alpha=0.6)
        ax1.set_xlabel('排列重要性 (准确率下降)', fontfamily='SimHei')
        ax1.set_title('特征重要性 (排列法)', fontweight='bold', fontfamily='SimHei')
        ax1.grid(True, alpha=0.3)
        
        # 2. 特征相关性分析
        ax2 = axes[0, 1]
        
        # 计算特征与目标变量的相关性
        correlations = []
        for i in range(self.X_train_scaled.shape[1]):
            corr = np.corrcoef(self.X_train_scaled[:, i], self.y_train)[0, 1]
            correlations.append(abs(corr))
        
        bars2 = ax2.barh(self.feature_names, correlations, color='lightgreen', alpha=0.8)
        ax2.set_xlabel('与目标变量的绝对相关性', fontfamily='SimHei')
        ax2.set_title('特征相关性分析', fontweight='bold', fontfamily='SimHei')
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征分布分析
        ax3 = axes[1, 0]
        
        # 为每个类别绘制特征分布
        for i, class_name in enumerate(self.target_names):
            class_mask = self.y_train == i
            for j, feature_name in enumerate(self.feature_names):
                if j == 0:  # 只绘制第一个特征
                    ax3.hist(self.X_train_scaled[class_mask, j], bins=20, alpha=0.6, 
                           label=class_name, color=self.colors[i])
        
        ax3.set_xlabel(f'标准化 {self.feature_names[0]}', fontfamily='SimHei')
        ax3.set_ylabel('频次', fontfamily='SimHei')
        ax3.set_title(f'特征分布: {self.feature_names[0]}', fontweight='bold', fontfamily='SimHei')
        ax3.legend(prop={'family': 'SimHei'})
        ax3.grid(True, alpha=0.3)
        
        # 4. 决策边界分析（2D投影）
        ax4 = axes[1, 1]
        
        # 选择两个最重要的特征
        important_features = np.argsort(perm_importance.importances_mean)[-2:]
        
        # 创建2D网格
        h = 0.02
        x_min, x_max = self.X_train_scaled[:, important_features[0]].min() - 1, self.X_train_scaled[:, important_features[0]].max() + 1
        y_min, y_max = self.X_train_scaled[:, important_features[1]].min() - 1, self.X_train_scaled[:, important_features[1]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # 预测网格点
        grid_points = np.zeros((xx.ravel().shape[0], self.X_train_scaled.shape[1]))
        grid_points[:, important_features[0]] = xx.ravel()
        grid_points[:, important_features[1]] = yy.ravel()
        
        # 对其他特征使用均值
        for i in range(self.X_train_scaled.shape[1]):
            if i not in important_features:
                grid_points[:, i] = self.X_train_scaled[:, i].mean()
        
        Z = rf_model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        ax4.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
        
        # 绘制数据点
        for i, class_name in enumerate(self.target_names):
            class_mask = self.y_train == i
            ax4.scatter(self.X_train_scaled[class_mask, important_features[0]], 
                       self.X_train_scaled[class_mask, important_features[1]],
                       c=self.colors[i], label=class_name, edgecolors='black', alpha=0.8)
        
        ax4.set_xlabel(f'标准化 {self.feature_names[important_features[0]]}', fontfamily='SimHei')
        ax4.set_ylabel(f'标准化 {self.feature_names[important_features[1]]}', fontfamily='SimHei')
        ax4.set_title('2D决策边界投影', fontweight='bold', fontfamily='SimHei')
        ax4.legend(prop={'family': 'SimHei'})
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('alternative_model_interpretability.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_contribution_analysis(self):
        """特征贡献分析"""
        print("\n特征贡献分析...")
        
        # 使用随机森林分析特征贡献
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        
        # 分析每个特征对分类的贡献
        contribution_analysis = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # 创建特征值范围
            feature_range = np.linspace(self.X_train_scaled[:, i].min(), 
                                      self.X_train_scaled[:, i].max(), 50)
            
            # 计算每个特征值对应的平均预测概率
            mean_probas = []
            for val in feature_range:
                # 创建测试样本
                test_sample = np.tile(self.X_train_scaled.mean(axis=0), (100, 1))
                test_sample[:, i] = val
                
                # 获取预测概率
                if hasattr(rf_model, 'predict_proba'):
                    probas = rf_model.predict_proba(test_sample)
                    mean_probas.append(probas.mean(axis=0))
                else:
                    predictions = rf_model.predict(test_sample)
                    mean_probas.append(np.mean(predictions == np.arange(len(self.target_names))))
            
            contribution_analysis[feature_name] = {
                'feature_range': feature_range,
                'contributions': np.array(mean_probas)
            }
        
        # 可视化特征贡献
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('特征贡献分析', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        for i, (feature_name, analysis) in enumerate(contribution_analysis.items()):
            ax = axes[i//2, i%2]
            
            feature_range = analysis['feature_range']
            contributions = analysis['contributions']
            
            # 绘制每个类别的贡献
            for class_idx, class_name in enumerate(self.target_names):
                if len(contributions.shape) > 1:
                    ax.plot(feature_range, contributions[:, class_idx], 
                           label=class_name, color=self.colors[class_idx], linewidth=2)
                else:
                    ax.plot(feature_range, contributions, 
                           label=class_name, color=self.colors[class_idx], linewidth=2)
            
            ax.set_xlabel(f'标准化 {feature_name}', fontfamily='SimHei')
            ax.set_ylabel('平均预测概率', fontfamily='SimHei')
            ax.set_title(f'{feature_name} 的贡献曲线', fontweight='bold', fontfamily='SimHei')
            ax.legend(prop={'family': 'SimHei'})
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_contribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def decision_path_analysis(self):
        """决策路径分析"""
        print("\n决策路径分析...")
        
        # 使用决策树进行决策路径分析
        dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)  # 限制深度以便可视化
        dt_model.fit(self.X_train_scaled, self.y_train)
        
        # 分析几个样本的决策路径
        sample_indices = [0, 10, 20]  # 选择几个不同的样本
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('决策路径分析', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        for i, sample_idx in enumerate(sample_indices):
            ax = axes[i]
            
            # 获取样本特征
            sample = self.X_test_scaled[sample_idx:sample_idx+1]
            true_label = self.y_test[sample_idx]
            predicted_label = dt_model.predict(sample)[0]
            
            # 获取决策路径
            decision_path = dt_model.decision_path(sample)
            
            # 获取决策路径的节点
            node_indices = decision_path.indices
            
            # 创建决策路径可视化
            self.visualize_decision_path(dt_model, sample, true_label, predicted_label, node_indices, ax)
            
            ax.set_title(f'样本 {sample_idx}\n真实: {self.target_names[true_label]} | 预测: {self.target_names[predicted_label]}', 
                        fontweight='bold', fontfamily='SimHei')
        
        plt.tight_layout()
        plt.savefig('decision_path_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_decision_path(self, model, sample, true_label, predicted_label, node_indices, ax):
        """可视化单个样本的决策路径"""
        # 简化的决策路径可视化
        
        # 获取特征重要性（用于路径中的重要特征）
        feature_importance = model.feature_importances_
        
        # 创建条形图显示特征值
        feature_values = sample[0]
        
        # 颜色映射：正确预测为绿色，错误预测为红色
        color = 'green' if true_label == predicted_label else 'red'
        
        bars = ax.barh(range(len(self.feature_names)), feature_values, 
                      color=color, alpha=0.7)
        
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)
        ax.set_xlabel('特征值 (标准化)')
        ax.grid(True, alpha=0.3)
        
        # 添加特征值标签
        for j, (bar, value) in enumerate(zip(bars, feature_values)):
            ax.text(value + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', va='center', fontsize=9)
        
        # 添加特征重要性信息
        for j, importance in enumerate(feature_importance):
            ax.text(-0.3, j, f'重要性: {importance:.3f}', 
                   va='center', fontsize=8, color='blue', fontweight='bold')
        
    def classification_difficulty_analysis(self):
        """分类难度区域分析"""
        
        # 1. 基于预测概率的分类难度分析
        print("\n1. 基于预测概率的分类难度分析...")
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        
        # 获取预测概率
        probas = rf_model.predict_proba(self.X_test_scaled)
        
        # 计算分类难度指标
        max_probas = np.max(probas, axis=1)
        entropy = -np.sum(probas * np.log2(probas + 1e-10), axis=1)
        
        # 识别分类困难的样本
        difficult_threshold = 0.8  # 最大概率小于0.8认为困难
        difficult_samples = max_probas < difficult_threshold
        
        print(f"分类困难样本数量: {np.sum(difficult_samples)} / {len(self.y_test)} ({np.mean(difficult_samples)*100:.1f}%)")
        
        # 2. 基于决策边界的分类难度分析
        print("\n2. 基于决策边界的分类难度分析...")
        
        # 使用多个分类器进行集成预测
        models = [
            ('逻辑回归', LogisticRegression(random_state=42)),
            ('SVM', SVC(random_state=42)),
            ('随机森林', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('K近邻', KNeighborsClassifier(n_neighbors=5))
        ]
        
        predictions = []
        for name, model in models:
            model.fit(self.X_train_scaled, self.y_train)
            pred = model.predict(self.X_test_scaled)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算预测一致性
        prediction_consensus = []
        for i in range(predictions.shape[1]):
            unique_preds, counts = np.unique(predictions[:, i], return_counts=True)
            consensus = np.max(counts) / len(models)
            prediction_consensus.append(consensus)
        
        consensus = np.array(prediction_consensus)
        
        # 识别低共识样本（分类困难）
        low_consensus_threshold = 0.75
        low_consensus_samples = consensus < low_consensus_threshold
        
        print(f"低共识样本数量: {np.sum(low_consensus_samples)} / {len(self.y_test)} ({np.mean(low_consensus_samples)*100:.1f}%)")
        
        # 3. 基于特征空间的分类难度分析
        print("\n3. 基于特征空间的分类难度分析...")
        
        # 计算样本到类中心的距离
        class_centers = []
        for class_idx in range(len(self.target_names)):
            class_mask = self.y_train == class_idx
            class_center = np.mean(self.X_train_scaled[class_mask], axis=0)
            class_centers.append(class_center)
        
        class_centers = np.array(class_centers)
        
        # 计算每个测试样本到各类中心的距离
        distances_to_centers = []
        for i, sample in enumerate(self.X_test_scaled):
            distances = []
            for center in class_centers:
                dist = np.linalg.norm(sample - center)
                distances.append(dist)
            distances_to_centers.append(distances)
        
        distances_to_centers = np.array(distances_to_centers)
        
        # 计算最小距离和第二小距离的比值（越小越困难）
        min_distances = np.min(distances_to_centers, axis=1)
        second_min_distances = np.partition(distances_to_centers, 1, axis=1)[:, 1]
        distance_ratios = min_distances / second_min_distances
        
        # 可视化分类难度分析结果
        self.visualize_classification_difficulty(max_probas, entropy, consensus, distance_ratios)
        
        return {
            'max_probas': max_probas,
            'entropy': entropy,
            'consensus': consensus,
            'distance_ratios': distance_ratios,
            'difficult_samples': difficult_samples,
            'low_consensus_samples': low_consensus_samples
        }
        
    def visualize_classification_difficulty(self, max_probas, entropy, consensus, distance_ratios):
        """可视化分类难度分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('分类难度区域分析', fontsize=16, fontweight='bold', fontfamily='SimHei')
        
        # 1. 预测概率分布
        ax1 = axes[0, 0]
        
        # 绘制直方图
        ax1.hist(max_probas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='困难阈值 (0.8)')
        ax1.set_xlabel('最大预测概率', fontfamily='SimHei')
        ax1.set_ylabel('频次', fontfamily='SimHei')
        ax1.set_title('预测概率分布', fontweight='bold', fontfamily='SimHei')
        ax1.grid(True, alpha=0.3)
        ax1.legend(prop={'family': 'SimHei'})
        
        # 添加统计信息
        mean_prob = np.mean(max_probas)
        ax1.text(0.02, 0.95, f'平均概率: {mean_prob:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontfamily='SimHei')
        
        # 2. 预测熵分布
        ax2 = axes[0, 1]
        
        ax2.hist(entropy, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=np.log2(len(self.target_names)), color='red', linestyle='--', 
                   linewidth=2, label='最大熵')
        ax2.set_xlabel('预测熵', fontfamily='SimHei')
        ax2.set_ylabel('频次', fontfamily='SimHei')
        ax2.set_title('预测不确定性 (熵)', fontweight='bold', fontfamily='SimHei')
        ax2.grid(True, alpha=0.3)
        ax2.legend(prop={'family': 'SimHei'})
        
        # 添加统计信息
        mean_entropy = np.mean(entropy)
        ax2.text(0.02, 0.95, f'平均熵: {mean_entropy:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontfamily='SimHei')
        
        # 3. 预测一致性分析
        ax3 = axes[1, 0]
        
        ax3.hist(consensus, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(x=0.75, color='red', linestyle='--', linewidth=2, label='低共识阈值 (0.75)')
        ax3.set_xlabel('预测一致性', fontfamily='SimHei')
        ax3.set_ylabel('频次', fontfamily='SimHei')
        ax3.set_title('多分类器预测一致性', fontweight='bold', fontfamily='SimHei')
        ax3.grid(True, alpha=0.3)
        ax3.legend(prop={'family': 'SimHei'})
        
        # 添加统计信息
        mean_consensus = np.mean(consensus)
        ax3.text(0.02, 0.95, f'平均一致性: {mean_consensus:.3f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontfamily='SimHei')
        
        # 4. 距离比分析
        ax4 = axes[1, 1]
        
        ax4.hist(distance_ratios, bins=20, alpha=0.7, color='gold', edgecolor='black')
        ax4.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='困难阈值 (0.8)')
        ax4.set_xlabel('最小距离/第二小距离', fontfamily='SimHei')
        ax4.set_ylabel('频次', fontfamily='SimHei')
        ax4.set_title('基于距离的样本难度', fontweight='bold', fontfamily='SimHei')
        ax4.grid(True, alpha=0.3)
        ax4.legend(prop={'family': 'SimHei'})
        
        # 添加统计信息
        mean_ratio = np.mean(distance_ratios)
        ax4.text(0.02, 0.95, f'平均比值: {mean_ratio:.3f}', transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontfamily='SimHei')
        
        plt.tight_layout()
        plt.savefig('classification_difficulty_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_advanced_analysis_report(self):
        """生成高级分析报告"""
        print("\n" + "="*80)
        print("高级分析与创新扩展 - 综合分析报告")
        print("="*80)
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 1. 特征重要性分析
        feature_importance_results = self.feature_importance_analysis()
        
        # 2. 集成学习分析
        ensemble_results = self.ensemble_learning_analysis()
        
        # 3. 模型解释性分析
        self.model_interpretability_analysis()
        
        # 4. 分类难度分析
        difficulty_results = self.classification_difficulty_analysis()
        
        
        return {
            'feature_importance': feature_importance_results,
            'ensemble_learning': ensemble_results,
            'classification_difficulty': difficulty_results,
            'generated_files': [
                'feature_importance_analysis.png',
                'ensemble_learning_analysis.png',
                'shap_model_interpretability.png',
                'alternative_model_interpretability.png',
                'feature_contribution_analysis.png',
                'decision_path_analysis.png',
                'classification_difficulty_analysis.png'
            ]
        }


def main():
    """主函数"""
    
    # 创建高级分析器
    analyzer = AdvancedAnalysis()
    
    # 生成高级分析报告
    results = analyzer.generate_advanced_analysis_report()


if __name__ == "__main__":
    main()