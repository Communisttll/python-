# 鸢尾花分类与可视化项目代码文件功能说明

## 核心分析文件

### 1. `data_preview.py`
**作用**：基础数据预览与可视化
- 加载鸢尾花数据集并进行基本的数据预处理
- 创建4个箱线图展示每个特征在不同类别中的分布
- 使用Plotly生成6个交互式散点图，展示特征对之间的分布关系
- 提供初步的数据探索功能，适合快速了解数据特征

### 2. `data_preview_enhanced.py`
**作用**：增强版数据探索与高级分析
- 全面的统计分析（描述性统计、正态性检验、方差齐性检验）
- 特征相关性分析（皮尔逊和斯皮尔曼相关系数）
- 数据降维可视化（PCA、LDA、t-SNE）
- 异常值检测（统计方法和孤立森林）
- 数据质量评估和分布分析
- 提供专业级的数据探索功能，适合深入理解数据特性

### 3. `classifier2d.py`
**作用**：2D分类器决策边界可视化
- 使用逻辑回归模型在2D特征空间中进行分类
- 可视化整体决策边界和每个类别的概率分布
- 创建4个子图：1个整体决策边界图 + 3个类别概率图
- 使用颜色映射展示分类概率的渐变效果
- 适合理解简单分类器在2D空间中的工作原理

### 4. `classifier_comparison.py`
**作用**：多分类器系统比较分析
- 实现6+种不同类型分类器的系统比较
- 包含超参数优化（GridSearchCV）和交叉验证
- 评估多个性能指标（准确率、精确率、召回率、F1分数等）
- 支持集成学习方法（投票分类器、堆叠分类器）
- 生成性能对比图表和详细评估报告
- 提供全面的分类器性能评估框架

### 5. `visualize_3d_boundary.py`
**作用**：三维特征空间决策边界可视化
- 在三维特征空间中训练多个分类器
- 可视化三维决策边界和分类曲面
- 支持8种不同分类器（逻辑回归、SVM、K近邻、随机森林等）
- 创建交互式3D可视化图表
- 展示分类器在三维空间中的决策过程
- 适合理解高维特征空间中的分类问题

### 6. `visualize_3d_probability.py`
**作用**：三维概率体可视化和等值面分析
- 实现三维空间中分类概率的体绘制
- 提取概率等值面进行可视化分析
- 支持4种分类器的概率分布比较
- 创建交互式3D概率可视化
- 展示分类器在不同区域的不确定性
- 提供概率分布的直观理解

### 7. `advanced_decision_viz.py`
**作用**：高级决策边界与概率可视化分析
- 概率热图可视化
- 不确定性量化分析
- 多分类器决策边界对比
- 置信度区域可视化
- 提供更精细的决策边界分析
- 适合深入理解分类器的决策过程

### 8. `advanced_analysis.py`
**作用**：高级分析与创新扩展模块
- 特征重要性分析（随机森林、排列重要性）
- 集成学习方法（硬投票、软投票、堆叠集成）
- 模型解释性分析（SHAP值分析）
- 分类难度区域分析
- 特征选择方法比较
- 提供深度分析和创新性研究

### 9. `final_summary.py`
**作用**：最终总结与综合分析报告生成器
- 综合性能对比分析
- 可视化汇总面板生成
- 失败分析与改进建议
- 交互式仪表板创建
- 生成最终综合报告
- 输出到 `final_report/` 目录

## 支持文件

### 10. `README.md`
**作用**：项目说明文档
- 项目概述和深化目标
- 文件结构规划说明
- 各阶段详细实施计划
- 技术栈和运行说明
- 确保高分的关键要素
- 项目总结和成果展示

## 输出文件（`final_report/` 目录）

### 13. `final_report/` 目录
**作用**：最终分析结果输出
- `final_comprehensive_report.md` - 最终综合分析报告
- `final_comprehensive_report.png` - PNG格式报告
- `comprehensive_dashboard.png` - 综合仪表板
- `performance_comparison.png` - 性能对比图
- `confusion_matrices.png` - 混淆矩阵汇总
- `roc_curves.png` - ROC曲线汇总
- `feature_importance.png` - 特征重要性分析
- `classification_difficulty.png` - 分类难度分析
- `interactive_*.png` - 各种交互式可视化
- `3d_isosurface_*.png` - 3D等值面可视化
- `interactive_scatter_matrix.png` - 交互式散点图矩阵
- `interactive_3d_scatter.png` - 交互式3D散点图
- `interactive_parallel_coordinates.png` - 平行坐标图
- `interactive_boxplots.png` - 交互式箱线图
- `interactive_probability_*.png` - 交互式概率可视化
- `project_summary.txt` - 项目总结文本
- `experiment_results/` - 实验配置和结果数据

## 文件执行顺序

建议的执行顺序为：
1. `data_preview.py` → 基础数据了解
2. `data_preview_enhanced.py` → 深度数据探索
3. `classifier2d.py` → 2D分类器理解
4. `classifier_comparison.py` → 多分类器比较
5. `visualize_3d_boundary.py` → 3D决策边界
6. `visualize_3d_probability.py` → 3D概率可视化
7. `advanced_decision_viz.py` → 高级决策分析
8. `advanced_analysis.py` → 高级分析方法
9. `final_summary.py` → 综合总结报告