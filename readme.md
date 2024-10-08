

# 项目名称：细胞网络分析

## 项目概述

本项目旨在通过构建和分析细胞骨架网络图，探索细胞内部结构的复杂性及其功能性。我们使用Python语言，结合NetworkX库，实现了网络的构建、可视化、社区检测以及中心性分析。

## 功能描述

### 1.0版本功能：

- **数据读取**：从Excel文件中读取节点位置数据。
- **网络构建**：基于节点间欧氏距离构建无向图。
- **社区检测**：应用Louvain算法对网络进行社区划分。
- **网络可视化**：绘制网络图，并根据社区划分结果着色。
- **中心性分析**：计算并输出节点的特征向量中心性、度中心性、接近中心性和中介中心性。
- **网络特性计算**：计算并输出网络的直径、平均路径长度、平均聚类系数和网络密度。

### 2.0版本（计划）：

- **边权重信息**：将考虑边的权重信息（例如路宽度）进行更深入的网络分析。
- **边的长度信息**：使用边的长度信息来调整网络的结构和分析。

### 3.0版本（计划）：

- **动态网络分析**：如果存在不同生命周期的网络数据，将构建动态网络模型并分析其变化。
- **三维时空网络构建**：引入Z轴维度，构建三维时空网络，以包含更多生物学信息。

## 使用指南

### 安装依赖：

请确保已安装以下Python库：

```python
pip install pandas numpy networkx matplotlib scikit-learn community
```

### 准备数据：

将节点的位置数据保存在Excel文件中，文件应包含两列，分别代表X和Y坐标。

### 运行代码：

将上述代码保存为Python脚本，确保Excel文件路径正确，然后运行脚本。

## 示例代码

```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import community as community_louvain

# 从Excel文件中读取数据
file_path = 'test.xlsx'  # Excel文件路径
positions = pd.read_excel(file_path, usecols=[0, 1])

# 构建图
G = nx.Graph()

# 添加节点
for i, (x, y) in enumerate(positions.values):
    G.add_node(i, pos=(x, y))

# 计算距离矩阵
dist_matrix = pairwise_distances(positions, metric='euclidean')

# 添加边（假设距离小于某个阈值的点之间有边）
threshold = 1.0  # 这个阈值可以根据需要调整
for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        if dist_matrix[i, j] < threshold:
            G.add_edge(i, j, weight=1/dist_matrix[i, j])  # 使用距离的倒数作为权重

# 应用Louvain算法进行社区检测
partition = community_louvain.best_partition(G)

# 可视化社区
pos = nx.get_node_attributes(G, 'pos')
colors = [partition[node] for node in G.nodes()]
nx.draw(G, pos, node_color=colors, with_labels=True, node_size=50, cmap=plt.cm.jet)
plt.show()

# 计算特征向量中心性，增加迭代次数和容忍度
max_iter = 1000  # 增加最大迭代次数
tolerance = 1e-3  # 调整容忍度
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tolerance)
print("Eigenvector Centrality:", eigenvector_centrality)

# 计算全局网络特性
if nx.is_connected(G):
    path_length = dict(nx.all_pairs_dijkstra_path_length(G))
    diameter = max([max(path) for path in path_length.values()])
    average_path_length = sum(sum(lengths) for lengths in path_length.values()) / (len(G) * (len(G) - 1))
else:
    diameter = None
    average_path_length = None

average_clustering = nx.average_clustering(G)
density = nx.density(G)

# 输出结果
print("Diameter:", diameter)
print("Average Path Length:", average_path_length)
print("Average Clustering Coefficient:", average_clustering)
print("Network Density:", density)

# 计算度中心性
degree_centrality = nx.degree_centrality(G)
print("Degree Centrality:", degree_centrality)

# 计算接近中心性
closeness_centrality = nx.closeness_centrality(G)
print("Closeness Centrality:", closeness_centrality)

# 计算中介中心性
betweenness_centrality = nx.betweenness_centrality(G)
print("Betweenness Centrality:", betweenness_centrality)
```

## 未来展望

### 增强网络分析

- **边权重信息**：在网络分析中引入边的权重信息，例如细胞骨架的厚度或强度，以更准确地反映生物学特性。
- **边的长度信息**：通过考虑边的实际长度，调整网络的几何结构和分析结果，以提高分析的精确性。

### 动态网络构建

- **时间序列数据**：收集不同时间点的细胞骨架数据，构建动态网络模型，分析其随时间的变化。
- **生命周期分析**：研究细胞骨架在不同生命周期阶段的变化，揭示其动态特性。

### 三维时空网络

- **Z轴数据整合**：通过显微镜或其他技术获取细胞的三维结构数据，构建三维网络模型。
- **时空分析**：结合时间维度，分析细胞骨架在三维空间中的动态变化，提供更全面的生物学洞察。

---
