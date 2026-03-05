"""
工具函数
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def visualize_features(features, labels, species_names, save_path=None):
    """
    使用t-SNE可视化特征分布
    
    Args:
        features: 特征矩阵 (n_samples, n_features)
        labels: 标签列表
        species_names: 物种名称列表
        save_path: 保存路径
    """
    print("正在计算t-SNE...")
    # 改进t-SNE参数：使用PCA初始化，增加迭代次数，自动学习率
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                init='pca', learning_rate='auto', max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 12))
    unique_labels = np.unique(labels)
    
    # 使用更多颜色的调色板
    if len(unique_labels) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    else:
        # 如果物种很多，使用gist_ncar或类似的
        colors = plt.cm.gist_ncar(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        # 获取物种名称
        if int(label) < len(species_names):
            name = species_names[int(label)]
        else:
            name = str(label)
            
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=name, 
                   alpha=0.7, s=60, edgecolors='w', linewidth=0.5)
    
    # 调整图例
    if len(unique_labels) > 30:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
    plt.title("Feature Space Visualization (t-SNE)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"特征可视化已保存到: {save_path}")
    plt.close()

def plot_distance_matrix(distance_matrix, species_names, save_path=None):
    """
    可视化距离矩阵
    
    Args:
        distance_matrix: 距离矩阵
        species_names: 物种名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix, 
                xticklabels=species_names,
                yticklabels=species_names,
                cmap='viridis_r',
                square=True,
                cbar_kws={'label': 'Distance'})
    plt.title("Inter-Species Distance Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"距离矩阵已保存到: {save_path}")
    plt.close()

def save_features(features, labels, species_names, filepath):
    """保存特征到文件"""
    np.savez(filepath, 
             features=features, 
             labels=labels, 
             species_names=species_names)
    print(f"特征已保存到: {filepath}")

def load_features(filepath):
    """从文件加载特征"""
    data = np.load(filepath, allow_pickle=True)
    return data['features'], data['labels'], data['species_names']

def calculate_species_features(image_features, image_labels, all_species_names=None, aggregation='mean'):
    """
    计算每个物种的平均特征
    
    Args:
        image_features: 图像特征矩阵 (n_images, n_features)
        image_labels: 图像标签 (n_images,)
        all_species_names: 所有物种名称列表（字符串），索引对应label
        aggregation: 聚合方式 ('mean' 或 'median')
    
    Returns:
        species_features: 物种特征矩阵 (n_species, n_features)
        species_names: 物种名称列表
    """
    unique_labels = np.unique(image_labels)
    species_features = []
    species_names = []
    
    for label in unique_labels:
        mask = image_labels == label
        species_feat = image_features[mask]
        
        if aggregation == 'mean':
            species_feat = np.mean(species_feat, axis=0)
        elif aggregation == 'median':
            species_feat = np.median(species_feat, axis=0)
        else:
            raise ValueError(f"不支持的聚合方式: {aggregation}")
        
        species_features.append(species_feat)
        
        # 如果提供了物种名称列表，则使用名称，否则使用标签索引
        if all_species_names is not None and int(label) < len(all_species_names):
            species_names.append(all_species_names[int(label)])
        else:
            species_names.append(str(label))
    
    return np.array(species_features), species_names

def plot_distance_distribution(distance_matrix, save_path=None):
    """
    绘制距离分布直方图
    
    Args:
        distance_matrix: 距离矩阵
        save_path: 保存路径
    """
    # 提取上三角矩阵（不包括对角线）
    n = distance_matrix.shape[0]
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(distance_matrix[i, j])
    distances = np.array(distances)
    
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distance Distribution Histogram', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(distances):.3f}')
    plt.axvline(np.median(distances), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(distances):.3f}')
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"距离分布图已保存到: {save_path}")
    plt.close()

def plot_feature_correlation(species_features, species_names, save_path=None, max_species=20):
    """
    绘制特征相关性矩阵
    
    Args:
        species_features: 物种特征矩阵
        species_names: 物种名称列表
        save_path: 保存路径
        max_species: 最大显示物种数量（如果太多则采样）
    """
    n_species = len(species_names)
    
    # 如果物种太多，随机采样
    if n_species > max_species:
        indices = np.random.choice(n_species, max_species, replace=False)
        species_features = species_features[indices]
        species_names = [species_names[i] for i in indices]
        n_species = max_species
    
    # 计算相关性矩阵
    correlation_matrix = np.corrcoef(species_features)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                xticklabels=species_names,
                yticklabels=species_names,
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"特征相关性矩阵已保存到: {save_path}")
    plt.close()

def plot_species_feature_comparison(species_features, species_names, save_path=None, top_n=10):
    """
    绘制物种特征对比图（使用PCA降维到2D）
    
    Args:
        species_features: 物种特征矩阵
        species_names: 物种名称列表
        save_path: 保存路径
        top_n: 显示前N个物种
    """
    from sklearn.decomposition import PCA
    
    n_species = len(species_names)
    if n_species > top_n:
        # 选择前top_n个物种
        species_features = species_features[:top_n]
        species_names = species_names[:top_n]
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(species_features)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(species_names)))
    
    for i, (name, color) in enumerate(zip(species_names, colors)):
        plt.scatter(features_2d[i, 0], features_2d[i, 1], 
                   c=[color], label=str(name), s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    # 避免文字重叠（简单处理）
    for i, name in enumerate(species_names):
        plt.annotate(str(name), (features_2d[i, 0], features_2d[i, 1]), 
                    fontsize=9, ha='center', va='bottom')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title("Species Feature Comparison (PCA)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"物种特征对比图已保存到: {save_path}")
    plt.close()

def plot_clustering_dendrogram(distance_matrix, species_names, save_path=None, max_species=30):
    """
    绘制聚类树状图
    
    Args:
        distance_matrix: 距离矩阵
        species_names: 物种名称列表
        save_path: 保存路径
        max_species: 最大显示物种数量
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    n_species = len(species_names)
    
    # 如果物种太多，随机采样
    if n_species > max_species:
        indices = np.random.choice(n_species, max_species, replace=False)
        distance_matrix = distance_matrix[np.ix_(indices, indices)]
        species_names = [species_names[i] for i in indices]
        n_species = max_species
    
    # 转换为压缩距离矩阵
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # 计算链接矩阵
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    plt.figure(figsize=(14, max(8, n_species * 0.4)))
    dendrogram(linkage_matrix, 
               labels=[str(name) for name in species_names],
               leaf_rotation=90,
               leaf_font_size=8)
    plt.title("Hierarchical Clustering Dendrogram", fontsize=14, fontweight='bold')
    plt.xlabel("Species", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"聚类树状图已保存到: {save_path}")
    plt.close()

def plot_feature_statistics(species_features, species_names, save_path=None):
    """
    绘制特征统计信息（箱线图）
    
    Args:
        species_features: 物种特征矩阵
        species_names: 物种名称列表
        save_path: 保存路径
    """
    n_species = len(species_names)
    
    # 计算每个物种的特征统计
    feature_means = np.mean(species_features, axis=1)
    feature_stds = np.std(species_features, axis=1)
    feature_maxs = np.max(species_features, axis=1)
    feature_mins = np.min(species_features, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 均值
    axes[0, 0].bar(range(n_species), feature_means, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Mean Feature Values', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Species Index', fontsize=10)
    axes[0, 0].set_ylabel('Mean Value', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, n_species, max(1, n_species//10)))
    
    # 标准差
    axes[0, 1].bar(range(n_species), feature_stds, color='coral', alpha=0.7)
    axes[0, 1].set_title('Feature Standard Deviation', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Species Index', fontsize=10)
    axes[0, 1].set_ylabel('Std Value', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(range(0, n_species, max(1, n_species//10)))
    
    # 最大值
    axes[1, 0].bar(range(n_species), feature_maxs, color='green', alpha=0.7)
    axes[1, 0].set_title('Maximum Feature Values', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Species Index', fontsize=10)
    axes[1, 0].set_ylabel('Max Value', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(0, n_species, max(1, n_species//10)))
    
    # 最小值
    axes[1, 1].bar(range(n_species), feature_mins, color='purple', alpha=0.7)
    axes[1, 1].set_title('Minimum Feature Values', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Species Index', fontsize=10)
    axes[1, 1].set_ylabel('Min Value', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(0, n_species, max(1, n_species//10)))
    
    plt.suptitle('Feature Statistics Across Species', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"特征统计图已保存到: {save_path}")
    plt.close()

def detect_outliers(features, labels, image_paths, species_names, output_dir, std_threshold=2.0):
    """
    检测并处理异常值（距离物种中心过远的图像）
    
    Args:
        features: 图像特征矩阵
        labels: 图像标签
        image_paths: 图像路径列表
        species_names: 物种名称列表
        output_dir: 输出目录
        std_threshold: 标准差阈值，超过该阈值被视为异常值
    """
    import shutil
    
    print(f"正在检测异常值 (阈值: {std_threshold} std)...")
    
    if len(image_paths) != len(labels):
        print(f"警告: 图像路径数量 ({len(image_paths)}) 与 标签数量 ({len(labels)}) 不匹配！")
        print("可能是因为使用了TripletDataset进行特征提取，导致路径未被收集。")
        print("跳过异常值检测。")
        return

    outlier_dir = os.path.join(output_dir, "outliers")
    os.makedirs(outlier_dir, exist_ok=True)
    
    unique_labels = np.unique(labels)
    outlier_count = 0
    total_images = len(labels)
    
    # 记录异常值信息的文本文件
    log_path = os.path.join(outlier_dir, "outliers_log.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("异常值检测报告\n")
        f.write("=================\n\n")
        
        for label in unique_labels:
            mask = labels == label
            species_feat = features[mask]
            species_paths = np.array(image_paths)[mask]
            
            if len(species_feat) < 2:
                continue
                
            # 计算物种中心（均值）
            centroid = np.mean(species_feat, axis=0)
            
            # 计算每张图像到中心的距离
            distances = np.linalg.norm(species_feat - centroid, axis=1)
            
            # 计算距离的均值和标准差
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # 确定异常值阈值
            threshold = mean_dist + std_threshold * std_dist
            
            # 找出异常值
            outlier_indices = np.where(distances > threshold)[0]
            
            if len(outlier_indices) > 0:
                species_name = species_names[int(label)]
                f.write(f"物种: {species_name} (ID: {label})\n")
                f.write(f"  均值距离: {mean_dist:.4f}, 标准差: {std_dist:.4f}, 阈值: {threshold:.4f}\n")
                
                # 为该物种创建异常值文件夹
                species_outlier_dir = os.path.join(outlier_dir, species_name)
                os.makedirs(species_outlier_dir, exist_ok=True)
                
                for idx in outlier_indices:
                    path = species_paths[idx]
                    dist = distances[idx]
                    filename = os.path.basename(path)
                    
                    # 复制文件
                    dest_path = os.path.join(species_outlier_dir, f"{dist:.4f}_{filename}")
                    try:
                        shutil.copy2(path, dest_path)
                        f.write(f"  - 异常图像: {filename}, 距离: {dist:.4f}\n")
                        outlier_count += 1
                    except Exception as e:
                        print(f"复制文件失败: {path}, 错误: {e}")
                
                f.write("\n")
    
    print(f"检测完成！共发现 {outlier_count} 个异常图像。")
    print(f"异常图像已保存到: {outlier_dir}")
    print(f"详细日志: {log_path}")
def numpy_to_nexus_file(data, names, fp):
    """
    将物种级的高维特征矩阵保存为带有连续性状声明的 Nexus (.nex) 文件
    
    Args:
        data (np.ndarray): 物种特征矩阵 (n_species, feature_dim)
        names (list or array): 物种名称列表 (n_species,)
        fp (str): 输出的 Nexus 文件路径
    """
    ntax, nchar = data.shape
    data = data.astype('float')

    with open(fp, 'w') as f:
        f.write('#NEXUS\n')
        f.write('Begin data;\n')
        f.write(f'Dimensions ntax={ntax} nchar={nchar};\n')
        f.write('Format datatype=Continuous missing=?;\n')
        f.write('Matrix\n')
        for tax, name in zip(data, names):
            # 将特征向量转为字符串并用双空格拼接
            features_str = '  '.join(tax.astype(str))
            f.write(f'{name}  {features_str}\n')
        f.write(';\n')
        f.write('end;\n')