"""
系统发育树构建模块
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo
from Bio.Phylo import draw
import matplotlib.pyplot as plt
import os

def _ensure_string_list(species_names):
    """
    确保 species_names 是 Python 字符串列表
    
    Args:
        species_names: 物种名称（可能是列表、numpy数组、pandas Series等）
    
    Returns:
        list: Python 字符串列表
    """
    # 如果是 numpy 数组，转换为列表
    if isinstance(species_names, np.ndarray):
        species_names = species_names.tolist()
    # 如果是其他可迭代对象但不是列表，转换为列表
    elif not isinstance(species_names, list):
        try:
            species_names = list(species_names)
        except TypeError:
            raise TypeError(f"无法将 species_names 转换为列表: {type(species_names)}")
    
    # 确保所有元素都是字符串
    species_names = [str(name) for name in species_names]
    
    return species_names

def calculate_distance_matrix(features, metric='euclidean'):
    """
    计算距离矩阵
    
    Args:
        features: 特征矩阵 (n_samples, n_features)
        metric: 距离度量 ('euclidean', 'cosine', 'manhattan')
    
    Returns:
        distance_matrix: 距离矩阵 (n_samples, n_samples)
    """
    from scipy.spatial.distance import pdist, squareform
    
    if metric == 'euclidean':
        distances = pdist(features, metric='euclidean')
    elif metric == 'cosine':
        distances = pdist(features, metric='cosine')
    elif metric == 'manhattan':
        distances = pdist(features, metric='cityblock')
    else:
        raise ValueError(f"不支持的距离度量: {metric}")
    
    distance_matrix = squareform(distances)
    return distance_matrix

def build_upgma_tree(distance_matrix, species_names):
    """
    使用UPGMA方法构建系统发育树
    
    Args:
        distance_matrix: 距离矩阵
        species_names: 物种名称列表
    
    Returns:
        tree: BioPython的Tree对象
    """
    # 确保 species_names 是 Python 字符串列表
    species_names = _ensure_string_list(species_names)
    
    # 验证输入
    n = len(species_names)
    if n < 2:
        raise ValueError(f"构建系统发育树至少需要2个物种，当前只有 {n} 个")
    
    # 确保距离矩阵是 numpy 数组
    distance_matrix = np.asarray(distance_matrix)
    
    # 验证距离矩阵维度
    if distance_matrix.shape[0] != n or distance_matrix.shape[1] != n:
        raise ValueError(f"距离矩阵维度 ({distance_matrix.shape}) 与物种数量 ({n}) 不匹配")
    
    # 转换为BioPython的DistanceMatrix格式
    dist_list = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(float(distance_matrix[i, j]))
        dist_list.append(row)
    
    # 创建DistanceMatrix对象
    dm = DistanceMatrix(names=species_names, matrix=dist_list)
    
    # 使用UPGMA方法构建树
    constructor = DistanceTreeConstructor()
    tree = constructor.upgma(dm)
    
    return tree

def build_nj_tree(distance_matrix, species_names):
    """
    使用Neighbor-Joining方法构建系统发育树
    
    Args:
        distance_matrix: 距离矩阵
        species_names: 物种名称列表
    
    Returns:
        tree: BioPython的Tree对象
    """
    # 确保 species_names 是 Python 字符串列表
    species_names = _ensure_string_list(species_names)
    
    # 验证输入
    n = len(species_names)
    if n < 2:
        raise ValueError(f"构建系统发育树至少需要2个物种，当前只有 {n} 个")
    
    # 确保距离矩阵是 numpy 数组
    distance_matrix = np.asarray(distance_matrix)
    
    # 验证距离矩阵维度
    if distance_matrix.shape[0] != n or distance_matrix.shape[1] != n:
        raise ValueError(f"距离矩阵维度 ({distance_matrix.shape}) 与物种数量 ({n}) 不匹配")
    
    # 转换为BioPython的DistanceMatrix格式
    dist_list = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(float(distance_matrix[i, j]))
        dist_list.append(row)
    
    # 创建DistanceMatrix对象
    dm = DistanceMatrix(names=species_names, matrix=dist_list)
    
    # 使用NJ方法构建树
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    
    return tree

def build_upgma_scipy(distance_matrix, species_names):
    """
    使用scipy的UPGMA方法构建系统发育树
    
    Args:
        distance_matrix: 距离矩阵
        species_names: 物种名称列表
    
    Returns:
        tree: Newick格式的树字符串
    """
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    
    # 转换为压缩距离矩阵
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # 使用UPGMA（average linkage）
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # 转换为树结构
    tree = to_tree(linkage_matrix)
    
    # 转换为Newick格式
    newick_str = tree_to_newick(tree, species_names)
    
    return newick_str, linkage_matrix

def tree_to_newick(node, species_names, parent_dist=0.0):
    """
    将树节点转换为Newick格式
    
    Args:
        node: 树节点
        species_names: 物种名称列表
        parent_dist: 父节点距离
    
    Returns:
        newick_str: Newick格式字符串
    """
    # 确保 species_names 是 Python 字符串列表
    species_names = _ensure_string_list(species_names)
    
    if node.is_leaf():
        # 叶子节点
        dist = max(0.0, parent_dist - (node.dist if hasattr(node, 'dist') else 0.0))
        name = species_names[node.id] if node.id < len(species_names) else f"species_{node.id}"
        return f"{name}:{dist:.6f}"
    else:
        # 内部节点
        node_dist = node.dist if hasattr(node, 'dist') else 0.0
        dist = max(0.0, parent_dist - node_dist)
        left = tree_to_newick(node.left, species_names, node_dist)
        right = tree_to_newick(node.right, species_names, node_dist)
        return f"({left},{right}):{dist:.6f}"

def visualize_tree(tree, species_names, method='upgma', save_path=None):
    """
    可视化系统发育树
    
    Args:
        tree: BioPython的Tree对象或Newick字符串
        species_names: 物种名称列表
        method: 方法名称
        save_path: 保存路径
    """
    # 确保 species_names 是 Python 字符串列表
    species_names = _ensure_string_list(species_names)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(species_names) * 0.3)))
    
    if isinstance(tree, str):
        # 如果是Newick字符串，需要先解析
        from io import StringIO
        tree_obj = Phylo.read(StringIO(tree), "newick")
    else:
        tree_obj = tree
    
    # 绘制树
    Phylo.draw(tree_obj, axes=ax, do_show=False)
    method_name = method.upper() if method.upper() == 'UPGMA' else 'Neighbor-Joining'
    ax.set_title(f"Phylogenetic Tree ({method_name})", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"系统发育树已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_tree_newick(tree, species_names, filepath):
    """
    保存树为Newick格式
    
    Args:
        tree: BioPython的Tree对象或Newick字符串
        species_names: 物种名称列表
        filepath: 保存路径
    """
    if isinstance(tree, str):
        newick_str = tree
    else:
        # 转换为Newick格式
        from io import StringIO
        handle = StringIO()
        Phylo.write(tree, handle, "newick")
        newick_str = handle.getvalue()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(newick_str)
    
    print(f"Newick格式树已保存到: {filepath}")

def save_tree_nexus(tree, species_names, filepath):
    """
    保存树为NEXUS格式（使用BioPython原生支持，确保兼容所有建树软件）
    
    Args:
        tree: BioPython的Tree对象或Newick字符串
        species_names: 物种名称列表
        filepath: 保存路径
    """
    from Bio import Phylo
    from io import StringIO
    
    # 1. 如果传进来的是字符串，先转成 Tree 对象
    if isinstance(tree, str):
        tree_obj = Phylo.read(StringIO(tree), "newick")
    else:
        tree_obj = tree
        
    # 2. 直接使用原生写入器导出 NEXUS 格式
    Phylo.write(tree_obj, filepath, "nexus")
    
    print(f"NEXUS格式树已安全保存到: {filepath}")
def save_distance_matrix_excel(distance_matrix, species_names, filepath):
    """
    保存距离矩阵为Excel文件
    
    Args:
        distance_matrix: 距离矩阵
        species_names: 物种名称列表
        filepath: 保存路径
    """
    import pandas as pd
    
    # 确保 species_names 是 Python 字符串列表
    species_names = _ensure_string_list(species_names)
    
    df = pd.DataFrame(distance_matrix, index=species_names, columns=species_names)
    df.to_excel(filepath)
    print(f"距离矩阵Excel文件已保存到: {filepath}")

def bootstrap_consensus_tree(features, labels, species_names, method='upgma', n_bootstraps=100):
    """
    执行Bootstrap分析并构建共识树
    
    Args:
        features: 原始图像特征矩阵 (n_images, n_features)
        labels: 图像标签 (n_images,)
        species_names: 物种名称列表
        method: 建树方法 ('upgma' 或 'nj')
        n_bootstraps: Bootstrap次数
        
    Returns:
        consensus_tree: 共识树
        support_tree: 带有支持度的树
    """
    from Bio.Phylo import Consensus
    
    # 确保 species_names 是 Python 字符串列表
    species_names = _ensure_string_list(species_names)
    
    print(f"开始Bootstrap分析 ({n_bootstraps}次)...")
    trees = []
    
    unique_labels = np.unique(labels)
    n_species = len(unique_labels)
    
    for b in range(n_bootstraps):
        # 对每个物种进行有放回抽样
        boot_species_features = []
        
        for label in unique_labels:
            mask = labels == label
            species_feat = features[mask]
            n_samples = len(species_feat)
            
            # 有放回抽样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sampled_feat = species_feat[indices]
            
            # 计算均值特征
            boot_species_features.append(np.mean(sampled_feat, axis=0))
        
        boot_species_features = np.array(boot_species_features)
        
        # 计算距离矩阵
        dist_matrix = calculate_distance_matrix(boot_species_features)
        
        # 构建树
        if method.lower() == 'upgma':
            tree = build_upgma_tree(dist_matrix, species_names)
        else:
            tree = build_nj_tree(dist_matrix, species_names)
            
        trees.append(tree)
    
    # 构建共识树 (Majority Rule)
    consensus_tree = Consensus.majority_consensus(trees, 0.5)
    
    return consensus_tree

def build_phylogenetic_trees(features, species_names, methods=['upgma', 'nj'], 
                            distance_metric='euclidean', output_dir='outputs/trees',
                            raw_features=None, raw_labels=None):
    """
    构建系统发育树的主函数
    
    Args:
        features: 物种特征矩阵 (n_species, n_features)
        species_names: 物种名称列表
        methods: 使用的方法列表
        distance_metric: 距离度量
        output_dir: 输出目录
        raw_features: 原始图像特征 (用于Bootstrap)
        raw_labels: 原始图像标签 (用于Bootstrap)
    
    Returns:
        trees: 字典，包含不同方法构建的树
    """
    # 统一转换 species_names 为 Python 字符串列表
    species_names = _ensure_string_list(species_names)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"计算距离矩阵 (度量: {distance_metric})...")
    distance_matrix = calculate_distance_matrix(features, metric=distance_metric)
    
    # 保存距离矩阵到Excel
    excel_path = os.path.join(output_dir, "distance_matrix.xlsx")
    save_distance_matrix_excel(distance_matrix, species_names, excel_path)
    
    trees = {}
    
    for method in methods:
        print(f"\n使用 {method.upper()} 方法构建系统发育树...")
        
        try:
            if method.lower() == 'upgma':
                tree = build_upgma_tree(distance_matrix, species_names)
            elif method.lower() == 'nj':
                tree = build_nj_tree(distance_matrix, species_names)
            else:
                print(f"警告: 不支持的方法 {method}，跳过")
                continue
            
            trees[method] = tree
            
            # 可视化
            viz_path = os.path.join(output_dir, f"tree_{method}.pdf")
            visualize_tree(tree, species_names, method, viz_path)
            
            # 保存Newick格式
            newick_path = os.path.join(output_dir, f"tree_{method}.newick")
            save_tree_newick(tree, species_names, newick_path)
            
            # 保存NEXUS格式
            nexus_path = os.path.join(output_dir, f"tree_{method}.nexus")
            save_tree_nexus(tree, species_names, nexus_path)
            
            # Bootstrap分析
            if raw_features is not None and raw_labels is not None:
                print(f"正在进行 {method.upper()} Bootstrap验证...")
                consensus_tree = bootstrap_consensus_tree(
                    raw_features, raw_labels, species_names, method=method
                )
                
                # 保存共识树
                cons_path = os.path.join(output_dir, f"tree_{method}_consensus.newick")
                save_tree_newick(consensus_tree, species_names, cons_path)
                
                # 可视化共识树
                cons_viz_path = os.path.join(output_dir, f"tree_{method}_consensus.pdf")
                visualize_tree(consensus_tree, species_names, f"{method} Consensus", cons_viz_path)
            
            print(f"✓ {method.upper()} 树构建完成")
            
        except Exception as e:
            print(f"错误: 构建 {method} 树时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return trees, distance_matrix

