"""
主程序：整合所有模块
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from utils import (set_seed, visualize_features, plot_distance_matrix, save_features,
                   plot_distance_distribution, plot_feature_correlation, 
                   plot_species_feature_comparison, plot_clustering_dendrogram,
                   plot_feature_statistics, detect_outliers, numpy_to_nexus_file)
from data_loader import load_dataset, create_dataloaders
from models import create_model
from trainer import train_model
from feature_extractor import extract_all_features, FeatureExtractor
from phylogeny import build_phylogenetic_trees, calculate_distance_matrix

# >>> 新增：设置类别数（用于 classification + triplet）
config.NUM_CLASSES = len(species_names)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='植物系统发育树构建')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['train', 'extract', 'phylogeny', 'full'],
                       help='运行模式: train(仅训练), extract(仅提取特征), phylogeny(仅构建树), full(完整流程)')
    parser.add_argument('--image_type', type=str, default='specimen',
                       choices=['specimen', 'habitat'],
                       help='图像类型: specimen(标本) 或 habitat(生境)')
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['resnet50', 'vit_b16', 'inception_resnet_v2'],
                       help='模型类型')
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练，直接使用预训练模型')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    config.MODEL_TYPE = args.model_type
    config.create_output_dirs()
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 确定数据路径
    if args.image_type == 'specimen':
        data_path = config.SPECIMEN_PATH
        # 优先使用my_dataset_blurred，如果不存在则使用results_plantsam
        if not os.path.exists(data_path):
            data_path = "数据集/标本图像/results_plantsam"
    else:
        data_path = config.HABITAT_PATH
    
    print("=" * 60)
    print("植物系统发育树构建系统")
    print("=" * 60)
    print(f"图像类型: {args.image_type}")
    print(f"数据路径: {data_path}")
    print(f"模型类型: {config.MODEL_TYPE}")
    print(f"特征维度: {config.FEATURE_DIM}")
    print(f"设备: {config.DEVICE}")
    print("=" * 60)
    
    # 加载数据集
    print("\n[1/5] 加载数据集...")
    image_paths, labels, species_names = load_dataset(data_path, args.image_type)
    
    if len(image_paths) == 0:
        print("错误: 没有找到图像数据！")
        return
    
    # 创建数据加载器
    print("\n[2/5] 创建数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        image_paths, labels,
        batch_size=config.BATCH_SIZE,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        use_triplet=True,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        model_type=config.MODEL_TYPE
    )
    
    # 训练模型
    model_path = os.path.join(config.OUTPUT_DIR, "models", f"{config.MODEL_TYPE}_best.pth")
    
    if args.mode in ['train', 'full'] and not args.skip_training:
        print("\n[3/5] 训练模型...")
        model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(config, train_loader, val_loader)
        
        # 绘制训练损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        loss_curve_path = os.path.join(config.OUTPUT_DIR, "figures", "training_loss_curve.pdf")
        plt.savefig(loss_curve_path, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"训练损失曲线已保存到: {loss_curve_path}")

        # 绘制训练准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label='Training Accuracy', linewidth=2)
        plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        acc_curve_path = os.path.join(config.OUTPUT_DIR, "figures", "training_accuracy_curve.pdf")
        plt.savefig(acc_curve_path, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"训练准确率曲线已保存到: {acc_curve_path}")
    else:
        print("\n[3/5] 跳过训练，使用预训练模型")
    
# 提取特征或加载特征
    if args.mode in ['extract', 'full']:
        print("\n[4/5] 提取特征...")
        
        # 使用训练好的模型或预训练模型
        if os.path.exists(model_path) and not args.skip_training:
            features, labels, image_paths = extract_all_features(
                config, test_loader, model_path
            )
        else:
            features, labels, image_paths = extract_all_features(
                config, test_loader, None
            )
            
    elif args.mode == 'phylogeny':
        print("\n[4/5] 模式为 phylogeny：正在从本地加载已提取的特征文件...")
        from utils import load_features
        feature_path = os.path.join(
            config.OUTPUT_DIR, 
            "features", 
            f"{args.image_type}_{config.MODEL_TYPE}_features.npz"
        )
        if not os.path.exists(feature_path):
            print(f"错误: 找不到特征文件 {feature_path}！请先运行 --mode extract")
            return
            
        features, labels, loaded_names = load_features(feature_path)
        # 注意：使用本地特征时，通常没有 image_paths，为了后续不报错，可以造一个空列表
        image_paths = [] 
        print(f"成功加载特征：共有 {len(features)} 个样本")

    # 注意这里的缩进！接下来的代码要和上面的 if/elif 同一个层级（即 args.mode在三种情况里都会执行）
    if args.mode in ['extract', 'full', 'phylogeny']:
        # 计算物种级特征
        from utils import calculate_species_features
        species_features, species_names_clean = calculate_species_features(
            features, labels, all_species_names=species_names, aggregation='mean'
        )
        
        print(f"提取了 {len(species_features)} 个物种的特征")
        # ==================== 新增：导出连续型性状矩阵 ====================
        import pandas as pd
        # 生成性状列名：Trait_1, Trait_2, ... Trait_512
        trait_columns = [f"Trait_{i+1}" for i in range(species_features.shape[1])]
        
        # 创建 DataFrame，行名是物种，列名是性状
        traits_df = pd.DataFrame(
            species_features, 
            index=species_names_clean, 
            columns=trait_columns
        )
        
        # 定义保存路径
        traits_csv_path = os.path.join(
            config.OUTPUT_DIR, 
            "features", 
            f"{args.image_type}_{config.MODEL_TYPE}_continuous_traits.csv"
        )
        
        # 保存为 CSV（最适合导入 R 或 RevBayes）
        traits_df.to_csv(traits_csv_path)
        print(f"✓ 连续型性状特征矩阵已保存至: {traits_csv_path}")
        # ==================== 新增：导出 Nexus 格式文件 ====================
        traits_nex_path = os.path.join(
            config.OUTPUT_DIR, 
            "features", 
            f"{args.image_type}_{config.MODEL_TYPE}_continuous_traits.nex"
        )
        # 安全机制：确保物种名中没有空格（用下划线替代），防止建树软件报错
        safe_species_names = [str(name).replace(' ', '_') for name in species_names_clean]
        
        # 写入 Nexus 文件
        numpy_to_nexus_file(species_features, safe_species_names, traits_nex_path)
        print(f"✓ 连续型性状 Nexus 文件已保存至: {traits_nex_path}")
        # ===================================================================
        # 检测异常值
        detect_outliers(features, labels, image_paths, species_names, config.OUTPUT_DIR)
        
        # 保存特征
        if config.SAVE_FEATURES:
            feature_path = os.path.join(
                config.OUTPUT_DIR, 
                "features", 
                f"{args.image_type}_{config.MODEL_TYPE}_features.npz"
            )
            save_features(features, labels, species_names, feature_path)
        
        # 可视化特征
        print("\n可视化特征分布...")
        tsne_path = os.path.join(
            config.OUTPUT_DIR, 
            "figures", 
            f"{args.image_type}_{config.MODEL_TYPE}_tsne.pdf"
        )
        visualize_features(features, labels, species_names, tsne_path)
        
        # 构建系统发育树
        if args.mode in ['phylogeny', 'full']:
            print("\n[5/5] 构建系统发育树...")
            
            # 计算距离矩阵
            distance_matrix = calculate_distance_matrix(
                species_features, 
                metric='euclidean'
            )
            
            # 可视化距离矩阵
            dist_matrix_path = os.path.join(
                config.OUTPUT_DIR,
                "figures",
                f"{args.image_type}_{config.MODEL_TYPE}_distance_matrix.pdf"
            )
            plot_distance_matrix(distance_matrix, species_names_clean, dist_matrix_path)
            
            # 生成额外的实验结果图
            print("\n生成实验结果图...")
            
            # 图1: 距离分布直方图
            dist_dist_path = os.path.join(
                config.OUTPUT_DIR,
                "figures",
                f"{args.image_type}_{config.MODEL_TYPE}_distance_distribution.pdf"
            )
            plot_distance_distribution(distance_matrix, dist_dist_path)
            
            # 图2: 特征相关性矩阵
            corr_matrix_path = os.path.join(
                config.OUTPUT_DIR,
                "figures",
                f"{args.image_type}_{config.MODEL_TYPE}_feature_correlation.pdf"
            )
            plot_feature_correlation(species_features, species_names_clean, corr_matrix_path)
            
            # 图3: 物种特征对比
            feature_comp_path = os.path.join(
                config.OUTPUT_DIR,
                "figures",
                f"{args.image_type}_{config.MODEL_TYPE}_species_feature_comparison.pdf"
            )
            plot_species_feature_comparison(species_features, species_names_clean, feature_comp_path)
            
            # 图4: 聚类树状图
            dendrogram_path = os.path.join(
                config.OUTPUT_DIR,
                "figures",
                f"{args.image_type}_{config.MODEL_TYPE}_clustering_dendrogram.pdf"
            )
            plot_clustering_dendrogram(distance_matrix, species_names_clean, dendrogram_path)
            
            # 图5: 特征统计信息
            feature_stats_path = os.path.join(
                config.OUTPUT_DIR,
                "figures",
                f"{args.image_type}_{config.MODEL_TYPE}_feature_statistics.pdf"
            )
            plot_feature_statistics(species_features, species_names_clean, feature_stats_path)
            
            # 构建树
            trees, distance_matrix = build_phylogenetic_trees(
                species_features,
                species_names_clean,
                methods=config.PHYLOGENY_METHODS,
                distance_metric='euclidean',
                output_dir=os.path.join(config.OUTPUT_DIR, "trees", args.image_type),
                raw_features=features,
                raw_labels=labels
            )
            
            print("\n" + "=" * 60)
            print("完成！")
            print("=" * 60)
            print(f"特征文件: {feature_path if config.SAVE_FEATURES else '未保存'}")
            print(f"系统发育树: {os.path.join(config.OUTPUT_DIR, 'trees', args.image_type)}")
            print(f"可视化结果: {os.path.join(config.OUTPUT_DIR, 'figures')}")
            print("=" * 60)
        else:
            print("\n跳过系统发育树构建")
    else:
        print("\n跳过特征提取和系统发育树构建")

if __name__ == '__main__':
    main()

