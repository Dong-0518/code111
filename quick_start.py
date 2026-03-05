import os
from config import Config
from utils import set_seed, calculate_species_features
from data_loader import load_dataset, create_dataloaders
from feature_extractor import extract_all_features
from phylogeny import build_phylogenetic_trees

def quick_start(image_type='specimen', model_type='resnet50', skip_training=True):
    """
    快速启动函数
    
    Args:
        image_type: 'specimen' 或 'habitat'
        model_type: 'resnet50', 'vit_b16', 或 'inception_resnet_v2'
        skip_training: 是否跳过训练（使用预训练模型）
    """
    # 创建配置
    config = Config()
    config.MODEL_TYPE = model_type
    config.create_output_dirs()
    set_seed(config.SEED)
    
    # 确定数据路径
    if image_type == 'specimen':
        data_path = config.SPECIMEN_PATH
        if not os.path.exists(data_path):
            data_path = "数据集/标本图像/results_plantsam"
    else:
        data_path = config.HABITAT_PATH
    
    print("=" * 60)
    print("快速启动：植物系统发育树构建")
    print("=" * 60)
    print(f"图像类型: {image_type}")
    print(f"模型类型: {model_type}")
    print(f"跳过训练: {skip_training}")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据集...")
    image_paths, labels, species_names = load_dataset(data_path, image_type)
    
    if len(image_paths) == 0:
        print("错误: 没有找到图像数据！")
        return
    
    # 创建数据加载器（使用测试集）
    print("\n[2/4] 创建数据加载器...")
    _, _, test_loader = create_dataloaders(
        image_paths, labels,
        batch_size=config.BATCH_SIZE,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        use_triplet=False,  # 特征提取不需要triplet
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        model_type=config.MODEL_TYPE
    )
    
    # 提取特征
    print("\n[3/4] 提取特征（使用预训练模型）...")
    features, labels, image_paths = extract_all_features(
        config, test_loader, None  # 使用预训练模型
    )
    
    # 计算物种级特征
    species_features, species_names_clean = calculate_species_features(
        features, labels, aggregation='mean'
    )
    
    print(f"提取了 {len(species_features)} 个物种的特征")
    
    # 构建系统发育树
    print("\n[4/4] 构建系统发育树...")
    trees, distance_matrix = build_phylogenetic_trees(
        species_features,
        species_names_clean,
        methods=config.PHYLOGENY_METHODS,
        distance_metric='euclidean',
        output_dir=os.path.join(config.OUTPUT_DIR, "trees", image_type)
    )
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"系统发育树保存在: {os.path.join(config.OUTPUT_DIR, 'trees', image_type)}")
    print("=" * 60)

if __name__ == '__main__':
    import os
    import sys
    
    # 默认参数
    image_type = 'specimen'
    model_type = 'resnet50'
    skip_training = True
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        image_type = sys.argv[1]
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    if len(sys.argv) > 3:
        skip_training = sys.argv[3].lower() == 'true'
    
    quick_start(image_type, model_type, skip_training)

