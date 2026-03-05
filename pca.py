import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

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
            # 使用 round() 可以控制一下精度，避免小数点后太长，这里默认保留6位小数
            features_str = '  '.join(tax.round(6).astype(str))
            f.write(f'{name}  {features_str}\n')
        f.write(';\n')
        f.write('end;\n')


# 1. 定义输入和输出路径
input_csv = '/data/yutong/line/codee/outputs/features/habitat_resnet50_continuous_traits.csv'
output_csv = '/data/yutong/line/codee/outputs/features/habitat_resnet50_pca_14d_traits.csv'
output_nex = '/data/yutong/line/codee/outputs/features/habitat_resnet50_pca_14d_traits.nex'

print(f"正在读取数据: {input_csv}")
# 读取 CSV 文件
df = pd.read_csv(input_csv)

# 2. 分离物种名称和数值特征
# 假设第一列是物种名字
species_names = df.iloc[:, 0].values
features = df.iloc[:, 1:].values

print(f"原始特征矩阵大小: {features.shape} (物种数, 特征维度)")

# 3. 数据标准化 (Standardization)
# 保证 PCA 的效果最佳
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. 执行 PCA 降维 (强制降至 14 维)
n_components = 14
pca = PCA(n_components=n_components) 
features_pca = pca.fit_transform(features_scaled)

print(f"PCA 降维完成！特征维度已固定降至 {features_pca.shape[1]} 维。")
print(f"这 14 个主成分总共解释了 {np.sum(pca.explained_variance_ratio_) * 100:.2f}% 的原始方差。")

# 5. 保存为降维后的 CSV 文件 (可选，方便你在本地查看)
pc_columns = [f'PC{i+1}' for i in range(n_components)]
df_pca = pd.DataFrame(features_pca, columns=pc_columns)
df_pca.insert(0, 'Species', species_names)
df_pca.to_csv(output_csv, index=False)
print(f"降维后的 CSV 已保存至: {output_csv}")

# 6. 直接调用函数，生成 RevBayes 需要的 Nexus 文件
numpy_to_nexus_file(features_pca, species_names, output_nex)
print(f"🎉 连续特征 Nexus 文件已成功生成: {output_nex}")