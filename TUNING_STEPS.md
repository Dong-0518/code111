# 联合训练调参步骤（针对“2.0 / 0.5 准确率下降”）

你提到 `CLASSIFICATION_WEIGHT=2.0`、`TRIPLET_WEIGHT=0.5` 时准确率下降。  
建议按下面顺序做最少量可复现实验（每组建议 3 个随机种子）：

## Step 1：先固定一个稳健基线

- 推荐基线：
  - `classification_weight=1.0`
  - `triplet_weight=1.0`
  - `margin=0.5`
  - `learning_rate=1e-4`
  - `num_epochs=40`

命令示例：

```bash
python main.py --mode full --image_type specimen --model_type resnet50 \
  --classification_weight 1.0 --triplet_weight 1.0 --margin 0.5 --learning_rate 1e-4 \
  --num_epochs 40 --notes "baseline_1.0_1.0"
```

## Step 2：只扫损失权重（最重要）

在其它参数不变情况下，只改权重，推荐顺序：

1. `0.8 / 1.2`
2. `1.0 / 1.0`
3. `1.2 / 0.8`
4. `1.5 / 0.5`
5. `2.0 / 0.5`（你的对照组）

目标：找到“验证准确率 + 度量质量”折中最优点。  
若你主要关心准确率，通常不会选到过大的分类权重。

## Step 3：再扫 margin

在最佳权重上，测试 `margin ∈ {0.3, 0.5, 0.7}`。  
记录验证准确率与树结构稳定性（相邻分支是否符合先验）。

## Step 4：最后微调学习率

在最佳“权重 + margin”上，测试 `learning_rate ∈ {3e-5, 1e-4, 3e-4}`。  
一般先保留 `1e-4` 作为主参考。

## Step 5：统一记录标准

每次实验至少记录：
- `best_val_acc`
- `best_val_total_loss`
- `best_val_triplet_loss`
- `best_val_classification_loss`
- `silhouette / davies_bouldin / calinski_harabasz / intra_over_inter_ratio`
- 系统发育树主观评分（1-5）

项目已经支持自动追加训练记录到：

`outputs/experiments/experiment_log.csv`

你也可以直接使用仓库根目录的 `experiment_record_template.csv` 作为 Excel 模板。
