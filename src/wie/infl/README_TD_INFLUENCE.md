# TD-Influence: 时间依赖影响力计算方法

本文档介绍如何使用TD-Influence（Temporal-Dependence Influence）方法进行样本影响力分析。

## 概述

TD-Influence是一种基于时间依赖的影响力计算方法，通过分析训练过程中样本的动态行为来识别可疑或重要的训练样本。该方法特别适用于：

- 错标检测
- 异常样本识别
- 数据质量评估
- 模型调试

## 核心特性

- **时间依赖分析**: 考虑训练过程中样本行为的时间演化
- **多信号融合**: 结合损失变化、遗忘事件、困难度和梯度对齐等多个信号
- **资格迹机制**: 使用资格迹积累历史记忆
- **可配置性**: 支持多种超参数配置和实验设置
- **完整评测**: 提供AUROC、PR-AUC、Precision@K等评测指标

## 安装和依赖

确保已安装以下依赖：

```bash
pip install torch numpy pandas scikit-learn pyyaml tqdm
```

## 快速开始

### 1. 基本使用

```bash
# 使用默认配置运行完整流程
python -m experiment.infl.td_main --mode all --data adult --model logreg --seeds 0 1 2

# 仅训练
python -m experiment.infl.td_main --mode train --data adult --model dnn --seeds 0 1 2 3 4

# 仅计算影响力（需要预训练模型）
python -m experiment.infl.td_main --mode influence --data adult --model logreg --seed 0

# 仅评测
python -m experiment.infl.td_main --mode eval --results_dir outputs/experiment_1
```

### 2. 使用预定义配置

```bash
# 高精度配置（使用全模型梯度）
python -m experiment.infl.td_main --mode all --config high_precision --data adult --model dnn

# 快速配置（使用投影和最后层）
python -m experiment.infl.td_main --mode all --config fast --data adult --model logreg

# 敏感配置（更关注梯度对齐）
python -m experiment.infl.td_main --mode all --config sensitive --data adult --model dnn

# 鲁棒配置（更关注损失变化）
python -m experiment.infl.td_main --mode all --config robust --data adult --model dnn
```

### 3. 自定义配置

```bash
# 使用命令行参数自定义
python -m experiment.infl.td_main --mode all \
    --data adult --model logreg \
    --lambda_trace 0.95 --gamma 0.99 \
    --alpha 1.0 0.8 0.3 0.3 \
    --use_projection --proj_dim 128 \
    --num_epochs 50 --batch_size 64

# 使用配置文件
python -m experiment.infl.td_main --mode all \
    --config_file my_config.yaml \
    --data adult --model dnn
```

## 配置说明

### 核心超参数

- `lambda_trace` (默认: 0.9): 资格迹衰减因子，控制历史记忆的保留程度
- `gamma` (默认: 0.97): 远期折扣因子，控制时间权重
- `alpha` (默认: [1.0, 0.5, 0.25, 0.25]): 各信号权重
  - `alpha[0]`: 损失上升信号权重
  - `alpha[1]`: 遗忘事件信号权重
  - `alpha[2]`: 困难度信号权重
  - `alpha[3]`: 梯度对齐信号权重

### 梯度计算设置

- `use_last_layer_only` (默认: True): 是否仅使用最后层梯度
- `use_projection` (默认: False): 是否使用随机投影降维
- `proj_dim` (默认: 128): 投影维度
- `proj_type` (默认: "gaussian"): 投影类型 ("gaussian" 或 "achlioptas")

### 训练设置

- `num_epochs` (默认: 100): 训练轮数
- `batch_size` (默认: 128): 训练批次大小
- `learning_rate` (默认: 0.01): 学习率
- `val_batch_size` (默认: 512): 验证批次大小

### 评测设置

- `k_percentage` (默认: 0.1): Precision@K中的K比例
- `confidence_level` (默认: 0.95): 置信水平

## 配置文件格式

### YAML格式示例

```yaml
# 高精度配置
lambda_trace: 0.95
gamma: 0.99
alpha: [1.0, 0.8, 0.3, 0.3]
use_last_layer_only: false
use_projection: false
num_epochs: 100
batch_size: 128
learning_rate: 0.01
seeds: [0, 1, 2, 3, 4]
```

### JSON格式示例

```json
{
  "lambda_trace": 0.9,
  "gamma": 0.97,
  "alpha": [1.0, 0.5, 0.25, 0.25],
  "use_last_layer_only": true,
  "use_projection": false,
  "num_epochs": 100,
  "batch_size": 128,
  "learning_rate": 0.01,
  "seeds": [0, 1, 2, 3, 4]
}
```

## 输出文件说明

### 训练输出

- `influence_scores.npy`: 影响力分数数组
- `epoch_logs.npz`: 训练过程中的日志信息
- `td_states.npz`: TD-Influence状态信息
- `results.csv`: 详细结果表格
- `config.json`: 使用的配置
- `best_model.pt`: 最佳模型权重
- `checkpoint_epoch_*.pt`: 定期检查点

### 评测输出

- `td_influence_metrics.json`: 评测指标（JSON格式）
- `td_influence_metrics.csv`: 评测指标（CSV格式）

## 评测指标

- **AUROC**: Area Under ROC Curve，衡量二分类性能
- **PR-AUC**: Precision-Recall AUC，适用于不平衡数据
- **Precision@K**: 前K个样本的精确率
- **跨种子统计**: 均值、标准差、置信区间

## 使用示例

### 示例1: 错标检测

```python
from wie.infl.td_evaluation import TDEvaluation, create_synthetic_noisy_dataset

# 创建合成噪声数据
noisy_data = create_synthetic_noisy_dataset(n_samples=1000, noise_ratio=0.1)

# 运行TD-Influence
# ... (训练和计算影响力分数) ...

# 评测
evaluator = TDEvaluation()
results = evaluator.evaluate_noise_detection(
    train_set=noisy_data,
    influence_scores=influence_scores,
    k_percentage=0.1
)

print(f"AUROC: {results['auroc']:.4f}")
print(f"PR-AUC: {results['prauc']:.4f}")
print(f"Precision@K: {results['p_at_k']:.4f}")
```

### 示例2: 自定义配置

```python
from wie.infl.td_config import TDInfluenceConfig, get_default_config

# 创建自定义配置
config = get_default_config()
config.lambda_trace = 0.95
config.gamma = 0.99
config.alpha = [1.0, 0.8, 0.3, 0.3]
config.use_projection = True
config.proj_dim = 64

# 保存配置
config.to_yaml("my_config.yaml")

# 使用配置
# ... (运行训练) ...
```

### 示例3: 批量实验

```python
from wie.infl.td_config import ConfigTemplateGenerator, get_default_config

# 生成网格搜索配置
base_config = get_default_config()
param_grid = {
    'lambda_trace': [0.8, 0.9, 0.95],
    'gamma': [0.9, 0.95, 0.97, 0.99],
    'use_projection': [True, False]
}

configs = ConfigTemplateGenerator.generate_grid_search_configs(
    base_config, param_grid
)

# 运行批量实验
for i, config in enumerate(configs):
    print(f"运行配置 {i+1}/{len(configs)}")
    # ... (运行实验) ...
```

## 性能优化建议

1. **内存优化**:
   - 使用 `use_last_layer_only=True` 减少梯度计算量
   - 使用 `use_projection=True` 降低梯度维度
   - 适当调整 `val_batch_size` 控制验证集大小

2. **计算优化**:
   - 使用 `proj_type="achlioptas"` 获得稀疏投影矩阵
   - 设置 `e_clip` 限制资格迹大小
   - 使用较小的 `proj_dim` 值

3. **精度优化**:
   - 使用 `use_last_layer_only=False` 获得全模型梯度
   - 增加 `num_epochs` 获得更稳定的结果
   - 使用更多种子进行统计

## 故障排除

### 常见问题

1. **内存不足**:
   - 减少 `batch_size` 和 `val_batch_size`
   - 启用 `use_projection` 和 `use_last_layer_only`
   - 减少 `proj_dim`

2. **计算时间过长**:
   - 使用 `config="fast"` 快速配置
   - 减少 `num_epochs`
   - 使用较少的种子

3. **结果不稳定**:
   - 增加种子数量
   - 调整超参数（特别是 `lambda_trace` 和 `gamma`）
   - 检查数据质量

### 调试模式

```bash
# 启用详细日志
python -m experiment.infl.td_main --mode all \
    --data adult --model logreg \
    --log_level DEBUG

# 使用小数据集测试
python -m experiment.infl.td_main --mode all \
    --data adult --model logreg \
    --num_epochs 10 --batch_size 32
```

## 引用

如果您使用了TD-Influence方法，请引用相关论文：

```bibtex
@article{td_influence_2024,
  title={Temporal-Dependence Influence: A Novel Approach for Sample Influence Analysis},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进TD-Influence方法。

## 联系方式

如有问题，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至: your.email@example.com