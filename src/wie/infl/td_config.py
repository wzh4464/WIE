import os
import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging


@dataclass
class TDInfluenceConfig:
    """
    TD-Influence方法的配置类

    包含所有超参数和设置选项
    """

    # 核心超参数
    lambda_trace: float = 0.9  # 资格迹衰减因子
    gamma: float = 0.97  # 远期折扣因子
    alpha: List[float] = (
        None  # 各信号权重 [dloss_pos, forgetting, hard_prob, align_val]
    )

    # 梯度计算设置
    norm_type: str = "cosine"  # 梯度对齐度量类型
    use_last_layer_only: bool = True  # 是否仅使用最后层梯度
    use_projection: bool = False  # 是否使用随机投影
    proj_dim: int = 128  # 投影维度
    proj_type: str = "gaussian"  # 投影类型: "gaussian" 或 "achlioptas"

    # 训练设置
    val_batch_size: int = 512  # 验证集批次大小
    e_clip: float = 10.0  # 资格迹裁剪上限

    # 评测设置
    k_percentage: float = 0.1  # Precision@K中的K比例
    confidence_level: float = 0.95  # 置信水平

    # 实验设置
    seeds: List[int] = None  # 随机种子列表
    num_epochs: int = 100  # 训练轮数
    batch_size: int = 128  # 训练批次大小
    learning_rate: float = 0.01  # 学习率

    def __post_init__(self):
        """初始化后处理"""
        if self.alpha is None:
            self.alpha = [1.0, 0.5, 0.25, 0.25]

        if self.seeds is None:
            self.seeds = [0, 1, 2, 3, 4]

        # 确保alpha是4个元素的列表
        if len(self.alpha) != 4:
            self.alpha = self.alpha[:4] + [0.25] * (4 - len(self.alpha))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_yaml(self, file_path: str) -> None:
        """保存为YAML文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def to_json(self, file_path: str) -> None:
        """保存为JSON文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TDInfluenceConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, file_path: str) -> "TDInfluenceConfig":
        """从YAML文件加载配置"""
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, file_path: str) -> "TDInfluenceConfig":
        """从JSON文件加载配置"""
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []

        # 检查超参数范围
        if not 0 <= self.lambda_trace <= 1:
            errors.append(f"lambda_trace必须在[0,1]范围内，当前值: {self.lambda_trace}")

        if not 0 <= self.gamma <= 1:
            errors.append(f"gamma必须在[0,1]范围内，当前值: {self.gamma}")

        if len(self.alpha) != 4:
            errors.append(f"alpha必须有4个元素，当前长度: {len(self.alpha)}")

        if any(a < 0 for a in self.alpha):
            errors.append("alpha的所有元素必须非负")

        # 检查投影设置
        if self.use_projection:
            if self.proj_dim <= 0:
                errors.append(f"proj_dim必须为正数，当前值: {self.proj_dim}")

            if self.proj_type not in ["gaussian", "achlioptas"]:
                errors.append(
                    f"proj_type必须是'gaussian'或'achlioptas'，当前值: {self.proj_type}"
                )

        # 检查其他参数
        if self.val_batch_size <= 0:
            errors.append(f"val_batch_size必须为正数，当前值: {self.val_batch_size}")

        if self.e_clip < 0:
            errors.append(f"e_clip必须非负，当前值: {self.e_clip}")

        if not 0 < self.k_percentage <= 1:
            errors.append(f"k_percentage必须在(0,1]范围内，当前值: {self.k_percentage}")

        if not 0 < self.confidence_level <= 1:
            errors.append(
                f"confidence_level必须在(0,1]范围内，当前值: {self.confidence_level}"
            )

        if not self.seeds:
            errors.append("seeds列表不能为空")

        if any(s < 0 for s in self.seeds):
            errors.append("所有种子必须非负")

        return errors


class TDInfluenceConfigManager:
    """TD-Influence配置管理器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.configs: Dict[str, TDInfluenceConfig] = {}

    def add_config(self, name: str, config: TDInfluenceConfig) -> None:
        """添加配置"""
        if errors := config.validate():
            raise ValueError(f"配置'{name}'验证失败: {'; '.join(errors)}")

        self.configs[name] = config
        self.logger.info(f"添加配置: {name}")

    def get_config(self, name: str) -> TDInfluenceConfig:
        """获取配置"""
        if name not in self.configs:
            raise KeyError(f"配置'{name}'不存在")
        return self.configs[name]

    def list_configs(self) -> List[str]:
        """列出所有配置名称"""
        return list(self.configs.keys())

    def remove_config(self, name: str) -> None:
        """删除配置"""
        if name in self.configs:
            del self.configs[name]
            self.logger.info(f"删除配置: {name}")

    def save_all_configs(self, base_dir: str) -> None:
        """保存所有配置到目录"""
        os.makedirs(base_dir, exist_ok=True)

        for name, config in self.configs.items():
            # 保存YAML格式
            yaml_path = os.path.join(base_dir, f"{name}.yaml")
            config.to_yaml(yaml_path)

            # 保存JSON格式
            json_path = os.path.join(base_dir, f"{name}.json")
            config.to_json(json_path)

        self.logger.info(f"保存了{len(self.configs)}个配置到: {base_dir}")

    def load_configs_from_dir(self, config_dir: str) -> None:
        """从目录加载所有配置"""
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"配置目录不存在: {config_dir}")

        loaded_count = 0
        for filename in os.listdir(config_dir):
            if filename.endswith((".yaml", ".yml")):
                name = os.path.splitext(filename)[0]
                config_path = os.path.join(config_dir, filename)
                try:
                    config = TDInfluenceConfig.from_yaml(config_path)
                    self.configs[name] = config
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"加载配置文件失败 {config_path}: {e}")

        self.logger.info(f"从{config_dir}加载了{loaded_count}个配置")


# 预定义配置
def get_default_config() -> TDInfluenceConfig:
    """获取默认配置"""
    return TDInfluenceConfig()


def get_high_precision_config() -> TDInfluenceConfig:
    """获取高精度配置（使用全模型梯度）"""
    return TDInfluenceConfig(
        use_last_layer_only=False,
        use_projection=False,
        lambda_trace=0.95,
        gamma=0.99,
        alpha=[1.0, 0.8, 0.3, 0.3],
    )


def get_fast_config() -> TDInfluenceConfig:
    """获取快速配置（使用投影和最后层）"""
    return TDInfluenceConfig(
        use_last_layer_only=True,
        use_projection=True,
        proj_dim=64,
        proj_type="achlioptas",
        val_batch_size=256,
        lambda_trace=0.85,
        gamma=0.95,
    )


def get_sensitive_config() -> TDInfluenceConfig:
    """获取敏感配置（更关注梯度对齐）"""
    return TDInfluenceConfig(
        alpha=[0.5, 0.3, 0.2, 1.0],  # 更重视梯度对齐
        lambda_trace=0.8,
        gamma=0.9,
        e_clip=5.0,
    )


def get_robust_config() -> TDInfluenceConfig:
    """获取鲁棒配置（更关注损失变化）"""
    return TDInfluenceConfig(
        alpha=[1.5, 0.8, 0.4, 0.1],  # 更重视损失变化
        lambda_trace=0.95,
        gamma=0.98,
        e_clip=15.0,
    )


def create_config_from_experiment(
    experiment_name: str, base_config: Optional[TDInfluenceConfig] = None
) -> TDInfluenceConfig:
    """
    根据实验名称创建配置

    Args:
        experiment_name: 实验名称
        base_config: 基础配置（可选）

    Returns:
        配置对象
    """
    if base_config is None:
        base_config = get_default_config()

    # 根据实验名称调整配置
    if "high_precision" in experiment_name.lower():
        base_config.use_last_layer_only = False
        base_config.use_projection = False
    elif "fast" in experiment_name.lower():
        base_config.use_last_layer_only = True
        base_config.use_projection = True
        base_config.proj_dim = 64
    elif "sensitive" in experiment_name.lower():
        base_config.alpha = [0.5, 0.3, 0.2, 1.0]
    elif "robust" in experiment_name.lower():
        base_config.alpha = [1.5, 0.8, 0.4, 0.1]

    return base_config


# 配置模板生成器
class ConfigTemplateGenerator:
    """配置模板生成器"""

    @staticmethod
    def generate_grid_search_configs(
        base_config: TDInfluenceConfig, param_grid: Dict[str, List[Any]]
    ) -> List[TDInfluenceConfig]:
        """
        生成网格搜索配置

        Args:
            base_config: 基础配置
            param_grid: 参数网格，格式为 {参数名: [值列表]}

        Returns:
            配置列表
        """
        import itertools

        # 获取所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        configs = []
        for combination in combinations:
            config_dict = base_config.to_dict()
            for name, value in zip(param_names, combination):
                config_dict[name] = value

            config = TDInfluenceConfig.from_dict(config_dict)
            configs.append(config)

        return configs

    @staticmethod
    def generate_ablation_configs(
        base_config: TDInfluenceConfig,
    ) -> Dict[str, TDInfluenceConfig]:
        """
        生成消融实验配置

        Args:
            base_config: 基础配置

        Returns:
            消融配置字典
        """
        configs = {}

        # 基础配置
        configs["baseline"] = base_config

        # 不使用投影
        config_no_proj = TDInfluenceConfig.from_dict(base_config.to_dict())
        config_no_proj.use_projection = False
        configs["no_projection"] = config_no_proj

        # 使用全模型梯度
        config_full_grad = TDInfluenceConfig.from_dict(base_config.to_dict())
        config_full_grad.use_last_layer_only = False
        configs["full_gradients"] = config_full_grad

        # 不同的alpha权重
        for i, component in enumerate(
            ["dloss_pos", "forgetting", "hard_prob", "align_val"]
        ):
            config_alpha = TDInfluenceConfig.from_dict(base_config.to_dict())
            config_alpha.alpha = [0.25, 0.25, 0.25, 0.25]
            config_alpha.alpha[i] = 1.0
            configs[f"alpha_{component}"] = config_alpha

        # 不同的lambda值
        for lambda_val in [0.7, 0.8, 0.9, 0.95]:
            config_lambda = TDInfluenceConfig.from_dict(base_config.to_dict())
            config_lambda.lambda_trace = lambda_val
            configs[f"lambda_{lambda_val}"] = config_lambda

        # 不同的gamma值
        for gamma_val in [0.9, 0.95, 0.97, 0.99]:
            config_gamma = TDInfluenceConfig.from_dict(base_config.to_dict())
            config_gamma.gamma = gamma_val
            configs[f"gamma_{gamma_val}"] = config_gamma

        return configs


# 便捷函数
def setup_td_influence_experiment(
    experiment_name: str,
    config_name: str = "default",
    output_dir: str = "outputs",
    save_config: bool = True,
) -> Tuple[TDInfluenceConfig, str]:
    """
    设置TD-Influence实验

    Args:
        experiment_name: 实验名称
        config_name: 配置名称
        output_dir: 输出目录
        save_config: 是否保存配置

    Returns:
        (配置对象, 输出目录路径)
    """
    # 创建输出目录
    exp_output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_output_dir, exist_ok=True)

    # 获取配置
    config_manager = TDInfluenceConfigManager()

    # 添加预定义配置
    config_manager.add_config("default", get_default_config())
    config_manager.add_config("high_precision", get_high_precision_config())
    config_manager.add_config("fast", get_fast_config())
    config_manager.add_config("sensitive", get_sensitive_config())
    config_manager.add_config("robust", get_robust_config())

    # 获取配置
    if config_name in config_manager.list_configs():
        config = config_manager.get_config(config_name)
    else:
        config = create_config_from_experiment(experiment_name)
        config_manager.add_config(config_name, config)

    # 保存配置
    if save_config:
        config_path = os.path.join(exp_output_dir, f"{config_name}_config.yaml")
        config.to_yaml(config_path)

    return config, exp_output_dir
