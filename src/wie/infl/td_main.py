#!/usr/bin/env python3
"""
TD-Influence主入口文件

提供完整的TD-Influence方法实现，包括：
- 训练循环
- 影响力计算
- 评测功能
- 配置管理

使用方法:
    python -m wie.infl.td_main --help
"""

import argparse
import json
import logging
import os
import sys

# 本模块应作为包模块运行：python -m wie.infl.td_main
from wie.infl.td_config import (
    TDInfluenceConfig,
    get_default_config,
    get_high_precision_config,
    get_fast_config,
    get_sensitive_config,
    get_robust_config,
    setup_td_influence_experiment,
)
from wie.infl.td_training import run_td_influence_training
from wie.infl.td_evaluation import run_td_influence_evaluation
from wie.infl.td_influence import TDInfluenceCalculator


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("td_influence.log")],
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TD-Influence: 时间依赖影响力计算方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置训练
  python -m wie.infl.td_main --mode train --data adult --model logreg --seeds 0 1 2

  # 使用高精度配置
  python -m wie.infl.td_main --mode train --config high_precision --data adult --model dnn

  # 评测结果
  python -m wie.infl.td_main --mode eval --results_dir outputs/experiment_1

  # 计算影响力（使用预训练模型）
  python -m wie.infl.td_main --mode influence --data adult --model logreg --seed 0
        """,
    )

    # 基本参数
    parser.add_argument(
        "--mode",
        choices=["train", "influence", "eval", "all"],
        required=True,
        help="运行模式",
    )
    parser.add_argument("--data", default="adult", help="数据集键")
    parser.add_argument("--model", default="logreg", help="模型类型")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="随机种子列表"
    )
    parser.add_argument("--device", default="cuda", help="计算设备")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    # 配置参数
    parser.add_argument(
        "--config",
        default="default",
        choices=["default", "high_precision", "fast", "sensitive", "robust"],
        help="预定义配置",
    )
    parser.add_argument("--config_file", help="自定义配置文件路径")
    parser.add_argument("--lambda_trace", type=float, help="资格迹衰减因子")
    parser.add_argument("--gamma", type=float, help="远期折扣因子")
    parser.add_argument(
        "--alpha",
        nargs=4,
        type=float,
        help="信号权重 [dloss_pos, forgetting, hard_prob, align_val]",
    )
    parser.add_argument("--use_projection", action="store_true", help="使用随机投影")
    parser.add_argument("--proj_dim", type=int, help="投影维度")
    parser.add_argument(
        "--use_last_layer_only", action="store_true", help="仅使用最后层梯度"
    )

    # 训练参数
    parser.add_argument("--num_epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--val_batch_size", type=int, help="验证批次大小")

    # 评测参数
    parser.add_argument("--results_dir", help="结果目录")
    parser.add_argument("--noise_labels_path", help="噪声标签文件路径")
    parser.add_argument(
        "--noise_ratio", type=float, default=0.1, help="噪声比例（用于合成数据）"
    )
    parser.add_argument(
        "--k_percentage", type=float, default=0.1, help="Precision@K中的K比例"
    )

    # 输出参数
    parser.add_argument("--output_dir", default="outputs", help="输出目录")
    parser.add_argument("--experiment_name", help="实验名称")
    parser.add_argument("--save_config", action="store_true", help="保存配置")

    return parser.parse_args()


def get_config_from_args(args: argparse.Namespace) -> TDInfluenceConfig:
    """从命令行参数获取配置"""
    # 获取基础配置
    config_map = {
        "default": get_default_config(),
        "high_precision": get_high_precision_config(),
        "fast": get_fast_config(),
        "sensitive": get_sensitive_config(),
        "robust": get_robust_config(),
    }

    if args.config_file:
        # 从文件加载配置
        if args.config_file.endswith(".yaml") or args.config_file.endswith(".yml"):
            config = TDInfluenceConfig.from_yaml(args.config_file)
        elif args.config_file.endswith(".json"):
            config = TDInfluenceConfig.from_json(args.config_file)
        else:
            raise ValueError(f"不支持的配置文件格式: {args.config_file}")
    else:
        # 使用预定义配置
        config = config_map[args.config]

    # 从命令行参数覆盖配置
    if args.lambda_trace is not None:
        config.lambda_trace = args.lambda_trace
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.use_projection:
        config.use_projection = True
    if args.proj_dim is not None:
        config.proj_dim = args.proj_dim
    if args.use_last_layer_only:
        config.use_last_layer_only = True
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.val_batch_size is not None:
        config.val_batch_size = args.val_batch_size
    if args.k_percentage is not None:
        config.k_percentage = args.k_percentage

    # 设置种子
    config.seeds = args.seeds

    return config


def run_training_mode(args: argparse.Namespace, logger: logging.Logger) -> None:
    """运行训练模式"""
    logger.info("开始训练模式")

    # 获取配置
    config = get_config_from_args(args)

    # 设置实验
    experiment_name = (
        args.experiment_name or f"td_influence_{args.data}_{args.model}_{args.config}"
    )
    config, output_dir = setup_td_influence_experiment(
        experiment_name=experiment_name,
        config_name=args.config,
        output_dir=args.output_dir,
        save_config=args.save_config,
    )

    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"配置: {config.to_dict()}")

    # 运行训练
    results = run_td_influence_training(
        config=config,
        data_key=args.data,
        model_type=args.model,
        seeds=args.seeds,
        device=args.device,
        logger=logger,
    )

    # 保存训练结果摘要
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"训练完成，结果保存到: {output_dir}")


def run_influence_mode(args: argparse.Namespace, logger: logging.Logger) -> None:
    """运行影响力计算模式"""
    logger.info("开始影响力计算模式")

    # 获取配置
    config = get_config_from_args(args)

    # 创建影响力计算器
    calculator = TDInfluenceCalculator(
        infl_type="td_influence",
        key=args.data,
        model_type=args.model,
        seed=args.seeds[0],  # 使用第一个种子
        gpu=0 if args.device == "cuda" else -1,
        save_dir=args.output_dir,
        **config.to_dict(),
    )

    # 运行计算
    calculator.run()

    logger.info("影响力计算完成")


def run_evaluation_mode(args: argparse.Namespace, logger: logging.Logger) -> None:
    """运行评测模式"""
    logger.info("开始评测模式")

    if not args.results_dir:
        raise ValueError("评测模式需要指定 --results_dir")

    # 运行评测
    run_td_influence_evaluation(
        results_dir=args.results_dir,
        seeds=args.seeds,
        noise_labels_path=args.noise_labels_path,
        noise_ratio=args.noise_ratio,
        output_dir=args.output_dir,
    )

    logger.info("评测完成")


def run_all_mode(args: argparse.Namespace, logger: logging.Logger) -> None:
    """运行完整模式（训练+影响力计算+评测）"""
    logger.info("开始完整模式")

    # 1. 训练
    logger.info("=== 步骤1: 训练 ===")
    run_training_mode(args, logger)

    # 2. 影响力计算
    logger.info("=== 步骤2: 影响力计算 ===")
    run_influence_mode(args, logger)

    # 3. 评测
    logger.info("=== 步骤3: 评测 ===")
    args.results_dir = args.output_dir  # 使用训练输出目录作为结果目录
    run_evaluation_mode(args, logger)

    logger.info("完整模式执行完成")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 设置日志
    logger = setup_logging(args.log_level)

    logger.info(f"TD-Influence启动，模式: {args.mode}")
    logger.info(f"参数: {vars(args)}")

    try:
        # 根据模式运行
        if args.mode == "train":
            run_training_mode(args, logger)
        elif args.mode == "influence":
            run_influence_mode(args, logger)
        elif args.mode == "eval":
            run_evaluation_mode(args, logger)
        elif args.mode == "all":
            run_all_mode(args, logger)
        else:
            raise ValueError(f"未知模式: {args.mode}")

        logger.info("TD-Influence执行成功")

    except Exception as e:
        logger.error(f"TD-Influence执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
