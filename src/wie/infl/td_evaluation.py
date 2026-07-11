import os
import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, List, Optional, Union
from sklearn.metrics import roc_auc_score, average_precision_score
import logging


class TDEvaluation:
    """
    TD-Influence方法的评测工具

    实现错标检测等评测任务，包括：
    - AUROC (Area Under ROC Curve)
    - PR-AUC (Precision-Recall AUC)
    - Precision@K
    - 跨种子统计
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def compute_auroc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算AUROC"""
        try:
            return roc_auc_score(y_true, y_score)
        except ValueError as e:
            self.logger.warning(f"AUROC计算失败: {e}")
            return 0.5  # 随机分类器的性能

    def compute_prauc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算PR-AUC"""
        try:
            return average_precision_score(y_true, y_score)
        except ValueError as e:
            self.logger.warning(f"PR-AUC计算失败: {e}")
            return 0.0

    def compute_precision_at_k(
        self, y_true: np.ndarray, y_score: np.ndarray, k: int
    ) -> float:
        """计算Precision@K"""
        if k <= 0 or k > len(y_true):
            return 0.0

        # 按分数降序排序
        sorted_indices = np.argsort(y_score)[::-1]
        top_k_indices = sorted_indices[:k]

        # 计算top-k中的正例比例
        precision = np.sum(y_true[top_k_indices]) / k
        return precision

    def evaluate_noise_detection(
        self,
        train_set: Union[np.ndarray, pd.DataFrame, Dict],
        influence_scores: np.ndarray,
        k_percentage: float = 0.1,
    ) -> Dict[str, float]:
        """
        评测错标检测性能

        Args:
            train_set: 训练集，包含噪声标签信息
            influence_scores: 影响力分数（越大越可疑）
            k_percentage: Precision@K中的K比例

        Returns:
            包含各种评测指标的字典
        """
        # 提取噪声标签
        if isinstance(train_set, dict):
            if "is_noisy" in train_set:
                y_true = train_set["is_noisy"]
            elif "noisy_labels" in train_set:
                y_true = train_set["noisy_labels"]
            else:
                raise ValueError("训练集字典中未找到噪声标签信息")
        elif isinstance(train_set, pd.DataFrame):
            if "is_noisy" in train_set.columns:
                y_true = train_set["is_noisy"].values
            elif "noisy_labels" in train_set.columns:
                y_true = train_set["noisy_labels"].values
            else:
                raise ValueError("训练集DataFrame中未找到噪声标签列")
        elif isinstance(train_set, np.ndarray):
            y_true = train_set
        else:
            raise ValueError("不支持的训练集格式")

        # 确保标签是二进制的
        y_true = np.array(y_true, dtype=int)
        if len(np.unique(y_true)) != 2:
            raise ValueError("噪声标签必须是二进制的（0=干净，1=噪声）")

        # 确保分数和标签长度一致
        if len(influence_scores) != len(y_true):
            raise ValueError(
                f"影响力分数长度({len(influence_scores)})与标签长度({len(y_true)})不匹配"
            )

        # 计算各种指标
        auroc = self.compute_auroc(y_true, influence_scores)
        prauc = self.compute_prauc(y_true, influence_scores)

        # Precision@K
        k = max(1, int(np.ceil(k_percentage * len(y_true))))
        p_at_k = self.compute_precision_at_k(y_true, influence_scores, k)

        results = {
            "auroc": auroc,
            "prauc": prauc,
            "p_at_k": p_at_k,
            "k": k,
            "k_percentage": k_percentage,
            "n_samples": len(y_true),
            "n_noisy": np.sum(y_true),
            "noise_ratio": np.mean(y_true),
        }

        self.logger.info(
            f"错标检测评测结果: AUROC={auroc:.4f}, PR-AUC={prauc:.4f}, P@{k}={p_at_k:.4f}"
        )
        self.logger.info(
            f"样本统计: 总数={len(y_true)}, 噪声={np.sum(y_true)}, 噪声比例={np.mean(y_true):.4f}"
        )

        return results

    def evaluate_multiple_seeds(
        self,
        results_dir: str,
        seeds: List[int],
        train_set: Union[np.ndarray, pd.DataFrame, Dict],
        k_percentage: float = 0.1,
        confidence_level: float = 0.95,
    ) -> Dict:
        """
        跨多个种子评测TD-Influence性能

        Args:
            results_dir: 结果目录
            seeds: 种子列表
            train_set: 训练集
            k_percentage: Precision@K中的K比例
            confidence_level: 置信水平

        Returns:
            包含跨种子统计结果的字典
        """
        all_results = []
        successful_seeds = []

        for seed in seeds:
            try:
                # 加载影响力分数
                influence_file = os.path.join(
                    results_dir, f"infl_td_influence_{seed:03d}.dat"
                )
                if not os.path.exists(influence_file):
                    self.logger.warning(f"种子{seed}的结果文件不存在: {influence_file}")
                    continue

                influence_scores = torch.load(influence_file, map_location="cpu")
                if isinstance(influence_scores, torch.Tensor):
                    influence_scores = influence_scores.numpy()

                # 评测
                results = self.evaluate_noise_detection(
                    train_set, influence_scores, k_percentage
                )
                all_results.append(results)
                successful_seeds.append(seed)

            except Exception as e:
                self.logger.error(f"种子{seed}评测失败: {e}")
                continue

        if not all_results:
            raise ValueError("没有成功的种子结果")

        # 计算统计量
        metrics = ["auroc", "prauc", "p_at_k"]
        summary = {
            "seeds": successful_seeds,
            "n_seeds": len(successful_seeds),
            "confidence_level": confidence_level,
        }

        for metric in metrics:
            values = [r[metric] for r in all_results]
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values,
            }

            # 计算置信区间
            if len(values) > 1:
                # 使用t分布计算置信区间
                from scipy import stats

                alpha = 1 - confidence_level
                t_val = stats.t.ppf(1 - alpha / 2, len(values) - 1)
                margin_error = t_val * np.std(values) / np.sqrt(len(values))
                summary[metric]["ci_lower"] = np.mean(values) - margin_error
                summary[metric]["ci_upper"] = np.mean(values) + margin_error
            else:
                summary[metric]["ci_lower"] = values[0]
                summary[metric]["ci_upper"] = values[0]

        # 添加其他统计信息
        summary["n_samples"] = all_results[0]["n_samples"]
        summary["n_noisy"] = all_results[0]["n_noisy"]
        summary["noise_ratio"] = all_results[0]["noise_ratio"]

        self.logger.info(f"跨种子评测完成: {len(successful_seeds)}个种子")
        for metric in metrics:
            mean_val = summary[metric]["mean"]
            std_val = summary[metric]["std"]
            ci_lower = summary[metric]["ci_lower"]
            ci_upper = summary[metric]["ci_upper"]
            self.logger.info(
                f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            )

        return summary

    def save_evaluation_results(
        self, results: Dict, output_path: str, format: str = "json"
    ) -> None:
        """保存评测结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == "json":
            # 转换numpy数组为列表以便JSON序列化
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            json_results[key][sub_key] = sub_value.tolist()
                        else:
                            json_results[key][sub_key] = sub_value
                elif isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            # 创建汇总表格
            summary_data = []
            for metric in ["auroc", "prauc", "p_at_k"]:
                if metric in results:
                    summary_data.append(
                        {
                            "metric": metric.upper(),
                            "mean": results[metric]["mean"],
                            "std": results[metric]["std"],
                            "ci_lower": results[metric]["ci_lower"],
                            "ci_upper": results[metric]["ci_upper"],
                            "min": results[metric]["min"],
                            "max": results[metric]["max"],
                        }
                    )

            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"不支持的格式: {format}")

        self.logger.info(f"评测结果保存到: {output_path}")


def create_synthetic_noisy_dataset(
    n_samples: int, noise_ratio: float = 0.1, seed: int = 42
) -> Dict:
    """
    创建合成噪声数据集用于测试

    Args:
        n_samples: 样本数量
        noise_ratio: 噪声比例
        seed: 随机种子

    Returns:
        包含噪声标签信息的字典
    """
    np.random.seed(seed)

    # 随机生成噪声标签
    n_noisy = int(n_samples * noise_ratio)
    is_noisy = np.zeros(n_samples, dtype=int)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    is_noisy[noisy_indices] = 1

    return {
        "is_noisy": is_noisy,
        "noisy_indices": noisy_indices,
        "n_samples": n_samples,
        "n_noisy": n_noisy,
        "noise_ratio": noise_ratio,
    }


def load_noise_labels_from_csv(
    csv_path: str, label_column: str = "is_noisy"
) -> np.ndarray:
    """
    从CSV文件加载噪声标签

    Args:
        csv_path: CSV文件路径
        label_column: 标签列名

    Returns:
        噪声标签数组
    """
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"CSV文件中未找到列: {label_column}")

    return df[label_column].values


# 示例使用函数
def run_td_influence_evaluation(
    results_dir: str,
    seeds: List[int] = [0, 1, 2, 3, 4],
    noise_labels_path: Optional[str] = None,
    noise_ratio: float = 0.1,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    运行TD-Influence评测的便捷函数

    Args:
        results_dir: TD-Influence结果目录
        seeds: 种子列表
        noise_labels_path: 噪声标签CSV文件路径（可选）
        noise_ratio: 如果使用合成数据，噪声比例
        output_dir: 输出目录（默认使用results_dir）

    Returns:
        评测结果字典
    """
    logger = logging.getLogger(__name__)
    evaluator = TDEvaluation(logger)

    # 准备训练集
    if noise_labels_path and os.path.exists(noise_labels_path):
        logger.info(f"从文件加载噪声标签: {noise_labels_path}")
        is_noisy = load_noise_labels_from_csv(noise_labels_path)
        train_set = {"is_noisy": is_noisy}
    else:
        logger.info(f"创建合成噪声数据集，噪声比例: {noise_ratio}")
        # 假设使用第一个种子的结果来确定样本数量
        first_seed_file = os.path.join(
            results_dir, f"infl_td_influence_{seeds[0]:03d}.dat"
        )
        if os.path.exists(first_seed_file):
            influence_scores = torch.load(first_seed_file, map_location="cpu")
            n_samples = len(influence_scores)
        else:
            n_samples = 1000  # 默认值
            logger.warning(f"无法确定样本数量，使用默认值: {n_samples}")

        train_set = create_synthetic_noisy_dataset(n_samples, noise_ratio)

    # 运行评测
    results = evaluator.evaluate_multiple_seeds(results_dir, seeds, train_set)

    # 保存结果
    if output_dir is None:
        output_dir = results_dir

    # 保存JSON格式
    json_path = os.path.join(output_dir, "td_influence_metrics.json")
    evaluator.save_evaluation_results(results, json_path, "json")

    # 保存CSV格式
    csv_path = os.path.join(output_dir, "td_influence_metrics.csv")
    evaluator.save_evaluation_results(results, csv_path, "csv")

    return results
