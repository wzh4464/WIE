import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import pandas as pd

from ..models.networks import get_network
from ..data.modules import fetch_data_module
from ..utils.paths import resolve_output_dir
from .td_config import TDInfluenceConfig


class TDTrainingLoop:
    """
    TD-Influence训练循环

    实现端到端的训练过程，包括：
    - 模型训练
    - 状态记录
    - 影响力计算
    - 结果保存
    """

    def __init__(
        self,
        config: TDInfluenceConfig,
        data_key: str,
        model_type: str,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.data_key = data_key
        self.model_type = model_type
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # 训练状态
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

        # 数据
        self.x_tr = None
        self.y_tr = None
        self.x_val = None
        self.y_val = None
        self.n_tr = None
        self.n_val = None

        # TD-Influence状态
        self.td_states = None
        self.epoch_logs = None

        # 输出目录
        self.output_dir = None
        # 投影矩阵（直积法可选降维）
        self.proj_R_a: Optional[torch.Tensor] = None  # 输入激活投影矩阵
        self.proj_R_d: Optional[torch.Tensor] = None  # 预激活梯度投影矩阵

    def setup_data(self, n_tr: int, n_val: int, n_test: int, seed: int) -> None:
        """设置数据"""
        self.logger.info(f"设置数据: 训练={n_tr}, 验证={n_val}, 测试={n_test}")

        # 获取数据模块
        module = fetch_data_module(
            self.data_key,
            data_dir=os.path.join(os.path.dirname(__file__), "..", "data"),
            logger=self.logger,
            seed=seed,
        )
        module.append_one = False

        # 获取数据
        z_tr, z_val, z_test = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr_np, y_tr_np), (x_val_np, y_val_np) = z_tr, z_val

        # 转换为tensor
        self.x_tr = torch.from_numpy(x_tr_np).to(torch.float32).to(self.device)
        self.y_tr = (
            torch.from_numpy(y_tr_np).to(torch.float32).unsqueeze(1).to(self.device)
        )
        self.x_val = torch.from_numpy(x_val_np).to(torch.float32).to(self.device)
        self.y_val = (
            torch.from_numpy(y_val_np).to(torch.float32).unsqueeze(1).to(self.device)
        )

        self.n_tr = len(self.x_tr)
        self.n_val = len(self.x_val)

        # 创建数据加载器
        train_dataset = TensorDataset(self.x_tr, self.y_tr)
        val_dataset = TensorDataset(self.x_val, self.y_val)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.logger.info(
            f"数据加载完成: 训练批次={len(self.train_loader)}, 验证批次={len(self.val_loader)}"
        )

    def setup_model(self, input_dim: Union[int, Tuple[int, ...]]) -> None:
        """设置模型"""
        self.logger.info(f"设置模型: {self.model_type}, 输入维度={input_dim}")

        self.model = get_network(self.model_type, input_dim, logger=self.logger).to(
            self.device
        )
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=0.0001,
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

        self.logger.info(
            f"模型设置完成: 参数数量={sum(p.numel() for p in self.model.parameters())}"
        )

    def setup_td_influence(self) -> None:
        """设置TD-Influence状态"""
        self.logger.info("设置TD-Influence状态")

        self.td_states = {
            "e": np.zeros(self.n_tr, dtype=np.float32),  # 资格迹
            "s": np.zeros(self.n_tr, dtype=np.float32),  # 累积影响力分数
            "last_correct": np.zeros(self.n_tr, dtype=np.int32),  # 上一轮正确性
            "loss_last": np.zeros(self.n_tr, dtype=np.float32),  # 上一轮损失
            "w": 0.0,  # 折扣权重
        }

        self.epoch_logs = {
            "mean_train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
            "gamma": self.config.gamma,
            "lambda": self.config.lambda_trace,
            "alpha": self.config.alpha,
            "norm": self.config.norm_type,
            "proj_dim": self.config.proj_dim if self.config.use_projection else None,
            "proj_type": self.config.proj_type if self.config.use_projection else None,
            "seed": None,  # 将在训练时设置
        }

    def setup_output_dir(self, output_dir: str, seed: int) -> None:
        """设置输出目录"""
        base_dir = str(resolve_output_dir(output_dir))
        self.output_dir = os.path.join(base_dir, f"seed_{seed:03d}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存配置
        config_path = os.path.join(self.output_dir, "config.json")
        self.config.to_json(config_path)

        self.logger.info(f"输出目录设置完成: {self.output_dir}")

    def compute_sample_metrics(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[float, float, bool]:
        """计算样本的损失、margin和正确性"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, y).item()

            # 计算margin和正确性
            if logits.dim() > 1 and logits.shape[1] > 1:
                # 多分类情况
                pred_class = torch.argmax(logits, dim=1)
                correct = (pred_class == y.long().flatten()).item()

                # 计算margin
                y_true = y.long().flatten()
                correct_logit = logits[0, y_true]
                other_logits = torch.cat([logits[0, :y_true], logits[0, y_true + 1 :]])
                max_other_logit = other_logits.max()
                margin = (correct_logit - max_other_logit).item()
            else:
                # 二分类情况
                pred = (logits > 0).long()
                correct = (pred == y.long()).item()
                margin = abs(logits.item())

        return loss, margin, correct

    def extract_gradients(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """提取梯度"""
        self.model.eval()
        self.model.zero_grad()

        output = self.model(x)
        loss = nn.BCEWithLogitsLoss(reduction="mean")(output, y)
        loss.backward()

        if self.config.use_last_layer_only:
            # 提取最后线性层的梯度
            last_layer_params = []
            for name, param in self.model.named_parameters():
                if ("weight" in name or "bias" in name) and param.grad is not None:
                    last_layer_params.append(param.grad.flatten())

            if not last_layer_params:
                # 如果没有找到，使用所有参数
                all_grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        all_grads.append(param.grad.flatten())
                return (
                    torch.cat(all_grads)
                    if all_grads
                    else torch.tensor([], device=self.device)
                )

            return torch.cat(last_layer_params)
        else:
            # 使用所有参数梯度
            all_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.flatten())
            return (
                torch.cat(all_grads)
                if all_grads
                else torch.tensor([], device=self.device)
            )

    def compute_validation_gradient(self, val_indices: List[int]) -> torch.Tensor:
        """计算验证集梯度"""
        self.model.eval()
        self.model.zero_grad()

        x_val_batch = self.x_val[val_indices]
        y_val_batch = self.y_val[val_indices]

        output = self.model(x_val_batch)
        loss = nn.BCEWithLogitsLoss(reduction="mean")(output, y_val_batch)
        loss.backward()

        grad_val = self.extract_gradients(x_val_batch, y_val_batch)
        self.model.zero_grad()
        return grad_val

    # ------- 直积法：最后线性层因子与对齐（内存节省） -------
    def _find_last_linear(self) -> nn.Linear:
        """定位模型中的最后一个线性层，用于直积法因子提取。"""
        last_linear = None
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise RuntimeError("未找到线性层用于直积法")
        return last_linear

    @torch.no_grad()
    def _maybe_setup_projection(self, d_in: int, d_out: int) -> None:
        """懒加载初始化因子空间的投影矩阵（JL）。"""
        if not self.config.use_projection:
            return
        if self.proj_R_a is not None and self.proj_R_d is not None:
            return
        torch.manual_seed(42)
        k_a = int(min(self.config.proj_dim, d_in))
        k_d = int(min(self.config.proj_dim, max(1, d_out)))

        def gaussian(k: int, d: int) -> torch.Tensor:
            return torch.randn(k, d, device=self.device, dtype=torch.float32) / (k**0.5)

        def achlioptas(k: int, d: int) -> torch.Tensor:
            s = 3
            probs = torch.rand(k, d, device=self.device)
            R = torch.zeros(k, d, device=self.device)
            R[probs < (1.0 / (2 * s))] = 1.0
            R[(probs >= (1.0 / (2 * s))) & (probs < (1.0 / s))] = -1.0
            R = R * ((s / k) ** 0.5)
            return R

        maker = gaussian if self.config.proj_type == "gaussian" else achlioptas
        self.proj_R_a = maker(k_a, d_in)
        self.proj_R_d = maker(k_d, max(1, d_out))

    def _extract_last_layer_factors(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取最后线性层因子 (a, δ)。返回形状分别为 (d_in,), (d_out,)。
        注意：该操作会执行一次前向与一次反向，但不保留参数梯度。
        """
        last_fc = self._find_last_linear()
        saved_a: Optional[torch.Tensor] = None
        saved_d: Optional[torch.Tensor] = None

        def fwd_hook(module, inp, out):
            nonlocal saved_a
            saved_a = inp[0].detach()  # (B, d_in)

        def bwd_hook(module, grad_input, grad_output):
            nonlocal saved_d
            saved_d = grad_output[0].detach()  # (B, d_out)

        h1 = last_fc.register_forward_hook(fwd_hook)
        h2 = last_fc.register_full_backward_hook(bwd_hook)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, y)
        loss.backward()

        # 清理
        h1.remove()
        h2.remove()
        self.model.zero_grad(set_to_none=True)

        assert saved_a is not None and saved_d is not None, "未捕获到最后层因子"
        a = saved_a.reshape(saved_a.size(0), -1)[0]  # (d_in,)
        d = saved_d.reshape(saved_d.size(0), -1)[0]  # (d_out,)
        return a, d

    def _compute_validation_factors(
        self, val_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取验证批次的最后层因子矩阵 A_val, D_val，形状 (B, d_in)、(B, d_out)。"""
        last_fc = self._find_last_linear()
        saved_a: Optional[torch.Tensor] = None
        saved_d: Optional[torch.Tensor] = None

        def fwd_hook(module, inp, out):
            nonlocal saved_a
            saved_a = inp[0].detach()  # (B, d_in)

        def bwd_hook(module, grad_input, grad_output):
            nonlocal saved_d
            saved_d = grad_output[0].detach()  # (B, d_out)

        h1 = last_fc.register_forward_hook(fwd_hook)
        h2 = last_fc.register_full_backward_hook(bwd_hook)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x_val_batch = self.x_val[val_indices]
        y_val_batch = self.y_val[val_indices]
        logits = self.model(x_val_batch)
        loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, y_val_batch)
        loss.backward()

        h1.remove()
        h2.remove()
        self.model.zero_grad(set_to_none=True)

        assert saved_a is not None and saved_d is not None, "未捕获到验证因子"
        A_val = saved_a  # (B, d_in)
        D_val = saved_d  # (B, d_out)

        # 懒加载投影矩阵
        self._maybe_setup_projection(d_in=A_val.size(1), d_out=D_val.size(1))
        return A_val, D_val

    @torch.no_grad()
    def _compute_factorized_cosine(
        self,
        a_i: torch.Tensor,
        d_i: torch.Tensor,
        A_val: torch.Tensor,
        D_val: torch.Tensor,
    ) -> float:
        """在直积表示下与验证梯度的余弦相似度（可选因子空间投影）。"""
        eps = 1e-12
        # 可选投影（在因子空间而非展开向量）
        if (
            self.config.use_projection
            and self.proj_R_a is not None
            and self.proj_R_d is not None
        ):
            a_i_eff = self.proj_R_a @ a_i  # (k_a,)
            d_i_eff = self.proj_R_d @ d_i  # (k_d,)
            A_eff = A_val @ self.proj_R_a.t()  # (B, k_a)
            D_eff = D_val @ self.proj_R_d.t()  # (B, k_d)
        else:
            a_i_eff, d_i_eff, A_eff, D_eff = a_i, d_i, A_val, D_val

        num_vec1 = A_eff @ a_i_eff  # (B,)
        num_vec2 = D_eff @ d_i_eff  # (B,)
        numerator = torch.sum(num_vec1.view(-1) * num_vec2.view(-1))

        norm_i = torch.norm(a_i_eff) * torch.norm(d_i_eff) + eps
        G_a = A_eff @ A_eff.t()  # (B,B)
        G_d = D_eff @ D_eff.t()  # (B,B)
        val_sq = torch.sum(G_a * G_d)
        norm_val = torch.sqrt(torch.clamp(val_sq, min=eps))

        cos = (numerator / (norm_i * norm_val + eps)).item()
        return float(cos)

    def compute_cosine_similarity(self, g1: torch.Tensor, g2: torch.Tensor) -> float:
        """计算两个梯度的余弦相似度"""
        eps = 1e-12
        num = (g1 * g2).sum()
        den = g1.norm() * g2.norm() + eps
        cos = (num / den).item()
        return cos

    def td_influence_update(
        self,
        sample_idx: int,
        epoch: int,
        loss_t: float,
        margin_t: float,
        correct_t: bool,
        grad_i_t: torch.Tensor,
        grad_val_t: torch.Tensor,
        align_val_override: Optional[float] = None,
    ) -> None:
        """TD-Influence核心更新算法"""

        # 1) 计算时间步信号 r_i^t
        # 忘记事件
        forgetting = (
            1
            if (self.td_states["last_correct"][sample_idx] == 1 and correct_t == 0)
            else 0
        )

        # 损失上升
        dloss_pos = max(0, loss_t - self.td_states["loss_last"][sample_idx])

        # 困难度（负margin）
        hard_prob = max(0, -margin_t)

        # 梯度对齐度
        if align_val_override is None:
            align_val = -self.compute_cosine_similarity(grad_i_t, grad_val_t)
        else:
            align_val = align_val_override
        align_val = max(0, align_val)  # 只考虑负对齐

        # 组合信号
        r = (
            self.config.alpha[0] * dloss_pos
            + self.config.alpha[1] * forgetting
            + self.config.alpha[2] * hard_prob
            + self.config.alpha[3] * align_val
        )

        # 2) 资格迹更新
        self.td_states["e"][sample_idx] = (
            self.config.lambda_trace * self.td_states["e"][sample_idx] + 1.0
        )
        # 可选裁剪
        if self.config.e_clip > 0:
            self.td_states["e"][sample_idx] = min(
                self.td_states["e"][sample_idx], self.config.e_clip
            )

        # 3) 折扣权重（在线实现）
        self.td_states["w"] = self.config.gamma * self.td_states["w"] + (
            1 - self.config.gamma
        )
        w_t = self.td_states["w"]

        # 4) 累积影响力分数
        self.td_states["s"][sample_idx] += w_t * self.td_states["e"][sample_idx] * r

        # 5) 更新状态
        self.td_states["last_correct"][sample_idx] = correct_t
        self.td_states["loss_last"][sample_idx] = loss_t

    def train_epoch(self, epoch: int, val_indices: List[int]) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        # 计算验证集梯度/因子
        use_factorized = self.config.use_last_layer_only
        if use_factorized:
            A_val, D_val = self._compute_validation_factors(val_indices)
        else:
            grad_val_t = self.compute_validation_gradient(val_indices)

        # 训练统计
        total_loss = 0.0
        total_correct = 0
        num_samples = 0

        # 训练循环
        for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
            batch_size = x_batch.size(0)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(x_batch)
            loss = nn.BCEWithLogitsLoss(reduction="mean")(output, y_batch)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * batch_size
            pred = (output > 0).long()
            total_correct += (pred == y_batch.long()).sum().item()
            num_samples += batch_size

            # TD-Influence更新（对每个样本）
            for i in range(batch_size):
                sample_idx = batch_idx * self.config.batch_size + i
                if sample_idx >= self.n_tr:
                    break

                x_i = x_batch[i : i + 1]
                y_i = y_batch[i : i + 1]

                # 计算样本指标
                loss_t, margin_t, correct_t = self.compute_sample_metrics(x_i, y_i)

                if use_factorized:
                    # 直积法：提取 (a_i, δ_i) 并在因子空间计算对齐
                    a_i, d_i = self._extract_last_layer_factors(x_i, y_i)
                    cos_val = self._compute_factorized_cosine(a_i, d_i, A_val, D_val)
                    align_val = max(0.0, -float(cos_val))
                    # 传递覆盖的对齐值（梯度占位）
                    self.td_influence_update(
                        sample_idx,
                        epoch,
                        loss_t,
                        margin_t,
                        correct_t,
                        grad_i_t=torch.tensor(0.0, device=self.device),
                        grad_val_t=torch.tensor(0.0, device=self.device),
                        align_val_override=align_val,
                    )
                else:
                    # 常规扁平梯度
                    grad_i_t = self.extract_gradients(x_i, y_i)
                    self.td_influence_update(
                        sample_idx,
                        epoch,
                        loss_t,
                        margin_t,
                        correct_t,
                        grad_i_t,
                        grad_val_t,
                    )

        # 计算验证集指标
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                output = self.model(x_batch)
                loss = nn.BCEWithLogitsLoss(reduction="mean")(output, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                pred = (output > 0).long()
                val_correct += (pred == y_batch.long()).sum().item()
                val_samples += x_batch.size(0)

        # 返回统计信息
        return {
            "train_loss": total_loss / num_samples,
            "train_acc": total_correct / num_samples,
            "val_loss": val_loss / val_samples,
            "val_acc": val_correct / val_samples,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def train(self, seed: int) -> Dict[str, Any]:
        """完整训练过程"""
        self.logger.info(f"开始训练，种子: {seed}")

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 设置输出目录
        self.setup_output_dir("outputs", seed)

        # 设置数据
        self.setup_data(
            n_tr=1000,  # 可以根据需要调整
            n_val=200,
            n_test=200,
            seed=seed,
        )

        # 设置模型
        input_dim = self.x_tr.shape[1] if self.x_tr.dim() == 2 else self.x_tr.shape[1:]
        self.setup_model(input_dim)

        # 设置TD-Influence
        self.setup_td_influence()
        self.epoch_logs["seed"] = seed

        # 设置验证集索引（固定子集）
        torch.manual_seed(42)  # 固定随机种子
        val_indices = torch.randperm(len(self.x_val))[
            : self.config.val_batch_size
        ].tolist()

        # 训练循环
        best_val_acc = 0.0
        for epoch in range(self.config.num_epochs):
            # 训练一个epoch
            epoch_stats = self.train_epoch(epoch, val_indices)

            # 更新学习率
            self.scheduler.step()

            # 记录日志
            self.epoch_logs["mean_train_loss"].append(epoch_stats["train_loss"])
            self.epoch_logs["train_acc"].append(epoch_stats["train_acc"])
            self.epoch_logs["val_loss"].append(epoch_stats["val_loss"])
            self.epoch_logs["val_acc"].append(epoch_stats["val_acc"])
            self.epoch_logs["learning_rate"].append(epoch_stats["learning_rate"])

            # 保存最佳模型
            if epoch_stats["val_acc"] > best_val_acc:
                best_val_acc = epoch_stats["val_acc"]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "best_model.pt"),
                )

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "td_states": self.td_states,
                    "epoch_logs": self.epoch_logs,
                    "config": self.config.to_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(self.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
                )

            # 日志输出
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                    f"训练损失={epoch_stats['train_loss']:.4f}, "
                    f"训练准确率={epoch_stats['train_acc']:.4f}, "
                    f"验证损失={epoch_stats['val_loss']:.4f}, "
                    f"验证准确率={epoch_stats['val_acc']:.4f}, "
                    f"学习率={epoch_stats['learning_rate']:.6f}"
                )

        # 保存最终结果
        self.save_results()

        self.logger.info(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")

        return {
            "best_val_acc": best_val_acc,
            "final_td_states": self.td_states,
            "epoch_logs": self.epoch_logs,
            "output_dir": self.output_dir,
        }

    def save_results(self) -> None:
        """保存结果"""
        # 保存影响力分数
        influence_scores = self.td_states["s"]
        np.save(os.path.join(self.output_dir, "influence_scores.npy"), influence_scores)

        # 保存epoch日志
        epoch_logs_path = os.path.join(self.output_dir, "epoch_logs.npz")
        np.savez(epoch_logs_path, **self.epoch_logs)

        # 保存TD状态
        td_states_path = os.path.join(self.output_dir, "td_states.npz")
        np.savez(td_states_path, **self.td_states)

        # 保存CSV结果
        csv_path = os.path.join(self.output_dir, "results.csv")
        df_data = {
            "sample_idx": np.arange(self.n_tr),
            "influence": influence_scores,
            "eligibility_trace": self.td_states["e"],
            "last_correct": self.td_states["last_correct"],
            "loss_last": self.td_states["loss_last"],
        }

        # 添加真实标签（如果有的话）
        if self.y_tr is not None:
            df_data["true_label"] = self.y_tr.cpu().numpy().flatten()

        df = pd.DataFrame(df_data)
        df["influence_rank"] = (
            df["influence"].rank(ascending=False, method="first").astype(int)
        )
        df["influence_percentile"] = df["influence"].rank(pct=True)
        df.to_csv(csv_path, index=False)

        self.logger.info(f"结果保存到: {self.output_dir}")


def run_td_influence_training(
    config: TDInfluenceConfig,
    data_key: str,
    model_type: str,
    seeds: List[int],
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    运行TD-Influence训练的便捷函数

    Args:
        config: TD-Influence配置
        data_key: 数据键
        model_type: 模型类型
        seeds: 种子列表
        device: 设备
        logger: 日志器

    Returns:
        训练结果字典
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    results = {}

    for seed in seeds:
        logger.info(f"开始训练种子 {seed}")

        try:
            # 创建训练循环
            trainer = TDTrainingLoop(
                config=config,
                data_key=data_key,
                model_type=model_type,
                device=device,
                logger=logger,
            )

            # 训练
            result = trainer.train(seed)
            results[seed] = result

            logger.info(f"种子 {seed} 训练完成")

        except Exception as e:
            logger.error(f"种子 {seed} 训练失败: {e}")
            results[seed] = {"error": str(e)}

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
