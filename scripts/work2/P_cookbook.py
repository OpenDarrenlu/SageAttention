"""
P（注意力 softmax 概率，取值 [0,1)）的 cookbook 标量量化工具。

设计要点（与 turboquant-pytorch/lloyd_max.py 同源思路：Lloyd / 1D k-means 最优质心）：
- P 在 0 附近概率质量大、在 1 附近更“重要”（对 PV 输出的贡献大），因此质心拟合使用**加权** MSE：
  默认权重 w ∝ p^importance_exp，使码本在较大 P 处分得更细。
- 码本从样本用加权 k-means（Lloyd）迭代得到；可从多个 .pt 汇总样本再 fit，得到更贴近真实分布的 cookbook。

主要入口（尽量少）：
- PCookbook：单层码本（质心 + 量化/反量化 + PV）
- PCookbookManager：按 layer_idx 管理多层；支持离线 fit、在线 observe、保存/加载
- attention_forward_p_quant_only：仅对 P 做 cookbook 量化的 attention，便于端到端评估

典型流程：
- 离线：多个 .pt → fit_p_cookbook_from_pt_files（可选先 evaluate_p_distribution_stability）
- 在线整模型：seed_placeholder_cookbooks(range(n_layers)) → 各层 forward 内 observe(layer_idx, P) →
  推理结束 finalize_all → save；或中途已 load 旧码本则可直接量化前向

自检：python P_cookbook.py --demo（随机 P/QKV，跑通全部主要路径）
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 内部：加权 1D Lloyd（与 lloyd_max 中“分区质心 = 条件期望”一致，这里是离散样本上的经验版）
# ---------------------------------------------------------------------------


def _weighted_lloyd_1d(
    samples: torch.Tensor,
    weights: torch.Tensor,
    n_levels: int,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    在 [0,1) 上的样本拟合 n_levels 个质心（升序），最小化 sum_i w_i (x_i - c_{k(i)})^2。
    samples, weights: 同形状，一维展开后使用。
    """
    x = samples.detach().float().reshape(-1)
    w = weights.detach().float().reshape(-1)
    if x.numel() == 0:
        raise ValueError("empty samples")
    eps = 1e-12
    w = w.clamp_min(eps)
    lo, hi = x.min().item(), x.max().item()
    if hi <= lo:
        return torch.linspace(lo, hi, n_levels, device=x.device, dtype=x.dtype)

    centroids = torch.linspace(lo, hi, n_levels, device=x.device, dtype=x.dtype)
    for _ in range(max_iter):
        dist = (x.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assign = dist.argmin(dim=1)
        new_centroids = centroids.clone()
        for j in range(n_levels):
            mask = assign == j
            if not mask.any():
                continue
            ww = w[mask]
            xx = x[mask]
            new_centroids[j] = (ww * xx).sum() / ww.sum()
        new_centroids, _ = torch.sort(new_centroids)
        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if shift < tol:
            break
    return centroids


def _p_importance_weights(p: torch.Tensor, importance_exp: float) -> torch.Tensor:
    """默认：w ∝ p^exp，强调较大 P；p=0 处权重为 0，仍参与分区但不拉质心。"""
    p = p.clamp(min=0.0, max=1.0 - 1e-7)
    return p.pow(importance_exp)


# ---------------------------------------------------------------------------
# 数据：从 .pt 取 P 或从 QKV 算 P
# ---------------------------------------------------------------------------


def load_p_tensor_from_pt(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    从 .pt 加载 P。支持：
    - 直接是 Tensor（视为 P）
    - dict 含 'P' / 'attn_probs'
    - dict 含 'query','key','value' 或 'q','k','v'：用 softmax(QK^T/sqrt(d)) 算 P（最后一维 softmax）
    """
    obj: Any = torch.load(path, map_location=device)
    if torch.is_tensor(obj):
        return obj.float()
    if not isinstance(obj, dict):
        raise TypeError(f"unsupported .pt content: {type(obj)}")

    if "P" in obj:
        return obj["P"].float()
    if "attn_probs" in obj:
        return obj["attn_probs"].float()

    q = obj.get("query", obj.get("q"))
    k = obj.get("key", obj.get("k"))
    if q is None or k is None:
        raise KeyError("dict .pt 需要 'P' / 'attn_probs' 或 ('query','key') / ('q','k')")

    if q.dim() < 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
    d = q.size(-1)
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(d))
    return F.softmax(s, dim=-1).float()


# ---------------------------------------------------------------------------
# 稳定性：跨多个 .pt / 多次 observe 的 P 分布是否足够一致
# ---------------------------------------------------------------------------


@dataclass
class StabilityReport:
    stable: bool
    cv_mean_p: float
    cv_std_p: float
    per_source_mean: List[float]
    message: str


def evaluate_p_distribution_stability(
    p_tensors: List[torch.Tensor],
    mean_cv_threshold: float = 0.35,
    std_cv_threshold: float = 0.50,
) -> StabilityReport:
    """
    用多份 P 的逐张量 mean/std 的变异系数判断分布是否“稳定”。
    若不稳定，使用固定 cookbook 风险较大，应报警。
    """
    means = [t.float().mean().item() for t in p_tensors]
    stds = [t.float().std().item() for t in p_tensors]
    m_bar = sum(means) / max(len(means), 1)
    s_bar = sum(stds) / max(len(stds), 1)
    cv_m = (torch.tensor(means).std().item() / (abs(m_bar) + 1e-12)) if len(means) > 1 else 0.0
    cv_s = (torch.tensor(stds).std().item() / (abs(s_bar) + 1e-12)) if len(stds) > 1 else 0.0
    stable = (cv_m <= mean_cv_threshold) and (cv_s <= std_cv_threshold)
    msg = (
        f"P 分布稳定性: {'通过' if stable else '不通过'} "
        f"(mean CV={cv_m:.4f}, std CV={cv_s:.4f}; 阈值 mean<={mean_cv_threshold}, std<={std_cv_threshold})"
    )
    if not stable:
        warnings.warn(msg, UserWarning)
    return StabilityReport(
        stable=stable,
        cv_mean_p=cv_m,
        cv_std_p=cv_s,
        per_source_mean=means,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Cookbook：单层
# ---------------------------------------------------------------------------


@dataclass
class PCookbook:
    """
    单层 P 的标量 cookbook：centroids 升序，长度 2^bits。
    importance_exp：拟合时 w ∝ p^exp；越大越重视靠近 1 的 P。
    """

    bits: int
    centroids: torch.Tensor
    importance_exp: float = 1.0
    layer_idx: Optional[int] = None
    notes: str = ""

    @property
    def n_levels(self) -> int:
        return self.centroids.numel()

    @classmethod
    def fit_from_p_samples(
        cls,
        p: torch.Tensor,
        bits: int,
        importance_exp: float = 1.0,
        layer_idx: Optional[int] = None,
        max_iter: int = 200,
    ) -> "PCookbook":
        n_levels = 2 ** bits
        w = _p_importance_weights(p, importance_exp)
        c = _weighted_lloyd_1d(p, w, n_levels, max_iter=max_iter)
        return cls(bits=bits, centroids=c.cpu(), importance_exp=importance_exp, layer_idx=layer_idx)

    def quantize(self, p: torch.Tensor) -> torch.Tensor:
        """最近质心索引，与 centroids 同 device 计算。"""
        c = self.centroids.to(device=p.device, dtype=p.dtype)
        dist = (p.unsqueeze(-1) - c).abs()
        return dist.argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(device=indices.device, dtype=torch.float32)[indices.long()]

    def quantize_dequantize(self, p: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(p))

    def expected_mse_on_samples(self, p: torch.Tensor) -> float:
        pq = self.quantize_dequantize(p)
        return ((p.float() - pq.float()) ** 2).mean().item()


def fit_p_cookbook_from_pt_files(
    pt_paths: List[str],
    bits: int,
    importance_exp: float = 1.0,
    max_samples: int = 2_000_000,
    layer_idx: Optional[int] = None,
    check_stability: bool = True,
) -> Tuple[PCookbook, Optional[StabilityReport]]:
    """
    从多个 .pt 汇总 P 样本，拟合一个 PCookbook。
    若 check_stability，先评估跨文件 P 的稳定性；不稳定仍会 fit，但会 warnings.warn。
    """
    tensors: List[torch.Tensor] = []
    chunks: List[torch.Tensor] = []
    for path in pt_paths:
        p = load_p_tensor_from_pt(path, device=torch.device("cpu"))
        tensors.append(p)
        flat = p.reshape(-1)
        chunks.append(flat)
    report: Optional[StabilityReport] = None
    if check_stability and len(tensors) >= 2:
        report = evaluate_p_distribution_stability(tensors)

    cat = torch.cat(chunks, dim=0)
    if cat.numel() > max_samples:
        idx = torch.randperm(cat.numel())[:max_samples]
        cat = cat[idx]

    book = PCookbook.fit_from_p_samples(
        cat, bits=bits, importance_exp=importance_exp, layer_idx=layer_idx
    )
    return book, report


# ---------------------------------------------------------------------------
# 多层管理 + 在线 observe + 保存/加载
# ---------------------------------------------------------------------------


@dataclass
class PCookbookManager:
    """
    按 layer_idx 保存多个 PCookbook；在线推理时反复 observe(layer_idx, P)，
    在 finalize_layer 或 finalize_all 时用累计样本 fit 并写回 cookbooks。
    """

    bits: int
    importance_exp: float = 1.0
    max_samples_per_layer: int = 500_000
    cookbooks: Dict[int, PCookbook] = field(default_factory=dict)
    _buffers: Dict[int, torch.Tensor] = field(default_factory=dict)
    _buffer_sizes: Dict[int, int] = field(default_factory=dict)

    def observe(self, layer_idx: int, p: torch.Tensor) -> None:
        """在线统计：将本层 P 的样本并入缓冲区（合并后若超上限则均匀随机保留 max_samples_per_layer 条）。"""
        flat = p.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        cap = self.max_samples_per_layer
        prev = self._buffers.get(layer_idx)
        merged = flat.cpu() if prev is None else torch.cat([prev, flat.cpu()], dim=0)
        if merged.numel() > cap:
            idx = torch.randperm(merged.numel(), device=merged.device)[:cap]
            merged = merged[idx]
        self._buffers[layer_idx] = merged
        self._buffer_sizes[layer_idx] = merged.numel()

    def finalize_layer(self, layer_idx: int, max_iter: int = 200) -> PCookbook:
        """用当前缓冲区样本 fit 该层 cookbook。"""
        buf = self._buffers.get(layer_idx)
        if buf is None or buf.numel() == 0:
            raise RuntimeError(f"layer {layer_idx}: 无样本，无法 finalize")
        book = PCookbook.fit_from_p_samples(
            buf,
            bits=self.bits,
            importance_exp=self.importance_exp,
            layer_idx=layer_idx,
            max_iter=max_iter,
        )
        self.cookbooks[layer_idx] = book
        return book

    def finalize_all(self, max_iter: int = 200) -> Dict[int, PCookbook]:
        for lid in list(self._buffers.keys()):
            if self._buffer_sizes.get(lid, 0) > 0:
                self.finalize_layer(lid, max_iter=max_iter)
        return dict(self.cookbooks)

    def clear_buffers(self) -> None:
        self._buffers.clear()
        self._buffer_sizes.clear()

    def get_cookbook(self, layer_idx: int) -> PCookbook:
        if layer_idx not in self.cookbooks:
            raise KeyError(f"no cookbook for layer {layer_idx}; call finalize_layer or load_state")
        return self.cookbooks[layer_idx]

    def seed_placeholder_cookbooks(self, layer_indices: List[int]) -> None:
        """
        尚无样本时，用 [0,1) 上均匀质心占位，使 attention_forward_p_quant_only_with_layer 能跑通；
        整模型推理结束后应 finalize_* 用真实 P 覆盖，勿长期用占位码本。
        """
        n = 2 ** self.bits
        c = torch.linspace(0.0, 1.0 - 1e-6, n)
        for lid in layer_indices:
            self.cookbooks[lid] = PCookbook(
                bits=self.bits,
                centroids=c.clone(),
                importance_exp=self.importance_exp,
                layer_idx=lid,
                notes="placeholder",
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "importance_exp": self.importance_exp,
            "max_samples_per_layer": self.max_samples_per_layer,
            "cookbooks": {
                k: {
                    "centroids": v.centroids,
                    "importance_exp": v.importance_exp,
                    "layer_idx": v.layer_idx,
                    "notes": v.notes,
                }
                for k, v in self.cookbooks.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.bits = int(state["bits"])
        self.importance_exp = float(state.get("importance_exp", 1.0))
        self.max_samples_per_layer = int(state.get("max_samples_per_layer", self.max_samples_per_layer))
        self.cookbooks.clear()
        for k, v in state["cookbooks"].items():
            lid = int(k)
            self.cookbooks[lid] = PCookbook(
                bits=self.bits,
                centroids=v["centroids"],
                importance_exp=float(v.get("importance_exp", self.importance_exp)),
                layer_idx=v.get("layer_idx", lid),
                notes=str(v.get("notes", "")),
            )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "PCookbookManager":
        state = torch.load(path, map_location="cpu")
        m = cls(bits=int(state["bits"]), importance_exp=float(state.get("importance_exp", 1.0)))
        m.load_state_dict(state)
        return m


# ---------------------------------------------------------------------------
# PV 与验证
# ---------------------------------------------------------------------------


def pv_matmul_with_p_cookbook(p: torch.Tensor, v: torch.Tensor, book: PCookbook) -> torch.Tensor:
    """O = P_q @ V，其中 P_q 为 P 经 cookbook 反量化后的近似。"""
    pq = book.quantize_dequantize(p)
    return torch.matmul(pq, v)


def verify_pv_accuracy(
    p: torch.Tensor,
    v: torch.Tensor,
    book: PCookbook,
) -> Dict[str, float]:
    """对比 P@V 与 P_q@V 的误差指标。"""
    ref = torch.matmul(p, v)
    out = pv_matmul_with_p_cookbook(p, v, book)
    ref_f = ref.float().reshape(-1)
    out_f = out.float().reshape(-1)
    cos = F.cosine_similarity(ref_f, out_f, dim=0).item()
    rmse = torch.sqrt(((ref_f - out_f) ** 2).mean()).item()
    mean_abs = ref_f.abs().mean().clamp_min(1e-12)
    rel_l1 = (ref_f - out_f).abs().mean().item() / mean_abs.item()
    return {"cosine_similarity": cos, "rmse": rmse, "relative_l1": rel_l1}


# ---------------------------------------------------------------------------
# Attention：仅 P 做 cookbook 量化（端到端内嵌）
# ---------------------------------------------------------------------------


def attention_forward_p_quant_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    book: PCookbook,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准 attention：softmax(QK^T * scale) @ V，其中只对 P 应用 cookbook（quantize-dequantize）。
    返回 (output, p_exact) 便于额外分析；计算图仍对 Q,K,V 可导（P 上为离散近似分支）。
    scale 默认 1/sqrt(head_dim)。
    """
    d = q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(float(d))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = F.softmax(scores, dim=-1)
    pq = book.quantize_dequantize(p)
    out = torch.matmul(pq, v)
    return out, p


def attention_forward_p_quant_only_with_layer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    manager: PCookbookManager,
    layer_idx: int,
    scale: Optional[float] = None,
    online_observe: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    与 attention_forward_p_quant_only 相同，但按 layer_idx 从 manager 取码本；
    online_observe=True 时在本前向中把真实 P 并入 manager（用于整模型跑完后 finalize）。
    """
    book = manager.get_cookbook(layer_idx)
    out, p = attention_forward_p_quant_only(q, k, v, book, scale=scale)
    if online_observe:
        manager.observe(layer_idx, p.detach())
    return out, p


# ---------------------------------------------------------------------------
# 可选：端到端跑完后一次性保存各层 cookbook
# ---------------------------------------------------------------------------


def run_inference_then_save_cookbooks(
    forward_fn: Callable[[], None],
    manager: PCookbookManager,
    save_path: str,
    max_iter: int = 200,
) -> PCookbookManager:
    """
    forward_fn：用户定义的“整模型前向”，内部应在每层 attention 调用 observe(layer_idx, P)。
    前向结束后本函数对每层 finalize 并 save(save_path)。
    """
    forward_fn()
    manager.finalize_all(max_iter=max_iter)
    manager.save(save_path)
    manager.clear_buffers()
    return manager


# ---------------------------------------------------------------------------
# 随机数据 demo：python P_cookbook.py --demo
# ---------------------------------------------------------------------------


def _random_attention_p_v(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    device: torch.device,
    logits_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """随机 Q,K,V，返回 (P, V, Q, K)（P 为 softmax 概率）。"""
    q = torch.randn(batch, heads, seq, dim, device=device) * logits_scale
    k = torch.randn(batch, heads, seq, dim, device=device) * logits_scale
    v = torch.randn(batch, heads, seq, dim, device=device)
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(dim))
    p = F.softmax(s, dim=-1)
    return p, v, q, k


def run_demo(
    seed: int = 42,
    bits: int = 4,
    n_layers: int = 3,
    device: Optional[str] = None,
) -> None:
    """
    用随机 QKV→P 走通本模块主要能力（打印到 stdout）。
    运行：python P_cookbook.py --demo
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    print(f"[demo] device={dev}, seed={seed}, bits={bits}, n_layers={n_layers}\n")

    B, H, L, D = 2, 4, 32, 64
    p1, v1, q1, k1 = _random_attention_p_v(B, H, L, D, dev, logits_scale=1.0)
    p2, _, _, _ = _random_attention_p_v(B, H, L, D, dev, logits_scale=0.3)

    # 1) PCookbook 拟合 + PV 验证 + attention 仅 P 量化
    print("--- 1) PCookbook.fit_from_p_samples + verify_pv_accuracy ---")
    book = PCookbook.fit_from_p_samples(p1, bits=bits, importance_exp=1.0)
    m_pv = verify_pv_accuracy(p1, v1, book)
    for k, v in m_pv.items():
        print(f"    {k}: {v:.6f}")
    print(f"    expected_mse_on_samples: {book.expected_mse_on_samples(p1):.8f}")

    out_q, p_exact = attention_forward_p_quant_only(q1, k1, v1, book)
    out_ref = torch.matmul(p_exact, v1)
    cos_attn = F.cosine_similarity(out_ref.flatten(), out_q.flatten(), dim=0).item()
    print(f"    attention_forward_p_quant_only vs P@V cosine: {cos_attn:.6f}\n")

    # 2) 跨分布稳定性
    print("--- 2) evaluate_p_distribution_stability (p1 vs p2) ---")
    rep = evaluate_p_distribution_stability([p1.cpu(), p2.cpu()])
    print(f"    {rep.message}\n")

    # 3) 临时 .pt 多文件 → fit_p_cookbook_from_pt_files
    print("--- 3) fit_p_cookbook_from_pt_files (temp .pt) ---")
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, t in enumerate([p1, p2]):
            path = os.path.join(td, f"p_{i}.pt")
            torch.save({"P": t.cpu()}, path)
            paths.append(path)
        book_m, rep2 = fit_p_cookbook_from_pt_files(
            paths, bits=bits, importance_exp=1.0, check_stability=True
        )
        print(f"    merged cookbook n_levels={book_m.n_levels}, report={rep2 is not None}\n")

    # 4) PCookbookManager：仅 observe → finalize_all → save/load
    print("--- 4) PCookbookManager observe → finalize_all → save/load ---")
    with tempfile.TemporaryDirectory() as td:
        save_path = os.path.join(td, "mgr.pt")
        mgr = PCookbookManager(bits=bits, importance_exp=1.0, max_samples_per_layer=100_000)
        for lid in range(n_layers):
            p_l, _, _, _ = _random_attention_p_v(B, H, L, D, dev, logits_scale=1.0 + 0.1 * lid)
            mgr.observe(lid, p_l)
        mgr.finalize_all()
        mgr.save(save_path)
        mgr2 = PCookbookManager.load(save_path)
        assert len(mgr2.cookbooks) == n_layers
        print(f"    saved/loaded layers: {sorted(mgr2.cookbooks.keys())}\n")

    # 5) 占位码本 + 在线 observe + attention_with_layer + finalize + run_inference_then_save_cookbooks
    print("--- 5) seed_placeholder + attention_forward_p_quant_only_with_layer(observe) + run_inference_then_save_cookbooks ---")
    with tempfile.TemporaryDirectory() as td:
        path_inf = os.path.join(td, "online.pt")
        mgr3 = PCookbookManager(bits=bits, importance_exp=1.0)
        mgr3.seed_placeholder_cookbooks(list(range(n_layers)))

        def forward_pass() -> None:
            for lid in range(n_layers):
                p_l, v_l, q_l, k_l = _random_attention_p_v(B, H, L, D, dev, logits_scale=1.2)
                attention_forward_p_quant_only_with_layer(
                    q_l, k_l, v_l, mgr3, layer_idx=lid, online_observe=True
                )

        run_inference_then_save_cookbooks(forward_pass, mgr3, path_inf)
        mgr4 = PCookbookManager.load(path_inf)
        print(f"    post-inference layers: {sorted(mgr4.cookbooks.keys())}")
        print("    (placeholders overwritten by finalize; notes may still be empty on PCookbook)\n")

    print("[demo] done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P cookbook 工具；加 --demo 运行随机自检。")
    p.add_argument(
        "--demo",
        action="store_true",
        help="随机生成 P/QKV，跑通拟合、稳定性、多.pt、Manager、端到端保存等自检并打印结果",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.demo:
        run_demo(seed=args.seed, bits=args.bits, n_layers=args.n_layers, device=args.device)
    else:
        print("用法: python P_cookbook.py --demo [--seed N] [--bits B] [--n-layers L] [--device cuda|cpu]")


__all__ = [
    "PCookbook",
    "PCookbookManager",
    "load_p_tensor_from_pt",
    "fit_p_cookbook_from_pt_files",
    "evaluate_p_distribution_stability",
    "StabilityReport",
    "pv_matmul_with_p_cookbook",
    "verify_pv_accuracy",
    "attention_forward_p_quant_only",
    "attention_forward_p_quant_only_with_layer",
    "run_inference_then_save_cookbooks",
    "run_demo",
]
