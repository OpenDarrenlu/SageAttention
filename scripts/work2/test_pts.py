#!/usr/bin/env python3
"""
从包含 .pt 文件的文件夹中读取并执行 P codebook 分析的完整流程。

主要功能：
1. 遍历文件夹，收集所有 .pt 文件
2. 加载并检查 P 分布稳定性
3. 拟合 codebook
4. 验证 PV 准确率
5. 保存结果
"""

import argparse
import os
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

# 导入 P_codebook 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from P_codebook import (
    load_p_tensor_from_pt,
    fit_p_codebook_from_pt_files,
    evaluate_p_distribution_stability,
    Pcodebook,
    PcodebookManager,
    verify_pv_accuracy,
    StabilityReport,
    attention_forward_p_quant_only,
)


def find_pt_files(folder_path: str, recursive: bool = True) -> List[str]:
    """
    从文件夹中查找所有 .pt 文件
    
    Args:
        folder_path: 文件夹路径
        recursive: 是否递归查找子文件夹
    
    Returns:
        .pt 文件路径列表
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    pattern = "**/*.pt" if recursive else "*.pt"
    pt_files = list(folder.glob(pattern))
    pt_files = [str(f) for f in pt_files]
    pt_files.sort()
    return pt_files


def load_all_p_tensors(pt_paths: List[str], device: Optional[str] = None) -> List[torch.Tensor]:
    """
    从多个 .pt 文件中加载所有 P tensor
    
    Args:
        pt_paths: .pt 文件路径列表
        device: 设备
    
    Returns:
        P tensor 列表
    """
    dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    p_tensors = []
    
    print(f"\n[加载 .pt 文件]")
    for path in pt_paths:
        try:
            p = load_p_tensor_from_pt(path, device=torch.device(dev))
            p_tensors.append(p)
            print(f"  ✓ {os.path.basename(path)}: shape={p.shape}, dtype={p.dtype}, "
                  f"mean={p.mean().item():.6f}, std={p.std().item():.6f}")
        except Exception as e:
            print(f"  ✗ {os.path.basename(path)}: 加载失败 - {e}")
    
    return p_tensors


def generate_random_v(p: torch.Tensor) -> torch.Tensor:
    """
    根据 P 的形状生成随机 V tensor 用于验证
    
    Args:
        p: P tensor (shape: ..., seq_len, seq_len)
    
    Returns:
        V tensor (shape: ..., seq_len, d)
    """
    seq_len = p.shape[-1]
    d = 64  # 常用的维度
    shape = list(p.shape[:-1]) + [d]
    return torch.randn(*shape, device=p.device)


def load_qkv_from_pt(path: str, device: Optional[str] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    从 .pt 文件中加载 Q, K, V
    
    Args:
        path: .pt 文件路径
        device: 设备
    
    Returns:
        (q, k, v) 元组，如果不存在则为 None
    """
    dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        obj: Any = torch.load(path, map_location=dev)
        if not isinstance(obj, dict):
            return None, None, None
        
        q = obj.get("query", obj.get("q"))
        k = obj.get("key", obj.get("k"))
        v = obj.get("value", obj.get("v"))
        
        if q is not None and q.dim() < 3:
            q = q.unsqueeze(0)
        if k is not None and k.dim() < 3:
            k = k.unsqueeze(0)
        if v is not None and v.dim() < 3:
            v = v.unsqueeze(0)
        
        return q, k, v
    except Exception:
        return None, None, None


def verify_attention_accuracy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    book: Pcodebook,
    scale: Optional[float] = None,
) -> Dict[str, float]:
    """
    验证端到端 Attention 精度：对比原始 Attention 和量化后 Attention 的输出
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        book: Pcodebook
        scale: Attention scale (默认 1/sqrt(d))
    
    Returns:
        精度指标字典
    """
    # import ipdb; ipdb.set_trace()
    # 原始 Attention
    d = q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(float(d))
    scores_orig = torch.matmul(q, k.transpose(-2, -1)) * scale
    p_orig = F.softmax(scores_orig, dim=-1)
    out_orig = torch.matmul(p_orig, v.to(p_orig.dtype))
    
    # 量化后的 Attention
    out_quant, p_exact = attention_forward_p_quant_only(q, k, v, book, scale=scale)
    
    # 计算误差指标
    out_orig_flat = out_orig.float().reshape(-1)
    out_quant_flat = out_quant.float().reshape(-1)
    
    cos = F.cosine_similarity(out_orig_flat, out_quant_flat, dim=0).item()
    rmse = torch.sqrt(((out_orig_flat - out_quant_flat) ** 2).mean()).item()
    mean_abs = out_orig_flat.abs().mean().clamp_min(1e-12)
    rel_l1 = (out_orig_flat - out_quant_flat).abs().mean().item() / mean_abs.item()
    
    return {
        "cosine_similarity": cos,
        "rmse": rmse,
        "relative_l1": rel_l1,
    }


def run_importance_exp_scan(
    pt_files: List[str],
    bits: int = 4,
    importance_exp_list: Optional[List[float]] = None,
    max_samples: int = 2_000_000,
    device: Optional[str] = None,
    include_zero: bool = False,
    zero_threshold: float = 0.01,
) -> Dict[float, Dict[str, Any]]:
    """
    扫描不同 importance_exp 值，分析精度差异
    
    Args:
        pt_files: .pt 文件路径列表
        bits: codebook 比特数
        importance_exp_list: 要扫描的 importance_exp 列表
        max_samples: 最大样本数
        device: 设备
    
    Returns:
        每个 importance_exp 对应的结果字典
    """
    if importance_exp_list is None:
        importance_exp_list = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    results = {}
    dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"[Importance Exp 扫描] 扫描 {len(importance_exp_list)} 个值: {importance_exp_list}")
    print(f"{'='*80}")
    
    # 先加载所有 P tensor（避免重复加载）
    p_tensors = []
    qkv_list = []
    for path in pt_files:
        try:
            p = load_p_tensor_from_pt(path, device=torch.device(dev))
            p_tensors.append(p)
            q, k, v = load_qkv_from_pt(path, device=dev)
            qkv_list.append((q, k, v))
        except Exception as e:
            print(f"  跳过文件 {os.path.basename(path)}: {e}")
    
    if not p_tensors:
        raise ValueError("没有可用的 P tensor")
    
    # 扫描每个 importance_exp
    for exp in importance_exp_list:
        print(f"\n[扫描] importance_exp = {exp}")
        try:
            # 拟合 codebook
            codebook, _ = fit_p_codebook_from_pt_files(
                pt_files,
                bits=bits,
                importance_exp=exp,
                max_samples=max_samples,
                check_stability=False,
                include_zero=include_zero,
                zero_threshold=zero_threshold,
            )
            
            # 验证 PV 准确率
            pv_metrics_list = []
            for p in p_tensors:
                v = generate_random_v(p)
                metrics = verify_pv_accuracy(p, v, codebook)
                pv_metrics_list.append(metrics)
            
            avg_pv_metrics = {
                k: sum(m[k] for m in pv_metrics_list) / len(pv_metrics_list)
                for k in pv_metrics_list[0].keys()
            }
            
            # 验证 Attention 精度（如果有 QKV）
            attn_metrics_list = []
            for i, (q, k, v) in enumerate(qkv_list):
                if q is not None and k is not None and v is not None:
                    try:
                        metrics = verify_attention_accuracy(q, k, v, codebook)
                        attn_metrics_list.append(metrics)
                    except Exception:
                        pass
            
            avg_attn_metrics = None
            if attn_metrics_list:
                avg_attn_metrics = {
                    k: sum(m[k] for m in attn_metrics_list) / len(attn_metrics_list)
                    for k in attn_metrics_list[0].keys()
                }
            
            results[exp] = {
                "importance_exp": exp,
                "codebook": codebook,
                "pv_metrics": avg_pv_metrics,
                "attn_metrics": avg_attn_metrics,
            }
            
            print(f"  ✓ PV - cosine: {avg_pv_metrics['cosine_similarity']:.6f}, "
                  f"rmse: {avg_pv_metrics['rmse']:.6f}, "
                  f"rel_l1: {avg_pv_metrics['relative_l1']:.6f}")
            if avg_attn_metrics:
                print(f"  ✓ Attention - cosine: {avg_attn_metrics['cosine_similarity']:.6f}, "
                      f"rmse: {avg_attn_metrics['rmse']:.6f}, "
                      f"rel_l1: {avg_attn_metrics['relative_l1']:.6f}")
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results[exp] = {"error": str(e)}
    
    # 打印对比总结
    print(f"\n{'='*80}")
    print(f"[扫描总结]")
    print(f"{'='*80}")
    print(f"{'Importance Exp':<15} {'PV Cosine':<12} {'PV RMSE':<12} {'Attn Cosine':<12}")
    print(f"{'-'*51}")
    
    for exp in sorted(results.keys()):
        res = results[exp]
        if "error" in res:
            print(f"{exp:<15} {'ERROR':<12}")
        else:
            pv_cos = res["pv_metrics"]["cosine_similarity"]
            pv_rmse = res["pv_metrics"]["rmse"]
            attn_cos = res["attn_metrics"]["cosine_similarity"] if res["attn_metrics"] else "N/A"
            print(f"{exp:<15} {pv_cos:<12.6f} {pv_rmse:<12.6f} {attn_cos:<12}")
    
    return results


def run_pt_folder_demo(
    pt_folder: str,
    bits: int = 4,
    importance_exp: float = 1.0,
    max_samples: int = 2_000_000,
    check_stability: bool = True,
    recursive: bool = True,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    verify_attention: bool = False,
    scan_importance_exp: bool = False,
    importance_exp_list: Optional[List[float]] = None,
    include_zero: bool = False,
    zero_threshold: float = 0.01,
) -> Dict[str, Any]:
    results = {}
    
    dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[配置] device={dev}, bits={bits}, importance_exp={importance_exp}, "
          f"max_samples={max_samples}")
    
    # 1. 查找 .pt 文件
    print(f"\n[1/6] 查找 .pt 文件 (文件夹: {pt_folder})")
    pt_files = find_pt_files(pt_folder, recursive=recursive)
    if not pt_files:
        raise ValueError(f"在文件夹 {pt_folder} 中未找到 .pt 文件")
    print(f"  找到 {len(pt_files)} 个 .pt 文件:")
    for f in pt_files:
        print(f"    - {os.path.basename(f)}")
    results["pt_files"] = pt_files
    
    # 2. 加载 P tensor
    print(f"\n[2/6] 加载 P tensor")
    p_tensors = load_all_p_tensors(pt_files, device=dev)
    if not p_tensors:
        raise ValueError("未能加载任何有效 P tensor")
    results["p_tensors_loaded"] = len(p_tensors)
    
    # 3. 检查分布稳定性
    if check_stability and len(p_tensors) >= 2:
        print(f"\n[3/6] 评估 P 分布稳定性")
        stability_report = evaluate_p_distribution_stability(p_tensors)
        print(f"  {stability_report.message}")
        results["stability"] = {
            "stable": stability_report.stable,
            "cv_mean_p": stability_report.cv_mean_p,
            "cv_std_p": stability_report.cv_std_p,
            "per_source_mean": stability_report.per_source_mean,
        }
    else:
        print(f"\n[3/6] 跳过分布稳定性检查 (少于 2 个文件或 check_stability=False)")
        results["stability"] = None
    
    # 4. 拟合 codebook
    print(f"\n[4/6] 拟合 P codebook")
    if include_zero:
        print(f"  [配置] 包含 0 (zero_threshold={zero_threshold})")
    codebook, rep = fit_p_codebook_from_pt_files(
        pt_files,
        bits=bits,
        importance_exp=importance_exp,
        max_samples=max_samples,
        check_stability=check_stability,
        include_zero=include_zero,
        zero_threshold=zero_threshold,
    )
    print(f"  ✓ Codebook 拟合完成: n_levels={codebook.n_levels}, bits={codebook.bits}")
    print(f"    centroids (升序): {codebook.centroids}")
    results["codebook"] = {
        "bits": codebook.bits,
        "n_levels": codebook.n_levels,
        "importance_exp": codebook.importance_exp,
        "centroids": codebook.centroids.tolist(),
    }
    
    # 5. 验证 PV 准确率
    total_steps = 8 if verify_attention else 6
    print(f"\n[5/{total_steps}] 验证 PV 准确率")
    v_tensors = []
    pv_metrics_list = []
    
    for i, p in enumerate(p_tensors):
        v = generate_random_v(p)
        v_tensors.append(v)
        metrics = verify_pv_accuracy(p, v, codebook)
        pv_metrics_list.append(metrics)
        print(f"  文件 {i+1} ({os.path.basename(pt_files[i])}):")
        for k, v_val in metrics.items():
            print(f"    {k}: {v_val:.6f}")
    
    # 计算平均指标
    if pv_metrics_list:
        avg_metrics = {
            k: sum(m[k] for m in pv_metrics_list) / len(pv_metrics_list)
            for k in pv_metrics_list[0].keys()
        }
        print(f"\n  平均指标:")
        for k, v_val in avg_metrics.items():
            print(f"    {k}: {v_val:.6f}")
        results["pv_metrics"] = {
            "per_file": pv_metrics_list,
            "average": avg_metrics,
        }
    
    # 6. 验证端到端 Attention 精度
    attn_metrics_list = []
    if verify_attention:
        print(f"\n[6/{total_steps}] 验证端到端 Attention 精度")
        
        for i, path in enumerate(pt_files):
            q, k, v = load_qkv_from_pt(path, device=dev)
            if q is not None and k is not None and v is not None:
                try:
                    metrics = verify_attention_accuracy(q, k, v, codebook)
                    attn_metrics_list.append(metrics)
                    print(f"  文件 {i+1} ({os.path.basename(pt_files[i])}):")
                    for kk, vv in metrics.items():
                        print(f"    {kk}: {vv:.6f}")
                except Exception as e:
                    print(f"  ✗ 文件 {i+1} ({os.path.basename(pt_files[i])}): 验证失败 - {e}")
            else:
                print(f"  ⚠ 文件 {i+1} ({os.path.basename(pt_files[i])}): 未找到 Q/K/V，跳过")
        
        if attn_metrics_list:
            avg_attn_metrics = {
                k: sum(m[k] for m in attn_metrics_list) / len(attn_metrics_list)
                for k in attn_metrics_list[0].keys()
            }
            print(f"\n  平均 Attention 指标:")
            for k, v_val in avg_attn_metrics.items():
                print(f"    {k}: {v_val:.6f}")
            results["attn_metrics"] = {
                "per_file": attn_metrics_list,
                "average": avg_attn_metrics,
            }
    
    # 7. Importance Exp 扫描（如果启用）
    if scan_importance_exp:
        print(f"\n[7/{total_steps}] 执行 Importance Exp 扫描")
        scan_results = run_importance_exp_scan(
            pt_files,
            bits=bits,
            importance_exp_list=importance_exp_list,
            max_samples=max_samples,
            device=device,
            include_zero=include_zero,
            zero_threshold=zero_threshold,
        )
        results["importance_exp_scan"] = scan_results
    
    # 8. 保存结果
    total_steps = 8 if verify_attention else 6
    if output_dir:
        print(f"\n[{total_steps}/{total_steps}] 保存结果到: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 codebook
        codebook_path = os.path.join(output_dir, "p_codebook.pt")
        torch.save(codebook.centroids, codebook_path)
        print(f"  ✓ Codebook 保存到: {codebook_path}")
        
        # 保存完整结果
        manager = PcodebookManager(bits=bits, importance_exp=importance_exp)
        manager.codebooks[0] = codebook  # 暂时用 layer_idx=0
        manager_path = os.path.join(output_dir, "p_codebook_manager.pt")
        manager.save(manager_path)
        print(f"  ✓ PcodebookManager 保存到: {manager_path}")
        
        # 保存结果字典（文本）
        result_txt_path = os.path.join(output_dir, "results.txt")
        with open(result_txt_path, "w") as f:
            f.write("=== P Codebook 分析结果 ===\n\n")
            f.write(f"文件夹: {pt_folder}\n")
            f.write(f".pt 文件数: {len(pt_files)}\n")
            f.write(f"Bits: {bits}\n")
            f.write(f"Importance Exp: {importance_exp}\n\n")
            
            f.write("Centroids:\n")
            f.write(f"  {codebook.centroids}\n\n")
            
            if "stability" in results and results["stability"]:
                f.write(f"分布稳定性: {'通过' if results['stability']['stable'] else '不通过'}\n")
                f.write(f"  CV(mean): {results['stability']['cv_mean_p']:.6f}\n")
                f.write(f"  CV(std): {results['stability']['cv_std_p']:.6f}\n\n")
            
            if "pv_metrics" in results:
                f.write("平均 PV 指标:\n")
                for k, v_val in results["pv_metrics"]["average"].items():
                    f.write(f"  {k}: {v_val:.6f}\n")
            
            if "attn_metrics" in results:
                f.write("\n平均 Attention 指标:\n")
                for k, v_val in results["attn_metrics"]["average"].items():
                    f.write(f"  {k}: {v_val:.6f}\n")
            
            if "importance_exp_scan" in results:
                f.write("\n=== Importance Exp 扫描结果 ===\n")
                scan_res = results["importance_exp_scan"]
                f.write(f"{'Importance Exp':<15} {'PV Cosine':<12} {'PV RMSE':<12} {'Attn Cosine':<12}\n")
                f.write(f"{'-'*51}\n")
                for exp in sorted(scan_res.keys()):
                    res = scan_res[exp]
                    if "error" in res:
                        f.write(f"{exp:<15} {'ERROR':<12}\n")
                    else:
                        pv_cos = res["pv_metrics"]["cosine_similarity"]
                        pv_rmse = res["pv_metrics"]["rmse"]
                        attn_cos = res["attn_metrics"]["cosine_similarity"] if res["attn_metrics"] else "N/A"
                        f.write(f"{exp:<15} {pv_cos:<12.6f} {pv_rmse:<12.6f} {attn_cos:<12}\n")
        
        print(f"  ✓ 结果摘要保存到: {result_txt_path}")
        results["output_dir"] = output_dir
        results["saved_files"] = [codebook_path, manager_path, result_txt_path]
    else:
        print(f"\n[{total_steps}/{total_steps}] 跳过保存结果 (output_dir 未指定)")
    
    print(f"\n[完成] 所有步骤执行完毕！")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="从包含 .pt 文件的文件夹中执行 P codebook 完整分析流程"
    )
    parser.add_argument(
        "pt_folder",
        type=str,
        help="包含 .pt 文件的文件夹路径"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="codebook 比特数 (默认: 4)"
    )
    parser.add_argument(
        "--importance-exp",
        type=float,
        default=1.0,
        help="重要性权重指数，越大越重视接近 1 的 P (默认: 1.0)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2_000_000,
        help="最大样本数 (默认: 2,000,000)"
    )
    parser.add_argument(
        "--no-stability",
        action="store_true",
        help="不检查 P 分布稳定性"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归查找子文件夹"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，保存 codebook 和结果 (默认: 不保存)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu，默认自动选择)"
    )
    parser.add_argument(
        "--verify-attention",
        action="store_true",
        help="验证端到端 Attention 精度 (需要 .pt 文件包含 Q/K/V)"
    )
    parser.add_argument(
        "--scan-importance-exp",
        action="store_true",
        help="扫描不同 importance_exp 值分析精度差异 (计算量较大)"
    )
    parser.add_argument(
        "--importance-exp-list",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        help="要扫描的 importance_exp 值列表 (默认: 0.2 0.5 0.8 1.0 1.5 2.0 3.0)"
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="在码表中包含 0 (推荐用于小概率值较多的场景)"
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.01,
        help="小于此值的 P 量化为 0 (默认: 0.01)"
    )
    
    args = parser.parse_args()
    
    try:
        run_pt_folder_demo(
            pt_folder=args.pt_folder,
            bits=args.bits,
            importance_exp=args.importance_exp,
            max_samples=args.max_samples,
            check_stability=not args.no_stability,
            recursive=not args.no_recursive,
            output_dir=args.output_dir,
            device=args.device,
            verify_attention=args.verify_attention,
            scan_importance_exp=args.scan_importance_exp,
            importance_exp_list=args.importance_exp_list,
            include_zero=args.include_zero,
            zero_threshold=args.zero_threshold,
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()