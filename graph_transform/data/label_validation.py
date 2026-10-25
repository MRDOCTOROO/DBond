"""
标签严格校验模块（P0-2）

PBCLA 生成的 true_multi 标签可能因为数据错误出现：非法 token、长度与 seq 不匹配、
空/缺失标签等。旧实现会在 _parse_labels/_prepare_labels 中静默丢弃非法 token、
补 0 或截断，把数据错误变成训练目标的一部分。

本模块在数据入口（GraphDataset._load_data）执行 fail-fast 校验：

  - 序列长度 >= 2（否则没有 bond）；
  - 标签 token 全部属于 {0, 1}；
  - 标签数量 == len(seq) - 1（只允许尾部分号）；
  - 不接受空 / NaN 标签。

校验模式由 data.strict_label_mode 控制：'drop'（丢弃坏样本并记录，默认）或
'raise'（直接抛异常）。被丢弃样本会输出完整清单到日志，保证可审计。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALID_TOKENS = {"0", "1"}


def parse_label_tokens(label_str: Any) -> List[str]:
    """把 true_multi 字符串切成 token 列表（保留空段信息，供错误定位）。"""
    if pd.isna(label_str):
        return []
    return str(label_str).split(";")


def label_error_reason(seq: Any, label_str: Any) -> str | None:
    """返回该行的标签错误原因；无错误返回 None。"""
    if seq is None or pd.isna(seq):
        return "missing_seq"
    seq = str(seq).strip()
    if len(seq) < 2:
        return f"seq_too_short(len={len(seq)})"

    if label_str is None or pd.isna(label_str) or str(label_str).strip() == "":
        return "missing_label"

    text = str(label_str).strip()
    tokens = text.split(";")
    # 只允许尾部分号造成的空尾段
    if tokens and tokens[-1] == "":
        tokens = tokens[:-1]
    elif "" in tokens:
        return "empty_token_mid_string"

    for tok in tokens:
        if tok not in _VALID_TOKENS:
            return f"invalid_token({tok!r})"

    n_bonds = len(seq) - 1
    if len(tokens) != n_bonds:
        return f"length_mismatch(label={len(tokens)}, expected={n_bonds})"
    return None


def label_error_report(frame: pd.DataFrame, seq_col: str = "seq", label_col: str = "true_multi") -> pd.DataFrame:
    """返回坏样本清单 DataFrame（row_idx, seq, reason），无坏样本返回空表。"""
    reasons: List[Dict[str, Any]] = []
    for row_idx, (seq, label) in enumerate(zip(frame[seq_col], frame[label_col])):
        reason = label_error_reason(seq, label)
        if reason is not None:
            reasons.append({"row_idx": row_idx, "seq": seq, "true_multi": label, "reason": reason})
    return pd.DataFrame(reasons, columns=["row_idx", "seq", "true_multi", "reason"])


def validate_label_frame(
    frame: pd.DataFrame,
    seq_col: str = "seq",
    label_col: str = "true_multi",
    on_error: str = "drop",
    source_name: str = "",
) -> pd.DataFrame:
    """校验整个标签矩阵，返回清洗后的 frame。

    Args:
        frame: 含 seq / true_multi 列的 DataFrame。
        on_error: 'drop' 丢弃坏样本（记录清单）或 'raise' 抛出 ValueError。
        source_name: 数据来源标识，仅用于日志。
    """
    if on_error not in ("drop", "raise"):
        raise ValueError(f"strict_label_mode 必须是 drop/raise，得到 {on_error!r}")

    errors = label_error_report(frame, seq_col, label_col)
    if errors.empty:
        return frame

    reason_counts = errors["reason"].value_counts().to_dict()
    logger.warning(
        "strict label check [%s]: %d/%d rows invalid, reasons=%s",
        source_name or frame.attrs.get("source", "unknown"),
        len(errors),
        len(frame),
        reason_counts,
    )
    # 打印前 20 条坏样本明细，保证可审计
    for _, row in errors.head(20).iterrows():
        logger.warning("  bad row %d seq=%s label=%r reason=%s",
                       row["row_idx"], row["seq"], row["true_multi"], row["reason"])

    if on_error == "raise":
        raise ValueError(f"strict label check failed: {len(errors)} bad rows, reasons={reason_counts}")

    bad_idx = errors["row_idx"].to_numpy()
    keep_mask = np.ones(len(frame), dtype=bool)
    keep_mask[bad_idx] = False
    return frame.iloc[keep_mask].reset_index(drop=True)
