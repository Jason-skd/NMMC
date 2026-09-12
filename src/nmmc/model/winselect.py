"""去超参（issue #6）：窗长逐日滚动伪样本外选择（1-SE 规则）+ 零参数变体。

W_L=4（L1 同周几窗）与 W_R=7（P1 近窗，兼题3 融合持续法窗）原为经验
定数（zzj 档定选），消融只在候选间裁决，窗长本身从未被论证。本模块把
窗长改为逐日数据驱动：

  对每个候选窗长 w，用全部可得历史重放伪样本外预报（L1/P1 均为封闭
  形式滚动均值，重放零成本），得逐日 MAE 曲面 MAE(i, w)；在公共评估
  集——所有候选都拥有完整窗的过去日——上算各候选均值 m_w 与标准误
  se_w，按 1-SE 规则取 {w : m_w ≤ m_best + se_best} 的最小窗。

只用过去（i < j）、无未来信息，与反方差融合"谁历史准听谁"（F12）
同一哲学；带内取最小窗防逐日抖动（候选间 MAE 差多为噪声，1-SE 带内
不分胜负时偏简单）。评估日不足 min_days 退回先验默认窗（冷启动纪律
同 fusion_weights）。

备选零参数路线 zp_load / zp_pv：L3/P3 全历史拟合 × 全历史形态（负载
按同周几、光伏全体），连形态窗 W_sh 一并消掉——若技能不输 L1/P1，
窗口问题整个消失。
"""
import numpy as np
import pandas as pd

from nmmc.data.year import YearData
from nmmc.model.forecast import (ForecastConfig, _load_regression,
                                 _pv_harmonic)

LOAD_GRID = tuple(range(1, 11))   # W_L 候选 1..10（同周几步，十周）
PV_GRID = tuple(range(1, 15))     # W_R 候选 1..14（两周）
MIN_EVAL_DAYS = 21                # 评估日下限（三周），不足退回先验窗
DEFAULT_WL, DEFAULT_WR = 4, 7     # 冷启动先验（原交付常数）

# 零参数变体借用 L3/P3 的回归核（harm_order=2 与一阶年谐波均同交付档）
_ZP_LOAD_CFG = ForecastConfig(load="L3", harm_order=2)
_ZP_PV_CFG = ForecastConfig(pv="P3", pv_fit_window=0)


def trailing_mean_mae(curves: np.ndarray, grid) -> np.ndarray:
    """逐位置 × 逐候选窗的伪样本外 MAE：位置 k 用前 w 条均值预报第 k 条。

    Args:
        curves: (n, 144) 按时间排序的同族日曲线（负载为同周几类内、
            光伏为全体日序）。
        grid: 候选窗长序列。

    Returns:
        (n, len(grid))，窗不足（历史不够 w 条）处 NaN。
    """
    n = len(curves)
    cs = np.vstack([np.zeros((1, curves.shape[1])), curves.cumsum(axis=0)])
    out = np.full((n, len(grid)), np.nan)
    for g, w in enumerate(grid):
        if w >= n:
            break                                  # 无任何位置能凑齐完整窗
        pred = (cs[w:n] - cs[0:n - w]) / w         # 位置 k 预报 = 前 w 条均值
        out[w:, g] = np.abs(curves[w:] - pred).mean(axis=1)
    return out


def day_mae_load(year: YearData, grid=LOAD_GRID) -> np.ndarray:
    """L1 族伪样本外 MAE 曲面 (365, G)：同周几类内重放近 w 同周几均值。"""
    out = np.full((len(year.dates), len(grid)), np.nan)
    for d in range(7):
        idx = np.flatnonzero(year.dow == d)
        out[idx] = trailing_mean_mae(year.ell[idx], grid)
    return out


def day_mae_pv(year: YearData, grid=PV_GRID) -> np.ndarray:
    """P1 族伪样本外 MAE 曲面 (365, G)：全体日序内重放近 w 日均值。"""
    return trailing_mean_mae(year.r, grid)


def select_one_se(mean: np.ndarray, se: np.ndarray, grid) -> int:
    """1-SE 规则：与最优一个标准误之内取最小窗。"""
    best = int(np.argmin(mean))
    ok = np.flatnonzero(mean <= mean[best] + se[best] + 1e-12)
    return grid[int(ok[0])]


def family_schedule(day_mae: np.ndarray, grid, default: int,
                    min_days: int = MIN_EVAL_DAYS) -> dict[str, np.ndarray]:
    """单族逐日滚动选择：第 j 日只用 i < j 的公共评估日。

    公共评估日 = 所有小候选都拥有完整窗的过去日（MAE 行无 NaN）。均值与
    标准误由滚动累加（sum / sumsq）增量维护，365 日 O(1) 推进。

    Args:
        day_mae: 伪样本外 MAE 曲面 (n, G)。
        grid: 候选窗长序列。
        default: 冷启动（评估日不足）先验窗。
        min_days: 评估日下限。

    Returns:
        逐日诊断字典：w（选中窗）、best（MAE 最优窗）、mae / se（最优
        候选的均值与标准误）、n_eval（评估日数）。
    """
    n = day_mae.shape[0]
    usable = ~np.isnan(day_mae).any(axis=1)

    # 滚动累加：处理第 j 日前并入 i = j−1，评估集恰为 i < j（无泄漏）
    s = np.zeros(day_mae.shape[1])
    s2 = np.zeros(day_mae.shape[1])
    cnt = 0
    diag: dict[str, np.ndarray] = {
        k: np.zeros(n) for k in ("w", "best", "mae", "se", "n_eval")}
    for j in range(n):
        if j >= 1 and usable[j - 1]:
            cnt += 1
            s += day_mae[j - 1]
            s2 += day_mae[j - 1] ** 2
        if cnt >= min_days:
            mean = s / cnt
            var = np.maximum((s2 - cnt * mean ** 2) / (cnt - 1), 0.0)
            se = np.sqrt(var / cnt)
            w = select_one_se(mean, se, grid)
            best = int(np.argmin(mean))
            diag["w"][j], diag["best"][j] = w, grid[best]
            diag["mae"][j], diag["se"][j] = mean[best], se[best]
        else:
            diag["w"][j], diag["best"][j] = default, 0
            diag["mae"][j], diag["se"][j] = np.nan, np.nan
        diag["n_eval"][j] = cnt
    return diag


def window_schedule(year: YearData, load_grid=LOAD_GRID, pv_grid=PV_GRID,
                    min_days: int = MIN_EVAL_DAYS
                    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """全年逐日选窗序列（论文可引用诊断）。

    Args:
        year: 附件2 全年实际值。
        load_grid / pv_grid: 两族候选窗长。
        min_days: 评估日下限。

    Returns:
        (逐日诊断表, 负载 MAE 曲面, 光伏 MAE 曲面)。表列：j、date、
        w_l / w_r（选中窗，冷启动为先验 4/7）、best_l / best_r（MAE
        最优窗）、mae_/se_（最优候选均值与标准误）、n_eval（评估日数）。
    """
    mae_l = day_mae_load(year, load_grid)
    mae_r = day_mae_pv(year, pv_grid)
    dl = family_schedule(mae_l, load_grid, DEFAULT_WL, min_days)
    dr = family_schedule(mae_r, pv_grid, DEFAULT_WR, min_days)
    df = pd.DataFrame({
        "j": np.arange(len(year.dates)),
        "date": pd.to_datetime(year.dates).date,
        "w_l": dl["w"].astype(int), "w_r": dr["w"].astype(int),
        "best_l": dl["best"].astype(int), "best_r": dr["best"].astype(int),
        "mae_l": dl["mae"], "se_l": dl["se"], "n_eval_l": dl["n_eval"].astype(int),
        "mae_r": dr["mae"], "se_r": dr["se"], "n_eval_r": dr["n_eval"].astype(int),
    })
    return df, mae_l, mae_r


def zp_load(year: YearData, j: int, typ: np.ndarray) -> np.ndarray:
    """零参数负载预报：L3 全历史对数线性总量 × 同周几全历史均值形态。"""
    hist = year.ell[:j]
    if len(hist) == 0:
        return typ.copy()

    # 形态：目标周几的全历史均值曲线（类空退化为全历史均值，冷启动防呆）
    same = hist[year.dow[:j] == year.dow[j]]
    shape = same.mean(axis=0) if len(same) else hist.mean(axis=0)
    return _load_regression(_ZP_LOAD_CFG, hist, year.dow[:j], year.doy[:j],
                            year.dow[j], year.doy[j], shape)


def zp_pv(year: YearData, j: int, typ: np.ndarray) -> np.ndarray:
    """零参数光伏预报：P3 全历史一阶年谐波总量 × 全历史均值形态。"""
    hist = year.r[:j]
    if len(hist) == 0:
        return typ.copy()
    return _pv_harmonic(_ZP_PV_CFG, hist, year.doy[:j], year.doy[j],
                        hist.mean(axis=0))
