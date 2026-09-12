"""题2 预报层：点预报候选（L1/L2/L3、P1/P2/P3）与残差场景族（R1/R2）。

候选制（文档 §3）：负载与光伏分开建模再相减得净负荷预报（统计结构不同）；
误差场景为伪样本外残差日曲线整条贴加，相关结构自动保留。全部只用
F_0^(j)（截至昨日的实际值 + 日历）；随机性固定种子。样本不足时用可得
样本，完全无样本（day 0）退化为附件 1 典型日先验，残差为 0 时场景退化
N_s=1（保守暖机的最简实现）。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.data.year import YearData
from nmmc.model.lp import T

R_EPS = 1.0  # 光伏相对残差防除零下限，kWh（文档 §3.3）


@dataclass(frozen=True)
class ForecastConfig:
    """预报层配置（文档 §3.4 网格三轴 + 各候选参数默认值）。

    Attributes:
        load: 负载候选，L1 近周几 / L2 同星期几 EWMA / L3 对数线性回归×形态。
        pv: 光伏候选，P1 近窗 / P2 长窗×持续性 / P3 谐波。
        resid: 残差场景配置，R1 28 日放回抽 ns 条 / R2 近窗 12 全取。
        load_lags: L1 同星期几窗 W_L。
        load_halflife: L2 半衰期 H_L（单位：同星期几步）。
        harm_order: L3 谐波阶数 n_h。
        shape_win: L3 / P3 形态窗 W_sh。
        pv_window: P1 近窗 W_R。
        pv_long_window: P2 长窗 W_R^L（< 一季度）。
        pv_ratio_window: P2 持续性比值窗 W_ρ。
        pv_ratio_clip: P2 比值截断 [ρ_lo, ρ_hi]。
        pv_fit_window: P3 谐波拟合窗（0 = 全部可得历史）。
        resid_window: R1 残差窗。
        ns: R1 抽样情景数 N_s。
        seed: 随机种子。
    """

    load: str = "L1"
    pv: str = "P1"
    resid: str = "R2"
    load_lags: int = 4
    load_halflife: float = 8.0
    harm_order: int = 2
    shape_win: int = 7
    pv_window: int = 7
    pv_long_window: int = 84
    pv_ratio_window: int = 7
    pv_ratio_clip: tuple[float, float] = (0.5, 1.2)
    pv_fit_window: int = 0
    resid_window: int = 28
    ns: int = 25
    seed: int = 0


def forecast_load(cfg: ForecastConfig, year: YearData, j: int,
                  typ: np.ndarray, hist_end: int | None = None) -> np.ndarray:
    """第 j 日（0 基）负载点预报，只用前 hist_end 日历史与日历。

    Args:
        cfg: 预报层配置。
        year: 全年数据（本函数只切片 [:hist_end]）。
        j: 0 基日索引（预报目标）。
        typ: 附件1 典型日负载曲线，无历史时的先验。
        hist_end: 历史截止（不含）；缺省 None 表示截至前 j 日（= j）。
            预报 j+1 而信息只到 j−1 时传 hist_end=j（ν 锚实验 V2b 用）。

    Returns:
        点预报 ℓ̂，shape (144,)。

    Raises:
        ValueError: 候选编号不在 {L1, L2, L3}。
    """
    he = j if hist_end is None else hist_end
    hist = year.ell[:he]

    # 无样本（day 0）：退化为附件 1 典型日先验
    if len(hist) == 0:
        return typ.copy()

    # L1：ℓ̂ = 近 W_L 个同星期几日的同点均值
    if cfg.load == "L1":
        idx = np.flatnonzero(year.dow[:he] == year.dow[j])[-cfg.load_lags:]
        return hist[idx].mean(axis=0) if len(idx) else hist.mean(axis=0)

    # L2：同星期几 EWMA——权重 w = 0.5^(steps/H_L)，steps = 距今同星期几步数
    if cfg.load == "L2":
        idx = np.flatnonzero(year.dow[:he] == year.dow[j])
        if len(idx) == 0:
            return hist.mean(axis=0)
        steps = np.arange(len(idx) - 1, -1, -1, dtype=float)  # 距今同星期几步数
        w = 0.5 ** (steps / cfg.load_halflife)
        return (hist[idx] * w[:, None]).sum(axis=0) / w.sum()

    # L3：对数线性回归总量 × 近窗归一化形态
    if cfg.load == "L3":
        return _load_regression(cfg, hist, year.dow[:he], year.doy[:he],
                                year.dow[j], year.doy[j])
    raise ValueError(f"未知负载候选: {cfg.load}")


def _load_regression(cfg: ForecastConfig, hist: np.ndarray,
                     dow: np.ndarray, doy: np.ndarray,
                     dow_j: int, doy_j: float) -> np.ndarray:
    """L3：全历史日总量对数线性回归（星期哑变量 + 年谐波同场），点预报 =
    预测总量 × 近窗归一化形态（文档 §3.1）。样本不足参数个数时退化为
    全历史同点均值。"""
    y = hist.sum(axis=1)
    n_par = 7 + 2 * cfg.harm_order

    # 样本不足参数个数：退化为全历史同点均值
    if len(hist) < n_par + 1:
        return hist.mean(axis=0)

    # 对数线性回归：log y = 设计矩阵·β + ε（同场最小二乘）
    x = _dow_harm_design(dow, doy, cfg.harm_order)
    beta, *_ = np.linalg.lstsq(x, np.log(y), rcond=None)

    # 总量预测 ŷ = exp(row·β)——对数可加 = 乘法分解（基准 × 星期效用）
    row = _dow_harm_design(dow_j, doy_j, cfg.harm_order)[0]
    total = float(np.exp(row @ beta))

    # 点预报 ℓ̂ = ŷ × s，s = 近窗形态（Σ_t s_t = 1）
    shape = hist[-cfg.shape_win:].mean(axis=0)
    return total * shape / shape.sum()


def _dow_harm_design(dow, doy, nh: int) -> np.ndarray:
    """回归设计矩阵：截距 + 星期哑变量（周一为基准，留一）+ 1..nh 阶年谐波。

    Args:
        dow: 星期几数组（或标量）。
        doy: 年内日序数组（或标量）。
        nh: 谐波阶数。

    Returns:
        设计矩阵，shape (n, 7 + 2*nh)。
    """
    dow = np.atleast_1d(dow)
    doy = np.atleast_1d(doy)

    # 截距 + 星期哑变量（周一为基准，留一防与截距共线）
    cols = [np.ones(len(dow))]
    for sigma in range(1, 7):
        cols.append((dow == sigma).astype(float))

    # 1..nh 阶年谐波：cos/sin(2πm·doy/365)
    ang = 2.0 * np.pi * doy / 365.0
    for m in range(1, nh + 1):
        cols.append(np.cos(m * ang))
        cols.append(np.sin(m * ang))
    return np.column_stack(cols)


def forecast_pv(cfg: ForecastConfig, year: YearData, j: int,
                typ: np.ndarray, hist_end: int | None = None) -> np.ndarray:
    """第 j 日（0 基）光伏点预报，只用前 hist_end 日历史与日历。

    Args:
        cfg: 预报层配置。
        year: 全年数据（本函数只切片 [:hist_end]）。
        j: 0 基日索引（预报目标）。
        typ: 附件1 典型日光伏曲线，无历史时的先验。
        hist_end: 历史截止（不含）；缺省 None 表示截至前 j 日（= j）。
            预报 j+1 而信息只到 j−1 时传 hist_end=j（ν 锚实验 V2b 用）。

    Returns:
        点预报 r̂，shape (144,)。

    Raises:
        ValueError: 候选编号不在 {P1, P2, P3}。
    """
    he = j if hist_end is None else hist_end
    hist = year.r[:he]

    # 无样本（day 0）：退化为附件 1 典型日先验
    if len(hist) == 0:
        return typ.copy()

    # P1：r̂ = 近 W_R 日同点均值
    if cfg.pv == "P1":
        return hist[-cfg.pv_window:].mean(axis=0)

    # P2：长窗基底 × 持续性比值 ρ = Σrecent/Σbase（截断 [ρ_lo, ρ_hi]）
    if cfg.pv == "P2":
        base = hist[-cfg.pv_long_window:].mean(axis=0)
        recent = hist[-cfg.pv_ratio_window:]
        den = len(recent) * base.sum()
        ratio = recent.sum() / den if den > 1e-9 else 1.0
        return base * np.clip(ratio, *cfg.pv_ratio_clip)

    # P3：日总量一阶年谐波 × 近窗归一化形态
    if cfg.pv == "P3":
        return _pv_harmonic(cfg, hist, year.doy[:he], year.doy[j])
    raise ValueError(f"未知光伏候选: {cfg.pv}")


def _pv_harmonic(cfg: ForecastConfig, hist: np.ndarray,
                 doy: np.ndarray, doy_j: float) -> np.ndarray:
    """P3：日总量一阶年谐波最小二乘，点预报 = 预测总量 × 近窗归一化形态
    （文档 §3.2）。拟合样本不足 3 个时退化为近窗均值。"""
    sub = hist[-cfg.pv_fit_window:] if cfg.pv_fit_window else hist
    y = sub.sum(axis=1)

    # 拟合样本不足 3 个：退化为近窗均值
    if len(sub) < 3:
        return hist[-cfg.shape_win:].mean(axis=0)

    # 一阶年谐波回归：log y = a + b·cos(2π·doy/365) + c·sin(2π·doy/365)
    ang = 2.0 * np.pi * doy[-len(sub):] / 365.0
    a = np.column_stack([np.ones(len(sub)), np.cos(ang), np.sin(ang)])
    beta, *_ = np.linalg.lstsq(a, np.log(np.maximum(y, 1e-9)), rcond=None)

    # 总量预测与形态：r̂ = ŷ × s（Σ_t s_t = 1；形态塌缩时均匀铺开）
    ang_j = 2.0 * np.pi * doy_j / 365.0
    total = float(np.exp(np.array([1.0, np.cos(ang_j), np.sin(ang_j)]) @ beta))
    shape = hist[-cfg.shape_win:].mean(axis=0)
    ssum = shape.sum()
    return total * shape / ssum if ssum > 1e-9 else np.full(T, total / T)


def build_scenarios(cfg: ForecastConfig, year: YearData,
                    fcasts: list[tuple[np.ndarray, np.ndarray]], j: int,
                    ell_hat: np.ndarray, r_hat: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """残差场景族（文档 §3.3）：把伪样本外残差日曲线整条贴到当日点预报。

    残差定义：负载加性 ε^L = ℓ − ℓ̂；光伏相对 ε^R = (r − r̂)/max(r̂, r_ε)。
    贴加约定：ℓ^(s) = [ℓ̂ + ε^L]_+，r^(s) = [r̂(1 + ε^R)]_+，等权。

    Args:
        cfg: 预报层配置。
        year: 全年数据。
        fcasts: 历史各日当时的点预报 (ℓ̂, r̂)，长度应为 j（滚动时逐日追加）。
        j: 0 基日索引。
        ell_hat / r_hat: 当日点预报。
        rng: 固定种子的随机源（R1 抽样）。

    Returns:
        净负荷场景 n^(s)，shape (N_s, 144)。
    """
    # day 0 无残差样本：场景退化 N_s = 1（点预报本身）
    if j == 0:
        return (ell_hat - r_hat)[None, :]

    # 残差池：R1 近 28 日带放回抽 ns 条 / R2 近窗 12 全取
    if cfg.resid == "R1":
        pool = np.arange(max(0, j - cfg.resid_window), j)
        days = rng.choice(pool, size=cfg.ns, replace=True)
    elif cfg.resid == "R2":
        days = np.arange(max(0, j - 12), j)
    else:
        raise ValueError(f"未知残差配置: {cfg.resid}")
    nets = np.empty((len(days), T))
    for row, i in enumerate(days):
        # 残差：ε^L = ℓ − ℓ̂（加性）；ε^R = (r − r̂)/max(r̂, r_ε)（相对）
        ell_i, r_i = fcasts[int(i)]
        eps_l = year.ell[i] - ell_i
        eps_r = (year.r[i] - r_i) / np.maximum(r_i, R_EPS)
        # 贴加：ℓ^(s) = [ℓ̂+ε^L]_+，r^(s) = [r̂(1+ε^R)]_+，n^(s) = ℓ^(s) − r^(s)
        nets[row] = (np.maximum(ell_hat + eps_l, 0.0)
                     - np.maximum(r_hat * (1.0 + eps_r), 0.0))
    return nets


def reference_path(scn: np.ndarray, alpha: float) -> np.ndarray:
    """确定性等价 MPC 的参考路径：逐时段情景净负荷的 α 分位（文档 §7.3）。"""
    return np.quantile(scn, alpha, axis=0)
