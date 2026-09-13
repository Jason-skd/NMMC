"""题2 预报层：点预报候选（L1/L2/L3、P1/P2/P3）与残差场景族（R1/R2）。

候选制（文档 §3）：负载与光伏分开建模再相减得净负荷预报（统计结构不同）；
误差场景为伪样本外残差日曲线整条贴加，相关结构自动保留。全部只用
F_0^(j)（截至昨日的实际值 + 日历）；随机性固定种子。样本不足时用可得
样本，完全无样本（day 0）退化为附件 1 典型日先验，残差为 0 时场景退化
N_s=1（保守暖机的最简实现）。

题4 扩展：build_scenarios 接受可选电价输入时返回 (nets, prices) 二元组
——价格残差 ε^P 与 (ε^L, ε^R) 取自同一历史日整条贴加（三元联合，题4 档
§4.2），prices 为 None 时即常数价口径（题2，逐位不变）。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.data.price import YearPrice
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
        shape = hist[-cfg.shape_win:].mean(axis=0)
        return _load_regression(cfg, hist, year.dow[:he], year.doy[:he],
                                year.dow[j], year.doy[j], shape)
    raise ValueError(f"未知负载候选: {cfg.load}")


def _load_regression(cfg: ForecastConfig, hist: np.ndarray,
                     dow: np.ndarray, doy: np.ndarray,
                     dow_j: int, doy_j: float,
                     shape: np.ndarray) -> np.ndarray:
    """L3：全历史日总量对数线性回归（星期哑变量 + 年谐波同场），点预报 =
    预测总量 × 归一化形态（文档 §3.1；形态由调用方给定——交付为近窗
    W_sh 均值，零参数变体为同周几全历史均值）。样本不足参数个数时退化
    为全历史同点均值。"""
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

    # 点预报 ℓ̂ = ŷ × s，s = 归一化形态（Σ_t s_t = 1）
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
        shape = hist[-cfg.shape_win:].mean(axis=0)
        return _pv_harmonic(cfg, hist, year.doy[:he], year.doy[j], shape)
    raise ValueError(f"未知光伏候选: {cfg.pv}")


def _pv_harmonic(cfg: ForecastConfig, hist: np.ndarray,
                 doy: np.ndarray, doy_j: float,
                 shape: np.ndarray) -> np.ndarray:
    """P3：日总量一阶年谐波最小二乘，点预报 = 预测总量 × 归一化形态
    （文档 §3.2；形态由调用方给定——交付为近窗 W_sh 均值，零参数变体
    为全历史均值）。拟合样本不足 3 个时退化为该形态。"""
    sub = hist[-cfg.pv_fit_window:] if cfg.pv_fit_window else hist
    y = sub.sum(axis=1)

    # 拟合样本不足 3 个：退化为调用方形态
    if len(sub) < 3:
        return shape

    # 一阶年谐波回归：log y = a + b·cos(2π·doy/365) + c·sin(2π·doy/365)
    ang = 2.0 * np.pi * doy[-len(sub):] / 365.0
    a = np.column_stack([np.ones(len(sub)), np.cos(ang), np.sin(ang)])
    beta, *_ = np.linalg.lstsq(a, np.log(np.maximum(y, 1e-9)), rcond=None)

    # 总量预测与形态：r̂ = ŷ × s（Σ_t s_t = 1；形态塌缩时均匀铺开）
    ang_j = 2.0 * np.pi * doy_j / 365.0
    total = float(np.exp(np.array([1.0, np.cos(ang_j), np.sin(ang_j)]) @ beta))
    ssum = shape.sum()
    return total * shape / ssum if ssum > 1e-9 else np.full(T, total / T)


def pool_days(cfg: ForecastConfig, j: int,
              rng: np.random.Generator) -> np.ndarray:
    """残差池日索引：R1 近 resid_window 日带放回抽 ns 条 / R2 近窗 12 全取。

    day 0 无残差样本，返回空数组（场景退化 N_s = 1 由调用侧处理）。
    """
    if j == 0:
        return np.empty(0, dtype=int)
    if cfg.resid == "R1":
        return rng.choice(np.arange(max(0, j - cfg.resid_window), j),
                          size=cfg.ns, replace=True)
    if cfg.resid == "R2":
        return np.arange(max(0, j - 12), j)
    raise ValueError(f"未知残差配置: {cfg.resid}")


def load_pv_residuals(year: YearData,
                      fcasts: list[tuple[np.ndarray, np.ndarray]],
                      i: int) -> tuple[np.ndarray, np.ndarray]:
    """第 i 日伪样本外残差：ε^L = ℓ − ℓ̂（加性）、ε^R = (r − r̂)/max(r̂, r_ε)（相对）。"""
    ell_i, r_i = fcasts[int(i)]
    return (year.ell[i] - ell_i,
            (year.r[i] - r_i) / np.maximum(r_i, R_EPS))


def attach(ell_hat: np.ndarray, r_hat: np.ndarray, eps_l: np.ndarray,
           eps_r: np.ndarray) -> np.ndarray:
    """残差贴加：ℓ^(s) = [ℓ̂+ε^L]_+，r^(s) = [r̂(1+ε^R)]_+，返回净负荷。"""
    return (np.maximum(ell_hat + eps_l, 0.0)
            - np.maximum(r_hat * (1.0 + eps_r), 0.0))


def price_residual(price: YearPrice, p_fcasts: list[np.ndarray],
                   i: int) -> np.ndarray:
    """第 i 日电价伪样本外相对残差 ε^P = (p − p̂)/p̂（题4 档 §4.2）。

    p̂ > 0 恒成立（附件4 无非正值、预报为正均值），无需防除零。
    """
    return (price.p[i] - p_fcasts[i]) / p_fcasts[i]


def decouple_price_days(days: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """把残差日索引独立打乱：专用 rng，不消耗情景主 rng。

    联合抽样（默认）用同一历史日贴加 (ε^L, ε^R, ε^P)；独立抽样保持
    负载/光伏残差日不变，只把 ε^P 的来源日在同池内置换——性质 2
    推论 2 的直接消融。salt 编码 (j, m)，两线/各阶段互不撞种子。
    """
    src = np.asarray(days)
    if src.size == 0:
        return src
    return np.random.default_rng(seed * 1_000_003 + salt).permutation(src)


def build_scenarios(cfg: ForecastConfig, year: YearData,
                    fcasts: list[tuple[np.ndarray, np.ndarray]], j: int,
                    ell_hat: np.ndarray, r_hat: np.ndarray,
                    rng: np.random.Generator, *, price: YearPrice | None = None,
                    p_fcasts: list[np.ndarray] | None = None,
                    p_hat: np.ndarray | None = None,
                    couple: bool = True
                    ) -> tuple[np.ndarray, np.ndarray | None]:
    """残差场景族（文档 §3.3）：把伪样本外残差日曲线整条贴到当日点预报。

    贴加约定：ℓ^(s) = [ℓ̂ + ε^L]_+，r^(s) = [r̂(1 + ε^R)]_+，等权。
    给定价格输入时贴加电价段 ε^P（题4 档 §4.2）：p^(s) = [p̂(1 + ε^P)]_+。
    couple=True（默认）时 (ε^L, ε^R, ε^P) 取自同一历史日；False 时 ε^P
    在同池内独立置换（性质 2 推论 2 消融）。

    Args:
        cfg: 预报层配置。
        year: 全年数据。
        fcasts: 历史各日当时的点预报 (ℓ̂, r̂)，长度应为 j（滚动时逐日追加）。
        j: 0 基日索引。
        ell_hat / r_hat: 当日点预报。
        rng: 固定种子的随机源（R1 抽样）。
        price: 附件4 全年实现电价（价格残差真值源；题4 口径）。
        p_fcasts: 历史各日当时的电价点预报 p̂_i（与 fcasts 同步滚动）。
        p_hat: 当日电价点预报。
        couple: True = 三元联合（默认）；False = 价格残差独立抽样。

    Returns:
        (nets, prices)：净负荷场景 n^(s)，shape (N_s, 144)；电价场景
        p^(s)，常数价口径（未给价格输入）为 None。
    """
    # day 0 无残差样本：场景退化 N_s = 1（点预报本身）
    days = pool_days(cfg, j, rng)
    if not len(days):
        prices = None if p_hat is None else p_hat[None, :].copy()
        return (ell_hat - r_hat)[None, :], prices

    nets = np.empty((len(days), T))
    prices = (np.empty((len(days), T))
              if p_hat is not None else None)
    p_days = (days if couple or prices is None
              else decouple_price_days(days, cfg.seed, j))
    for row, i in enumerate(days):
        eps_l, eps_r = load_pv_residuals(year, fcasts, int(i))
        nets[row] = attach(ell_hat, r_hat, eps_l, eps_r)
        if prices is not None:
            # 联合：ε^P 与 (ε^L, ε^R) 同日；独立：同池置换后的残差日
            eps_p = price_residual(price, p_fcasts, int(p_days[row]))
            prices[row] = np.maximum(p_hat * (1.0 + eps_p), 0.0)
    return nets, prices


def reference_path(scn: np.ndarray, alpha: float) -> np.ndarray:
    """确定性等价 MPC 的参考路径：逐时段情景净负荷的 α 分位（文档 §7.3）。"""
    return np.quantile(scn, alpha, axis=0)
