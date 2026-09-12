"""论文图表包：style 为共享样式层（配色槽位 / 统一样式 / 图例 / 保存），
q1 / q2 为各问图类。

图类满足各自 report 模块的导出协议（q1: export(data, sol, out_dir)；
q2: export(days, year, p, out_dir)），与表格导出器并列挂进 CLI 导出链。
"""
