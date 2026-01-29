# Implementation TODO v5 — Dual‑Stream + Bridge Coupling Attention（Stable QK × Content V）

> 原则：  
> - **尽量少的实验，快速判断核心模块是否有效**。  
> - 每一步都保留 “可退化” 开关（至少通过 init 让效果≈关闭），这样任何退化都能快速定位。  
> - 参数只测 **少量、间距大** 的点（避免网格爆炸）。  
> - 指标：MSE / MAE + 必要诊断（alpha / gate / entropy / topk_mass / adj_diff）。

---

## 0. Anchor 选择（v5 的对照基线）

建议把 v5 的 anchor 定义为：

- **Dual‑Stream（EMA 0.1） + Trend linear(shared) + Season TemporalEncoder + Head**
- 不启用跨变量耦合（Bridge Coupling 关闭）
- 不启用 Decomp Gate（等价 g=1）或让 gate 初始化为接近 1（见 3.2）

也就是你现有实验体系中的 “DS only” 最干净对照（对应你 v4 的 B3/Anchor）。

> 为什么 anchor 不是带 FactorMix 的？  
> 因为 v5 的目标是验证 **Bridge Coupling** 是否能替代/超越 FactorMix；  
> 所以对照最好是“无跨变量模块”的干净基线，否则难判断收益来源。

---

## 1. v5.0 — 代码落地：Bridge Coupling 模块（先做到“可退化”）

### 1.1 新增模块文件
新增 `modules/bridge_coupler.py`（或你现有 `modules/` 目录风格下的新文件），包含 3 个子模块：

1) `StableTokenDetrend(window)`  
- 输入：`H [B,C,N,D]`  
- 输出：`H_stable [B,C,N,D]`  
- 实现：token 维 avgpool detrend（replicate padding）

2) `StatsFiLM(d_model, hidden=16)`  
- 输入：`mu [B,C]`, `sigma [B,C]`, `H [B,C,N,D]`  
- 输出：`H_content [B,C,N,D]`  
- 输出公式：`H * (1+gamma) + beta`（gamma/beta 通过 MLP 产生，broadcast）

3) `BridgeCouplingAttention(d_model, rank, scale, topk, alpha_init)`  
- 输入：`H_stable, H_content`  
- 输出：`H_out` + 日志：`A_entropy, adj_diff, topk_mass, alpha`
- 内部：按 scale 分段 → 池化得 `z_k` → 低秩 Q/K → row-softmax 得 A_k → (A_k @ H_content_seg) 注入

### 1.2 与现有模型的集成点（Residual 分支）
在 residual/season 分支里，TemporalEncoder 得到 `H` 后：

- `H_stable = detrend(H)`
- `H_content = film(mu,sigma,H)`（先做出来，后面再决定是否保留）
- `H_out = bridge_coupler(H_stable, H_content)`
- `Y_res = head(H_out)`

> 注意：BridgeCoupler 的输出 token 形状必须与 head 期待一致（仍是 `[B,C,N,D]`）。

### 1.3 “可退化”要求（必做）
为了保证回归测试与快速定位：

- 把 `alpha_init` 设为一个很小的初值（例如 -20），确保 $$\alpha \approx 0$$  
  这时 `H_out ≈ H_content`，BridgeCoupling 等价关闭。
- FiLM 初始要接近恒等：`gamma≈0, beta≈0`  
  最简单：MLP 最后一层权重初始化为 0，bias 初始化为 0。

---

## 2. v5.1 — 最小回归实验：确认“加了模块但关掉时不变”

### 2.1 实验（每个数据集 1 次即可）
- **E0**：Anchor（原来 DS-only，未引入 BridgeCoupler 的旧实现）
- **E1**：v5 新实现，但 `alpha_init=-20`（BridgeCoupler 关闭），FiLM 恒等，Decomp Gate 关闭/恒等

数据集建议（覆盖你最关键的三类现象）：
- ETTm1（DS 强收益）
- weather（跨变量可能有效）
- national_illness（DS 会退化的典型）

验收标准：
- E1 与 E0 的 MSE/MAE 差异应非常小（比如 ≤ 0.001 量级，允许随机性浮动）
- `alpha_mean≈0`，`A_entropy` 仍可算但对输出无影响

---

## 3. v5.2 — 核心验证：Bridge Coupling 是否值得保留？

### 3.1 只开 Bridge Coupling（FiLM 先保持恒等）
固定其它训练超参 = anchor。

做 2~3 个强度点（间距大）：

- **E2**：`alpha_init=-6`（非常弱注入）
- **E3**：`alpha_init=-4`（弱注入，推荐默认）
- **E4**：`alpha_init=-2`（中等注入，用于观察是否开始污染）

观察点：
- 性能是否提升（至少在 weather / flotation 这类“耦合可能有用”的数据集上出现趋势）
- `alpha_mean` 是否学到非零且稳定
- `A_entropy` 是否过低（过低容易过拟合）或过高（几乎平均搅拌）

快速决策规则：
- 若 E3 相比 anchor **稳定不差**，且至少在 1~2 个数据集上有提升趋势 → 进入 3.2  
- 若 E4 明显变差、且 alpha 学很大 → 说明耦合注入过强，默认就用 -4/-6，不再尝试更强。

### 3.2 再开 FiLM（把“Content V”补齐）
在 3.1 的最优 alpha_init 上，只做 on/off：

- **E5**：FiLM 恒等（对照，等价不用 stats）
- **E6**：FiLM 启用（用 mu/logsigma 产生 gamma/beta）

如果 E6 优于 E5（哪怕只在若干数据集上），说明 “Content V 带非平稳上下文” 是有效机制；  
否则可以把 FiLM 砍掉，BridgeCoupling 退化成 “Stable QK × V=H” 的形式（仍然有价值）。

---

## 4. v5.3 — 加入 Decomp Gate（解决 DS 会退化的数据集）

> 这一步的目标非常明确：  
> 在 national_illness / flotation 这类 DS 会退化的数据集上，让模型自动回退到“更像 non‑DS”的行为。

### 4.1 gate 实现（与 v4 一致即可）
- 输入统计：`e_res, e_delta, rho`（按变量）  
- 小 MLP：`3→16→1` 输出 `g [B,C]`  
- 融合：`Y = Y_trend + g ⊙ Y_res`

### 4.2 实验（只做 2 个）
基于你在 3.x 得到的最优 BridgeCoupling 设置：

- **E7**：Decomp Gate 关闭（g=1，等价 DS-sum）
- **E8**：Decomp Gate 启用（learnable）

关注：
- 在 DS 会退化的数据集（national_illness）是否能显著降低退化程度  
- 在 DS 有益的数据集（ETT/weather）是否不明显变差

> 如果 E8 在 national_illness 明显更好且其他数据集不掉点，那么 v5 的固定结构就可以把 Decomp Gate 作为“必选件”。

---

## 5. v5.4 — 固定最终结构（给后续大规模最终实验用）

当你完成 3~4 步的快速验证后，固定 v5 的默认配置：

- `decomp_alpha=0.1`
- `trend_head=linear, share=1`
- BridgeCoupling：
  - `rank=8`
  - `scale=8`
  - `topk=min(6,C)`
  - `alpha_init` 取你在 3.1 的冠军（大概率 -4 或 -6）
  - `stable_window=patch_len (16)`
  - FiLM：根据 3.2 的结论决定保留与否（若保留就作为固定结构的一部分）
- Decomp Gate：根据 4.x 的结论决定是否固定保留（若 national_illness 明显改善，建议固定保留）

---

## 6. 日志与可解释性（建议固定输出）

每次实验至少记录：

- `mse, mae`
- `alpha_mean`（耦合强度）
- `gate_mean`（分解门控）
- `A_entropy_mean/p50/p90`
- `topk_mass_mean`
- `adj_diff_mean`

并保留一小段样本的 `A_k`（例如每个 epoch 保存 1 个 batch 的 2 个 segment），方便你肉眼检查“是否在学结构”。

---

## 7. 最终你应该得到的结论形态（方便写论文）

- Bridge Coupling 的收益是否来自：
  - “稳定域路由”（stable detrend）  
  - “内容域 value”（FiLM 注入 stats）  
  - “长尺度耦合”（scale=8 的段级更新）
- Decomp Gate 是否能稳定解决 DS 在少数数据集上的退化

只要这两条证据链成立，v5 就值得作为下一阶段最终大实验的固定结构。  
如果 BridgeCoupling 在多数数据集都无收益且 alpha 学到 0，那么就说明当前任务下“跨变量耦合”不是瓶颈（或需要更强的外生/工况信息），后续应把精力转向 temporal backbone 或更强的输入上下文建模。

