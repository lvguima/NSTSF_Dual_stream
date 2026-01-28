# Implementation TODO v4 — Adaptive Dual‑Stream + Decomp Gate + Learnable Alpha + Factor Mixing

> 指导原则：  
> - **一次只验证一个关键新增模块**（先能退化到 anchor，再开功能）。  
> - 实验不做过密网格：对关键超参只测 **少量、间距大的点**。  
> - 每一步都要保留 **on/off 对照**，以便快速判断“值不值得继续”。  
> - 仍按你现有口径：单 seed，指标记录 MSE / MAE + 必要诊断量。

---

## 0. Anchor Baseline（必须先锁定）

### 0.1 选择 v4 的 anchor（不含图相关模块）
以你 B0–B5 消融里“**Dual‑Stream only**”为 v4 anchor（对应 B3）：

- `decomp_mode=ema`
- `decomp_alpha=0.1`
- `trend_head=linear`
- `trend_head_share=1`
- **关闭图传播**：`gate_init=-20`（或等价开关）
- `graph_map_norm=none`（即便代码里存在也不启用）
- `use_norm=1`
- 其余训练超参保持与 v3 一致（seq/pred=96/96，TCN，d_model=128，15 epochs …）

> 说明：你的最新消融已经表明 graph propagation/SMGP 的边际收益极小，因此 v4 的对照不再以 B5 为 anchor，而以 B3 作为最干净的“无图基线”。

### 0.2 Anchor 复现实验（必做）
至少在 3 个代表数据集复现一次（单 seed）：

- **ETTm1**（DS 有益、快速）
- **Weather**（中通道、DS 有益）
- **national_illness 或 flotation**（DS 可能退化，用于验证 Decomp Gate 的必要性）

输出必须包含：
- MSE / MAE
- （如果 DS）trend/res 能量比统计
- 日志文件与命令行配置快照

---

## 1. v4.0 — FactorMix 模块落地（先做到“可退化”）

### 1.1 代码任务：新增 `FactorMixer`
建议放置：`models/modules/factor_mixer.py`（或整合进现有 `mixer.py`）

**输入/输出约定**
- 输入：`H ∈ [B, C, N, D]`
- 输出：`H_out ∈ [B, C, N, D]`

**核心公式**
- `A = RowSoftmax( (P @ Q^T) / sqrt(r) )`
- `alpha = sigmoid(a)`（标量）
- `H_out = H + alpha * (A @ H)`（在变量维做 matmul）

### 1.2 回归测试（必须）
- 当 `alpha_init` 设为很小（如 `a=-8`，使 `alpha≈0`）时：
  - v4.0 的输出必须与 anchor 的 season 分支几乎一致（数值差异应只来自浮点误差）
- 写一个最小 shape test：不同 `C/N/D` 都能跑通

### 1.3 日志（必须加）
- `alpha_mix = sigmoid(a)`
- `A_entropy`（对每行算熵再取均值）
- `||A-I||_F`

---

## 2. v4.1 — 验证 FactorMix 是否值得保留（少量大间距实验）

> 目的：快速判断“静态低秩混合”能否在不引入图传播风险的情况下提供增益。

### 2.1 实验设计（每个数据集只做 3 个点）
固定所有训练设置 = anchor，仅改变 FactorMix 的 `alpha_init`。

- **E1（对照）**：anchor（无 FactorMix 或 `alpha≈0`）
- **E2（弱混合）**：`alpha_init=-4`（`alpha≈0.018`）
- **E3（中等混合）**：`alpha_init=-2`（`alpha≈0.119`）

> 不建议直接测 `alpha_init=0`（`alpha=0.5`）作为第一轮，因为强混合更容易把模型带崩，除非你特别想看“上限/灾难点”。

### 2.2 数据集（建议与 0.2 相同）
- ETTm1
- Weather
- national_illness 或 flotation

### 2.3 通过/停止规则
- 若 E2/E3 在“DS 有益”的数据集上有一致改善趋势（哪怕只有 0.001 级别），并且 `alpha_mix` 不坍缩到 0：进入 v4.2。
- 若所有数据集上都无改善且 `alpha_mix` 训练后趋近 0：FactorMix 可能不值，后续可把重点放在 Decomp Gate（仍继续 v4.2）。

---

## 3. v4.2 — Decomp Gate 落地（核心：让 DS 具备自动回退能力）

### 3.1 代码任务：新增 `DecompGate`
建议放置：`models/modules/decomp_gate.py`

**输入（固定）**  
从输入窗口计算每个变量的统计量：
- `e_res = mean_t |x_res|`
- `e_diff = mean_t |x_t - x_{t-1}|`
- `rho = e_res / (e_res + e_diff + eps)`
- 送入：`[rho, log(e_res+eps), log(e_diff+eps)]`

**网络结构（固定）**
- MLP：`3 → 16 → 1`，GELU + sigmoid
- 输出：`g ∈ [B, C]`

**融合方式（固定）**
- `Y = Y_trend + g ⊙ Y_res`（广播到 pred_len）

### 3.2 必须做的 sanity 对照（3 个点）
在 **不引入 FactorMix** 的情况下先验证 gate 的必要性（避免耦合干扰）。

- **G0（对照）**：原始 DS：`g ≡ 1`（等价 v3 Dual‑Stream 相加）
- **G1（趋势上限）**：`g ≡ 0`（trend-only，帮助理解 DS 在该数据集是否本就不适合）
- **G2（可学习 gate）**：开启 DecompGate（默认 bias 让初始 `g` 稍偏大，例如 `b_init=+2`）

### 3.3 观察点（非常关键）
- 在 **DS 会退化**的数据集上（national_illness / flotation）：
  - 期望 `g` 会显著小于 1（甚至接近 0），从而使性能接近 G1 或至少显著缓解退化。
- 在 **DS 有益**的数据集上（ETTm1 / Weather）：
  - 期望 `g` 不会塌缩到 0（否则说明 gate 学到了错误策略）。

> 如果 G2 在 DS 有益的数据集上把 `g` 压到很小，通常意味着：  
> - gate 的输入特征不够区分（需要更好的非平稳度量）；或  
> - gate 初始化太保守（bias 太小）；或  
> - trend head 太强导致 residual 分支被忽略（需要检查 trend/res 能量比）。

---

## 4. v4.3 — 组合：FactorMix + DecompGate（最终 v4 结构）

> 只有当 v4.1 / v4.2 都至少“不明显退化”时再做组合，否则先修单模块。

### 4.1 实验矩阵（保持轻量，但足够判断交互）
对每个数据集做 3 个实验即可：

- **C1**：DecompGate only（v4.2 的 G2）
- **C2**：FactorMix only（v4.1 的最佳 alpha_init）
- **C3**：DecompGate + FactorMix（v4.3 full）

### 4.2 判定逻辑
- 若 C3 明显优于 C1/C2：说明两者互补，v4 方向成立。
- 若 C3 ≈ C1 且 `alpha_mix→0`：说明 cross‑var 混合不重要，v4 主要贡献来自“DS 自适应回退”。
- 若 C3 退化且 `alpha_mix` 变大：说明 FactorMix 学到了有害耦合，需要加强正则或降低 rank。

---

## 5. v4.4 — 必要的正则与 rank（只做最少验证）

### 5.1 加入 `||A-I||_F^2` 正则（只测 2 个间距大点）
在 v4.3 的 full 配置上：

- **R0**：`lambda_factor=0`
- **R1**：`lambda_factor=1e-3`

> 如果 R1 明显更稳或更好，就固定使用；否则保持 0，避免多余约束。

### 5.2 rank（只在你怀疑“过拟合/欠表达”时再做）
只测两档：

- `r = min(4, C)`
- `r = min(16, C)`

并且只在 **一个代表数据集**上测（建议 Weather 或你后续的 Traffic）。

---

## 6. 最终交付（v4 最终实验建议）

当你确定 v4.3 的 full 配置后，再做最终跑表：

- 数据集：你论文需要的全套（先不含 Traffic 也可以）
- 预测长度：按论文目标（96、192、336、720 等）
- 输出：一张总表 + gate/alpha 的统计可解释性图

最少需要保存：
- 每个数据集每个 horizon 的 MSE/MAE
- `alpha_mix` 的均值与分位数
- `decomp_gate` 的均值与分位数
- （可选）`A_entropy` 与 `||A-I||_F`（解释 FactorMix 行为）

---

## 7. 你可以直接用的 CLI 参数建议（草案）

你可以按你现有 `run.py` 风格新增这些参数（命名可按代码习惯微调）：

- Dual‑Stream：
  - `--decomp_mode ema`
  - `--decomp_alpha 0.1`
  - `--trend_head linear`
  - `--trend_head_share 1`

- Decomp Gate：
  - `--decomp_gate 1`
  - `--decomp_gate_hidden 16`
  - `--decomp_gate_bias_init 2`

- FactorMix：
  - `--factor_mix 1`
  - `--factor_rank 8`
  - `--factor_alpha_init -4`
  - `--factor_reg_lambda 1e-3`

- 日志：
  - `--log_decomp_gate 1`
  - `--log_factor_stats 1`

> 注意：为了让每一步都能做严格消融，你需要保留开关：  
> - `decomp_gate=0` 时强制 `g=1`  
> - `factor_mix=0` 时跳过 FactorMix 或强制 `alpha=0`

