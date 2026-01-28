# Model Design v4 — Adaptive Dual‑Stream + Decomp Gate + Learnable Alpha + Factor Mixing

> 目标：在**不依赖动态 GNN/动态图传播**的前提下，继续围绕“**非平稳**”这一核心动机做深挖，把 v3 里已经被验证最有效的 **Dual‑Stream 分解**做成更鲁棒、更可控、对不同数据集更“自适应”的最终主干。

---

## 0. 设计动机（来自现有结论）

从 v3 及后续消融现象我们已经看到两个关键信号（用你的实验口径表述）：

1) **Dual‑Stream（趋势/残差分解）是目前唯一稳定带来显著变化的主线能力**：在多数数据集上能显著降低误差；但也存在少数数据集会退化（说明“分解+相加”并不是普适最优策略，需要自适应控制）。
2) **图传播（graph propagation）与 SMGP 的边际收益很小**：在你最近的 DS 消融里，“几乎不做图传播”的结果与 baseline 几乎一致，甚至强传播会变差——这暗示当前设置下，跨变量关系建模不应再依赖“时变邻接+消息传递”的路径，而应换一种更稳、更低风险的 cross‑var 建模方式。

因此 v4 的主线是：

- **保留 Dual‑Stream 的“非平稳建模抓手”**；
- 用 **Decomp Gate** 把 “trend + season 直接相加” 改成“可学习、可解释的注入强度”，让模型在 DS 不适用的数据集上能自动回退；
- 用 **Factor Mixing** 取代动态图传播：用一个**静态/慢变、低秩、可控强度**的跨变量混合模块来建模变量间耦合（降低错边带来的系统性风险）；
- 用 **Learnable Alpha** 控制 Factor Mixing 的注入强度，让模型可以在“需要耦合”的数据集上放大作用，在“耦合会带来伪相关污染”的数据集上自动减弱。

---

## 1. 任务与记号

输入多变量序列：

$$
X \in \mathbb{R}^{B \times L \times C}
$$

输出未来 $$H$$ 步预测：

$$
\hat{Y} \in \mathbb{R}^{B \times H \times C}.
$$

其中：
- $$B$$：batch size
- $$L$$：输入长度（如 96）
- $$H$$：预测长度（如 96）
- $$C$$：变量数

---

## 2. v4 总体结构（单一固定架构）

v4 的 forward 可以概括为 5 步：

1. **可选归一化**（沿用你现有 pipeline 的 `use_norm=1` 逻辑）
2. **EMA Dual‑Stream 分解**：得到 trend 与 season/residual
3. **Trend 分支**：轻量线性头预测未来 trend
4. **Season 分支**：TemporalEncoder 提取残差的时序表征 + Factor Mixing 做跨变量混合 + ForecastHead 输出未来残差
5. **Decomp Gate 融合**：最终预测为 trend + gated residual

---

## 3. Adaptive Dual‑Stream：EMA 分解（固定形式）

沿用 v3 已验证有效的 EMA 分解（按变量、沿时间轴）：

**(1) 趋势分量**

$$
x^{\mathrm{trend}}_{t} = \alpha_{\mathrm{ema}} x_{t} + (1-\alpha_{\mathrm{ema}})\,x^{\mathrm{trend}}_{t-1}
$$

其中 $$\alpha_{\mathrm{ema}} \in (0,1)$$ 为分解超参。  
在 v4 默认推荐：$$\alpha_{\mathrm{ema}}=0.1$$（你在 v3 多数据集上的经验最稳区域）。

**(2) 残差/季节分量**

$$
x^{\mathrm{res}}_{t} = x_t - x^{\mathrm{trend}}_t
$$

> 直觉：trend 捕捉慢变非平稳（均值漂移/缓慢工况变化），res 捕捉更接近“弱平稳/近似平稳”的短期结构。

---

## 4. Trend 分支（固定）

采用 DLinear 风格的共享线性预测头：

对每个变量单独沿时间维做线性映射（参数可共享）：

$$
\hat{Y}^{\mathrm{trend}}_{:, :, c} = W_{\mathrm{trend}}\, X^{\mathrm{trend}}_{:, :, c} + b_{\mathrm{trend}}
$$

其中：

- $$W_{\mathrm{trend}} \in \mathbb{R}^{H \times L}$$
- $$b_{\mathrm{trend}} \in \mathbb{R}^{H}$$
- 默认：**跨变量共享**（与 v3 中 `trend_head_share=1` 的经验一致，通常更稳）

---

## 5. Season 分支：TemporalEncoder + Factor Mixing + ForecastHead

### 5.1 TemporalEncoder（变量内时序编码）

把残差序列输入现有 TemporalEncoder（如 TCN）得到 token 表征：

$$
H = f_{\mathrm{time}}(X^{\mathrm{res}}) \in \mathbb{R}^{B \times C \times N \times D}
$$

其中：
- $$N$$：token 数（不使用 patch 时 $$N=L$$；使用 patch 则为 token 数）
- $$D$$：隐空间维度（如 128）

> v4 不改变你当前的时序骨干（TCN/Transformer），只替换“跨变量耦合”的建模方式。

### 5.2 Factor Mixing（跨变量低秩混合）

我们引入一个**静态的、低秩的跨变量混合矩阵**，替代“动态图 + 消息传递”。

**(1) 低秩参数化**

学习两个因子矩阵：

$$
P \in \mathbb{R}^{C \times r}, \quad Q \in \mathbb{R}^{C \times r}
$$

构造 logits：

$$
S = \frac{1}{\sqrt{r}} P Q^\top \in \mathbb{R}^{C \times C}
$$

做行归一化得到混合权重（行随机矩阵）：

$$
A = \mathrm{RowSoftmax}(S)
$$

其中 $$A_{ij}$$ 表示“变量 $$j$$ 对变量 $$i$$ 的贡献权重”。

**(2) 对 token 表征做混合**

对每个 token 位置 $$n$$：

$$
\tilde{H}_{:, :, n, :} = A \, H_{:, :, n, :}
$$

即在变量维上做线性组合。

### 5.3 Learnable Alpha（Factor Mixing 注入强度）

为了让模型能在不同数据集上自适应“是否需要跨变量耦合”，我们为 Factor Mixing 引入一个可学习的强度系数：

$$
\alpha = \sigma(a) \in (0,1)
$$

最终输出采用残差式注入：

$$
H^{\mathrm{out}} = H + \alpha \cdot \tilde{H}
$$

- 当 $$\alpha \approx 0$$：Factor Mixing 基本关闭，season 分支退化为“纯 TemporalEncoder”；
- 当 $$\alpha$$ 学到较大：模型将显式利用跨变量耦合。

> 这一步是 v4 的关键：它把“跨变量模块”的风险变成**可学习的强度**，避免像动态图传播那样一旦错就持续污染。

### 5.4 ForecastHead（残差预测）

沿用你现有的 head（flatten tokens → linear），得到未来残差预测：

$$
\hat{Y}^{\mathrm{res}} = f_{\mathrm{head}}(H^{\mathrm{out}}) \in \mathbb{R}^{B \times H \times C}
$$

---

## 6. Decomp Gate：控制残差注入（自适应 Dual‑Stream）

在 v3 的 Dual‑Stream 中，最终输出是简单相加：

$$
\hat{Y} = \hat{Y}^{\mathrm{trend}} + \hat{Y}^{\mathrm{res}}
$$

但从多数据集消融看，这会导致：
- DS 在某些数据集上非常有效；
- 但在另一些数据集上可能退化（说明残差分支会“带偏”）。

因此 v4 引入 **Decomp Gate**，把残差分支当作“可控增量”注入：

$$
\hat{Y} = \hat{Y}^{\mathrm{trend}} + g \odot \hat{Y}^{\mathrm{res}}
$$

其中 gate：

$$
g \in (0,1)^{B \times C}
$$

并广播到预测步：

$$
g \odot \hat{Y}^{\mathrm{res}} \Rightarrow \text{broadcast to } \mathbb{R}^{B \times H \times C}.
$$

### 6.1 gate 的输入（固定：能量/非平稳强度统计）

为了让 gate 的行为可解释且不引入过多自由度，我们使用输入窗口上的简洁统计特征：

- 残差能量：$$e_{\mathrm{res}} = \mathrm{Mean}_t \left|x^{\mathrm{res}}_t\right|$$  
- 趋势变化强度：$$e_{\Delta} = \mathrm{Mean}_t \left|x_t - x_{t-1}\right|$$  
- 归一化比例：  
  $$
  \rho = \frac{e_{\mathrm{res}}}{e_{\mathrm{res}} + e_{\Delta} + \epsilon}
  $$

对每个变量得到 $$\rho \in [0,1]$$。

### 6.2 gate 的形式（固定：小 MLP）

对每个变量做一个轻量 MLP：

$$
g = \sigma\left(\mathrm{MLP}([\rho, \log(e_{\mathrm{res}}+\epsilon), \log(e_{\Delta}+\epsilon)])\right)
$$

建议 MLP 结构：`3 → 16 → 1`（GELU + sigmoid）。

> 这样 gate 在“残差主导的相对平稳段”倾向更大，在“强非平稳/趋势主导”时倾向更小，从而自动调节 DS 的有效性。

---

## 7. 训练目标与正则（固定）

主损失：预测 MSE（或你现有的 loss 定义）：

$$
\mathcal{L}_{\mathrm{pred}} = \mathrm{MSE}(\hat{Y}, Y)
$$

为避免 Factor Mixing 学成“全平均搅拌”，加一个保守正则（推荐很小权重）：

$$
\mathcal{L}_{\mathrm{factor}} = \left\|A - I\right\|_F^2
$$

总损失：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{pred}} + \lambda_{\mathrm{factor}} \mathcal{L}_{\mathrm{factor}}
$$

默认建议：$$\lambda_{\mathrm{factor}} \in \{10^{-4}, 10^{-3}\}$$。

---

## 8. 诊断日志（必须记录）

为了判断 v4 是否“真的在发挥作用”，建议最少记录：

- `alpha_mix = sigmoid(a)`（FactorMix 强度）
- `decomp_gate_mean / p10 / p50 / p90`
- `A_entropy`：行熵（观察 FactorMix 是否退化成均匀）
- `||A - I||_F`：观察正则是否在起作用
- trend/res 能量比：`E_res / (E_res + E_trend)`（用于解释 gate）

---

## 9. 与 v3 的关系：我们保留什么、替换什么

- **保留**：Dual‑Stream（EMA 分解）+ trend 线性头 + season 的 TemporalEncoder + 预测 head（这些已经在多数数据集上验证有效）
- **替换**：动态图学习/传播（GraphLearner、SMGP、gate_init 等）→ **FactorMix（静态低秩混合）**
- **新增**：Decomp Gate（让 Dual‑Stream 具备“自动回退能力”）、Learnable Alpha（让跨变量混合具备“自动强度控制”）

---

## 10. 预期现象与成功标准（写论文也好用）

成功的 v4 应体现为：

1) 在 **DS 明显有益**的数据集上：不回退，并且 `alpha_mix` 有机会学到非零（表明 cross‑var 混合在用）。
2) 在 **DS 会退化**的数据集上：`decomp_gate` 自发变小（残差注入减弱），从而显著减少退化幅度，甚至逼近 non‑DS baseline。
3) 不需要引入动态图传播，就能取得“接近 v3 最终版本”的性能；同时结构更简单、稳定性更强、解释更直观。
