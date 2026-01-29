# Model Design v5 — Dual‑Stream + Bridge Coupling Attention（Stable QK × Content V）

> 这版 v5 的定位：**不丢弃 Dual‑Stream**，而是在 v4 的“低风险可控”框架上，吸收你那套 Attention 方法里最核心、最可迁移的机制——  
> **在更稳定的空间计算相似度/路由（Q,K），在保留非平稳信息的空间搬运内容（V）**。  
> 它本质上是一种“Attention 化的跨变量耦合器”，用于替代/升级 v4 的静态 FactorMix（并保持同样的“可控注入强度”思想）。

---

## 0. 先回答你的关键问题：会丢弃 Dual‑Stream 吗？

不会丢弃，而是一次**融合**：

- **Dual‑Stream（EMA trend/res）**解决的是：输入级非平稳（慢变趋势/工况漂移）与短期结构混叠的问题。它已经被你的多数据集消融证明：在大多数数据集上是决定性收益来源，但在少数数据集会带来退化，因此需要 **Decomp Gate** 让它“可回退、可控”。  
- **Bridge Coupling Attention（Stable QK × Content V）**解决的是：跨变量耦合建模的“伪相关风险”。它并不替代 Dual‑Stream，而是让 season/residual 分支里“跨变量信息交互”更稳、更可解释。

所以 v5 的核心是：  
**Dual‑Stream 把“非平稳”拆开；Bridge Coupling 把“跨变量耦合”做得更安全、更自适应。**

---

## 1. 任务与记号

输入多变量序列：

$$
X \in \mathbb{R}^{B \times L \times C}
$$

输出未来预测：

$$
\hat{Y} \in \mathbb{R}^{B \times H \times C}
$$

其中：batch $$B$$，历史长度 $$L$$，预测长度 $$H$$，变量数 $$C$$。

---

## 2. v5 总体结构（一句话）

$$
\hat{Y} = \hat{Y}^{\mathrm{trend}} + g \odot \hat{Y}^{\mathrm{res}}
$$

- $$\hat{Y}^{\mathrm{trend}}$$：趋势分支（线性头）  
- $$\hat{Y}^{\mathrm{res}}$$：残差/季节分支（TemporalEncoder + Bridge Coupling + Head）  
- $$g \in (0,1)^{B\times C}$$：Decomp Gate（每变量一个门控），用于在 DS 可能退化的数据集上自动减弱 residual 注入。

> 注：这里的 $$g$$ 是“分解融合门控”，不同于跨变量耦合的强度系数 $$\alpha$$（后文）。

---

## 3. 模块 1：EMA Dual‑Stream 分解（固定）

### 3.1 趋势分量（EMA）

对每个变量沿时间轴做 EMA：

$$
x^{\mathrm{trend}}_{t} = \alpha_{\mathrm{ema}} x_t + (1-\alpha_{\mathrm{ema}}) x^{\mathrm{trend}}_{t-1}
$$

默认推荐（沿用你 v3/v4 最稳区域）：

$$
\alpha_{\mathrm{ema}} = 0.1
$$

### 3.2 残差/季节分量

$$
x^{\mathrm{res}} = x - x^{\mathrm{trend}}
$$

---

## 4. 模块 2：Trend 分支（线性头，固定）

采用 DLinear 风格线性映射（沿时间维）：

$$
\hat{Y}^{\mathrm{trend}}_{:,:,c} = W_{\mathrm{trend}} X^{\mathrm{trend}}_{:,:,c} + b_{\mathrm{trend}}
$$

其中：

- $$W_{\mathrm{trend}}\in \mathbb{R}^{H \times L}$$  
- $$b_{\mathrm{trend}}\in \mathbb{R}^{H}$$  
- **跨变量共享**（trend_head_share=1），保持你在 v3 的经验结论：更稳、更省参数。

---

## 5. 模块 3：Residual/Season 分支（核心）

Residual 分支负责预测 $$\hat{Y}^{\mathrm{res}}$$，由三段组成：

1) TemporalEncoder（变量内时间建模）  
2) Bridge Coupling Attention（跨变量耦合器：Stable QK × Content V）  
3) Forecast Head（输出 pred_len）

### 5.1 TemporalEncoder（沿用你现有最稳骨干）

对 $$X^{\mathrm{res}}$$ 做变量内编码，得到 token 表征：

$$
H = f_{\mathrm{time}}(X^{\mathrm{res}})\in \mathbb{R}^{B \times C \times N \times D}
$$

- $$N$$：token 数（不 patch 时 $$N=L$$；patch 时为 token 数）  
- $$D$$：隐空间维度（如 128）  

> v5 不强制把 backbone 换成 Transformer；TCN/SSM 都可以。v5 的新增能力主要体现在“跨变量耦合器”。

---

## 6. 模块 4：Bridge Coupling Attention（v5 的关键新件）

### 6.1 为什么要引入 Bridge Coupling？

你在 v3/v4 的证据链已经很清楚：

- 动态图传播/SMGP 的边际收益小，甚至模型会把传播强度压到接近 0（说明“错边风险”远大于收益）。  
- v4 的静态 FactorMix + 可学习强度 $$\alpha$$ 是一条更低风险路线，但它的表达力上限可能不足（尤其在“耦合结构随工况变化”的非平稳场景）。

因此我们把 FactorMix 升级为一种更“Attention 化”的耦合器：

- **相似度/路由图（Q,K）在更稳定的空间计算**：降低伪相关  
- **被搬运的内容（V）保留非平稳信息**：避免过度平稳导致输出变钝  
- 仍然用 **可学习注入强度 $$\alpha$$** 把风险变成“可控增量”

---

### 6.2 Map / Value 解耦：Stable QK × Content V

我们在 residual 分支内部构造两套特征：

- **Stable 表征（用于 Q,K）**：$$H_{\mathrm{stable}}$$  
- **Content 表征（用于 V）**：$$H_{\mathrm{content}}$$

它们来自同一个 $$H$$，但经过不同处理：

#### (A) Stable：局部去趋势（token 维 detrend）

对 token 维做滑动平均去趋势：

$$
H_{\mathrm{stable}} = H - \mathrm{MA}(H; w)
$$

其中窗口 $$w$$ 采用一个**不引入额外超参**的固定策略：

- 若启用 patch：$$w = \mathrm{patch\_len}$$（默认 16）  
- 否则：$$w=16$$（对 96 长度输入是一个合理默认）

> 这是从你 v3 的 SMGP 经验里抽象出来的“可迁移形式”，但 v5 只把它用于 **相似度计算**，风险更低。

#### (B) Content：统计量调制（FiLM）让 V 带上非平稳上下文

从输入窗口提取每变量统计量（实例级）：

$$
\mu_c = \mathrm{Mean}_t(x_{t,c}),\quad \sigma_c = \mathrm{Std}_t(x_{t,c})
$$

构造一个轻量调制器（MLP）输出 FiLM 参数：

$$
[\gamma_c,\beta_c] = \mathrm{MLP}_{\mathrm{stat}}\big([\mu_c,\log(\sigma_c+\epsilon)]\big)
$$

把它注入到 token 表征作为内容 value：

$$
H_{\mathrm{content}} = H \odot (1+\gamma) + \beta
$$

其中 $$\gamma,\beta \in \mathbb{R}^{B\times C\times 1\times 1}$$ 并 broadcast 到 token 与 embedding 维。

> 直觉：QK 仍然尽量“看结构/形状”，V 允许带上“量纲/尺度/工况”信息，从而避免“过度平稳 → 过平滑”。

---

### 6.3 Long‑scale Coupling：只在长尺度上建模跨变量关系（抗伪相关）

我们不在每个时间点做跨变量 attention，而是在更粗的 token 段上做耦合图更新。

设长尺度分段大小为 $$s$$（对应你已有 `graph_scale` 的思想）：

- 第 $$k$$ 段 token 区间：$$[ks,(k+1)s)$$  
- 段内做均值池化得到每变量的段级摘要：

$$
z_k = \mathrm{MeanPool}\left(H_{\mathrm{stable}}[:,:,ks:(k+1)s,:]\right)\in \mathbb{R}^{B\times C\times D}
$$

---

### 6.4 Cross‑Variable Attention（低秩形式，计算耦合矩阵 A）

对每个段 $$k$$：

1) 线性投影得到 Q/K（低秩 $$r$$）：

$$
Q_k = z_k W_Q,\quad K_k = z_k W_K
$$

其中 $$W_Q,W_K\in \mathbb{R}^{D\times r}$$，默认 $$r=8$$。

2) 相似度矩阵（跨变量）：

$$
S_k = \frac{1}{\sqrt{r}} Q_k K_k^\top \in \mathbb{R}^{B\times C\times C}
$$

3) 行归一化得到“路由图”（跨变量权重）：

$$
A_k = \mathrm{RowSoftmax}(S_k)
$$

> 你可以把 $$A_k$$ 看成“Attention map”，也可以看成“动态耦合图”。  
> 在 v5 的叙事里，我们强调它是 **由稳定表征驱动的路由矩阵**。

（可选但建议作为 v5 固定实现细节）加入 top‑k 稀疏化增强可解释性与稳健性：

- 对每行仅保留 top‑k 权重并重归一化。  
- 默认 $$k=\min(6,C)$$（对小通道几乎等价全连接，对大通道提供稀疏先验）。

---

### 6.5 用 A 搬运内容：对段内所有 token 做耦合

对第 $$k$$ 段的内容 tokens：

$$
H^{\mathrm{mix}}_{k} = A_k \cdot H_{\mathrm{content},k}
$$

最终残差式注入：

$$
H^{\mathrm{out}}_{k} = H_{\mathrm{content},k} + \alpha \cdot H^{\mathrm{mix}}_{k}
$$

其中耦合强度：

$$
\alpha=\sigma(a)\in (0,1)
$$

- $$a$$ 为可学习标量参数（也可做 per‑var，但 v5 固定为 scalar 更稳）。  
- 初始化建议让 $$\alpha$$ 很小（例如 $$a=-4$$ 对应弱注入），保证训练早期不被错耦合污染。

将所有段拼接回：

$$
H^{\mathrm{out}} = \mathrm{Concat}_k(H^{\mathrm{out}}_k)
$$

---

### 6.6 输出头（Residual Head）

沿用你现有 head（flatten tokens → linear）得到：

$$
\hat{Y}^{\mathrm{res}} = f_{\mathrm{head}}(H^{\mathrm{out}})\in \mathbb{R}^{B\times H\times C}
$$

---

## 7. 模块 5：Decomp Gate（让 Dual‑Stream 在“会退化”的数据集上自动回退）

你的 ablation 已经证明：Dual‑Stream 不是“对所有数据集都好”。  
所以 v5 固定保留 Decomp Gate，把 residual 分支当成“可控增量”：

$$
\hat{Y} = \hat{Y}^{\mathrm{trend}} + g \odot \hat{Y}^{\mathrm{res}}
$$

### 7.1 gate 的输入：残差能量 + 非平稳强度（固定）

从输入窗提取每变量统计（与 v4 一致）：

$$
e_{\mathrm{res}}=\mathrm{Mean}_t|x^{\mathrm{res}}_t|,\quad
e_{\Delta}=\mathrm{Mean}_t|x_t-x_{t-1}|
$$

构造比例：

$$
\rho=\frac{e_{\mathrm{res}}}{e_{\mathrm{res}}+e_{\Delta}+\epsilon}
$$

### 7.2 gate 的形式：小 MLP（固定）

$$
g=\sigma\left(\mathrm{MLP}_{\mathrm{gate}}\big([\rho,\log(e_{\mathrm{res}}+\epsilon),\log(e_{\Delta}+\epsilon)]\big)\right)
$$

推荐结构：`3 → 16 → 1`（GELU + sigmoid），输出形状 $$[B,C]$$。

> 解释性：  
> - 当输入窗“强趋势/强漂移”时，$$e_{\Delta}$$ 大、$$\rho$$ 小，gate 倾向变小 → 下调 residual 注入，避免 DS 退化。  
> - 当输入更接近“残差主导/弱漂移”时，gate 倾向更大 → 更信任 residual 分支。

---

## 8. v5 的固定默认超参（建议）

这是一套“开箱即用”的 v5 默认（你后续可以微调，但 v5 的结构不再变化）：

- `decomp_alpha = 0.1`
- `trend_head = linear`, `trend_head_share = 1`
- residual temporal backbone：继续用你当前最稳（TCN, d_model=128, e_layers=2 …）
- Bridge Coupling：
  - `coupling_scale = 8`
  - `coupling_rank = 8`
  - `stable_window = patch_len (default 16)`
  - `topk = min(6, C)`
  - `alpha_init = -4`（弱注入起步）
- Decomp Gate：
  - `MLP_gate: 3→16→1`
  - 初始化 bias 让 gate 初始靠近 1（更接近 v3 的 DS-sum 行为），再由训练自动调整

---

## 9. 需要记录的诊断量（建议固定输出到 log）

为了把“这个耦合器到底有没有用”讲清楚，建议固定记录：

- `alpha_mean`（耦合强度）
- `gate_mean`（Decomp Gate 均值）
- `A_entropy`（耦合矩阵行熵的均值/分位数）
- `A_topk_mass`（top‑k 权重和）
- `adj_diff`（段间 $$\|A_k-A_{k-1}\|_1$$ 的均值）
- `corr(alpha, metric)`：alpha 与验证误差的相关趋势（粗看即可）

---

## 10. v5 的方法叙事（写论文时很顺）

- 非平稳分解（Dual‑Stream）把慢变趋势从残差里拿走，降低耦合学习的伪相关风险。  
- 跨变量耦合采用 Bridge Coupling：稳定域学路由（Stable QK），内容域搬信息（Content V）。  
- 长尺度更新耦合矩阵 + 可学习注入强度 $$\alpha$$，让跨变量交互“可控、可解释、可回退”。  
- Decomp Gate 让 Dual‑Stream 在不适用的数据集上自动弱化，从而提升跨数据集鲁棒性。

