# NoahPy 数学物理原理总结清单

## 核心控制方程（2个）

### 1. 热传导方程（Heat Diffusion Equation）
**方程形式**：
$$\frac{\partial(\rho C_p T)}{\partial t} = \frac{\partial}{\partial z}\left(\lambda \frac{\partial T}{\partial z}\right) + Q$$

**关键特征**：
- 控制土壤温度随深度和时间的变化
- Q项包含相变潜热（冻融过程）
- 实现函数：`SHFLX()` (Module_sf_noahlsm.py:1118-1205)
- 数值方法：隐式-显式（IEEC）方案 + 三对角矩阵求解

### 2. Richards方程（Unsaturated-Saturated Water Flow）
**方程形式**：
$$\frac{\partial \theta}{\partial t} = \frac{\partial}{\partial z}\left[D(\theta)\frac{\partial \theta}{\partial z}\right] + \frac{\partial K(\theta)}{\partial z} + S$$

**关键特征**：
- 控制土壤水分运动
- D(θ)和K(θ)依赖水分含量（非线性）
- S为蒸散发吸收项
- 实现函数：`SMFLX()` + `SRT()` (Module_sf_noahlsm.py:1209-1263)
- 参数化：Campbell (1974) 方案

---

## 热学过程（3个）

### 3. 土壤热容量计算
**方程**：
$$C_p = \theta_l C_{water} + \theta_i C_{ice} + (1-\phi)C_{soil} + (1-\phi-\theta_l-\theta_i)C_{air}$$

**参数值**：
| 分量 | 比热容 (J/(kg·K)) | 体积热容 (J/(m³·K)) |
|------|------------------|-------------------|
| 液态水 | 4218 | 4.218×10⁶ |
| 冰 | 2106 | 2.106×10⁶ |
| 土壤 | ~800 | 2.0×10⁶ |
| 空气 | 1004 | 1004 |

**代码位置**：Module_sf_noahlsm.py:1162

### 4. 土壤热导率参数化（McCumber & Pielke 1981）
**分步计算**：
1. **干土热导率**（与土壤类型有关）
2. **饱和热导率**：$\lambda_{sat} = \lambda_{soil}^{(1-\theta_s)} \cdot \lambda_{ice}^{(\theta_s - \theta_l)} \cdot \lambda_{water}^{\theta_l}$
3. **相对饱和度相关性**：$K_e = f(S_e, \theta_i)$
4. **最终结果**：$\lambda = K_e(\lambda_{sat} - \lambda_{dry}) + \lambda_{dry}$

**代码位置**：`TDFCND_C05_Tensor()` (Module_sf_noahlsm.py:778-816)

### 5. Stefan-Boltzmann辐射定律
**方程**：
$$L \uparrow = \epsilon \sigma T^4$$

**参数**：
- 表面发射率：ε ≈ 0.95 (土壤/植被)
- Stefan-Boltzmann常数：σ = 5.67×10⁻⁸ W/(m²·K⁴)

**代码位置**：Module_model_constants.py:17,22

---

## 水文过程（4个）

### 6. Campbell水分参数化方案
**导水率关系**：
$$K(\theta) = K_s\left(\frac{\theta}{\theta_s}\right)^{2b+3}$$

**基质势关系**：
$$\psi(\theta) = \psi_s\left(\frac{\theta}{\theta_s}\right)^{-b}$$

**土壤参数来源**：SOILPARM.TBL（19种土壤）
- b: 孔隙大小分布参数（2.71-11.55）
- Ks: 饱和导水率（10⁻⁷-10⁻⁴ m/s）
- ψs: 进气值基质势（-0.05 to -3.0 m）
- θs: 孔隙度/饱和含水量（0.337-0.468）

**代码位置**：Module_sf_noahlsm.py:20-89, REDSTP函数

### 7. 水分扩散率计算
**定义**：
$$D(\theta) = K(\theta)\frac{d\psi}{d\theta} = K(\theta)\frac{d\psi_s}{d\theta_s}\frac{d\theta_s}{d\theta}$$

**应用**：Richards方程中的扩散项
**代码**：Module_sf_noahlsm.py:1020-1037

### 8. 冰阻抗效应（Frozen Soil Reduction）
**物理机制**：冰占据孔隙，阻断液态水流动

**参数化公式**：
$$FRZFACT = \exp(-\alpha \cdot (SICE/SMC)^2), \quad \alpha \approx 9.0$$

**修正后导水率**：
$$K_{eff} = K_{unf} \times FRZFACT$$

**后果**：冻土导水率可降至原值的1%

**代码位置**：Module_sf_noahlsm.py:1244-1252 (SMFLX函数)

### 9. 土壤水分-基质势-温度关系
**Clausius-Clapeyron效应**：土壤水冻点下降
$$T_f = T_0 - \frac{\psi(\theta)}{\rho_w c_m g}$$

其中 $c_m \approx 0.01$ K·Pa⁻¹（Clausius-Clapeyron系数）

**物理意义**：微孔土壤中水可在零下多度仍保持液态

---

## 蒸散发过程（4个）

### 10. Penman-Monteith方程
**标准形式**：
$$ET_p = \frac{\Delta(R_n - G) + \rho c_p(e_s - e_a)/r_a}{\Delta + \gamma(1 + r_s/r_a)}$$

**参数含义**：
- Δ: 饱和水汽压曲线斜率 (dq_s/dT × P)
- γ: 干湿常数 ≈ 66 Pa/K
- r_a: 空气动力学阻力
- r_s: 冠层阻力

**代码位置**：`PENMAN()` (Module_sf_noahlsm.py:1269+)

### 11. 冠层阻力模型（Jarvis-Stewart）
**基本形式**：
$$r_s = \frac{r_{s,min}}{LAI} \times \frac{1}{f_R \cdot f_T \cdot f_D \cdot f_\theta}$$

**环境胁迫因子**：
- 光照胁迫（f_R）：依赖于吸收的短波辐射
- 温度胁迫（f_T）：在T_min和T_max之间为二次函数
- 水汽压差胁迫（f_D）：大气干燥程度
- 土壤水分胁迫（f_θ）：根层加权土壤含水量

**代码位置**：`CANRES()` (Module_sf_noahlsm.py:405-437)

### 12. 蒸散发分配
**总蒸散发分解**：
$$ETA = EDIR + EC + ET$$

其中：
- **EDIR**（直接蒸发）：仅来自第一层土壤，受植被遮荫影响
- **EC**（冠层蒸发）：来自截留降水，受冠层含水量限制
- **ET**（植被蒸腾）：来自根层吸水，受根系分布和植被胁迫影响

**代码位置**：`EVAPO()` (Module_sf_noahlsm.py:440+)

### 13. 饱和水汽压和相对湿度
**August-Roche-Magnus公式**：
$$e_s(T) = 610.78 \exp\left(\frac{17.27(T-273.15)}{T-35.85}\right) \text{ Pa}$$

**饱和比湿**：
$$q_s = \frac{\epsilon e_s}{P - (1-\epsilon)e_s}, \quad \epsilon = 0.622$$

**代码**：Module_sf_noahlsm.py 中多处调用

---

## 冻融过程（4个）

### 14. 超冷液态水计算（FRH2O）
**Koren et al. (1999)方案**：
通过Newton-Raphson迭代求解非线性方程

**Niu-Yang (2006)简化方案**：
$$\theta_l = \min\left[\theta_s \left(\frac{SMP}{-1}\right)^{-1/b}, \theta\right]$$

其中 $SMP = \frac{L_f}{g}\frac{T_f - T}{T \times \psi_s}$ （冻点压抑）

**物理意义**：
- 零下温度下，土壤孔隙中存在液态水膜
- 液态水比例依赖于：温度、土壤质地、含水量
- 冰含量：θ_i = θ_s - θ_l（假设没有空气）

**代码位置**：`FRH2O_tensor()` (Module_sf_noahlsm.py:962-1020)

### 15. 相变潜热耦合（Phase Change Source Term）
**能量守恒**：
$$Q_{phase} = L_f \rho_i \frac{\partial \theta_i}{\partial t}$$

**融化潜热**：L_f = 3.335×10⁵ J/kg

**典型量化**：
- 冰含量变化 Δθ_i = 0.1 m³/m³
- 释放潜热：Q = 3.335×10⁵ × 917 × 0.1 ≈ 3.06×10⁷ J/m³
- 可导致温度变化：ΔT ≈ 14.6 K

**代码位置**：`SNKSRC()` (Module_sf_noahlsm.py:1086-1115)

### 16. 温度平均化（TMPAVG）
**方法**：三点平均法
$$T_{avg} = \frac{T_{up} + T_{mid} + T_{down}}{3}$$

其中：
- T_up = (T_surf + T_mid)/2
- T_mid = T_layer
- T_down = (T_layer + T_below)/2

**用途**：在冻融层计算相变潜热时，使用平均温度判断水冰混合物状态

### 17. 冻融循环能量学
**活跃层物理过程**：

| 季节 | 热源 | 主过程 | 结果 |
|------|------|--------|------|
| 冬季 | 冷源 | T↓、水冻结、潜热释放 | 活跃层冻结深化 |
| 春夏 | 热源 | T↑、冰融化、潜热消耗 | 融化深度增加 |
| 秋冬 | 冷源 | T↓、新水冻结 | 冻融循环形成 |

**多年冻土形成条件**：需要长期（>2年）年平均温度 < 0°C

---

## 积雪过程（3个）

### 18. 新雪密度计算（SNOW_NEW）
**温度相关的新雪密度**：
$$\rho_{new} = \begin{cases}
50 \text{ kg/m}^3 & T > 2°C \\
100 \text{ kg/m}^3 & -2°C \leq T \leq 2°C \\
150 \text{ kg/m}^3 & T < -2°C
\end{cases}$$

**降雪后密度更新**：
$$\rho = \frac{\rho_{old} \times d_{old} + \rho_{new} \times \Delta d}{\rho_{old} \times d_{old} + \Delta d}$$

**代码位置**：`SNOW_NEW()` (Module_sf_noahlsm.py:819+)

### 19. 积雪覆盖度参数化（SNFRAC）
**Niu-Yang方案**：
$$f_{snow} = 1 - \exp(-S_c \times SWE/S_{max})$$

**参数**：
- S_c = 2.6（曲线参数）
- SWE = 积雪水当量 (kg/m² = mm)
- S_max ≈ 20 kg/m² （完全覆盖时的SWE）

**意义**：平滑过度从无雪到完全覆盖

**代码位置**：`SNFRAC()` (Module_sf_noahlsm.py:1165-1178)

### 20. 积雪反照率效应（ALCALC）
**综合反照率**：
$$\alpha = f_{snow} \times \alpha_{snow} + (1-f_{snow}) \times \alpha_{soil}$$

**积雪参数**：
- 新雪反照率：α_snow ≈ 0.8-0.85
- 老雪反照率：随龄期降低 (≈0.4-0.6)
- 土壤反照率：α_soil ≈ 0.2-0.3 (纹理相关)

**物理后果**：
- 高反照率 → 短波辐射反射多 → 地表升温慢
- 反照率随融化季节降低 → 正反馈，加速融化

---

## 大气边界层耦合（2个）

### 21. Monin-Obukhov相似性理论
**无量纲稳定度参数**：
$$\zeta = \frac{z}{L_{MO}} = \frac{z g \overline{w'\theta'}}{u_*^3 \rho T}$$

**三个稳定度制度**：
| ζ范围 | 条件 | 特征 | 修正函数 |
|-------|------|------|---------|
| ζ < 0 | 不稳定 | 浮力驱动对流，湍流强 | ψ = f(x), x=(1-16ζ)^1/4 |
| ζ = 0 | 中立 | 纯机械湍流，无浮力 | ψ = 0 |
| ζ > 0 | 稳定 | 浮力抑制湍流 | ψ = -5ζ |

**传输系数**：
$$C_D = \left[\frac{\kappa}{\ln(z/z_{0m}) - \psi_m}\right]^2, \quad C_H = \frac{\kappa^2}{[\ln(z/z_{0m})-\psi_m][\ln(z/z_{0h})-\psi_h]}$$

其中κ = 0.4（von Kármán常数）

**代码位置**：`SFCDIF_MYJ()` (Module_sfcdif_wrf.py:235-323)

### 22. 粗糙度长度参数化（z0mz0h）
**植被相关的动量粗糙度**：
$$z_{0m} = \max(z_{0sea}, LAI \times C_h) + z_{0min}$$

**参数值**：
- z_0sea = 0.001 m （水面）
- C_h ≈ 0.1 m/单位LAI
- z_0min = 0.001 m （最小值）

**热粗糙度**（通常较小）：
$$z_{0h} \approx 0.1 \times z_{0m}$$

**物理意义**：植被越密集，表面粗糙度越大，摩擦效应越强

---

## 数值方法（3个）

### 23. 隐式-显式耦合方案（IEEC）
**Kalnay & Kanamitsu (1988)**

**关键特征**：
- **隐式**：未来时刻状态（T^{t+Δt}）在方程左边
- **显式**：当前时刻系数（λ^t）在右边
- **优势**：无条件稳定，O(n)复杂度，无需迭代
- **误差**：O(Δt²)，适合扩散主导过程

**标准形式**：
$$\frac{S^{t+\Delta t} - S^t}{\Delta t} = \mathcal{L}^t(S^{t+\Delta t} - S^t)$$

**适用范围**：日尺度（Δt=86400s）土壤温湿度模拟

**代码**：Module_sf_noahlsm.py:1194-1200

### 24. 三对角矩阵求解（Thomas Algorithm）
**系统形式**：
$$A_i X_{i-1} + B_i X_i + C_i X_{i+1} = R_i, \quad i=1,2,...,n$$

**PyTorch实现**：
```
CO_Matrix = torch.diag(A[1:], -1) + torch.diag(B) + torch.diag(C[:-1], 1)
X = torch.linalg.solve(CO_Matrix, R)
```

**计算复杂度**：O(n)，其中n=20层

**数值稳定性**：对角占优矩阵，数值稳定

### 25. 网格离散化结构
**20层变厚度分层**：
- **第1-3层**（0-0.16 m）：日循环主要区间，厚度~0.05 m
- **第4-10层**（0.16-0.96 m）：季节循环，厚度递增
- **第11-20层**（0.96-3.8 m）：年际循环，厚度>0.3 m

**特点**：
- 根部主要分布：第1-4层（0-0.3 m）
- 季节冻融：第1-6层（0-0.6 m）
- 多年冻土：第7-20层（0.6-3.8 m）

---

## 约束与稳定化（2个）

### 26. 物理约束条件
**土壤含水量**：
$$0 \leq \theta_l \leq \theta_s - \theta_i$$

**冰含量**：
$$0 \leq \theta_i \leq \theta_s$$

**总含水量**：
$$\theta_s = \theta_l + \theta_i + \theta_{air}$$

**积雪覆盖度**：
$$0 \leq f_{snow} \leq 1$$

**实现方法**：`torch.clamp()` 函数限制数值范围

### 27. 梯度裁剪（Gradient Clipping）
**目的**：防止反向传播时梯度爆炸

**实现方式**：
```python
if k_time % 30 == 0:
    SH2O = SH2O.detach()  # 打破计算图
    STC = STC.detach()
```

**频率**：每30天（2,592,000秒）重新初始化一次梯度

**物理含义**：长期记忆衰减，周期性重启

---

## PyTorch特有特性（3个）

### 28. 自动微分（Automatic Differentiation）
**工作原理**：
1. 前向传播：计算输出 y = f(x)，同时记录计算图
2. 反向传播：从输出梯度反向计算输入梯度

**应用**：
```python
SH2O = torch.tensor([...], requires_grad=True)
Date, STC, SH2O = model.noah_main('forcing.txt')
loss = (STC - obs_STC).pow(2).sum()
loss.backward()  # 计算 ∂loss/∂所有参数
```

**优势**：
- 无需手推导梯度公式
- 支持复杂的非线性耦合过程
- 与参数优化（Adam、SGD等）无缝集成

### 29. 参数可学习性
**传统方法**：参数固定为查表值

**NoahPy创新**：
```python
self.BB = nn.Parameter(torch.tensor(...))  # 可学习参数
self.MAXSMC = nn.Parameter(torch.tensor(...))
```

**应用场景**：
- 参数反演：通过观测数据优化参数
- 地区适应：为特定地区调整参数
- 混合模型：融合物理模型与机器学习

### 30. 矩阵操作向量化
**低效循环**：
```python
for k in range(NSOIL):
    DENOM2[k] = ZSOIL[k] - ZSOIL[k+1]
```

**高效向量化**：
```python
DENOM2[1:] = ZSOIL[:-1] - ZSOIL[1:]
```

**性能提升**：
- CPU：5-10倍加速
- GPU：50-100倍加速（CUDA并行）

---

## 总结表

### 主要物理过程分类

| 类别 | 控制变量 | 关键方程 | 参数化方案 | 代码函数 |
|------|---------|---------|----------|---------|
| **热过程** | T(z,t) | 热扩散方程 | McCumber热导率 | SHFLX, TDFCND |
| **水过程** | θ(z,t) | Richards方程 | Campbell参数化 | SMFLX, SRT |
| **相变** | θ_i, θ_l | 能量守恒 | FRH2O超冷液水 | SNKSRC |
| **蒸散** | ETA | Penman-Monteith | Jarvis冠层阻力 | PENMAN, CANRES |
| **积雪** | SWE, ρ | 能量-水平衡 | 温度-密度关系 | SNOW_NEW |
| **边界层** | u*, θ*, q* | 相似性理论 | M-O稳定度修正 | SFCDIF_MYJ |

### 关键参数统计

| 参数类型 | 数量 | 来源 | 范围 |
|---------|------|------|------|
| **物理常数** | 30+ | 国际标准 | 精确值 |
| **土壤参数** | 5/类型 | SOILPARM.TBL | 19种 |
| **植被参数** | 15/类型 | VEGPARM.TBL | 17种 |
| **经验参数** | 10+ | 文献拟合 | 可调范围 |
| **总参数数** | ~100 | 混合 | 覆盖全球 |

---

**文档版本**：v1.0
**创建日期**：2025年10月28日
**涵盖原理数**：30个核心原理
**相关代码行数**：~4900行
**适用模型**：Noah LSM v3.4.1 PyTorch版
**目标用户**：学生、研究者、模型开发者
