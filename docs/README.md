## Build Documentation

1. Clone MMYOLO

   ```bash
   git clone https://github.com/open-mmlab/mmyolo.git
   cd mmyolo
   ```

2. Install the building dependencies of documentation

   ```bash
   pip install -r requirements/docs.txt
   ```

3. Change directory to `docs/en` or `docs/zh_cn`

   ```bash
   cd docs/en  # or docs/zh_cn
   ```

4. Build documentation

   ```bash
   make html
   ```

5. Open `_build/html/index.html` with browser

---

# MessDet 训练性能调研（2026-07）

> 本节的性能实验、根因分析与文档由 [Claude](https://claude.com/claude-code)（Anthropic）协助完成。
>
> ⚠️ **当前状态：结论已实测验证，代码尚未落地。** e2cnn / escnn 的源码没有修改，下述加速需要手动应用补丁才能生效。

## 1. 一句话结论

**等变卷积不慢，但是 BatchNorm 很慢；而且三分之二的开销用纯 PyTorch 就能拿回来，不需要写 CUDA kernel。**

## 2. 单 step 时间去向

A100 80GB（**空闲**），真实 backbone + neck（107 个 `R2Conv`，N=8），batch 2 @ 1024²，15 次中位数：

```
297 ms  ┌── 130 ms  不可避免（同形状普通 CNN 的开销）
        └── 167 ms  等变税
                    ├── 112 ms  实现方式造成的，纯 PyTorch 可消除
                    └──  55 ms  等变性固有
                                ├── ~40 ms  群内共享 BN 相对独立 BN 的代价
                                └── ~15 ms  基展开
```

| 变体 | fwd | fwd+bwd | vs 基线 |
| --- | --- | --- | --- |
| 原样（基线） | 132 ms | **297 ms** | 1.00× |
| 去掉 BN 的 428 次隐式同步 | 86 ms | 251 ms | 1.19× |
| **BN 改手写 reduction　[数学等价]** | 79 ms | **209 ms** | **1.42×** |
| **上一行再套 `torch.compile`　[数学等价]** | 81 ms | **185 ms** | **1.61×** |
| BN 不做群内共享　[非等价，仅用于定价] | 52 ms | 145 ms | 2.05× |
| ≈ 同形状普通 CNN　[非等价，仅用于定价] | 42 ms | **130 ms** | 2.28× |

标 **[非等价]** 的变体改变了数学，只用来给某个结构选择定价，**不是修复方案**。

## 3. 两个根因

### 3.1 群内共享 BatchNorm 把 cuDNN 的并行度打掉 8 倍

**BN 占 GPU 时间的 57%，是卷积的 2.6 倍。**

等变性要求一个 field 内的 N 个群元素**共享统计量**（否则不同旋转分量被独立缩放，等变性立刻破坏）。所以 e2cnn 把 `(B, C, H, W)` 看成 `(B, C/N, N, H, W)` 调 `BatchNorm3d(C/N)`。而 cuDNN 的 batchnorm 是**按通道并行**的，通道数从 C 变成 C/8，并行度直接掉 8 倍。

| 张量 | fields (=C/8) | `BatchNorm2d(C)`<br>非等变参照 | `BatchNorm3d(C/8)`<br>**现状** | 手写 reduction |
| --- | --- | --- | --- | --- |
| B2 C128 256² | 16 | 0.849 ms | **3.112 ms (3.67×)** | 1.384 ms (1.63×) |
| B2 C256 128² | 32 | 0.463 ms | **0.970 ms (2.10×)** | 0.926 ms (2.00×) |
| B2 C512 64² | 64 | 0.378 ms | **0.443 ms (1.17×)** | 0.702 ms (1.86×) |
| B2 C1024 32² | 128 | 0.369 ms | **0.386 ms (1.05×)** | 0.670 ms (1.82×) |

**惩罚随 field 数反比放大**：fields=128 时几乎没有惩罚（1.05×），fields=16 时高达 3.67×。而 field 少 = 通道少 = 分辨率高 = **张量最大的浅层**。最贵的地方恰好在最费带宽的地方，两个因素相乘。

注意手写 reduction 只在 fields ≤ 32 时赢，fields ≥ 64 反而输给 cuDNN —— 通用补丁需要按形状分派，或全交给 `torch.compile`。

### 3.2 每 step 428 次隐式 GPU→CPU 同步

`InnerBatchNorm` 把切片下标存成 registered buffer（会随 `.cuda()` 搬到显存），然后拿它当 Python 切片边界：

```python
# e2cnn/nn/modules/batchnormalization/inner.py:136-141
indices = getattr(self, f"indices_{s}")          # ← CUDA 上的 LongTensor
output[:, indices[0]:indices[1], :, :] = batchnorm(...)
```

`indices[0]` 作为切片边界必须是 host 端 Python int → 隐式 `.item()` → `cudaMemcpyDtoH` + `cudaStreamSynchronize`，排干整个异步队列。每个 BN 4 次，107 个 BN = **每 step 正好 428 次**。只把下标改成 Python int（约 5 行，数学完全不变）就有 **fwd 1.54× / 整 step 1.19×**。

> 这是全库范围的写法（e2cnn 35 处 / escnn 40 处），不是个案。但 `output[:, indices, ...]` 这种 fancy indexing **不同步**，可以留着。

## 4. 建模型 80 秒：78% 是 He-init 的纯 Python 循环

cProfile 真实 MessDet 构建：`generalized_he_init` 占 90.4 s / 115 s。根因是它**为每一个参数构造一个 Python dict** —— 模型有 13.60 M 参数，`get_basis_info()` 恰好 yield 了 13.42 M 次，其中 26.8 M 次 `str.format` 只为拼一个字符串 id。

而真正要算的方差只依赖 `(in_irrep, out_irrep_idx)`，即**只依赖 representation 对**，与"第几个输入/输出 field"无关。512→512 层是 64×64 = **4096 倍冗余**。

改成从 block 级 basis info 算一次再 tile 展开：**整模型构建 80.3 s → 10.3 s（7.8×），13.60 M 参数逐位相同（`torch.equal`）**。

> 库里已有的 `cache=True` 解决不了这个问题：缓存键是 basismanager 对象本身，每个 `R2Conv` 一个实例，单次构建内部零命中。

## 5. 正确性验证

BN 改写做了三重验证，这是能不能用在训练上的前提：

| 验证项 | 结果 |
| --- | --- |
| **数值等价** | 单层 fwd / dx / dw / db / running stats 全部匹配到 1e-7 |
| **等变性不变** | 纯纤维作用 ρ(g)，N=8：原版 7.51e-8，改写后 **7.51e-8**（完全相同） |
| **checkpoint 兼容** | 只改 `forward`，`state_dict` 键名/形状/语义一个没变，**旧权重直接加载，不需要重训** |

整模型端到端的偏差看起来大（输出 6.06e-2），但对照实验显示这是模型自身的条件数问题：拿**原版代码**把输入扰动百万分之一，偏差是 9.20e-2 —— 比补丁造成的还大。107 层 BN 叠起来本身就是混沌系统。



## 6. 不要做的事（已否证，勿重复劳动）

| 假设 | 实测 |
| --- | --- |
| 基展开是瓶颈，值得手写 CUDA kernel | 只占 5%（消融只省 24/297 ms），**理论上限约 1%，不值得做** |
| 等变卷积本身比普通卷积慢 | 把 filter 冻成普通 `Parameter` 后 297→273 ms，卷积部分几乎无等变开销 |
| 瓶颈是 kernel launch / dispatch 数量 | GPU 已 94% 忙，不是 dispatch bound（batch 2 @1024²） |
| 打开已有的 `cache=True` 能解决初始化慢 | 缓存键是对象本身，单次构建零命中 |
| 初始化慢是在构造 kernel basis | basis 全缓存，90.4/115 s 在 `generalized_he_init` |

**唯一值得考虑手写 CUDA kernel 的目标**：185 ms → 145 ms 那 40 ms，即"群内共享 BN 相对独立 BN 的固有代价"。即便完全做到，也只是在已拿到的 1.61× 之上再加约 1.28×。

## 7. 落地前还差什么

目前两项修复都只以**运行时 monkeypatch** 的形式验证过，跑真实训练时不会生效。要真正用上，还需处理本模型用不到但库必须支持的情况：多 field group、非连续下标、`track_running_stats=False`、`affine=False`、异构 FieldType。

`torch.compile` 那一档需要设 `torch._dynamo.config.recompile_limit`：模型有约 20 种不同的 BN 形状，默认上限 8 会让大部分层退回 eager —— 实测差别是 200 ms vs 185 ms。

