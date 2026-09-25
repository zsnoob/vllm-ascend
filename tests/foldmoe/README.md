# FoldMoE 1A1M —— 注意力 / MoE 流水线（推理 prefill）

把 FoldMoE（Zhu et al., *FoldMoE: Efficient Long Sequence MoE Training via
Attention-MoE Pipelining*, ACL 2025）的 **1A1M** 调度从训练搬到 vllm-ascend 的
推理 prefill，用注意力和专家计算去掩盖 MoE 的 all-to-all。

论文：https://aclanthology.org/2025.acl-long.186.pdf

## 做了什么

一层内把 token 切成 `d` 个 micro-batch，计算流按下面的顺序下发，通信流就自然错开一拍：

```
计算流   A0  A1  M0  A2  M1  A3  M2  …  M(d-1)
通信流       D0  D1  C0  D2  C1  D3  …  C(d-1)

A = 注意力   M = 专家 FFN   D = A2A dispatch   C = A2A combine
```

除首尾两拍外，每个 dispatch / combine 底下都压着一段计算。
首尾那两拍没有配对对象，是结构性的，论文 Figure 7 也是这么画的。

注意力的切分靠一块 **token buffer**：已算出的 `k_nope / k_pe / value` 留在预分配
buffer 里，第 `i` 块的 query 对 `buf[:end_i]` 做一次 `sparse_mode=3`
（rightDownCausal）的融合注意力。这和整段因果注意力数值等价，FLOPs 不变，
既不重算 `kv_b_proj` 也不做 LSE 合并。

## 实测

DeepSeek-V2-Lite，TP=2 / EP=2，`FORCE_MOE_COMM=alltoall`，纯 prefill（`max_tokens=1`），
2×Ascend 910B2。输出 token 与基线逐个一致。

| 序列 | 最优配置 | 基线 | 1A1M | 加速 | MoE 通信被掩盖 |
|---|---|---|---|---|---|
| 64k  | d=2, v1 | 4.867 s  | 4.272 s  | +12.2% | — |
| 128k | d=4, v1 | 13.656 s | 12.129 s | +11.2% | 0% → 57.2% |
| 256k | d=8, v2 | 43.288 s | 38.658 s | +10.7% | 0% → 75.1% |

覆盖率的分母只算 MoE 的 dispatch 和 combine。TP 的 all-reduce 在结构上无法重叠，
计入分母只会稀释指标。

拆开看，**dispatch 能到 86~91%，combine 只有 36%（128k）**。短板是每拍收尾那一段
（等 combine 回来、跑共享专家、再做一次 TP all-reduce）会把计算流卡住，它挨着哪个
通信，哪个就盖不住。`v2` 调度把注意力拆成 `Acore` / `Apost` 两段，让收尾段落在
"没有 MoE 通信在飞"的空档，256k 下 combine 因此从 22% 提到 66.7%。

**重叠是有代价的**：128k 下 MoE 通信总量从 115.4 ms 涨到 131.9 ms（+14%），
计算和通信抢带宽，通信本身会变慢。所以"盖住 57%"不等于"省下 57% 的通信时间"。

## 用法

```bash
export FORCE_MOE_COMM=alltoall     # 强制走 all-to-all，否则单机可能走 allgather
export FOLDMOE_1A1M=4              # micro-batch 数，<=1 关闭（不设变量时行为与原版一致）
python your_bench.py
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `FOLDMOE_1A1M` | `0` | micro-batch 数 `d`，`<=1` 关闭。各长度最优：64k→2，128k→4，256k→8 |
| `FOLDMOE_SCHED` | `auto` | `auto` = `d>=8` 用 `v2`，否则 `v1`。v2 有固定下发开销，`d<=4` 时得不偿失 |
| `FOLDMOE_UNIFORM` | `0` | `1` = 注意力按 time-uniform 切（论文 Algorithm 1），MoE 仍是等长，中间靠 token buffer 解耦 |
| `FOLDMOE_MERGE` | `1` | 把每块的 `mlp()` 调用从 3 次降到 2 次 |
| `FOLDMOE_OUT` | `.` | 测试脚本的输出目录 |

只有 `FOLDMOE_1A1M>1` 时补丁才会挂载，否则代码路径与上游完全一致。

## 一个负面结果：time-uniform 在这里没有收益

论文 §4.2 用 token buffer 把注意力和 MoE 的分块解耦：注意力可以按等时长切得不均匀，
MoE 仍按固定 `L/d` 先进先出地取，约束 `Σ_{i<=j} l_i >= (j/d)·L` 保证 buffer 不欠数。
本实现照此实现（`FOLDMOE_UNIFORM=1`）。

效果确实出现了——注意力各段被拉平，dispatch / combine / 专家计算恢复等宽——
但墙钟始终在噪声范围内。原因是它要解决的问题在这里不存在：等长切分下通信就已经
盖住九成以上，把注意力拉平换不来更多覆盖，而非等长切分让算子效率下降，
计算总时间涨 3~4%，正好抵消。论文的收益场景是跨机训练，那里 all-to-all 远厚于计算。

另外，**Algorithm 1 的 FLOPs 模型用的是矩形 `l·c`，没扣掉因果掩码的三角形**，
所以 `d=2` 和 `d=4` 时它会精确退化成等长切分，只有 `d>=8` 才切得不一样。
在 `d=4` 上做 time-uniform 的 A/B 对照其实什么都没测到。

## 文件

| 文件 | 说明 |
|---|---|
| `vllm_ascend/ops/fused_moe/foldmoe_1a1m.py` | 实现主体，`apply()` 打四个补丁点 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | `TokenDispatcherWithAll2AllV` 增加 prep / issue / finish 拆分，让 a2a 能异步发射 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 末尾的挂载钩子 |
| `tests/foldmoe/fold_sweep.py` | 端到端 prefill 基准，扫不同的 `d` |
| `tests/foldmoe/gantt.py` | 从 `kernel_details.csv` 还原单层真实时间线，可直接对照论文 Figure 2c |
| `tests/foldmoe/beat.py` | 每一拍的 D/C 覆盖率，并区分被注意力还是被专家 FFN 盖住 |
| `tests/foldmoe/tl_json.py` | 把单层时间线导成 JSON |
| `tests/foldmoe/foldmoe_micro.py` | 单层微基准（DeepSeek-V2-Lite 形状），只要两张卡，不依赖 vLLM |
| `tests/foldmoe/foldmoe_micro_v4.py` | 单层微基准（DeepSeek-V4 形状，DSA 稀疏注意力） |

## 微基准：上限在哪

微基准去掉了 TP all-reduce 和共享专家，只留真实算子和真实 all-to-all，用来判断
端到端剩下的差距是调度问题还是结构问题。

| 场景 | MoE 通信被掩盖 | 加速 |
|---|---|---|
| 端到端 vLLM（128k d=4） | 57.2% | +11.2% |
| 端到端 vLLM（256k d=8） | 75.1% | +10.7% |
| 微基准 V2-Lite 形状（256k d=8） | 95.8% | +15.4% |
| 微基准 V4 形状（64k d=8） | 100.0% | +16.6% |

结论：**调度本身已接近极限，端到端剩下的差距来自结构上无法重叠的部分。**
128k 一层里 all-reduce 加 all-gather 有 108.1 ms，占该层 24%，重叠率是 0。

V4 形状（DSA 稀疏注意力）同样可迁移，输出逐位相同，但收益在 64k 见顶
（16k +15.3%，64k +16.6%，128k +11.8%，256k +4.5%）——序列越长注意力占比越高，
通信占比越低，而加速的上限就是通信占比。

## 已知限制

- 只在 prefill 且 `num_prefills == 1` 时生效，遇到 decode、chunked context 会自动回退。
- 只覆盖 `DeepseekV2DecoderLayer` + `AscendMLAImpl` 这条路径。
- `d` 需要手动选；最优 `d` 随序列长度单调上移。
- fp16 会回退（只在 bf16 上验证过）。
