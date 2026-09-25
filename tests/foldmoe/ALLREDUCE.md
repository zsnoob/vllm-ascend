# 下一步：TP all-reduce 能不能藏起来

> **状态：进行中，尚无硬件实测。**
> `sched_sim.py` 可以直接跑（纯 Python，无需 NPU）。
> `foldmoe_micro_ar.py` **一次都没在硬件上跑过**，只做过语法检查和调度顺序干跑。
> vLLM 侧的异步 all-reduce **尚未实现**，`foldmoe_1a1m.py` 没有改动。

## 问题

1A1M 把 MoE 的 all-to-all 藏起来之后，下一个瓶颈是 TP 的 all-reduce。
实测 128k d=4 单层（Ascend profiler）：

| 分项 | 时长 | 与计算重叠 |
|---|---|---|
| 注意力 A | 245.0 ms | — |
| 专家 FFN M | 31.5 ms | — |
| 零散计算 x | 59.3 ms | — |
| **all-reduce R** | **85.9 ms** | **0.0%** |
| **all-gather G** | **22.3 ms** | **0.0%** |
| dispatch D | 51.4 ms | 90.7% |
| combine C | 80.4 ms | 35.8% |

层耗时 448.2 ms ≈ 计算流的串行总和 A+M+x+R+G = 444.1 ms。
**也就是说 dispatch/combine 几乎全被盖住了，层耗时基本就是"计算 + all-reduce"串起来。**

0% 重叠的原因是这些 all-reduce 同步地压在计算流上：
`o_proj` 是 RowParallelLinear，forward 里直接调 `tensor_model_parallel_all_reduce`；
MoE 侧是 `fused_moe.py` 的 `maybe_all_reduce_tensor_model_parallel`。
d=4 有 8 次 R，正好每块两次。

## 关键观察：该用注意力去盖，不是专家计算

真实系统里每块的注意力是 61 ms，专家计算只有 8.7 ms。
而 `A(i+1)` 并不依赖 `R_attn(i)`（下一块的注意力只需要层输入），
所以正确的排法是先把 `A(i+1)` 发出去，让它盖住 `R_attn(i)`，再回头等。

**注意**：`foldmoe_micro.py` 这个微基准里专家计算约 44 ms，比真实系统大 5 倍。
所以微基准可以验证机制（异步化写对没有、数值一致不一致），
**但收益量级不可外推到真实系统**。

## 模拟器预测

`sched_sim.py` 按 NPU 流的真实语义模拟（每条流按下发顺序执行，
op 开始时刻 = max(流空闲时刻, 所等事件完成时刻)），用上面的实测时长标定。

```bash
python3 sched_sim.py            # all-reduce 与 D/C 共用一条通信流
python3 sched_sim.py --split-ar # all-reduce 单独一条流
python3 sched_sim.py -v         # 附时间线
```

| 排法 | 共用一条通信流 | 单独一条流 |
|---|---|---|
| sync（当前 vLLM） | 443.8 ms | 443.8 ms |
| async·藏在 M 底下 | 424.3 ms (+4.4%) | 372.2 ms (+16.1%) |
| async·藏在 A 底下 | 406.5 ms (+8.4%) | 370.2 ms (+16.6%) |

模拟器复现 sync 得 443.8 ms，实测 448.2 ms，差 1%。

**怎么读这些数**：
- 全藏起来的理论地板是 335.8 ms（−24.3%），**拿不到**——首尾空拍占 34.4 ms。
- **+16.6% 是上界，不是预测**：模拟器没有带宽模型，假设两条流完全并发；
  而 TP 组和 EP 组是同一对卡、共用同一条物理链路，并发只会分带宽。
- 保守估计 +8.4%，真实值应落在两者之间。
- 标定说明：443.8 ≈ 448.2 这个吻合有一部分靠拟合零散计算的时长；
  但"R/G 串行占着计算流"这个结构是从 profiler 观察到的，不是拟合的，
  不同调度之间的相对差值也不依赖那个拟合。

## 微基准的设计（尚未验证）

`foldmoe_micro_ar.py` 在 1A1M 上补齐两次 all-reduce，五种模式构成单变量对照链：

| 模式 | 说明 |
|---|---|
| `seq` | 不切块不流水，all-reduce 同步 —— 绝对基线 |
| `sync` | 1A1M 原样，all-reduce 同步 —— 当前 vLLM 的形状 |
| `deep_sync` | 深流水（注意力提前两拍、F 回到流水线），all-reduce 仍同步 |
| `deep_async1` | 深流水，all-reduce 异步，与 D/C 共用一条流 |
| `deep_async2` | 深流水，all-reduce 异步，单独一条流 |

- `deep_sync` vs `sync` = 改调度本身的代价
- `deep_async` vs `deep_sync` = **all-reduce 异步化的净收益**（唯一变量）
- `deep_async2` vs `deep_async1` = 共用物理链路的影响

这个设计是被多视角审查改过的。第一版有两个会让结论完全跑偏的缺陷：

1. naive 的 `Acore(i); M(i-1); Apost(i); D(i)` 因为 Apost 要等 R_attn，
   把 D(i) 挤到了 M(i-1) 之后，**丢掉了 dispatch 的重叠**。丢掉的约 200 MB
   比想藏的 all-reduce（约 33 MB）还大，sync↔async 不再是单变量对照。
2. F 全堆在最后一个 M 之后，导致 **R_moe 在任何模式下都没有计算可盖**，
   结构上限只有 (d-1)/2d。

干跑已确认 `sync` 与 `deep_sync` 的通信流入队顺序完全一致
（D0 D1 C0 D2 C1 D3 C2 C3），说明 dispatch 的重叠保住了。

```bash
FOLDMOE_OUT=. torchrun --nproc_per_node 2 foldmoe_micro_ar.py 131072 4
```

## 待办

- [ ] 在硬件上跑通 `foldmoe_micro_ar.py`（先小规模冒烟，再 128k d=4）
- [ ] 拿实测去对模拟器预测的 +8.4% ~ +16.6%；差太远说明理解有洞
- [ ] 按"藏在 A 底下"改微基准的调度（当前实现是藏在 M 底下）
- [ ] vLLM 侧实现：`o_proj.reduce_results=False` 取未归约部分和 →
      侧流异步 all-reduce → 计算流先发 A(i+1) → 回头等事件
