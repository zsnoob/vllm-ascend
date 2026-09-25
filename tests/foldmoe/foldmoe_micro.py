"""FoldMoE 单层微基准：1A1M 流水线 + time-uniform 切分 + token buffer。

不依赖 vLLM，两张卡各当一个 EP rank，只复现一层 Transformer-MoE：
    A = attention（FIA，sparse_mode=3 因果）+ o_proj
    D = all-to-all dispatch     M = 专家 FFN（GroupedMatmul）
    C = all-to-all combine      F = 收回 combine 结果做加权求和
形状取 DeepSeek-V2-Lite：hidden 2048，每卡 8 个头（= 真实 TP=2 时每卡的头数），
64 个路由专家 top-6（每卡 32 个），moe_intermediate 1408。

对比四种模式：
    serial     不切分，A D M C F 顺序执行（基线）
    uniform    1A1M，attention 与 MoE 都等长切
    tu_nobuf   1A1M，attention 按 time-uniform 切，MoE 也跟着同样的切法（我之前的错误实现）
    tu_buf     1A1M，attention 按 time-uniform 切，MoE 经 token buffer 取固定 L/d（论文 §4.2）

路由在计时前算好（随机 gate，固定种子），计时区只剩真实的计算与通信。
用法：torchrun --nproc_per_node 2 foldmoe_micro.py <L> <d> [每卡头数，默认 8]
"""

import functools
import json
import os

# 输出目录，用 FOLDMOE_OUT 覆盖
OUT = os.environ.get("FOLDMOE_OUT", ".")
import statistics
import sys

import torch
import torch.distributed as dist
import torch_npu

H, DN, DR, DV = 2048, 128, 64, 128
HEADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # 每卡头数，调小可让 attention 变薄
E, TOPK, INTER = 64, 6, 1408


@functools.lru_cache(None)
def slices_uniform(L, d):
    q, r = divmod(L, d)
    out, s = [], 0
    for i in range(d):
        e = s + q + (1 if i < r else 0)
        out.append((s, e))
        s = e
    return out


@functools.lru_cache(None)   # 纯 Python 循环要几十 ms，和 vLLM 版一样缓存
def slices_time_uniform(L, d, Hm=2048, h=16):
    """论文 Algorithm 1（quick-start time-uniform attention slicing），逐行照搬。"""
    flops = lambda l, c: (4 * Hm + 3 * h) * l * c + 8 * Hm * Hm * l
    m = -(-L // d)
    S = [m]
    t_hat = ((4 * Hm + 3 * h) * L * (L + 1) / 2 + 8 * Hm * Hm * L) / d
    start = m
    while start < L:
        end = max(start + 1, (len(S) + 1) * m)
        if L - end >= d - len(S):
            best, best_err = end, None
            for i in range(end, L + 1):
                err = abs(flops(i - start, i) - t_hat)
                if best_err is None or err < best_err:
                    best, best_err = i, err
                else:
                    break
            end = best
        end = min(end, L)
        S.append(end - start)
        start = end
    out, s = [], 0
    for n in S:
        out.append((s, s + n))
        s += n
    return out


def main():
    L, d = int(sys.argv[1]), int(sys.argv[2])
    rank = int(os.environ["RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    dev = torch.device("npu")
    bf = torch.bfloat16
    comp = torch.npu.current_stream()
    comm = torch.npu.Stream()
    E_loc = E // world

    # ---------------- 权重与输入（每 rank 不同的序列）
    g = torch.Generator().manual_seed(1234 + rank)
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(bf).to(dev)
    q_nope, q_pe = rnd(L, HEADS, DN), rnd(L, HEADS, DR)
    k_nope, k_pe, v = rnd(L, HEADS, DN), rnd(L, HEADS, DR), rnd(L, HEADS, DV)
    w_o = rnd(HEADS * DV, H)
    gw = torch.Generator().manual_seed(99 + rank)
    # 布局与 vllm 的 process_weights_after_loading 一致：transpose 后 .contiguous()，
    # 即连续的 [E, 入, 出]。（vllm 另有 npu_format_cast 到 FRACTAL_NZ，但只在
    # enable_fused_mc2 时才做，我们跑的 alltoall 路径不走它。）
    w13 = (torch.randn(E_loc, H, 2 * INTER, generator=gw) * 0.02).to(bf).to(dev).contiguous()
    w2 = (torch.randn(E_loc, INTER, H, generator=gw) * 0.02).to(bf).to(dev).contiguous()
    mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(dev)
    logits = torch.randn(L, E, generator=g)
    topw, topi = logits.softmax(-1).topk(TOPK, dim=-1)

    # ---------------- 预先算好某个 token 区间 [s, e) 作为一个 MoE micro-batch 的路由
    plan_cache = {}

    def plan(s, e):
        if (s, e) in plan_cache:
            return plan_cache[(s, e)]
        ids = topi[s:e].reshape(-1)
        order = torch.argsort(ids, stable=True)                  # 按专家排好 = 按目的 rank 排好
        send_cnt = torch.bincount(ids // E_loc, minlength=world)
        cnt_all = [torch.zeros(world, dtype=torch.long, device=dev) for _ in range(world)]
        dist.all_gather(cnt_all, send_cnt.to(dev))
        recv_cnt = [int(c[rank]) for c in cnt_all]
        # 接收方需要知道每行属于哪个本地专家：交换一次专家 id
        ids_sorted = (ids[order] % E_loc).to(torch.int32).to(dev)
        rid = torch.empty(sum(recv_cnt), dtype=torch.int32, device=dev)
        dist.all_to_all_single(rid, ids_sorted, recv_cnt, send_cnt.tolist())
        rorder = torch.argsort(rid.long(), stable=True)
        p = dict(
            tok=(order // TOPK).to(dev),                          # 发送行对应的 token（相对 s）
            w=topw[s:e].reshape(-1)[order].to(bf).to(dev)[:, None],
            sc=send_cnt.tolist(), rc=recv_cnt,
            rorder=rorder, grp=torch.bincount(rid.long(), minlength=E_loc).to(torch.int64),
        )
        plan_cache[(s, e)] = p
        return p

    # ---------------- 各阶段
    def attn(s, e):
        n = e - s
        o, _ = torch_npu.npu_fused_infer_attention_score(
            q_nope[s:e], k_nope[:e], v[:e], query_rope=q_pe[s:e], key_rope=k_pe[:e],
            num_heads=HEADS, num_key_value_heads=HEADS, input_layout="TND",
            atten_mask=mask, sparse_mode=3, scale=(DN + DR) ** -0.5,
            antiquant_mode=0, antiquant_scale=None, block_table=None, block_size=0,
            softmax_lse_flag=False, actual_seq_lengths=[n], actual_seq_lengths_kv=[e])
        return o.reshape(n, HEADS * DV) @ w_o

    def experts(x, p):
        x = x.index_select(0, p["rorder"])
        y = torch_npu.npu_grouped_matmul(x=[x], weight=[w13], split_item=2, group_list_type=1,
                                         group_type=0, group_list=p["grp"])[0]
        y = torch_npu.npu_swiglu(y)
        y = torch_npu.npu_grouped_matmul(x=[y], weight=[w2], split_item=2, group_list_type=1,
                                         group_type=0, group_list=p["grp"])[0]
        out = torch.empty_like(y)
        out.index_copy_(0, p["rorder"], y)
        return out

    def run(mode, rec):
        """跑一层。rec(name, idx, stream_kind, fn) 负责打事件并执行 fn。"""
        if mode == "serial":
            abnd = mbnd = [(0, L)]
        else:
            ub = slices_uniform(L, d)
            tu = slices_time_uniform(L, d)
            abnd = ub if mode == "uniform" else tu
            mbnd = tu if mode == "tu_nobuf" else ub
        n_mb = len(mbnd)
        fifo, st = [], [dict() for _ in range(n_mb)]
        out = torch.empty(L, H, dtype=bf, device=dev)

        def take(s, e):
            parts = [t[max(s, a) - a:min(e, b) - a] for a, b, t in fifo if a < e and b > s]
            while fifo and fifo[0][1] <= e:
                fifo.pop(0)
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        def A(i):
            if i < len(abnd):
                s, e = abnd[i]
                fifo.append((s, e, rec("A", i, "comp", lambda: attn(s, e))))

        def D(j):
            s, e = mbnd[j]
            p = plan(s, e)
            x = rec("P", j, "comp", lambda: take(s, e).index_select(0, p["tok"]))
            ev = comp.record_event()
            recv = torch.empty(sum(p["rc"]), H, dtype=bf, device=dev)
            with torch.npu.stream(comm):
                comm.wait_event(ev)
                rec("D", j, "comm", lambda: dist.all_to_all_single(recv, x, p["rc"], p["sc"]))
                st[j]["evd"] = comm.record_event()
            x.record_stream(comm)
            recv.record_stream(comm)
            st[j]["recv"], st[j]["p"] = recv, p

        def M(j):
            comp.wait_event(st[j]["evd"])
            y = rec("M", j, "comp", lambda: experts(st[j].pop("recv"), st[j]["p"]))
            ev = comp.record_event()
            back = torch.empty(sum(st[j]["p"]["sc"]), H, dtype=bf, device=dev)
            with torch.npu.stream(comm):
                comm.wait_event(ev)
                rec("C", j, "comm", lambda: dist.all_to_all_single(
                    back, y, st[j]["p"]["sc"], st[j]["p"]["rc"]))
                st[j]["evc"] = comm.record_event()
            y.record_stream(comm)
            back.record_stream(comm)
            st[j]["back"] = back

        def F(j):
            s, e = mbnd[j]
            p = st[j]["p"]
            comp.wait_event(st[j]["evc"])

            def fin():
                o = out[s:e]
                o.zero_()
                o.index_add_(0, p["tok"], st[j].pop("back") * p["w"])
            rec("F", j, "comp", fin)

        # 1A1M：计算流 A0 A1 M0 A2 M1 ...，通信流 D0 D1 C0 D2 C1 ...
        A(0)
        D(0)
        for i in range(1, n_mb):
            A(i)
            D(i)
            M(i - 1)
        M(n_mb - 1)
        for j in range(n_mb):
            F(j)
        return out, abnd, mbnd

    def plain_rec(name, i, kind, fn):
        return fn()

    modes = ["serial", "uniform", "tu_nobuf", "tu_buf"]
    res = {}
    ref = None
    for mode in modes:
        for _ in range(2):                     # 预热（也把路由计划算进缓存）
            run(mode, plain_rec)
        torch.npu.synchronize()
        times = []
        for _ in range(5):
            dist.barrier()
            torch.npu.synchronize()
            t0, t1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            t0.record()
            o, abnd, mbnd = run(mode, plain_rec)
            t1.record()
            torch.npu.synchronize()
            times.append(t0.elapsed_time(t1))
        # 再跑一次打细粒度事件，拿时间线
        evs = []

        def ev_rec(name, i, kind, fn):
            stream = comm if kind == "comm" else comp
            a = torch.npu.Event(enable_timing=True)
            b = torch.npu.Event(enable_timing=True)
            a.record(stream)
            r = fn()
            b.record(stream)
            evs.append((name, i, kind, a, b))
            return r

        dist.barrier()
        torch.npu.synchronize()
        base = torch.npu.Event(enable_timing=True)
        base.record()
        o, abnd, mbnd = run(mode, ev_rec)
        torch.npu.synchronize()
        tl = [(n, i, k, base.elapsed_time(a), base.elapsed_time(b)) for n, i, k, a, b in evs]
        if mode == "serial":
            ref = o
        diff = (o.float() - ref.float()).abs().max().item()
        res[mode] = dict(ms=statistics.median(times), all=times, diff=diff, tl=tl,
                         abnd=[e - s for s, e in abnd], mbnd=[e - s for s, e in mbnd])
        del o
        torch.npu.empty_cache()

    if rank == 0:
        path = OUT + "/micro_L%d_d%d_h%d.json" % (L, d, HEADS)
        json.dump(res, open(path, "w"))
        base_ms = res["serial"]["ms"]
        print("L=%d d=%d heads=%d  (rank0，5 次中位数)" % (L, d, HEADS))
        for m in modes:
            r = res[m]
            print("  %-9s %8.2f ms  vs serial %+6.1f%%  max|diff|=%.2e" % (
                m, r["ms"], 100 * (base_ms - r["ms"]) / base_ms, r["diff"]))
        print("  attention 切片:", res["tu_buf"]["abnd"])
        print("  saved", path)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
