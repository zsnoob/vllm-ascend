"""FoldMoE 单层微基准 —— DeepSeek-V4 形状版。

和 foldmoe_micro.py 同一套流水线（1A1M / time-uniform / token buffer），只把形状和
attention 换成 V4 的：
    A = DSA 稀疏注意力：lightning indexer 选 top-512，再做 sparse flash attention
    D = all-to-all dispatch     M = 专家 FFN（GroupedMatmul）
    C = all-to-all combine      F = 收回 combine 结果做加权求和

形状取 /docker/models/DeepSeek-V4-W8A8 的 config（TP=2/EP=2 时的每卡份额）：
    hidden 4096，每卡 32 个注意力头，head_dim 512 + rope 64，KV 是共享的单头潜变量
    256 个路由专家 top-6（每卡 128 个），moe_intermediate 2048
    indexer：64 头（每卡 32）× 128 维，index_topk 512

**这不是 V4 的完整复现**：只做了 DSA 这一条主路径，没有实现 compressor（逐层
交替的 4/128 压缩）、128 的滑动窗口、Hyper-Connections 残差和 W8A8 量化。
目的是回答「V4 的形状下 1A1M 还能省多少通信」，不是对齐 V4 的数值。

用法：torchrun --nproc_per_node 2 foldmoe_micro_v4.py <L> <d>
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

H = 4096
HEADS = 32           # 每卡头数（TP=2 时 64/2）
DN, DR = 512, 64     # head_dim / qk_rope_head_dim
IDX_D, IDX_TOPK = 128, 512
E, TOPK, INTER = 256, 6, 2048


@functools.lru_cache(None)
def slices_uniform(L, d):
    q, r = divmod(L, d)
    out, s = [], 0
    for i in range(d):
        e = s + q + (1 if i < r else 0)
        out.append((s, e))
        s = e
    return out


@functools.lru_cache(None)
def slices_time_uniform(L, d, Hm=H, h=64):
    """论文 Algorithm 1。注意它的 FLOPs 模型假设的是稠密因果注意力，
    V4 是稀疏的，所以这里算出来的切法未必合适——正是要测的东西。"""
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

    g = torch.Generator().manual_seed(1234 + rank)
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(bf).to(dev)
    # attention：Q 每卡 32 头，KV 是 MLA 吸收后的共享单头潜变量
    q_nope, q_pe = rnd(L, HEADS, DN), rnd(L, HEADS, DR)
    kv, k_pe = rnd(L, 1, DN), rnd(L, 1, DR)
    # indexer：每头一个 128 维的小 query，打分后选 top-512
    q_li, k_li = rnd(L, HEADS, IDX_D), rnd(L, 1, IDX_D)
    w_li = rnd(L, HEADS)
    w_o = rnd(HEADS * DN, H)
    gw = torch.Generator().manual_seed(99 + rank)
    # 布局与 vllm 的 process_weights_after_loading 一致：transpose 后 .contiguous()，
    # 即连续的 [E, 入, 出]。（vllm 另有 npu_format_cast 到 FRACTAL_NZ，但只在
    # enable_fused_mc2 时才做，我们跑的 alltoall 路径不走它。）
    w13 = (torch.randn(E_loc, H, 2 * INTER, generator=gw) * 0.02).to(bf).to(dev).contiguous()
    w2 = (torch.randn(E_loc, INTER, H, generator=gw) * 0.02).to(bf).to(dev).contiguous()
    logits = torch.randn(L, E, generator=g)
    topw, topi = logits.softmax(-1).topk(TOPK, dim=-1)

    plan_cache = {}

    def plan(s, e):
        if (s, e) in plan_cache:
            return plan_cache[(s, e)]
        ids = topi[s:e].reshape(-1)
        order = torch.argsort(ids, stable=True)
        send_cnt = torch.bincount(ids // E_loc, minlength=world)
        cnt_all = [torch.zeros(world, dtype=torch.long, device=dev) for _ in range(world)]
        dist.all_gather(cnt_all, send_cnt.to(dev))
        recv_cnt = [int(c[rank]) for c in cnt_all]
        ids_sorted = (ids[order] % E_loc).to(torch.int32).to(dev)
        rid = torch.empty(sum(recv_cnt), dtype=torch.int32, device=dev)
        dist.all_to_all_single(rid, ids_sorted, recv_cnt, send_cnt.tolist())
        rorder = torch.argsort(rid.long(), stable=True)
        p = dict(
            tok=(order // TOPK).to(dev),
            w=topw[s:e].reshape(-1)[order].to(bf).to(dev)[:, None],
            sc=send_cnt.tolist(), rc=recv_cnt,
            rorder=rorder, grp=torch.bincount(rid.long(), minlength=E_loc).to(torch.int64),
        )
        plan_cache[(s, e)] = p
        return p

    def attn(s, e):
        """chunk [s, e) 的 query 打整段 buffer[:e]，rightDownCausal 等价于整段因果。"""
        n = e - s
        asq = torch.tensor([n], dtype=torch.int32, device=dev)
        ask = torch.tensor([e], dtype=torch.int32, device=dev)
        idx, _ = torch_npu.npu_lightning_indexer(
            query=q_li[s:e], key=k_li[:e], weights=w_li[s:e],
            actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
            layout_query="TND", layout_key="TND",
            sparse_count=IDX_TOPK, sparse_mode=3)
        o = torch_npu.npu_sparse_flash_attention(
            query=q_nope[s:e], key=kv[:e], value=kv[:e], sparse_indices=idx,
            scale_value=(DN + DR) ** -0.5, sparse_block_size=1,
            actual_seq_lengths_query=asq, actual_seq_lengths_kv=ask,
            query_rope=q_pe[s:e], key_rope=k_pe[:e],
            layout_query="TND", layout_kv="TND", sparse_mode=3,
            attention_mode=2)
        if isinstance(o, tuple):
            o = o[0]
        return o.reshape(n, HEADS * DN) @ w_o

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
        if mode == "serial":
            abnd = mbnd = [(0, L)]
        elif mode == "seq":
            # 分块但不重叠：每块 A→D→M→F 走完再进下一块。
            # 256k 时不切分的 dispatch 张量要 12.9GB 放不下，用它当基线，
            # 并在 64k/128k 上和 serial 对照验证两者一致。
            abnd = mbnd = slices_uniform(L, d)
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

        if mode == "seq":
            for i in range(n_mb):
                A(i)
                D(i)
                M(i)
                F(i)
            return out, abnd, mbnd
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

    # 256k 时不切分的 dispatch 张量放不下，跳过 serial，用 seq 当基线
    modes = ["seq", "uniform", "tu_nobuf", "tu_buf"]
    if L <= 131072:
        modes.insert(0, "serial")
    res = {}
    ref = None
    for mode in modes:
        for _ in range(2):
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
        if ref is None:
            ref = o
        diff = (o.float() - ref.float()).abs().max().item()
        res[mode] = dict(ms=statistics.median(times), all=times, diff=diff, tl=tl,
                         abnd=[e - s for s, e in abnd], mbnd=[e - s for s, e in mbnd])
        del o
        torch.npu.empty_cache()

    if rank == 0:
        path = OUT + "/microv4_L%d_d%d.json" % (L, d)
        json.dump(res, open(path, "w"))
        base_ms = res[modes[0]]["ms"]
        print("V4 形状  L=%d d=%d  (rank0，5 次中位数)" % (L, d))
        for m in modes:
            r = res[m]
            print("  %-9s %8.2f ms  vs %-6s %+6.1f%%  max|diff|=%.2e" % (
                m, r["ms"], modes[0], 100 * (base_ms - r["ms"]) / base_ms, r["diff"]))
        ua = [e - s for n, i, k, s, e in res["uniform"]["tl"] if n == "A"]
        print("  等长切分下各段 attention 耗时:", " ".join("%.1f" % x for x in ua))
        print("  （第一段记为 1.0 时：%s）" % " ".join("%.2f" % (x / ua[0]) for x in ua))
        print("  saved", path)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
