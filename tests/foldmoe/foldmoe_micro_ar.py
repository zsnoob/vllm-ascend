"""FoldMoE 单层微基准 —— TP all-reduce 能不能藏起来。

在 1A1M 流水线上补齐 TP=2 的两次 all-reduce（端到端 profiler 里每块正好两次：
注意力 o_proj 出口一次 R_attn、MoE 出口一次 R_moe）。

五种模式，设计成只差一个变量的链条：

    seq         不切块不流水，all-reduce 同步                —— 绝对基线
    sync        1A1M（Acore→D→M），all-reduce 同步压在计算流  —— 当前 vLLM 的样子
    deep_sync   深流水（见下），all-reduce 仍然同步
    deep_async1 深流水，all-reduce 异步，与 D/C 共用一条通信流
    deep_async2 深流水，all-reduce 异步，单独一条流

    deep_sync  vs sync         = 改调度本身的代价/收益
    deep_async vs deep_sync    = **all-reduce 异步化的净收益**（唯一变量）
    async2     vs async1       = "TP 和 EP 共用同一条物理链路"假设

为什么要深流水：naive 的做法是 `Acore(i); M(i-1); Apost(i); D(i)`——让 R_attn(i)
躲在 M(i-1) 底下。但 Apost(i) 必须等 R_attn(i)，于是 D(i) 被挤到 M(i-1) 之后，
1A1M 原本"dispatch 压在专家计算底下"的重叠就丢了，代价比想藏的 all-reduce 还大。
所以把注意力再提前一拍：

    每拍 k：  Acore(k+2)   算 chunk k+2 的注意力，发 R_attn(k+2)
             M(k)        chunk k 的专家计算（约 40ms），盖住 R_attn(k+2) 和 R_moe(k-1)
             F(k-1)      收 chunk k-1 的 combine，发 R_moe(k-1)
             Apost(k+2)  等 R_attn(k+2)
             D(k+2)      发 dispatch，两拍后才被 M(k+2) 消费 → 重新压回专家计算底下

首尾的 R_attn(0)、R_attn(1) 和最后一个 R_moe 没有配对对象，是结构性的。

形状取 DeepSeek-V2-Lite（TP=2/EP=2 每卡份额），与端到端实测同构：
hidden 2048，每卡 8 个头，64 个路由专家 top-6（每卡 32 个），moe_intermediate 1408。

用法：torchrun --nproc_per_node 2 foldmoe_micro_ar.py <L> <d>   (d >= 4)
"""

import functools
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch_npu

OUT = os.environ.get("FOLDMOE_OUT", ".")
H, HEADS, DN, DR, DV = 2048, 8, 128, 64, 128
E, TOPK, INTER = 64, 6, 1408

DEEP = ("deep_sync", "deep_async1", "deep_async2")
ASYNC = ("deep_async1", "deep_async2")


@functools.lru_cache(None)
def slices_uniform(L, d):
    q, r = divmod(L, d)
    out, s = [], 0
    for i in range(d):
        e = s + q + (1 if i < r else 0)
        out.append((s, e))
        s = e
    return out


def main():
    L, d = int(sys.argv[1]), int(sys.argv[2])
    assert d >= 4, "深流水需要 d>=4"
    rank = int(os.environ["RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    # all-reduce 用独立通信域，和 all-to-all 的默认域分开（物理链路仍是同一条）
    tp = dist.new_group(list(range(world)))
    dev = torch.device("npu")
    bf = torch.bfloat16
    comp = torch.npu.current_stream()
    comm = torch.npu.Stream()      # D / C
    ars = torch.npu.Stream()       # deep_async2 专用的 all-reduce 流
    E_loc = E // world

    g = torch.Generator().manual_seed(1234 + rank)
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(bf).to(dev)
    q_nope, q_pe = rnd(L, HEADS, DN), rnd(L, HEADS, DR)
    k_nope, k_pe, v = rnd(L, HEADS, DN), rnd(L, HEADS, DR), rnd(L, HEADS, DV)
    # o_proj 行并行切片：每卡吃自己那 8 个头，出来是部分和，要 all-reduce
    w_o = rnd(HEADS * DV, H)
    gw = torch.Generator().manual_seed(99 + rank)
    # 权重布局必须和 vllm 的 unquant_apply_mlp 一致：按 [E, 出, 入] 存，调用前 transpose(1,2)。
    # 直接按 [E, 入, 出] 连续存虽然形状相同，但 npu_grouped_matmul 会走慢路径——
    # 实测同样的 FLOPs 慢 5.5 倍（39 TFLOPS vs 真实 vLLM 的 215 TFLOPS）。
    w13 = (torch.randn(E_loc, 2 * INTER, H, generator=gw) * 0.02).to(bf).to(dev).transpose(1, 2)
    w2 = (torch.randn(E_loc, H, INTER, generator=gw) * 0.02).to(bf).to(dev).transpose(1, 2)
    mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(dev)
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
        """注意力 + o_proj。返回本卡那几个头的部分和，尚未 all-reduce。"""
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
        bnd = slices_uniform(L, d)
        n = len(bnd)
        fifo, st = [], [dict() for _ in range(n)]
        out = torch.empty(L, H, dtype=bf, device=dev)
        ar_stream = comm if mode == "deep_async1" else ars
        is_async = mode in ASYNC
        pend_ar = []

        def take(s, e):
            parts = [t[max(s, a) - a:min(e, b) - a] for a, b, t in fifo if a < e and b > s]
            while fifo and fifo[0][1] <= e:
                fifo.pop(0)
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        def ar_sync(x, role, i):
            rec(role, i, "comp", lambda: dist.all_reduce(x, group=tp))

        def ar_async(x, role, i):
            ev0 = comp.record_event()
            with torch.npu.stream(ar_stream):
                ar_stream.wait_event(ev0)
                rec(role, i, "ar", lambda: dist.all_reduce(x, group=tp))
                ev = ar_stream.record_event()
            x.record_stream(ar_stream)
            return ev

        # ---------------- 各阶段
        def Acore(i):
            """注意力 + o_proj 部分和；async 下顺手把 R_attn 发出去。"""
            s, e = bnd[i]
            x = rec("A", i, "comp", lambda: attn(s, e))
            if is_async:
                st[i]["evr"] = ar_async(x, "Ra", i)
                st[i]["x"] = x
            else:
                ar_sync(x, "Ra", i)
                fifo.append((s, e, x))

        def Apost(i):
            """async 下等 R_attn 落地，再把这块交给 MoE。"""
            if not is_async:
                return
            s, e = bnd[i]
            comp.wait_event(st[i]["evr"])
            fifo.append((s, e, st[i].pop("x")))

        def D(j):
            s, e = bnd[j]
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
            """收 combine，加权求和，然后是 MoE 出口那次 all-reduce。"""
            s, e = bnd[j]
            p = st[j]["p"]
            comp.wait_event(st[j]["evc"])

            def fin():
                o = out[s:e]
                o.zero_()
                o.index_add_(0, p["tok"], st[j].pop("back") * p["w"])
            rec("F", j, "comp", fin)
            if is_async:
                pend_ar.append(ar_async(out[s:e], "Rm", j))
            else:
                ar_sync(out[s:e], "Rm", j)

        # ---------------- 调度
        if mode == "seq":
            for i in range(n):
                Acore(i); Apost(i); D(i); M(i); F(i)
        elif mode == "sync":
            # 1A1M 原样：D 紧跟 Acore，F 全堆在层尾（当前 vLLM 的形状）
            Acore(0); Apost(0); D(0)
            for i in range(1, n):
                Acore(i); Apost(i); D(i); M(i - 1)
            M(n - 1)
            for j in range(n):
                F(j)
        else:
            # 深流水：注意力提前两拍，D 依然压在专家计算底下，F 回到流水线里
            for i in (0, 1):
                Acore(i); Apost(i); D(i)
            for k in range(n):
                if k + 2 < n:
                    Acore(k + 2)          # 发 R_attn(k+2)
                M(k)                       # 盖住 R_attn(k+2) 与 R_moe(k-1)
                if k >= 1:
                    F(k - 1)               # 发 R_moe(k-1)
                if k + 2 < n:
                    Apost(k + 2)           # 等 R_attn(k+2)
                    D(k + 2)               # 两拍后才被 M(k+2) 消费
            F(n - 1)
        for ev in pend_ar:
            comp.wait_event(ev)
        return out

    def plain(name, i, kind, fn):
        return fn()

    modes = ["seq", "sync", "deep_sync", "deep_async1", "deep_async2"]
    res, ref = {}, None
    for mode in modes:
        for _ in range(2):
            run(mode, plain)
        torch.npu.synchronize()
        times = []
        for _ in range(5):
            dist.barrier()
            torch.npu.synchronize()
            t0 = torch.npu.Event(enable_timing=True)
            t1 = torch.npu.Event(enable_timing=True)
            t0.record()
            o = run(mode, plain)
            t1.record()
            torch.npu.synchronize()
            times.append(t0.elapsed_time(t1))
        evs = []

        def ev_rec(name, i, kind, fn):
            stream = {"comp": comp, "comm": comm,
                      "ar": (comm if mode == "deep_async1" else ars)}[kind]
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
        o = run(mode, ev_rec)
        torch.npu.synchronize()
        tl = [(nm, i, k, base.elapsed_time(a), base.elapsed_time(b)) for nm, i, k, a, b in evs]
        if ref is None:
            ref = o.clone()
        diff = (o.float() - ref.float()).abs().max().item()
        res[mode] = dict(ms=statistics.median(times), all=times, diff=diff, tl=tl)
        del o
        torch.npu.empty_cache()

    if rank == 0:
        path = OUT + "/micro_ar_L%d_d%d.json" % (L, d)
        json.dump(res, open(path, "w"))
        b = res["seq"]["ms"]

        def cov(tl, roles):
            """这些 all-reduce 有多少被计算盖住。"""
            iv = sorted([(s, e) for nm, i, k, s, e in tl if k == "comp"])
            u = []
            for s, e in iv:
                if u and s <= u[-1][1]:
                    u[-1][1] = max(u[-1][1], e)
                else:
                    u.append([s, e])
            tot = c = 0
            for nm, i, k, s, e in tl:
                if nm not in roles:
                    continue
                tot += e - s
                c += sum(max(0, min(e, y) - max(s, x)) for x, y in u)
            return tot, c

        print("TP all-reduce 微基准  L=%d d=%d  (rank0，5 次中位数)" % (L, d))
        for m in modes:
            r = res[m]
            print("  %-12s %8.2f ms  vs seq %+6.1f%%  max|diff|=%.2e" % (
                m, r["ms"], 100 * (b - r["ms"]) / b, r["diff"]))
        print("  --- all-reduce 拆开看（同步模式下它占着计算流，覆盖率按定义是 0）")
        for m in modes:
            ta, ca = cov(res[m]["tl"], ("Ra",))
            tm, cm = cov(res[m]["tl"], ("Rm",))
            pa = 100 * ca / ta if ta else 0
            pm = 100 * cm / tm if tm else 0
            print("  %-12s R_attn %6.1f ms 覆盖 %5.1f%%   R_moe %6.1f ms 覆盖 %5.1f%%" % (
                m, ta, pa, tm, pm))
        print("  --- 单变量对比")
        s_, ds, a1, a2 = (res[k]["ms"] for k in ("sync", "deep_sync", "deep_async1", "deep_async2"))
        print("  改调度     deep_sync vs sync        %+6.2f%%" % (100 * (s_ - ds) / s_))
        print("  异步AR(同流) deep_async1 vs deep_sync %+6.2f%%" % (100 * (ds - a1) / ds))
        print("  异步AR(独流) deep_async2 vs deep_sync %+6.2f%%" % (100 * (ds - a2) / ds))
        print("  独流 vs 同流 deep_async2 vs deep_async1 %+6.2f%%" % (100 * (a1 - a2) / a1))
        print("  结构上限：2d=%d 次 all-reduce 中，首尾 %d 次无配对对象" % (2 * d, 3))
        print("  saved", path)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
