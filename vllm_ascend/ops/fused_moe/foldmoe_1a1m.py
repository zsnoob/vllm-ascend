"""FoldMoE 1A1M —— 把 attention 拉进 MoE 的通信流水线（推理 prefill 版）。

参考 Zhu et al., "FoldMoE: Efficient Long Sequence MoE Training via
Attention-MoE Pipelining", ACL 2025, Figure 2c。原文是训练场景，这里搬到
推理 prefill。

一层内把 token 切成 d 个 micro-batch，计算流按

    A0 A1 M0 A2 M1 A3 M2 ... A(d-1) M(d-3) M(d-2) M(d-1)

下发，通信流自然排成

    D0 D1 C0 D2 C1 D3 C2 ... D(d-1) C(d-2) C(d-1)

于是除了首尾两拍，每个 dispatch/combine 都压在一段 attention 或专家计算下面。

attention 用 token buffer：已算出的 k_nope/k_pe/value 留在一块预分配 buffer 里，
chunk i 的 query 对 buffer[:end_i] 做一次 sparse_mode=3（rightDownCausal）的 FIA。
和整段因果注意力数值等价（tests/fia_slice_check.py 验证过），FLOPs 相同，
不重算 kv_b_proj、不做 LSE 合并。

环境变量：
    FOLDMOE_1A1M    micro-batch 数，<=1 关闭
    FOLDMOE_UNIFORM 1 = attention 用 time-uniform 切分（Algorithm 1），默认 0 = 等长切分。
                    MoE 永远是等长 micro-batch，两者之间靠 token buffer（论文 §4.2 /
                    Figure 6b）解耦：attention 输出先入队，MoE 按固定大小 FIFO 取。
"""

import dataclasses
import os

import torch
import torch_npu

_ORIG = {}
_COMM = None


class _Yield(Exception):
    """在 fused_experts 里提前返回，跳过后面的 shared experts / all_reduce。"""


class _Done(Exception):
    """v2 调度跑完，跳到 finally 做清理。"""


class _S:
    kv = None        # attention token buffer 状态，None = 不拦截
    stage = None     # ("D"|"M"|"F", chunk_idx)，None = 不拦截
    pend = {}
    n_log = 0


def _comm_stream():
    global _COMM
    if _COMM is None:
        _COMM = torch.npu.Stream()
    return _COMM


def _log(msg):
    _S.n_log += 1
    if _S.n_log in (1, 10, 100, 1000, 5000):
        print("[1A1M] %s (#%d)" % (msg, _S.n_log), flush=True)


def _degree():
    try:
        return int(os.environ.get("FOLDMOE_1A1M", "0"))
    except ValueError:
        return 0


# ---------------------------------------------------------------- 切分方案


_SLICE_CACHE = {}


def _slices_uniform(total, d):
    q, r = divmod(total, d)
    out, s = [], 0
    for i in range(d):
        e = s + q + (1 if i < r else 0)
        out.append((s, e))
        s = e
    return out


def _slices_time_uniform(total, d, hidden, n_heads):
    """FoldMoE Algorithm 1: quick-start time-uniform attention slicing.

    FLOPs(l, c) = (4H + 3h) * l * c + 8 * H^2 * l
    第一片取 L/d 让首个 A2A 尽早发出，其余片按 attention 延迟均匀切。
    """
    H, h = hidden, n_heads
    flops = lambda l, c: (4 * H + 3 * h) * l * c + 8 * H * H * l
    ck = (total, d, H, h)
    if ck in _SLICE_CACHE:
        return _SLICE_CACHE[ck]
    m = -(-total // d)
    sizes = [m]
    L = total
    t_hat = ((4 * H + 3 * h) * L * (L + 1) / 2 + 8 * H * H * L) / d
    start = m
    while start < total:
        end = max(start + 1, (len(sizes) + 1) * m)
        if total - end >= d - len(sizes):
            best, best_err = end, None
            for i in range(end, total + 1):
                err = abs(flops(i - start, i) - t_hat)
                if best_err is None or err < best_err:
                    best, best_err = i, err
                else:
                    break
            end = best
        end = min(end, total)
        sizes.append(end - start)
        start = end
    out, s = [], 0
    for n in sizes:
        out.append((s, s + n))
        s += n
    _SLICE_CACHE[ck] = out
    return out


# ------------------------------------------------------- attention: 切片 + buffer


def _mla_forward(self, layer_name, hidden_states, kv_cache, attn_metadata,
                 need_gather_q_kv=False, output=None):
    kv = _S.kv
    if kv is None or attn_metadata is None:
        return _ORIG["mla_fwd"](self, layer_name, hidden_states, kv_cache,
                                attn_metadata, need_gather_q_kv, output)
    s, e = kv["s"], kv["e"]
    n = e - s
    md = attn_metadata
    pm = md.prefill
    sl = lambda t: None if t is None else t[s:e]
    md_i = dataclasses.replace(
        md,
        num_actual_tokens=n,
        num_actual_tokens_pcp_padded=n,
        num_input_tokens=n,
        slot_mapping=md.slot_mapping[s:e],
        prefill=dataclasses.replace(
            pm,
            cos=sl(pm.cos),
            sin=sl(pm.sin),
            input_positions=sl(pm.input_positions),
            actual_seq_lengths_q=[n],
            max_query_len=n,
            max_seq_lens=e,
        ),
    )
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    old = _EXTRA_CTX.num_tokens
    _EXTRA_CTX.num_tokens = n
    try:
        return _ORIG["mla_fwd"](self, layer_name, hidden_states, kv_cache, md_i,
                                need_gather_q_kv, output)
    finally:
        _EXTRA_CTX.num_tokens = old


def _forward_prefill(self, q_nope, q_pe, k_nope, k_pe, value,
                     kv_c_and_k_pe_cache, attn_metadata):
    kv = _S.kv
    if kv is None:
        return _ORIG["fwd_prefill"](self, q_nope, q_pe, k_nope, k_pe, value,
                                    kv_c_and_k_pe_cache, attn_metadata)
    s, e, T = kv["s"], kv["e"], kv["T"]
    key = id(self)
    b = kv["bufs"].get(key)
    if b is None:
        mk = lambda t: torch.empty((T,) + tuple(t.shape[1:]), dtype=t.dtype,
                                   device=t.device)
        b = {"k": mk(k_nope), "r": mk(k_pe), "v": mk(value)}
        kv["bufs"][key] = b
    b["k"][s:e] = k_nope
    b["r"][s:e] = k_pe
    b["v"][s:e] = value

    num_tokens = q_nope.size(0)
    orig_dtype = q_nope.dtype
    if orig_dtype != torch.bfloat16:
        q_nope, q_pe = q_nope.to(torch.bfloat16), q_pe.to(torch.bfloat16)

    kk, rr, vv = b["k"][:e], b["r"][:e], b["v"][:e]
    kwargs = {
        "num_heads": self.num_heads,
        "num_key_value_heads": self.num_heads,
        "input_layout": "TND",
        "atten_mask": attn_metadata.prefill.attn_mask,
        "sparse_mode": 3,
        "scale": self.scale,
        "antiquant_mode": 0,
        "antiquant_scale": None,
        "block_table": None,
        "block_size": 0,
        "softmax_lse_flag": True,
        "actual_seq_lengths": [num_tokens],
        "actual_seq_lengths_kv": [e],
    }
    if self.head_padding > 0:
        query = torch.cat((q_nope, q_pe), dim=-1)
        keyt = torch.cat((kk, rr), dim=-1)
    else:
        kwargs["query_rope"] = q_pe
        kwargs["key_rope"] = rr
        query, keyt = q_nope, kk

    attn_out, _ = torch_npu.npu_fused_infer_attention_score(query, keyt, vv, **kwargs)
    attn_out = attn_out.reshape([num_tokens, self.num_heads * self.v_head_dim])
    if orig_dtype != torch.bfloat16:
        attn_out = attn_out.to(orig_dtype)
    return attn_out


# ------------------------------------------------------------ MoE: 三段式


def _finish(self, idx, fused_experts_input):
    """收 combine(idx)，组回 FusedExpertsResult，后面的 shared experts / all_reduce 正常跑。"""
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

    p = _S.pend[idx]
    torch.npu.current_stream().wait_event(p["evc"])
    routed_out = self.token_dispatcher.token_combine_finish(p["stc"])
    tdo = p["tdo"]
    _S.pend.pop(idx, None)
    return FusedExpertsResult(
        routed_out=routed_out,
        before_dispatch_evt=p["bd"],
        before_gmm2_evt=p["g2"],
        before_combine_evt=p["bc"],
        group_list_type=tdo.group_list_type,
        expert_tokens=tdo.group_list,
        swiglu_limit=fused_experts_input.swiglu_limit,
    )


def _fused_experts(self, fused_experts_input):
    stg = _S.stage
    if stg is None:
        return _ORIG["fused_experts"](self, fused_experts_input)
    mode, idx, fin = stg
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        build_mlp_compute_input,
        build_token_dispatch_input,
    )

    cur = torch.npu.current_stream()
    comm = _comm_stream()

    if mode == "D":
        routed = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed = fused_experts_input.routing.log2phy[routed]
        tdi = build_token_dispatch_input(fused_experts_input=fused_experts_input,
                                         topk_ids=routed)
        st = {}
        ev0 = cur.record_event()
        with torch.npu.stream(comm):
            comm.wait_event(ev0)
            self.token_dispatcher.token_dispatch(token_dispatch_input=tdi,
                                                 _defer_state=st)
            evd = comm.record_event()
        _S.pend[idx] = {"st": st, "evd": evd, "bd": ev0, "fei": fused_experts_input}
        _log("dispatch issued")
        raise _Yield()

    if mode == "F":
        return _finish(self, idx, fused_experts_input)

    p = _S.pend[idx]
    if True:
        # 合并模式下本次调用的 fused_experts_input 属于 chunk `fin`，
        # chunk `idx` 的专家计算要用 stage D 存下来的那份。
        fei_i = p.get("fei") or fused_experts_input
        cur.wait_event(p["evd"])
        tdo = self.token_dispatcher.token_dispatch_finish(p["st"])
        mci = build_mlp_compute_input(fused_experts_input=fei_i,
                                      token_dispatch_output=tdo,
                                      use_fusion_ops=self.use_fusion_ops)
        mlp_out, gmm2 = self._apply_mlp(mci)
        evm = cur.record_event()
        stc = {}
        with torch.npu.stream(comm):
            comm.wait_event(evm)
            self.token_dispatcher.token_combine(hidden_states=mlp_out,
                                                combine_metadata=tdo.combine_metadata,
                                                _defer_state=stc)
            evc = comm.record_event()
        p.update(stc=stc, evc=evc, tdo=tdo, g2=gmm2, bc=evm, fei=None)
        if fin is None:
            raise _Yield()
        return _finish(self, fin, fused_experts_input)


def _moe(layer, x, mode, idx, fin=None):
    """一个 chunk 要进三次 mlp（发 dispatch / 跑专家发 combine / 收 combine），
    但 vllm 的 moe_layer_index 每次 moe_forward 都会自增，所以这里存档回滚，
    整层只在最后统一 +1（见 _layer_forward）。"""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    saved = getattr(fc, "moe_layer_index", None)
    _S.stage = (mode, idx, fin)
    try:
        return layer.mlp(x)
    except _Yield:
        return None
    finally:
        _S.stage = None
        if saved is not None:
            fc.moe_layer_index = saved


# --------------------------------------------------------------- 层调度


def _get_md():
    from vllm.forward_context import get_forward_context

    try:
        fc = get_forward_context()
    except Exception:
        return None
    md = getattr(fc, "attn_metadata", None)
    if isinstance(md, dict):
        md = next(iter(md.values())) if md else None
    return md


def _applicable(layer, md, total):
    if md is None or total < 1024:
        return "no_md_or_small"
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not hasattr(mlp, "experts"):
        return "dense_layer"
    if getattr(layer, "use_mha", False):
        return "mha"
    if getattr(md, "num_decodes", 0) or getattr(md, "num_decode_tokens", 0):
        return "has_decode"
    if getattr(md, "num_prefills", 0) != 1:
        return "num_prefills!=1"
    pm = getattr(md, "prefill", None)
    if pm is None:
        return "no_prefill_md"
    if getattr(pm, "chunked_context", None) is not None:
        return "chunked_context"
    if getattr(md, "num_actual_tokens", -1) != total:
        return "token_mismatch"
    return None


def _layer_forward(self, positions, hidden_states, residual, llama_4_scaling=None):
    d = _degree()
    total = hidden_states.shape[0]
    if d <= 1:
        return _ORIG["layer"](self, positions, hidden_states, residual, llama_4_scaling)
    why = _applicable(self, (md := _get_md()), total)
    if why is not None or hidden_states.dtype == torch.float16:
        _log("FALLBACK: " + (why or "fp16"))
        return _ORIG["layer"](self, positions, hidden_states, residual, llama_4_scaling)
    d = min(d, max(1, total // 512))
    if d <= 1:
        return _ORIG["layer"](self, positions, hidden_states, residual, llama_4_scaling)

    mbnd = _slices_uniform(total, d)
    if os.environ.get("FOLDMOE_UNIFORM", "0") == "1":
        cfg = self.self_attn
        abnd = _slices_time_uniform(total, d,
                                    getattr(cfg, "hidden_size", 2048),
                                    getattr(cfg, "num_heads", 16))
    else:
        abnd = mbnd

    if residual is None:
        residual = hidden_states.clone()
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    kv = {"T": total, "bufs": {}, "s": 0, "e": 0}
    hs = [None] * d
    rs = [None] * d
    outs = [None] * d

    fifo = []  # token buffer：attention 输出按 (s, e, tensor) 入队

    def take(s, e):
        """从 token buffer 取 [s, e) 组成一个 MoE micro-batch，取完的块出队。"""
        parts = [t[max(s, a) - a:min(e, b) - a] for a, b, t in fifo if a < e and b > s]
        while fifo and fifo[0][1] <= e:
            fifo.pop(0)
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def Acore(i):
        """只跑 attention（含 o_proj），输出进 token buffer。
        Algorithm 1 可能切出少于 d 片，多出来的拍没有 attention。"""
        if i >= len(abnd):
            return
        s, e = abnd[i]
        kv["s"], kv["e"] = s, e
        _S.kv = kv
        try:
            kwargs = {"positions": positions[s:e], "hidden_states": hidden_states[s:e]}
            if not getattr(self, "use_mha", False):
                kwargs["llama_4_scaling"] = llama_4_scaling
            fifo.append((s, e, self.self_attn(**kwargs)))
        finally:
            _S.kv = None

    def Apost(i):
        """post_ln + 发 dispatch。单独成段，好把它排到阻塞的 F 之后。"""
        s, e = mbnd[i]
        hs[i], rs[i] = self.post_attention_layernorm(take(s, e), residual[s:e])
        _moe(self, hs[i], "D", i)

    def A(i):
        Acore(i)
        Apost(i)

    def M(i):
        _moe(self, hs[i], "M", i)

    def MF(i, j):
        outs[j] = _moe(self, hs[j], "M", i, fin=j)
        hs[j] = None

    def F(i):
        outs[i] = _moe(self, hs[i], "F", i)
        hs[i] = None

    merge = os.environ.get("FOLDMOE_MERGE", "1") == "1"
    # v2 把阻塞的 F 挪到通信空档，每多一拍就多救一个 combine，但自身有固定下发开销。
    # 实测 d<=4 时开销盖过收益（64k/128k 均略输），d>=8 才净赚（256k +9.0%->+10.7%）。
    sched = os.environ.get("FOLDMOE_SCHED", "auto")
    if sched == "auto":
        sched = "v2" if d >= 8 else "v1"
    try:
        _log("PIPELINED 1A1M: d=%d tokens=%d merge=%d sched=%s attn=%s moe=%s"
             % (d, total, merge, sched, [e - s for s, e in abnd], [e - s for s, e in mbnd]))
        if sched == "v2":
            # 把阻塞的 F 夹在 Acore 和 Apost 之间：F 执行时没有 dispatch 在飞，
            # 而 C_i 已经被前面那段 Acore 盖住了。
            A(0)
            if d > 1:
                A(1)
            M(0)
            if d > 2:
                Acore(2)
                Apost(2)
            for i in range(1, d):
                M(i)
                if i + 2 < d:
                    Acore(i + 2)
                F(i - 1)
                if i + 2 < d:
                    Apost(i + 2)
            F(d - 1)
            from vllm.forward_context import get_forward_context

            _fc2 = get_forward_context()
            if getattr(_fc2, "moe_layer_index", None) is not None:
                _fc2.moe_layer_index += 1
            raise _Done()
        A(0)
        if d > 1:
            A(1)
        if merge:
            M(0)
            if 2 < d:
                A(2)
            for i in range(1, d):
                MF(i, i - 1)
                if i + 2 < d:
                    A(i + 2)
        else:
            for i in range(d):
                M(i)
                if i + 2 < d:
                    A(i + 2)
                if i >= 1:
                    F(i - 1)
        F(d - 1)
        from vllm.forward_context import get_forward_context

        _fc = get_forward_context()
        if getattr(_fc, "moe_layer_index", None) is not None:
            _fc.moe_layer_index += 1
    except _Done:
        pass
    finally:
        _S.kv = None
        _S.stage = None
        _S.pend.clear()
        kv["bufs"].clear()

    out = torch.cat(outs, dim=0)
    if rs[0].data_ptr() == residual.data_ptr():
        res = residual                 # RMSNorm 原地写回，slice 即原张量
    else:
        res = torch.cat(rs, dim=0)
    return out, res


# ------------------------------------------------------------------- 安装


def apply():
    if _ORIG:
        return False
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl
    from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod

    _ORIG["layer"] = DeepseekV2DecoderLayer.forward
    _ORIG["mla_fwd"] = AscendMLAImpl.forward
    _ORIG["fwd_prefill"] = AscendMLAImpl._forward_prefill
    _ORIG["fused_experts"] = MoECommMethod.fused_experts

    DeepseekV2DecoderLayer.forward = _layer_forward
    AscendMLAImpl.forward = _mla_forward
    AscendMLAImpl._forward_prefill = _forward_prefill
    MoECommMethod.fused_experts = _fused_experts
    return True
