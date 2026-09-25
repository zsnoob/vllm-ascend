"""用实测数据标定的流/事件离散事件模拟器。

目的：在拿不到机器的时候，先把"all-reduce 异步化能省多少"从聚合可行性界
（240ms 通信 < 336ms 计算）变成一个可证伪的数字。

做法：按 NPU 流的真实语义模拟——每条流按下发顺序执行，一个 op 开始的时刻是
  max(该流空闲时刻, 它等的所有事件的完成时刻)
先用实测的 128k d=4 逐算子时长跑当前调度，看能不能复现实测的 448.2ms；
能复现才敢用它预测别的调度。

时长全部取自 /docker/zzx/vllm-ascend/prof_W131072_d4_v1 的 kernel_details.csv
（单层，Ascend profiler，见 e2e_all.json）。
"""

import json
import sys


class Sim:
    """三条流：comp（计算）、comm（dispatch/combine）、ar（all-reduce，可与 comm 合并）。"""

    def __init__(self, merge_ar_into_comm=True):
        self.t = {"comp": 0.0, "comm": 0.0, "ar": 0.0}
        self.merge = merge_ar_into_comm
        self.log = []
        self.ev = {}
        self._n = 0

    def _s(self, stream):
        if stream == "ar" and self.merge:
            return "comm"
        return stream

    def issue(self, stream, name, dur, waits=()):
        """在 stream 上排一个 op；waits 是它依赖的事件 id。返回完成事件 id。"""
        s = self._s(stream)
        start = self.t[s]
        for w in waits:
            if w is not None:
                start = max(start, self.ev[w])
        end = start + dur
        self.t[s] = end
        self._n += 1
        eid = self._n
        self.ev[eid] = end
        self.log.append((name, s, start, end))
        return eid

    def wait(self, stream, eid):
        """host 在 stream 上插一个等待（stream.wait_event）。"""
        if eid is None:
            return
        s = self._s(stream)
        self.t[s] = max(self.t[s], self.ev[eid])

    def span(self):
        return max(self.t.values())

    def busy(self, s):
        iv = sorted((a, b) for _, st, a, b in self.log if st == s)
        u = []
        for a, b in iv:
            if u and a <= u[-1][1]:
                u[-1][1] = max(u[-1][1], b)
            else:
                u.append([a, b])
        return sum(b - a for a, b in u)

    def covered(self, prefix):
        """带某前缀的通信 op 有多少被计算流盖住。"""
        iv = sorted((a, b) for _, st, a, b in self.log if st == "comp")
        u = []
        for a, b in iv:
            if u and a <= u[-1][1]:
                u[-1][1] = max(u[-1][1], b)
            else:
                u.append([a, b])
        tot = cov = 0.0
        for nm, st, a, b in self.log:
            if not nm.startswith(prefix) or st == "comp":
                continue
            tot += b - a
            cov += sum(max(0.0, min(b, y) - max(a, x)) for x, y in u)
        return tot, cov


# ---------------- 实测时长（128k d=4，单层，Ascend profiler）
# 标定目标：sync 模式下计算流的串行总和应 ≈ 实测层耗时 448.2 ms。
# 实测分项：A=245.0  M=31.5  x=59.3  R=85.9  G=22.3
#          其中 A+M+x=335.8 是真正的计算，R+G=108.2 占着计算流但不算计算。
P = dict(
    A=[13.79, 46.25, 78.14, 106.90],   # 注意力（含 o_proj 矩阵乘）
    M=[8.71, 9.10, 8.90, 4.70],        # 专家 FFN（两个 GroupedMatmul 合并），实测只有 31.4 ms
    D=[12.88, 12.74, 12.97, 12.90],    # A2A dispatch
    C=[15.07, 23.60, 19.80, 21.90],    # A2A combine
    Ra=[7.26, 7.24, 7.26, 12.30],      # o_proj 出口 all-reduce
    Rm=[15.90, 10.20, 13.20, 12.40],   # MoE 出口 all-reduce
    G=[3.55, 3.80, 6.40, 8.50],        # all-gather，同样同步压在计算流
    x_head=11.0,                        # 层首的零散计算
    x_pre=4.0, x_post=8.08,             # 每块零散计算，(59.3-11.0)/4 拆成两段
)
D_ = 4


def sched_sync(P, d, merge):
    """当前 vLLM 的样子：A(i) → R_attn(i) 阻塞 → D(i) → M(i-1)，F 全在层尾。"""
    s = Sim(merge)
    s.issue("comp", "x.head", P["x_head"])
    evd, evc = [None] * d, [None] * d
    for i in range(d):
        s.issue("comp", "A%d" % i, P["A"][i])
        # 同步 all-reduce：直接占着计算流
        s.issue("comp", "Ra%d" % i, P["Ra"][i])
        s.issue("comp", "x.pre%d" % i, P["x_pre"])
        s.issue("comp", "G%d" % i, P["G"][i])
        evd[i] = s.issue("comm", "D%d" % i, P["D"][i])
        if i >= 1:
            j = i - 1
            s.wait("comp", evd[j])
            em = s.issue("comp", "M%d" % j, P["M"][j])
            evc[j] = s.issue("comm", "C%d" % j, P["C"][j], waits=[em])
    j = d - 1
    s.wait("comp", evd[j])
    em = s.issue("comp", "M%d" % j, P["M"][j])
    evc[j] = s.issue("comm", "C%d" % j, P["C"][j], waits=[em])
    for j in range(d):
        s.wait("comp", evc[j])
        s.issue("comp", "F%d" % j, P["x_post"])
        s.issue("comp", "Rm%d" % j, P["Rm"][j])      # 同步
    return s


def sched_async_underM(P, d, merge):
    """把 R_attn 藏在专家计算底下（我原来的深流水设计）。"""
    s = Sim(merge)
    s.issue("comp", "x.head", P["x_head"])
    evd, evc, evr = [None] * d, [None] * d, [None] * d
    pend = []

    def acore(i):
        s.issue("comp", "A%d" % i, P["A"][i])
        evr[i] = s.issue("ar", "Ra%d" % i, P["Ra"][i])

    def apost_d(i):
        s.wait("comp", evr[i])
        s.issue("ar", "G%d" % i, P["G"][i])
        s.issue("comp", "x.pre%d" % i, P["x_pre"])
        evd[i] = s.issue("comm", "D%d" % i, P["D"][i])

    for i in (0, 1):
        acore(i); apost_d(i)
    for k in range(d):
        if k + 2 < d:
            acore(k + 2)
        s.wait("comp", evd[k])
        em = s.issue("comp", "M%d" % k, P["M"][k])
        evc[k] = s.issue("comm", "C%d" % k, P["C"][k], waits=[em])
        if k >= 1:
            s.wait("comp", evc[k - 1])
            s.issue("comp", "F%d" % (k - 1), P["x_post"])
            pend.append(s.issue("ar", "Rm%d" % (k - 1), P["Rm"][k - 1]))
        if k + 2 < d:
            apost_d(k + 2)
    s.wait("comp", evc[d - 1])
    s.issue("comp", "F%d" % (d - 1), P["x_post"])
    pend.append(s.issue("ar", "Rm%d" % (d - 1), P["Rm"][d - 1]))
    for e in pend:
        s.wait("comp", e)
    return s


def sched_async_underA(P, d, merge):
    """把 R_attn(i) 藏在下一块的注意力 A(i+1) 底下——真实系统里 A 才是大块。

    A(i+1) 不依赖 R_attn(i)，所以可以先发 A(i+1)，再回头等 R_attn(i)。
    """
    s = Sim(merge)
    s.issue("comp", "x.head", P["x_head"])
    evd, evc, evr = [None] * d, [None] * d, [None] * d
    pend = []

    def acore(i):
        s.issue("comp", "A%d" % i, P["A"][i])
        evr[i] = s.issue("ar", "Ra%d" % i, P["Ra"][i])

    def apost_d(i):
        s.wait("comp", evr[i])
        s.issue("ar", "G%d" % i, P["G"][i])
        s.issue("comp", "x.pre%d" % i, P["x_pre"])
        evd[i] = s.issue("comm", "D%d" % i, P["D"][i])

    acore(0)
    for k in range(d):
        # 先把下一块的注意力发出去，它会盖住 R_attn(k)
        if k + 1 < d:
            acore(k + 1)
        apost_d(k)                       # 这时 R_attn(k) 早已在 A(k+1) 底下跑完
        s.wait("comp", evd[k])
        em = s.issue("comp", "M%d" % k, P["M"][k])
        evc[k] = s.issue("comm", "C%d" % k, P["C"][k], waits=[em])
        if k >= 1:
            s.wait("comp", evc[k - 1])
            s.issue("comp", "F%d" % (k - 1), P["x_post"])
            pend.append(s.issue("ar", "Rm%d" % (k - 1), P["Rm"][k - 1]))
    s.wait("comp", evc[d - 1])
    s.issue("comp", "F%d" % (d - 1), P["x_post"])
    pend.append(s.issue("ar", "Rm%d" % (d - 1), P["Rm"][d - 1]))
    for e in pend:
        s.wait("comp", e)
    return s


def sched_async_keepD(P, d, merge):
    """同时保住两件事：R_attn(k) 藏在 A(k+1) 底下，D(k) 仍压在计算底下。

    关键是把 apost_d(k+1) 排在 acore(k+2) 之后、M(k) 之前：
    A(k+2) 先盖住 R_attn(k+1)，随后发出的 D(k+1) 又被 M(k) 盖住，
    而 M(k) 消费的 D(k) 是上一轮发的，早已在 A(k+1) 底下跑完。
    """
    s = Sim(merge)
    s.issue("comp", "x.head", P["x_head"])
    evd, evc, evr = [None] * d, [None] * d, [None] * d
    pend = []

    def acore(i):
        s.issue("comp", "A%d" % i, P["A"][i])
        evr[i] = s.issue("ar", "Ra%d" % i, P["Ra"][i])

    def apost_d(i):
        s.wait("comp", evr[i])
        s.issue("ar", "G%d" % i, P["G"][i])
        s.issue("comp", "x.pre%d" % i, P["x_pre"])
        evd[i] = s.issue("comm", "D%d" % i, P["D"][i])

    acore(0)
    acore(1)
    apost_d(0)
    for k in range(d):
        if k + 2 < d:
            acore(k + 2)          # A(k+2) 盖住 R_attn(k+1)
        if k + 1 < d:
            apost_d(k + 1)        # 发 D(k+1)，随后被 M(k) 盖住
        s.wait("comp", evd[k])
        em = s.issue("comp", "M%d" % k, P["M"][k])
        evc[k] = s.issue("comm", "C%d" % k, P["C"][k], waits=[em])
        if k >= 1:
            s.wait("comp", evc[k - 1])
            s.issue("comp", "F%d" % (k - 1), P["x_post"])
            pend.append(s.issue("ar", "Rm%d" % (k - 1), P["Rm"][k - 1]))
    s.wait("comp", evc[d - 1])
    s.issue("comp", "F%d" % (d - 1), P["x_post"])
    pend.append(s.issue("ar", "Rm%d" % (d - 1), P["Rm"][d - 1]))
    for e in pend:
        s.wait("comp", e)
    return s


SCHEDS = [
    ("sync（当前 vLLM）", sched_sync),
    ("async·藏在M底下", sched_async_underM),
    ("async·藏在A底下", sched_async_underA),
    ("async·A底下+保住D", sched_async_keepD),
]

if __name__ == "__main__":
    merge = "--split-ar" not in sys.argv
    print("all-reduce 与 dispatch/combine %s" % ("共用一条通信流" if merge else "各用一条流"))
    print("实测参照：128k d=4 单层 = 448.2 ms\n")
    base = None
    for name, fn in SCHEDS:
        s = fn(P, D_, merge)
        span = s.span()
        if base is None:
            base = span
        ta, ca = s.covered("Ra")
        tm, cm = s.covered("Rm")
        print("%-18s 层 %7.1f ms  vs sync %+6.1f%%   计算流忙 %6.1f  通信流忙 %6.1f" % (
            name, span, 100 * (base - span) / base, s.busy("comp"), s.busy("comm")))
        if ta or tm:
            print("%-18s   R_attn %5.1f ms 覆盖 %5.1f%%   R_moe %5.1f ms 覆盖 %5.1f%%" % (
                "", ta, 100 * ca / ta if ta else 0, tm, 100 * cm / tm if tm else 0))
    if "-v" in sys.argv:
        s = SCHEDS[-1][1](P, D_, merge)
        print("\n最后一种调度的时间线：")
        for nm, st, a, b in sorted(s.log, key=lambda r: r[2]):
            print("  %-8s %-5s %8.2f → %8.2f  (%6.2f)" % (nm, st, a, b, b - a))
