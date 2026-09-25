"""从 kernel_details.csv 还原单层真实时间线，和 FoldMoE Figure 2c 直接对照。

用法: gantt.py <kernel_details.csv> <d> [层序号,默认取中间一层]

计算流 A = FusedInferAttentionScore, M = GroupedMatmul
通信流 D/C = hcom_alltoallv（按发射序定角色）, R = allReduce, G = allGather
"""
import bisect
import csv
import sys

ROLE = {
    "FusedInferAttentionScore": ("A", "comp"),
    "aclnnGroupedMatmulV5_GroupedMatmul_GroupedMatmul": ("M", "comp"),
}


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            st = float(r["Start Time(us)"])
            du = float(r["Duration(us)"])
        except (TypeError, ValueError):
            continue
        rows.append([st, st + du, r["Name"]])
    rows.sort()
    return rows


def merged_compute(rows):
    m = []
    for s, e, n in rows:
        if n.startswith("hcom"):
            continue
        if m and s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return m


def main():
    rows = load(sys.argv[1])
    comp = merged_compute(rows)
    ends = [x[1] for x in comp]

    def ov(s, e):
        i = bisect.bisect_left(ends, s)
        t = 0.0
        while i < len(comp) and comp[i][0] < e:
            t += max(0.0, min(e, comp[i][1]) - max(s, comp[i][0]))
            i += 1
        return t

    a2a = [r for r in rows if r[2].startswith("hcom_alltoallv")]
    if not a2a:
        print("没有 alltoallv —— 不是 EP all-to-all 路径")
        return
    # d 由调用方给出（每层恰好 2d 个 alltoallv），比按间隔分组稳
    d = int(sys.argv[2])
    n2 = 2 * d
    groups = [a2a[i:i + n2] for i in range(0, len(a2a) - n2 + 1, n2)]
    print("alltoallv 共 %d 个，按每层 %d 个切成 %d 层" % (len(a2a), n2, len(groups)))

    gi = int(sys.argv[3]) if len(sys.argv) > 3 else len(groups) // 2
    g = groups[gi]
    if d == 1:
        order = ["D0", "C0"]
    else:
        order = ["D0", "D1"] + sum([["C%d" % i, "D%d" % (i + 2)] for i in range(d - 2)], []) + [
            "C%d" % (d - 2), "C%d" % (d - 1)]
    t0, t1 = g[0][0], g[-1][1]

    # 窗口左界取上一层最后一个 a2a 结束处，保证把本层的 A0 圈进来又不串层
    win0 = groups[gi - 1][-1][1] if gi > 0 else t0 - 60000
    ev = []
    for s, e, n in rows:
        if e < win0 or s > t1:
            continue
        if n.startswith("hcom_alltoallv"):
            continue
        key = None
        for k, (lab, lane) in ROLE.items():
            if n == k:
                key = (lab, lane)
        if key is None:
            if n.startswith("hcom_allReduce"):
                key = ("R", "comm")
            elif n.startswith("hcom_allGather"):
                key = ("G", "comm")
            else:
                continue
        ev.append([s, e, key[0], key[1]])
    for (s, e, _n), lab in zip(g, order):
        ev.append([s, e, lab, "comm"])
    ev.sort()
    # A0 之前的都不要
    firstA = next((i for i, x in enumerate(ev) if x[2] == "A"), 0)
    ev = ev[firstA:]
    base = ev[0][0]

    print("\n=== 第 %d 层真实时间线（d=%d，相对 ms）===" % (gi, d))
    print("%-5s %-5s %9s %9s %8s %8s" % ("op", "lane", "start", "end", "dur", "重叠"))
    for s, e, lab, lane in ev:
        o = ov(s, e) if lane == "comm" else 0.0
        print("%-5s %-5s %9.2f %9.2f %8.2f %8s"
              % (lab, lane, (s - base) / 1000, (e - base) / 1000, (e - s) / 1000,
                 ("%.2f" % (o / 1000)) if lane == "comm" else "-"))

    span = ev[-1][1] - base
    W = 110
    def bar(items):
        line = [" "] * W
        for s, e, lab, _ in items:
            a = int((s - base) / span * W)
            b = max(a + 1, int((e - base) / span * W))
            txt = lab.center(b - a)[: b - a]
            for i, ch in enumerate(txt):
                if a + i < W:
                    line[a + i] = ch
        return "".join(line)

    print("\ncomp |%s|" % bar([x for x in ev if x[3] == "comp"]))
    print("comm |%s|" % bar([x for x in ev if x[3] == "comm"]))
    print("     总跨度 %.1f ms" % (span / 1000))

    tot = sum(e - s for s, e, lab, lane in ev if lane == "comm")
    cov = sum(ov(s, e) for s, e, lab, lane in ev if lane == "comm")
    print("\n本层通信 %.1f ms，被计算盖住 %.1f ms (%.1f%%)" % (tot / 1000, cov / 1000, 100 * cov / tot))
    for pref in ("D", "C", "R", "G"):
        sel = [x for x in ev if x[3] == "comm" and x[2].startswith(pref)]
        if sel:
            t = sum(e - s for s, e, _, _ in sel)
            c = sum(ov(s, e) for s, e, _, _ in sel)
            print("   %s x%-2d  %7.2f ms  重叠 %6.2f ms (%5.1f%%)"
                  % (pref, len(sel), t / 1000, c / 1000, 100 * c / t))


main()
