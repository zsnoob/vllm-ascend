"""每一拍的 D/C 覆盖率，并区分是被 attention 还是被专家 FFN 盖住的。
用法: beat.py <kernel_details.csv> <d>
"""
import bisect, csv, sys, collections

ATT = "FusedInferAttentionScore"
GMM = "aclnnGroupedMatmulV5_GroupedMatmul_GroupedMatmul"

rows = []
for r in csv.DictReader(open(sys.argv[1])):
    try:
        st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
    except (TypeError, ValueError):
        continue
    rows.append([st, st + du, r["Name"]])
rows.sort()
d = int(sys.argv[2])

def merge(iv):
    m = []
    for s, e in sorted(iv):
        if m and s <= m[-1][1]: m[-1][1] = max(m[-1][1], e)
        else: m.append([s, e])
    return m

lanes = {
    "A": merge([(s, e) for s, e, n in rows if n == ATT]),
    "M": merge([(s, e) for s, e, n in rows if n == GMM]),
    "其他": merge([(s, e) for s, e, n in rows if not n.startswith("hcom") and n not in (ATT, GMM)]),
}
idx = {k: [x[1] for x in v] for k, v in lanes.items()}

def ov(lane, s, e):
    L = lanes[lane]; i = bisect.bisect_left(idx[lane], s); t = 0.0
    while i < len(L) and L[i][0] < e:
        t += max(0.0, min(e, L[i][1]) - max(s, L[i][0])); i += 1
    return t

a2a = [r for r in rows if r[2].startswith("hcom_alltoallv")]
n2 = 2 * d
groups = [a2a[i:i + n2] for i in range(0, len(a2a) - n2 + 1, n2)]
order = ["D0", "D1"] + sum([["C%d" % i, "D%d" % (i + 2)] for i in range(d - 2)], []) + ["C%d" % (d - 2), "C%d" % (d - 1)]
# 1A1M 里每一拍的「搭档」：D0 配 A1，D(i+1) 配 M(i)，C(i) 配 A(i+2)，末尾配 M
pair = {"D0": "A1"}
for i in range(1, d): pair["D%d" % i] = "M%d" % (i - 1)
for i in range(d):
    pair["C%d" % i] = ("A%d" % (i + 2)) if i + 2 < d else ("M%d" % (d - 1) if i == d - 2 else "—")

agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for g in groups:
    for k, (s, e, _) in zip(order, g):
        a = agg[k]; a[0] += 1; a[1] += (e - s) / 1000
        a[2] += ov("A", s, e) / 1000; a[3] += ov("M", s, e) / 1000; a[4] += ov("其他", s, e) / 1000

print("共 %d 层" % len(groups))
print("%-4s %-6s %9s %9s %8s %9s %9s" % ("拍", "应配", "总ms", "覆盖ms", "覆盖率", "其中A", "其中M"))
for k in order:
    v = agg[k]
    cov = max(v[2], v[3], v[4]) if False else 0
    # 三条计算泳道互斥（同一时刻只有一条在跑），直接相加
    cov = v[2] + v[3] + v[4]
    print("%-4s %-6s %9.1f %9.1f %7.1f%% %9.1f %9.1f" % (k, pair[k], v[1], cov, 100 * cov / v[1], v[2], v[3]))

# 每一拍的 attention / 专家 FFN 时长
att = [x for x in rows if x[2] == ATT]
gm = [x for x in rows if x[2] == GMM]
def per_beat(lst, k, label):
    du = [(e - s) / 1000 for s, e, _ in lst]
    off = next((i for i in range(1, len(du)) if du[i] < du[i - 1] / 2), 0)
    gg = [du[i:i + k] for i in range(off, len(du) - k + 1, k)]
    mid = [x for x in gg[len(gg) // 2:len(gg) // 2 + 6] if len(x) == k]
    if not mid: return None
    return [sum(l[i] for l in mid) / len(mid) for i in range(k)]
A = per_beat(att, d, "A")
M = per_beat(gm, 2 * d, "M")
if A: print("\n每层 A_i (ms):  " + "  ".join("%s=%.1f" % ("A%d" % i, x) for i, x in enumerate(A)))
if M: print("每层 M_i (ms):  " + "  ".join("%s=%.1f" % ("M%d" % i, M[2*i] + M[2*i+1]) for i in range(d)))
