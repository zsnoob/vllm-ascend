"""从 kernel_details.csv 抽一层的时间线，输出 JSON。用法: tl_json.py <csv> <d> <标签>"""
import csv, json, sys

def load(p):
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            st = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except (TypeError, ValueError):
            continue
        rows.append([st, st + du, r["Name"]])
    rows.sort()
    return rows

rows = load(sys.argv[1]); d = int(sys.argv[2]); tag = sys.argv[3]
a2a = [r for r in rows if r[2].startswith("hcom_alltoallv")]
n2 = 2 * d
groups = [a2a[i:i+n2] for i in range(0, len(a2a)-n2+1, n2)]
gi = len(groups)//2; g = groups[gi]
order = ["D0","C0"] if d == 1 else (["D0","D1"] +
    sum([["C%d"%i, "D%d"%(i+2)] for i in range(d-2)], []) + ["C%d"%(d-2), "C%d"%(d-1)])
t1 = g[-1][1]
win0 = groups[gi-1][-1][1] if gi > 0 else g[0][0]-60000
role = {r[0]: order[i] for i, r in enumerate(g)}
out, ai, mi = [], 0, 0
for s, e, n in rows:
    if e < win0 or s > t1: continue
    if n.startswith("hcom_alltoallv"):
        if s in role: out.append([role[s][0], int(role[s][1:]), s, e, "comm"])
    elif n.startswith("hcom_allReduce"): out.append(["R", 0, s, e, "other"])
    elif n.startswith("hcom_allGather"): out.append(["G", 0, s, e, "other"])
    elif n.startswith("hcom"): pass
    elif n == "FusedInferAttentionScore": out.append(["A", ai, s, e, "comp"]); ai += 1
    elif n.startswith("aclnnGroupedMatmulV5"): out.append(["M", mi, s, e, "comp"]); mi += 1
    else: out.append(["x", 0, s, e, "comp"])
t0 = min(x[2] for x in out)
out = [[n, i, round((s-t0)/1000, 2), round((e-t0)/1000, 2), k] for n, i, s, e, k in out]
json.dump({"tag": tag, "d": d, "tl": out}, open("/tmp/tl_%s.json" % tag, "w"))
comp = sorted([(s, e) for n,i,s,e,k in out if k == "comp"])
u = []
for s, e in comp:
    if u and s <= u[-1][1]: u[-1][1] = max(u[-1][1], e)
    else: u.append([s, e])
tot = cov = 0
for n,i,s,e,k in out:
    if k != "comm": continue
    c = sum(max(0, min(e,b)-max(s,a)) for a,b in u); tot += e-s; cov += c
print("%s  层跨度 %.1f ms  算子 %d  MoE通信 %.1f ms 覆盖 %.1f%%" % (
    tag, max(x[3] for x in out), len(out), tot, 100*cov/tot))
