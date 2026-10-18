"""Print compact val/test mean+-std tables from the regenerated summaries (read-only report helper)."""
import csv
O = "outputs/ablation/"
def L(n): return list(csv.DictReader(open(O + n, encoding="utf-8-sig")))
V = L("summary_val_mean_std.csv"); T = L("test_summary.csv")
print("nseeds", set(r["n_seeds"] for r in V + T))
key = lambda r: (r["setting"], r["task"], r["model"], r["condition"])
v = {key(r): r for r in V}; t = {key(r): r for r in T}
order = ["b3_L123","b4_L123","b5_L123","b6_L123","b7_L123","b8_L123","b9_L123","b10_L123","b12_L123","b16_L123","b6_L1","b6_L12","b6_L1234","b6_L12345"]
for cond in ["mean", "20_0", "30_2"]:
    print("==", cond, "acc val / test  [bearing cnn | bearing gcn | gear cnn | gear gcn]")
    for s in order:
        cells = []
        for task in ["bearing", "gear"]:
            for m in ["cnn", "gcn"]:
                a = v[(s, task, m, cond)]; b = t[(s, task, m, cond)]
                cells.append(f"{float(a['acc_mean']):.3f}+-{float(a['acc_std']):.3f} / {float(b['acc_mean']):.3f}+-{float(b['acc_std']):.3f}")
        print(s.ljust(10), " | ".join(cells))
print("== mean cond sens/spec/f1")
for s in order:
    for task in ["bearing", "gear"]:
        for m in ["cnn", "gcn"]:
            a = v[(s, task, m, "mean")]; b = t[(s, task, m, "mean")]
            print(s, task, m, "val", a["sens_mean"], a["spec_mean"], a["f1_mean"], "| test", b["sens_mean"], b["spec_mean"], b["f1_mean"])
