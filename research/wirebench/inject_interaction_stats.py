# オラクル注入×AA の交互作用の統計（研究メモ §9ar-1・§9ao。2026-10-05 の統計修正）。
# それまでセッション内のインライン計算だけだった数値をコードに残す：
#   (1) AA無しの注入効果：素3DGS 7k の12run（4条件のプール。監査R-05の循環の指摘）と、注入runと同一設定（newdef＝現行3DGS既定）の3runの両方と比べる
#   (2) AA有りの注入効果（Mip 7k、各9run）
#   (3) 交互作用 (注入−素)_AA無し − (注入−素)_AA有り を Welch–Satterthwaite の t（自由度は4群から合成）で検定
#       （旧版の z=2.66・p=0.008 は正規近似で、n が小さいので甘い）
#   (4) §9ao の「検出力80%で拾える最小効果」を非心 t で計算し直す（旧記述 0.34dB は再現不能）
# 指標は電線のみPSNR（eval_wires.py の psnr_wireonly、md2）。summary.json が無い run は eval_wires.py をその場で実行する。
# 使い方: python3 research/wirebench/inject_interaction_stats.py   → 標準出力＋ inject_interaction_stats.json（このdir）
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

E = Path("/workspace/experiments")
HERE = Path(__file__).parent
BASE = "20260704_wb_r15mm_f80_open"
INJ = "20260710_wb_r15mm_f80_open_inject"
IT = 7000

# 群名 → [(実験, モデルdir)]
GROUPS = {
    "3dgs_pool12": [(BASE, f"output_gen_{c}_r{r}") for c in ("newdef", "noexp", "old", "oplr05") for r in (1, 2, 3)],
    # 注入と同一設定の対照。r1〜r3 は §9x、r4〜r7 は 2026-10-05 追加（§9ar-12 の決着用）。レンダ済みの run を自動で拾う
    "3dgs_newdef": [(BASE, f"output_gen_newdef_r{r}") for r in range(1, 10)
                    if (E / BASE / f"output_gen_newdef_r{r}" / "test" / f"ours_{IT}" / "renders").is_dir()],
    "3dgs_inj": [(INJ, "output")] + [(INJ, f"output_3dgs_r{r}") for r in range(2, 8)],
    "mip": [(BASE, "output_mip")] + [(BASE, f"output_mip_r{r}") for r in range(2, 10)],
    "mip_inj": [(INJ, "output_mip")] + [(INJ, f"output_mip_r{r}") for r in range(2, 10)],
}


def wire_psnr(exp, model):
    name = "wire_eval_" + ("" if model == "output" else model + "_") + str(IT)
    f = E / exp / "gt" / name / "summary.json"
    if not f.exists():
        subprocess.run([sys.executable, str(HERE / "eval_wires.py"), str(E / exp), str(IT), model],
                       check=True, capture_output=True)
    return json.load(open(f))["psnr_wireonly"]


def contrast(groups, coefs):
    """線形対比 Σc_i·mean_i の推定値・SE・Welch–Satterthwaite 自由度・両側p・95%CI"""
    est = sum(c * v.mean() for c, v in zip(coefs, groups))
    terms = [c * c * v.var(ddof=1) / len(v) for c, v in zip(coefs, groups)]
    se = np.sqrt(sum(terms))
    df = sum(terms) ** 2 / sum(t * t / (len(v) - 1) for t, v in zip(terms, groups))
    t = est / se
    p = 2 * stats.t.sf(abs(t), df)
    h = stats.t.ppf(0.975, df) * se
    return {"est": est, "se": se, "df": df, "t": t, "p": p, "ci95": [est - h, est + h]}


def mde(sd, n1, n2, power=0.8, alpha=0.05, two_sided=True):
    """2群（等分散 t）で検出力 power を得る最小の差（非心 t による）"""
    df = n1 + n2 - 2
    k = np.sqrt(1 / n1 + 1 / n2)
    tc = stats.t.ppf(1 - alpha / (2 if two_sided else 1), df)

    def f(d):
        nc = d / (sd * k)
        pw = stats.nct.sf(tc, df, nc) + (stats.nct.cdf(-tc, df, nc) if two_sided else 0)
        return pw - power
    return optimize.brentq(f, 1e-6, 4 * sd)


def main():
    v = {g: np.array([wire_psnr(e, m) for e, m in runs]) for g, runs in GROUPS.items()}
    out = {"metric": "psnr_wireonly (md2)", "iteration": IT, "groups": {}, "tests": {}}
    print("群                n   平均     sd   個別")
    for g, x in v.items():
        print(f"{g:16s} {len(x):2d} {x.mean():6.2f} {x.std(ddof=1):6.2f}   " + " / ".join(f"{a:.2f}" for a in x))
        out["groups"][g] = {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "values": x.round(3).tolist(),
                            "runs": [f"{e}/{m}" for e, m in GROUPS[g]]}

    # 4条件の一元配置分散分析（プールしてよいかの確認。監査R-05）
    conds = [v["3dgs_pool12"][i * 3:(i + 1) * 3] for i in range(4)]
    an = stats.f_oneway(*conds)
    print(f"\n素3DGS 7k の4条件（newdef/noexp/old/oplr05）一元配置ANOVA: F={an.statistic:.2f} p={an.pvalue:.3f}")
    out["tests"]["anova_4cond"] = {"F": an.statistic, "p": an.pvalue}

    def show(key, label, groups, coefs):
        c = contrast(groups, coefs)
        print(f"{label:44s} {c['est']:+6.2f}  95%CI [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"
              f"  t={c['t']:.2f} df={c['df']:.1f} p={c['p']:.4f}")
        out["tests"][key] = c

    print()
    show("inj_noAA_vs_pool12", "AA無し 注入効果（対 素12run プール）", [v["3dgs_inj"], v["3dgs_pool12"]], [1, -1])
    show("inj_noAA_vs_newdef", "AA無し 注入効果（対 同一設定 newdef）", [v["3dgs_inj"], v["3dgs_newdef"]], [1, -1])
    show("inj_AA", "AA有り 注入効果（Mip 9 vs 9）", [v["mip_inj"], v["mip"]], [1, -1])
    show("interaction_pool12", "交互作用（素側=12run プール）",
         [v["3dgs_inj"], v["3dgs_pool12"], v["mip_inj"], v["mip"]], [1, -1, -1, 1])
    show("interaction_newdef", "交互作用（素側=同一設定 newdef）",
         [v["3dgs_inj"], v["3dgs_newdef"], v["mip_inj"], v["mip"]], [1, -1, -1, 1])

    # 旧版（正規近似 z）との対照
    c = out["tests"]["interaction_pool12"]
    pz = 2 * stats.norm.sf(abs(c["t"]))
    print(f"  参考：同じ交互作用を正規近似 z で計算すると p={pz:.4f}（旧記述の 0.008）")
    out["tests"]["interaction_pool12"]["p_normal_approx"] = pz

    # §9ao の最小検出効果（Mip 9 vs 9）
    sp = np.sqrt((v["mip"].var(ddof=1) + v["mip_inj"].var(ddof=1)) / 2)
    m2 = mde(sp, 9, 9)
    m1 = mde(sp, 9, 9, two_sided=False)
    ms = mde(v["mip"].std(ddof=1), 9, 9)
    print(f"\n§9ao 検出力80%の最小効果（Mip 9 vs 9、α=0.05）：プールsd {sp:.3f} → 両側 {m2:.2f}dB / 片側 {m1:.2f}dB"
          f"（参考：Mip素のsd {v['mip'].std(ddof=1):.3f} だけを使うと両側 {ms:.2f}dB）")
    out["tests"]["mde_9ao"] = {"pooled_sd": sp, "two_sided": m2, "one_sided": m1, "two_sided_mip_sd_only": ms}

    json.dump(out, open(HERE / "inject_interaction_stats.json", "w"), indent=1, ensure_ascii=False,
              default=float)
    print(f"\n保存: {HERE / 'inject_interaction_stats.json'}")


if __name__ == "__main__":
    main()
