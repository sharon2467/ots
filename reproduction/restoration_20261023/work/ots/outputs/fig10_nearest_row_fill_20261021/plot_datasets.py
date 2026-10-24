#!/usr/bin/env python3
"""Draw exactly three individual spectra from saved nearest-row-filled datasets."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/fig10_nearest_mpl")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/fig10_nearest_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT=Path(__file__).resolve().parent


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def edges(x):
    mid=(x[1:]+x[:-1])/2
    return np.r_[2*x[0]-mid[0],mid,2*x[-1]-mid[-1]]


def main():
    plotdir=ROOT/"plots"
    plotdir.mkdir(exist_ok=True)
    m=json.loads((ROOT/"processing_manifest.json").read_text())
    with np.load(m["datasets"]["fig10b"]["source_path"],allow_pickle=False) as raw:
        cmap=ListedColormap(raw["palette_rgb"][::-1]/255.)
    font=FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"font.size":11,"axes.unicode_minus":False,"savefig.facecolor":"white"})
    datasets={}
    for group,r in m["datasets"].items():
        assert sha256(Path(r["output_path"]))==r["output_sha256"]
        with np.load(r["output_path"],allow_pickle=False) as z:
            datasets[group]={k:z[k] for k in ("intensity","time_paper_ns","wavelength_nm")}
    xlim=(min(edges(d["wavelength_nm"])[0] for d in datasets.values())-263.3,
          max(edges(d["wavelength_nm"])[-1] for d in datasets.values())-263.3)
    ylim=(max(edges(d["time_paper_ns"])[-1] for d in datasets.values()),
          min(edges(d["time_paper_ns"])[0] for d in datasets.values()))
    titles={"fig10a":"(a) 实验：原强度 − 1691","fig10b":"(b) 无探针模拟：同一行最近值填补",
            "fig10c":"(c) 有探针模拟：同一行最近值填补"}
    artifacts={}
    combined=ROOT/"fig10_nearest_row_review.pdf"
    with PdfPages(combined) as pdf:
        for group,d in datasets.items():
            is_a=group=="fig10a"
            fig,ax=plt.subplots(figsize=(7.5,9.5),layout="constrained")
            fig.suptitle("Fig.10  时间—波长二维光谱",fontsize=17,fontweight="bold")
            mesh=ax.pcolormesh(edges(d["wavelength_nm"])-263.3,edges(d["time_paper_ns"]),d["intensity"],
                               cmap=cmap,norm=Normalize(0,3200 if is_a else 1),shading="flat",
                               antialiased=False,rasterized=True)
            assert np.array_equal(np.asarray(mesh.get_array()),d["intensity"])
            ax.set(xlim=xlim,ylim=ylim,xlabel="波长偏移 Δλ (nm)",ylabel="时间 (ns)")
            ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks(np.arange(1.5,4.51,.5))
            ax.set_box_aspect(1.5)
            ax.set_title(titles[group],fontsize=12.5,fontweight="bold",pad=51)
            top=ax.secondary_xaxis("top",functions=(lambda x:x+263.3,lambda x:x-263.3))
            top.set_xticks([262.3,263.3,264.3]);top.set_xlabel("绝对波长 (nm)",labelpad=5)
            cb=fig.colorbar(mesh,ax=ax,location="right",pad=.035,fraction=.045,shrink=.83)
            cb.set_label("扣背景后强度 (counts)" if is_a else "抠图色条位置 (0–1)")
            cb.set_ticks([0,800,1600,2400,3200] if is_a else np.arange(0,1.01,.2))
            note=("保留原强度 − 1691 counts；与上一版实验数据一致。" if is_a else
                  "刻度与标志区域取同一行最近的区域外原值；区域外不变。")
            fig.supxlabel(note+"\n原时间与波长网格；未平滑或归一化。",fontsize=10)
            for ext in ("png","pdf"):
                p=plotdir/f"{group}_nearest_row_spectrum.{ext}"
                fig.savefig(p,dpi=300)
                artifacts[str(p.relative_to(ROOT))]=sha256(p)
            pdf.savefig(fig,dpi=300)
            plt.close(fig)
    artifacts[combined.name]=sha256(combined)
    for r in m["datasets"].values():
        assert sha256(Path(r["output_path"]))==r["output_sha256"]
        assert sha256(Path(r["source_path"]))==r["source_sha256"]
    out={"status":"complete","figure_count":3,"combined_pdf_pages":3,
         "plotted_values_exactly_equal_saved_data":True,"artifacts_sha256":artifacts,
         "processing_manifest_sha256":sha256(ROOT/"processing_manifest.json"),"script_sha256":sha256(Path(__file__))}
    (ROOT/"plotting_manifest.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":"complete","figures":3,"pdf_pages":3}))


if __name__=="__main__":
    main()
