#!/usr/bin/env python3
"""Plot saved processed datasets for visual review; no inference or data edits."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/tmp/fig10_cleaned_mpl")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/fig10_cleaned_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parent
PLOTS=ROOT/"plots"
GROUPS=("fig10a","fig10b","fig10c")
TITLES=("(a) 实验：原强度 − 1691","(b) 无探针模拟：标记置零","(c) 有探针模拟：标记置零")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges(x):
    mid=.5*(x[1:]+x[:-1])
    return np.r_[2*x[0]-mid[0],mid,2*x[-1]-mid[-1]]


def main():
    PLOTS.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/"processing_manifest.json").read_text())
    before_sha=sha256(ROOT/"processing_manifest.json")
    processed,original=[],[]
    for group in GROUPS:
        record=manifest["datasets"][group]
        for field,hashfield in (("output_path","output_sha256"),("source_path","source_sha256")):
            assert sha256(Path(record[field]))==record[hashfield]
        with np.load(record["output_path"],allow_pickle=False) as src:
            processed.append({k:src[k] for k in src.files})
        with np.load(record["source_path"],allow_pickle=False) as src:
            original.append({k:src[k] for k in src.files})
    assert np.array_equal(original[1]["palette_rgb"],original[2]["palette_rgb"])
    cmap=ListedColormap(original[1]["palette_rgb"][::-1]/255.)
    font=FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"font.size":11,"axes.unicode_minus":False,
                         "savefig.facecolor":"white"})
    xlim=(min(edges(d["wavelength_nm"])[0] for d in processed)-263.3,
          max(edges(d["wavelength_nm"])[-1] for d in processed)-263.3)
    ylim=(max(edges(d["time_paper_ns"])[-1] for d in processed),
          min(edges(d["time_paper_ns"])[0] for d in processed))
    artifacts,checks={},[]

    def decorate(ax,title):
        ax.set(xlim=xlim,ylim=ylim,xlabel="波长偏移 Δλ (nm)",ylabel="时间 (ns)")
        ax.set_xticks([-1,-.5,0,.5,1])
        ax.set_yticks(np.arange(1.5,4.51,.5))
        ax.set_title(title,pad=51,fontsize=12.5,fontweight="bold")
        ax.set_box_aspect(1.5)
        top=ax.secondary_xaxis("top",functions=(lambda x:x+263.3,lambda x:x-263.3))
        top.set_xticks([262.3,263.3,264.3])
        top.set_xlabel("绝对波长 (nm)",labelpad=5)
        ax.tick_params(direction="out",top=False,right=False)

    def panel(fig,ax,i):
        d=processed[i]
        mesh=ax.pcolormesh(edges(d["wavelength_nm"])-263.3,edges(d["time_paper_ns"]),d["intensity"],
                           shading="flat",antialiased=False,rasterized=True,cmap=cmap,
                           norm=Normalize(0,3200 if i==0 else 1))
        assert np.array_equal(np.asarray(mesh.get_array()),d["intensity"])
        checks.append({"group":GROUPS[i],"plotted_values_equal_saved_dataset":True})
        decorate(ax,TITLES[i])
        cb=fig.colorbar(mesh,ax=ax,location="right",pad=.035,fraction=.045,shrink=.83)
        cb.set_label("扣背景后强度 (counts)" if i==0 else "抠图色条位置 (0–1)")
        cb.set_ticks([0,800,1600,2400,3200] if i==0 else np.arange(0,1.01,.2))

    def save(fig,stem,pdf):
        for ext in ("png","pdf"):
            path=PLOTS/f"{stem}.{ext}"
            fig.savefig(path,dpi=300)
            artifacts[str(path.relative_to(ROOT))]=sha256(path)
        pdf.savefig(fig,dpi=300)
        plt.close(fig)

    combined=ROOT/"fig10_cleaned_review.pdf"
    with PdfPages(combined) as pdf:
        fig,axes=plt.subplots(1,3,figsize=(15,8.3),layout="constrained")
        fig.suptitle("Fig.10(a/b/c)  处理后的时间—波长二维光谱\n已另存三组数据 · 供肉眼核对",
                     fontsize=18,fontweight="bold")
        for i,ax in enumerate(axes):
            panel(fig,ax,i)
        fig.supxlabel("a 全部像素减去 1691 counts；b/c 的边缘刻度、右上角标志及其局部边界置零。\n"
                      "坐标与网格不变；未归一化、插值或平滑。b/c 共用色标，a 保留 counts 单位。",fontsize=10)
        save(fig,"fig10abc_cleaned_spectra",pdf)
        for i,group in enumerate(GROUPS):
            fig,ax=plt.subplots(figsize=(7.5,9.5),layout="constrained")
            fig.suptitle("另存数据 · 时间—波长光谱",fontsize=17,fontweight="bold")
            panel(fig,ax,i)
            detail="全量强度减 1691；未截断、归一化或平滑。" if i==0 else "仅标志及刻度区域置零；区域外逐元素保持不变。"
            fig.supxlabel(detail+"\nΔλ = λ − 263.3 nm；时间向下增加。",fontsize=10)
            save(fig,group+"_cleaned_spectrum",pdf)
        fig,axes=plt.subplots(1,2,figsize=(11,10.2))
        fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.80,wspace=.25)
        fig.suptitle("模拟数据置零范围核对\n原始 RGB 上的红框：本次置零矩形（含抗锯齿边缘）",
                     fontsize=17,fontweight="bold")
        for ax,i in zip(axes,(1,2)):
            d=original[i]
            we,te=edges(d["wavelength_nm"])-263.3,edges(d["time_paper_ns"])
            ax.pcolormesh(we,te,d["rgb"],shading="flat",antialiased=False,rasterized=True)
            for r in manifest["datasets"][GROUPS[i]]["rectangles"]:
                r0,r1,c0,c1=r["bounds_row_start_stop_col_start_stop"]
                ax.add_patch(Rectangle((we[c0],te[r0]),we[c1]-we[c0],te[r1]-te[r0],
                                       facecolor="none",edgecolor="#ff345e",linewidth=1.1,clip_on=False))
            decorate(ax,"(b) 置零区域" if i==1 else "(c) 置零区域")
        fig.supxlabel("数据集同时保存 zeroed_region_mask，可逐像素追溯。\n"
                      "本页是处理范围示意；处理后光谱请看前四页。",fontsize=10)
        save(fig,"fig10bc_zeroed_regions",pdf)
    artifacts[combined.name]=sha256(combined)
    assert sha256(ROOT/"processing_manifest.json")==before_sha
    for group,record in manifest["datasets"].items():
        assert sha256(Path(record["output_path"]))==record["output_sha256"]
        assert sha256(Path(record["source_path"]))==record["source_sha256"]
    result={"status":"complete","processing_manifest_sha256":before_sha,"artifacts_sha256":artifacts,
            "checks":checks,"saved_data_and_sources_unchanged_by_plotting":True,
            "combined_pdf_pages":5,"script_sha256":sha256(Path(__file__))}
    (ROOT/"plotting_manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":"complete","figures":5,"pdf_pages":5}))


if __name__=="__main__":
    main()
