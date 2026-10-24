#!/usr/bin/env python3
"""Plot the three uploaded spectral arrays on their native grids; no inference."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fig10_spectra_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fig10_spectra_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INPUT = REPO / "outputs/fig10_input_alignment_20261021"
REFERENCE_NM = 263.3
GROUPS = ("fig10a", "fig10b", "fig10c")
TITLES = ("(a) 实验光谱", "(b) 模拟：无探针光束", "(c) 模拟：有探针光束")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges(x):
    mid = .5*(x[1:] + x[:-1])
    return np.r_[2*x[0]-mid[0], mid, 2*x[-1]-mid[-1]]


def main():
    plan = json.loads((REPO / "outputs/recovery_input_preparation/protected_inputs.json").read_text())
    for path, sha in plan["protected_sha256"].items():
        assert sha256(Path(path)) == sha, path
    inputs, records = [], {}
    for group in GROUPS:
        path = next((INPUT / group / "original").glob("*.npz"))
        with np.load(path, allow_pickle=False) as src:
            data = {key: src[key] for key in src.files}
        key = "measured_raw" if group == "fig10a" else "intensity"
        data["plot_values"] = data[key]
        t, w, z = data["time_paper_ns"], data["wavelength_nm"], data[key]
        assert z.shape == (len(t), len(w))
        assert np.all(np.diff(t)>0) and np.all(np.diff(w)>0) and np.isfinite(z).all()
        for array in data.values():
            array.flags.writeable = False
        inputs.append(data)
        records[group] = {"path": str(path), "sha256": sha256(path), "array": key,
                          "shape_time_wavelength": list(z.shape),
                          "time_range_ns": t[[0,-1]].tolist(),
                          "wavelength_range_nm": w[[0,-1]].tolist(),
                          "intensity_range": [float(z.min()),float(z.max())],
                          "unit": "counts" if group == "fig10a" else "digitized colorbar position (0–1)"}
    assert np.array_equal(inputs[1]["palette_rgb"], inputs[2]["palette_rgb"])
    palette = inputs[1]["palette_rgb"][::-1] / 255.
    cmap = ListedColormap(palette, name="uploaded_simulation_colorbar")
    # This maps the saved intensity back to the saved palette entry exactly.
    for data in inputs[1:]:
        colors = np.rint(cmap(data["plot_values"])[...,:3] * 255)
        assert np.array_equal(colors, data["palette_rgb"][data["palette_index"]])
    xlim = (min(edges(d["wavelength_nm"])[0] for d in inputs)-REFERENCE_NM,
            max(edges(d["wavelength_nm"])[-1] for d in inputs)-REFERENCE_NM)
    ylim = (max(edges(d["time_paper_ns"])[-1] for d in inputs),
            min(edges(d["time_paper_ns"])[0] for d in inputs))
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11,
                         "axes.unicode_minus": False, "savefig.facecolor": "white"})
    artifacts, checks = {}, []

    def decorate(ax, title):
        ax.set(xlim=xlim, ylim=ylim, xlabel="波长偏移 Δλ (nm)", ylabel="时间 (ns)")
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_yticks(np.arange(1.5,4.51,.5))
        ax.set_title(title, pad=51, fontsize=13, fontweight="bold")
        ax.set_box_aspect(1.5)
        top = ax.secondary_xaxis("top", functions=(lambda x: x+REFERENCE_NM, lambda x: x-REFERENCE_NM))
        top.set_xticks([262.3, 263.3, 264.3])
        top.set_xlabel("绝对波长 (nm)", labelpad=5)
        ax.tick_params(direction="out", top=False, right=False)
        return top

    def scalar_panel(fig, ax, index):
        data = inputs[index]
        z = data["plot_values"]
        norm = Normalize(float(z.min()), float(z.max())) if index == 0 else Normalize(0,1)
        mesh = ax.pcolormesh(edges(data["wavelength_nm"])-REFERENCE_NM,
                             edges(data["time_paper_ns"]), z, cmap=cmap, norm=norm,
                             shading="flat", antialiased=False, rasterized=True)
        assert np.array_equal(np.asarray(mesh.get_array()), z)
        checks.append({"group": GROUPS[index], "rendered_values_equal_original": True})
        decorate(ax, TITLES[index])
        cb = fig.colorbar(mesh, ax=ax, location="right", pad=.035, fraction=.045, shrink=.83)
        cb.set_label("原始强度 (counts)" if index == 0 else "抠图色条位置 (0–1)")
        if index:
            cb.set_ticks(np.arange(0,1.01,.2))
        else:
            cb.set_ticks([1691,2500,3300,4100,4891])
        return mesh

    def save(fig, stem, pdf):
        for ext in ("png", "pdf"):
            path = ROOT / f"{stem}.{ext}"
            fig.savefig(path, dpi=300)
            artifacts[path.name] = sha256(path)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    note = ("上传文件原始网格与原始强度；不使用 mask，不逐行归一化、扣背景、平滑或再次对齐。\n"
            "Δλ = λ − 263.3 nm 仅为坐标显示；a 色标为 counts，b/c 共用 0–1 色条位置尺度，不能跨组比较绝对强度。")
    combined = ROOT / "fig10_spectra_all.pdf"
    with PdfPages(combined) as pdf:
        fig, axes = plt.subplots(1,3,figsize=(15,8.3),layout="constrained")
        fig.suptitle("Fig.10(a/b/c)  时间—波长二维光谱\n原始上传数据 · 时间向下增加，与论文方向一致",
                     fontsize=18, fontweight="bold")
        for i, ax in enumerate(axes):
            scalar_panel(fig,ax,i)
        fig.supxlabel(note,fontsize=10)
        save(fig,"fig10abc_original_spectra",pdf)
        for i, group in enumerate(GROUPS):
            fig, ax = plt.subplots(figsize=(7.5,9.5),layout="constrained")
            fig.suptitle("原始上传数据 · 时间—波长光谱",fontsize=17,fontweight="bold")
            scalar_panel(fig,ax,i)
            fig.supxlabel("原始网格与强度；不筛 mask、不归一化或平滑。\nΔλ = λ − 263.3 nm；时间向下增加。",fontsize=10)
            save(fig,group+"_original_spectrum",pdf)
        fig, axes = plt.subplots(1,2,figsize=(11,10.2))
        fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.80,wspace=.25)
        fig.suptitle("Fig.10(b/c)  原 NPZ 保存的 RGB 像素\n直接显示颜色，不经过强度—色标的重新映射",
                     fontsize=17,fontweight="bold")
        for ax, i in zip(axes,(1,2)):
            data=inputs[i]
            mesh=ax.pcolormesh(edges(data["wavelength_nm"])-REFERENCE_NM,
                               edges(data["time_paper_ns"]),data["rgb"],shading="flat",
                               antialiased=False,rasterized=True)
            assert np.array_equal(np.asarray(mesh.get_array()),data["rgb"])
            checks.append({"group":GROUPS[i],"rendered_RGB_equal_original":True})
            decorate(ax,TITLES[i])
        fig.supxlabel("保留边缘暗色标记及全部原像素；不按 valid 标记遮挡。\n"
                      "此图用于对照论文外观；RGB 颜色本身不是散射功率数值。",fontsize=10)
        save(fig,"fig10bc_original_rgb",pdf)
    artifacts[combined.name]=sha256(combined)
    for path, sha in plan["protected_sha256"].items():
        assert sha256(Path(path))==sha,path
    result={"status":"complete", "inputs":records,"artifacts_sha256":artifacts,
            "checks":checks,"original_protected_files_unchanged":len(plan["protected_sha256"]),
            "paper_reference":{"file":"Zhao 等 - 2019 - Progress in optical Thomson scattering diagnostics for ICF gas-filled hohlraums.pdf",
                               "pdf_page_1based":10,"printed_page":"055201-9",
                               "fig10_time_increases_downward":True,
                               "experimental_colorbar_counts":[0,3000],
                               "simulation_colorbar_is_scattered_power_not_NPZ_0_to_1_position":True},
            "rendering":{"wavelength_reference_nm":REFERENCE_NM,"time_direction":"down",
                         "mask":"ignored; all stored values displayed","native_cells":True,
                         "scalar_palette":"reversed saved b/c palette_rgb, also used to display a",
                         "bc_palette_lookup_exactly_reproduced":True,
                         "new_interpolation":False,"new_normalization":False,"new_inference":False},
            "script_sha256":sha256(Path(__file__))}
    (ROOT/"plotting_manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":"complete","figures":5,"combined_pdf_pages":5,
                      "protected_files_unchanged":result["original_protected_files_unchanged"]}))


if __name__ == "__main__":
    main()
