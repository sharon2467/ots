#!/usr/bin/env python3
"""Save only user-authorized Fig.10 background subtraction / annotation zeroing."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = REPO / "outputs/fig10_input_alignment_20261021"
DATASETS = ROOT / "datasets"
GROUPS = ("fig10a", "fig10b", "fig10c")
TICK_ROWS = ((77,81),(157,162),(237,242),(318,323),(398,403),(478,480))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regions_for(group, shape):
    """Reviewed native-grid half-open rectangles, not an automatic valid mask."""
    if group == "fig10a":
        return []
    b = group == "fig10b"
    regions = [
        ("panel_label", (16,58,252,303) if b else (16,58,257,305)),
        ("top_center_tick", (0,6,157,162) if b else (0,6,158,163)),
        ("bottom_center_tick", (474,480,157,162) if b else (474,480,158,163)),
    ]
    for idx, (r0,r1) in enumerate(TICK_ROWS):
        suffix = f"{(2+idx*.5):g}ns" if idx < 5 else "bottom_corner"
        regions.extend([(f"left_tick_{suffix}",(r0,r1,0,5 if b else 6)),
                        (f"right_tick_{suffix}",(r0,r1,313 if b else 314,shape[1]))])
    return regions


def main():
    DATASETS.mkdir(parents=True,exist_ok=True)
    if any(DATASETS.glob("*.npz")) or (ROOT/"processing_manifest.json").exists():
        raise FileExistsError("Processed datasets already exist; refusing to overwrite them.")
    plan = json.loads((REPO/"outputs/recovery_input_preparation/protected_inputs.json").read_text())
    protected = dict(plan["protected_sha256"])
    for path, sha in protected.items():
        assert sha256(Path(path)) == sha, path
    records = {}
    for group in GROUPS:
        path = next((SOURCE/group/"original").glob("*.npz"))
        source_sha = sha256(path)
        with np.load(path,allow_pickle=False) as src:
            original = {k:src[k] for k in src.files}
        field = "measured_raw" if group=="fig10a" else "intensity"
        x = original[field]
        assert x.dtype == np.float64 and np.isfinite(x).all()
        mask = np.zeros(x.shape,dtype=bool)
        region_records = []
        for name,(r0,r1,c0,c1) in regions_for(group,x.shape):
            assert 0<=r0<r1<=x.shape[0] and 0<=c0<c1<=x.shape[1]
            mask[r0:r1,c0:c1] = True
            region_records.append({"name":name,"bounds_row_start_stop_col_start_stop":[r0,r1,c0,c1],
                                   "time_centers_ns":original["time_paper_ns"][[r0,r1-1]].tolist(),
                                   "wavelength_centers_nm":original["wavelength_nm"][[c0,c1-1]].tolist(),
                                   "pixel_count":(r1-r0)*(c1-c0)})
        background = 1691.0 if group=="fig10a" else 0.0
        processed = x-background
        processed[mask] = 0.0
        if group=="fig10a":
            assert np.array_equal(processed,x-1691.0)
            assert processed.min()==0 and processed.max()==3200
        else:
            assert np.array_equal(processed[~mask],x[~mask])
            assert np.all(processed[mask]==0)
        units = "counts" if group=="fig10a" else "digitized_colorbar_position_0_to_1"
        saved = {"intensity":processed,"time_paper_ns":original["time_paper_ns"],
                 "wavelength_nm":original["wavelength_nm"],"source_row_index":original["source_row_index"],
                 "zeroed_region_mask":mask,"background_subtracted_counts":np.array(background),
                 "intensity_units":np.array(units),"source_file_sha256":np.array(source_sha),
                 "source_intensity_key":np.array(field),"processing_version":np.array("fig10_user_cleanup_v1")}
        if "source_col_index" in original:
            saved["source_col_index"] = original["source_col_index"]
        output = DATASETS/f"{group}.npz"
        np.savez_compressed(output,**saved)
        with np.load(output,allow_pickle=False) as check:
            for key, value in saved.items():
                assert np.array_equal(check[key],value), (group,key)
        assert sha256(path)==source_sha
        records[group] = {"source_path":str(path),"source_sha256":source_sha,"source_field":field,
                          "output_path":str(output),"output_sha256":sha256(output),
                          "shape_time_wavelength":list(x.shape),"intensity_units":units,
                          "operation":"subtract_constant_1691" if group=="fig10a" else "set_reviewed_annotation_rectangles_to_zero",
                          "background_subtracted_counts":background,
                          "before_range":[float(x.min()),float(x.max())],
                          "after_range":[float(processed.min()),float(processed.max())],
                          "zeroed_region_pixels":int(mask.sum()),
                          "changed_value_pixels":int(np.count_nonzero(processed!=x)),
                          "zeroed_region_fraction":float(mask.mean()),
                          "rectangles":region_records,"coordinates_and_source_indices_unchanged":True,
                          "all_pixels_outside_rectangles_unchanged":None if group=="fig10a" else True}
        if group!="fig10a":
            invalid=~original["valid"]
            records[group]["original_invalid_pixels_total"]=int(invalid.sum())
            records[group]["original_invalid_pixels_within_reviewed_rectangles"]=int((invalid&mask).sum())
    for path,sha in protected.items():
        assert sha256(Path(path))==sha,path
    manifest={"status":"saved_for_user_visual_review","user_authorized_operations":[
                "a measured_raw minus 1691 counts everywhere",
                "b/c edge ticks and panel labels set to zero in reviewed native-grid rectangles"],
              "datasets":records,"rectangle_index_convention":"0-based half-open row/column intervals",
              "region_selection":"Original RGB and spatial positions reviewed independently; includes a small antialias margin, no global valid-mask filtering.",
              "new_interpolation":False,"new_normalization":False,"clipping":False,"smoothing":False,
              "new_model_inference":False,"original_protected_files_unchanged":len(protected),
              "script_sha256":sha256(Path(__file__))}
    (ROOT/"processing_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({g:{k:r[k] for k in ("shape_time_wavelength","after_range","zeroed_region_pixels","changed_value_pixels")}
                      for g,r in records.items()},ensure_ascii=False))


if __name__=="__main__":
    main()
