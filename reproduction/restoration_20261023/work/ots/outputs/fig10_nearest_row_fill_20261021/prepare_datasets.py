#!/usr/bin/env python3
"""Replace reviewed Fig.10 annotations with nearest original value in same row."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PREVIOUS = REPO / "outputs/fig10_cleaned_20261021"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest = ROOT / "datasets"
    dest.mkdir(exist_ok=True)
    if any(dest.glob("*.npz")) or (ROOT/"processing_manifest.json").exists():
        raise FileExistsError("New datasets already exist; refusing to overwrite.")
    previous = json.loads((PREVIOUS/"processing_manifest.json").read_text())
    protected = dict(json.loads((REPO/"outputs/recovery_input_preparation/protected_inputs.json").read_text())["protected_sha256"])
    previous_completion = json.loads((PREVIOUS/"completion.json").read_text())
    protected.update({str(PREVIOUS/p):h for p,h in previous_completion["sha256"].items()})
    protected[str(PREVIOUS/"completion.json")] = sha256(PREVIOUS/"completion.json")
    for p,h in protected.items():
        assert sha256(Path(p))==h,p
    records={}
    for group,prev in previous["datasets"].items():
        path=Path(prev["source_path"])
        assert sha256(path)==prev["source_sha256"]
        with np.load(path,allow_pickle=False) as src:
            raw={k:src[k] for k in src.files}
        x=raw[prev["source_field"]]
        mask=np.zeros(x.shape,dtype=bool)
        for r in prev["rectangles"]:
            r0,r1,c0,c1=r["bounds_row_start_stop_col_start_stop"]
            mask[r0:r1,c0:c1]=True
        with np.load(prev["output_path"],allow_pickle=False) as src:
            assert np.array_equal(mask,src["zeroed_region_mask"])
            previous_intensity=src["intensity"]
        y=x-1691.0 if group=="fig10a" else x.copy()
        donor=np.full(x.shape,-1,dtype=np.int32)
        ties=0
        for row in np.flatnonzero(mask.any(axis=1)):
            normal=np.flatnonzero(~mask[row])
            targets=np.flatnonzero(mask[row])
            assert normal.size
            distances=abs(targets[:,None]-normal[None,:])
            # normal is sorted; argmin picks the left column when distances tie.
            nearest=normal[np.argmin(distances,axis=1)]
            donor[row,targets]=nearest
            y[row,targets]=x[row,nearest]
            assert raw["valid"][row,nearest].all()
            ties+=int(np.count_nonzero((distances==distances.min(axis=1)[:,None]).sum(axis=1)>1))
        if group=="fig10a":
            assert np.array_equal(y,x-1691.0) and np.array_equal(y,previous_intensity)
        else:
            assert np.array_equal(y[~mask],x[~mask])
            rows,cols=np.where(mask)
            assert np.array_equal(y[mask],x[rows,donor[mask]])
            assert not mask[rows,donor[mask]].any()
        saved={"intensity":y,"time_paper_ns":raw["time_paper_ns"],"wavelength_nm":raw["wavelength_nm"],
               "source_row_index":raw["source_row_index"],"replacement_mask":mask,
               "donor_column_index":donor,"background_subtracted_counts":np.array(1691.0 if group=="fig10a" else 0.0),
               "intensity_units":np.array(prev["intensity_units"]),
               "source_file_sha256":np.array(prev["source_sha256"]),"source_intensity_key":np.array(prev["source_field"]),
               "processing_version":np.array("fig10_same_row_nearest_v1")}
        if "source_col_index" in raw:
            saved["source_col_index"]=raw["source_col_index"]
        out=dest/f"{group}.npz"
        np.savez_compressed(out,**saved)
        with np.load(out,allow_pickle=False) as z:
            for k,v in saved.items():
                assert np.array_equal(z[k],v),(group,k)
        records[group]={"source_path":str(path),"source_sha256":prev["source_sha256"],"source_field":prev["source_field"],
                        "output_path":str(out),"output_sha256":sha256(out),"shape_time_wavelength":list(x.shape),
                        "intensity_units":prev["intensity_units"],
                        "operation":"original_measured_raw_minus_1691" if group=="fig10a" else "copy_nearest_original_unmasked_value_within_same_row",
                        "rectangles":prev["rectangles"],"replacement_pixel_count":int(mask.sum()),
                        "values_changed_from_original":int(np.count_nonzero(y!=x)),
                        "values_changed_from_previous_zero_version":int(np.count_nonzero(y!=previous_intensity)),
                        "tie_count_left_preferred":ties,"all_donors_original_valid":True if group!="fig10a" else None,
                        "intensity_range":[float(y.min()),float(y.max())],
                        "coordinates_and_source_indices_unchanged":True,
                        "outside_replacement_mask_unchanged":True if group!="fig10a" else None}
    for p,h in protected.items():
        assert sha256(Path(p))==h,p
    result={"status":"saved_for_visual_review","datasets":records,
            "mask_source":str(PREVIOUS/"processing_manifest.json"),
            "mask_source_sha256":sha256(PREVIOUS/"processing_manifest.json"),
            "nearest_rule":"Minimum native column index distance among unmasked columns of the same original row; equal distance takes left.",
            "donors_are_original_values_not_previously_filled_values":True,
            "global_valid_mask_filtering":False,"smoothing":False,"renormalization":False,"coordinate_resampling":False,
            "new_inference":False,"protected_file_count":len(protected),"protected_files_sha256":protected,
            "script_sha256":sha256(Path(__file__))}
    (ROOT/"processing_manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({g:{k:r[k] for k in ("replacement_pixel_count","values_changed_from_original","values_changed_from_previous_zero_version","intensity_range")} for g,r in records.items()}))


if __name__=="__main__":
    main()
