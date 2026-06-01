#!/usr/bin/env python3
"""BirdNET-Analyzer v2.4 proxy-stream builder for the BirdCLEF-2026 diversity scout.

Forked/reproduced from the public baseline
  `ahmadzulfiqar001/birdclef-2026-birdnet-baseline` (birdnetlib + BirdNET-Analyzer v2.4),
with the label-mapping/branch-wiring reference
  `yaroslavkholmirzayev/birdnet-third-branch-site-hour-prior-restore`.

Differences from the public baseline (documented for provenance):
  * Emits DENSE per-segment probability vectors (min_conf=0.01, max confidence over
    BirdNET's 3s windows overlapping each 5s segment) instead of the thresholded
    min_conf=0.1 detections, so the output is a usable 234-class score matrix.
  * Maps BirdNET's 6522-label scientific-name space -> the competition's 234 species via
    taxonomy.csv (scientific_name; common-name fallback; one taxonomic synonym).
  * Unmapped classes (non-Aves taxa + Dwarf Tinamou) get a constant low prior.
  * Output row_id schema / column order is forced to match the canonical proxy CSV.

Location-aware inference uses the Pantanal centroid (lat -17.0, lon -57.0), matching the
public baseline.  No LB submission; read-only proxy build.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

PANTANAL_LAT = -17.0
PANTANAL_LON = -57.0
SEG = 5
LOW_PRIOR = 0.0001  # constant low prior for species BirdNET cannot represent


def build_mapping(analyzer: Analyzer, taxonomy: pd.DataFrame) -> tuple[dict[str, str], dict, list[str]]:
    """Return sci_lower -> competition primary_label, plus diagnostics."""
    bn_sci = {l.split("_")[0].lower().strip() for l in analyzer.labels}
    bn_common = {l.split("_", 1)[1].lower().strip(): l.split("_")[0].lower().strip()
                 for l in analyzer.labels}
    tax = taxonomy.copy()
    tax["sci"] = tax["scientific_name"].str.lower().str.strip()
    tax["cn"] = tax["common_name"].str.lower().str.strip()

    # Manual taxonomic synonyms (competition name -> BirdNET scientific name)
    synonyms = {"tyto furcata": "tyto alba"}  # American Barn Owl split from Barn Owl

    sci_to_code: dict[str, str] = {}
    mapped_codes: list[str] = []
    how = {"direct": [], "common": [], "synonym": [], "unmapped": []}
    for _, r in tax.iterrows():
        code = str(r["primary_label"])
        sci = r["sci"]
        cn = r["cn"]
        if sci in bn_sci:
            sci_to_code[sci] = code
            mapped_codes.append(code)
            how["direct"].append(code)
        elif cn in bn_common:
            sci_to_code[bn_common[cn]] = code
            mapped_codes.append(code)
            how["common"].append(code)
        elif sci in synonyms and synonyms[sci] in bn_sci:
            sci_to_code[synonyms[sci]] = code
            mapped_codes.append(code)
            how["synonym"].append(code)
        else:
            how["unmapped"].append(code)
    return sci_to_code, how, mapped_codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--proxy-csv", type=Path, required=True,
                    help="canonical proxy CSV defining row_id schema + species columns")
    ap.add_argument("--taxonomy-csv", type=Path, default=Path("data/taxonomy.csv"))
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--map-json", type=Path, required=True)
    ap.add_argument("--min-conf", type=float, default=0.01)
    ap.add_argument("--no-location", action="store_true",
                    help="omit the Pantanal geo-prior (location-agnostic re-score) to "
                         "test the orthogonality/competence domain-shift hypothesis")
    args = ap.parse_args()
    use_lat = None if args.no_location else PANTANAL_LAT
    use_lon = None if args.no_location else PANTANAL_LON
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    proxy = pd.read_csv(args.proxy_csv, nrows=0)
    species_cols = [c for c in proxy.columns if c != "row_id"]
    full = pd.read_csv(args.proxy_csv, usecols=["row_id"])
    canon_row_ids = full["row_id"].astype(str).tolist()
    # file stem -> ordered list of end-seconds we must emit
    want: dict[str, list[int]] = {}
    for rid in canon_row_ids:
        parts = rid.split("_")
        stem = "_".join(parts[:-1])
        end = int(parts[-1])
        want.setdefault(stem, []).append(end)

    taxonomy = pd.read_csv(args.taxonomy_csv)
    print("Loading BirdNET analyzer (v2.4)...")
    analyzer = Analyzer()
    print(f"BirdNET labels: {len(analyzer.labels)}")
    sci_to_code, how, mapped_codes = build_mapping(analyzer, taxonomy)
    mapped_set = set(mapped_codes)
    print(f"mapped={len(mapped_set)} direct={len(how['direct'])} "
          f"common={len(how['common'])} synonym={len(how['synonym'])} "
          f"unmapped={len(how['unmapped'])}")

    rows: dict[str, dict[str, float]] = {}
    for stem, ends in want.items():
        ogg = args.audio_dir / f"{stem}.ogg"
        if not ogg.exists():
            raise FileNotFoundError(f"missing audio for proxy stem: {ogg}")
        if use_lat is None:
            rec = Recording(analyzer, str(ogg), min_conf=args.min_conf)
        else:
            rec = Recording(analyzer, str(ogg), lat=use_lat, lon=use_lon,
                            week_48=-1, min_conf=args.min_conf)
        rec.analyze()
        dets = rec.detections
        # init dense rows for this file with low prior on mapped species (0 elsewhere later)
        for end in ends:
            rid = f"{stem}_{end}"
            row = {c: 0.0 for c in species_cols}
            rows[rid] = row
        # assign max BirdNET confidence to overlapping 5s segments
        for det in dets:
            sci = str(det.get("scientific_name", "")).lower().strip()
            code = sci_to_code.get(sci)
            if not code or code not in mapped_set:
                continue
            ds, de = det["start_time"], det["end_time"]
            for end in ends:
                seg_start = end - SEG
                # overlap between [ds,de] and [seg_start, end]
                if ds < end and de > seg_start:
                    rid = f"{stem}_{end}"
                    rows[rid][code] = max(rows[rid][code], float(det["confidence"]))
        print(f"  {stem}: {len(dets)} detections over {len(ends)} segments")

    # constant low prior for unmapped species (so they are never structurally identical)
    unmapped = set(how["unmapped"])
    for rid in canon_row_ids:
        for c in species_cols:
            if c in unmapped:
                rows[rid][c] = LOW_PRIOR

    out = pd.DataFrame([{"row_id": rid, **rows[rid]} for rid in canon_row_ids])
    out = out[["row_id"] + species_cols]
    assert out["row_id"].tolist() == canon_row_ids, "row_id order mismatch"
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} shape={out.shape}")

    args.map_json.write_text(json.dumps({
        "birdnet_labels": len(analyzer.labels),
        "competition_species": len(species_cols),
        "mapped_total": len(mapped_set),
        "direct_sciname": len(how["direct"]),
        "common_name_fallback": len(how["common"]),
        "synonym": how["synonym"],
        "synonym_detail": {"tyto furcata->tyto alba (brnowl)": "American Barn Owl split"},
        "common_name_codes": how["common"],
        "unmapped_count": len(how["unmapped"]),
        "unmapped_codes": how["unmapped"],
        "low_prior": LOW_PRIOR,
        "min_conf": args.min_conf,
        "lat": use_lat, "lon": use_lon,
        "location_agnostic": bool(args.no_location),
    }, indent=2))
    print(f"wrote mapping diagnostics {args.map_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
