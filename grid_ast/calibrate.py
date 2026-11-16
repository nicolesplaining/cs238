"""Fit and validate a joint residual Markov model from OPSD quarter-hour data.

Load error uses the published day-ahead forecast. Solar error uses a causal
previous-24-hour capacity-factor baseline, NOT an operational solar forecast.
The national/regional errors are transferred to a toy feeder; this is not
calibration of actual feeder risk. No raw source data is stored in the repository.
"""

import argparse
import hashlib
import json
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import requests

from .model import Config
from .empirical import MarkovDisturbances


SOURCE_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_15min_singleindex.csv"
SOURCE_SHA256 = "194f3ee7d110d04389751602132899172cdd8b7ca73c07d6e9009ddac066b2a1"
SOURCE_DOI = "https://doi.org/10.25832/time_series/2020-10-06"
COLUMNS = ["utc_timestamp", "DE_load_actual_entsoe_transparency", "DE_load_forecast_entsoe_transparency",
           "DE_solar_generation_actual", "DE_solar_capacity"]


def download_source(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".partial")
        with requests.get(SOURCE_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1 << 20):
                    stream.write(chunk)
        temporary.replace(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"Unexpected source checksum: {actual}")
    return path


def prepare_residuals(path):
    data = pd.read_csv(path, usecols=COLUMNS, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique and sorted.")
    data = data.asfreq("15min")
    load = data[COLUMNS[1]]
    forecast = data[COLUMNS[2]]
    solar = data[COLUMNS[3]]
    capacity = data[COLUMNS[4]]
    cf = (solar/capacity).where((capacity>0) & (solar>=0) & (solar<=capacity))
    residual = pd.DataFrame({"load": (load/forecast-1).where((forecast>0) & (load>=0)),
                             "solar": cf-cf.shift(96)}, index=data.index)
    # Missing rows are retained as NaN, never interpolated or silently bridged.
    local = data.index.tz_convert("Europe/Berlin")
    residual["step"] = local.hour*4+local.minute//15
    residual["year"] = local.year
    residual["date"] = local.strftime("%Y-%m-%d")
    residual["daylight"] = (cf>.01) | (cf.shift(96)>.01)
    return residual


def encode(values, edges):
    a = np.searchsorted(edges[0], values[:,0], side="right")
    b = np.searchsorted(edges[1], values[:,1], side="right")
    return a*(len(edges[1])+1)+b


def fit_model(values, steps, valid, train_mask, daylight, period_steps=24, prior_strength=5.0):
    """All fitted parameters depend only on rows selected by train_mask."""
    fit = valid & train_mask
    quantiles = [.05, .25, .75, .95]
    edges = [np.quantile(values[fit,0], quantiles),
             np.quantile(values[fit & daylight,1], quantiles)]
    if any(np.any(np.diff(e)<=0) for e in edges):
        raise ValueError("Insufficient residual variation for five bins per component.")
    axes = []
    for j in range(2):
        labels = np.searchsorted(edges[j], values[fit,j], side="right")
        axes.append(np.array([values[fit,j][labels==k].mean() for k in range(5)]))
    centers = np.array(list(product(*axes)))
    ids = encode(np.nan_to_num(values), edges)
    counts = np.zeros((96//period_steps,25,25))
    pairs = np.flatnonzero(fit[1:] & fit[:-1])+1
    np.add.at(counts, (steps[pairs]//period_steps, ids[pairs-1], ids[pairs]), 1)
    # Five pseudo-observations shrink sparse rows toward the period's observed
    # marginal distribution. A 0.5 marginal pseudocount keeps all paths scoreable.
    marginal = counts.sum(axis=1)+.5
    marginal /= marginal.sum(axis=1, keepdims=True)
    transitions = (counts + prior_strength*marginal[:,None,:]) / (counts.sum(axis=-1,keepdims=True)+prior_strength)
    model = {"kind":"joint_markov_v1", "period_steps":period_steps, "edges":[e.tolist() for e in edges],
             "centers":centers.tolist(), "transitions":transitions.tolist(),
             "solar_error_unit":"capacity_fraction", "load_error_unit":"forecast_fraction",
             "prior_strength":prior_strength, "marginal_pseudocount":.5,
             "provenance":{"source_url":SOURCE_URL,"source_sha256":SOURCE_SHA256,"source_doi":SOURCE_DOI,
                           "attribution":"Open Power System Data. 2020. Data Package Time series. Version 2020-10-06. Primary source: ENTSO-E Transparency.",
                           "fit_years":[2017,2018],"heldout_year":2019,"timezone":"Europe/Berlin",
                           "load_forecast":"Published day-ahead load forecast",
                           "solar_forecast":"Previous-24-hour solar capacity factor; a causal baseline constructed for this project",
                           "scope":"German aggregate errors transferred to ANM6 forecasts; shared load error, deterministic wind; not feeder validation."}}
    return model, counts, ids


def diagnostics(model, values, steps, valid, mask, ids):
    pairs = np.flatnonzero(valid[1:] & valid[:-1] & mask[1:] & mask[:-1])+1
    noise = MarkovDisturbances(model)
    p = steps[pairs]//model["period_steps"]
    current, target = ids[pairs-1], ids[pairs]
    probs = noise.transitions[p,current,target]
    prediction = noise.transitions[p,current] @ noise.values
    x = values[valid & mask]
    return {"valid_rows":int(len(x)),"adjacent_pairs":int(len(pairs)),
            "mean":x.mean(axis=0).tolist(),"std":x.std(axis=0).tolist(),
            "quantiles_01_50_99":np.quantile(x,[.01,.5,.99],axis=0).tolist(),
            "load_solar_correlation":float(np.corrcoef(x.T)[0,1]),
            "lag1_correlation":[float(np.corrcoef(values[pairs-1,j],values[pairs,j])[0,1]) for j in range(2)],
            "markov_mean_log_probability":float(np.log(probs).mean()),
            "conditional_mean_rmse":np.sqrt(((prediction-values[pairs])**2).mean(axis=0)).tolist(),
            "zero_error_forecast_rmse":np.sqrt((values[pairs]**2).mean(axis=0)).tolist(),
            "quantization_mae":np.abs(noise.values[ids[valid & mask]]-x).mean(axis=0).tolist()}


def calibrate(source, cache, config_path, report_path):
    source = download_source(source)
    frame = prepare_residuals(source)
    values = frame[["load","solar"]].to_numpy()
    valid = np.isfinite(values).all(axis=1)
    years, steps = frame.year.to_numpy(), frame.step.to_numpy()
    train_mask, test_mask = np.isin(years,[2017,2018]), years==2019
    model, counts, ids = fit_model(values,steps,valid,train_mask,frame.daylight.to_numpy())
    report = {"source_url":SOURCE_URL,"source_sha256":SOURCE_SHA256,"split":"2017-2018 fit; 2019 held out",
              "train":diagnostics(model,values,steps,valid,train_mask,ids),
              "heldout":diagnostics(model,values,steps,valid,test_mask,ids),
              "train_missing_rows":int(sum(train_mask & ~valid)),"heldout_missing_rows":int(sum(test_mask & ~valid)),
              "transition_counts":counts.astype(int).tolist(),
              "limitations":["Aggregate national measurements, not feeder data.","Solar forecast is previous-day persistence, not an operational forecast.",
                             "Quantization compresses extreme errors; small-count transitions use smoothing.","Four time-of-day periods; no seasonal or hidden weather state.",
                             "All load buses share one error; wind remains deterministic."]}
    config_path, report_path, cache = Path(config_path),Path(report_path),Path(cache)
    for p in (config_path,report_path,cache): p.parent.mkdir(parents=True,exist_ok=True)
    Config(disturbance_model=model).save(config_path)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    # Local derived-data cache for held-out daily trace replay. Never committed.
    np.savez_compressed(cache,values=values,valid=valid,steps=steps,years=years,
                        dates=frame.date.to_numpy(dtype="U10"),timestamps=frame.index.asi8)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",default="../data/time_series_15min_singleindex.csv")
    parser.add_argument("--cache",default="../data/opsd_residuals.npz")
    parser.add_argument("--config",default="configs/empirical_opsd.json")
    parser.add_argument("--report",default="results/v2/calibration.json")
    args=parser.parse_args()
    result=calibrate(args.source,args.cache,args.config,args.report)
    print(json.dumps({k:v for k,v in result.items() if k!="transition_counts"},indent=2))


if __name__=="__main__":
    main()
