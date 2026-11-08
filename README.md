# Aircraft Engine RUL Prediction

Predicting the remaining useful life (RUL), in flights, of 100 in-service aircraft engines from the run-to-retirement histories of 260 others. High-pressure compressor (HPC) degradation is the cause of retirement in every engine. This is Project 4 of the Industrial AI 2026 final project; the brief is in [`References/Project4_Engine_2026.pdf`](References/Project4_Engine_2026.pdf).

This README covers the data exploration, feature extraction, feature selection and dimensionality reduction done so far. RUL modeling is the next step.

## Repository layout

| Path | Contents |
|---|---|
| [`main.ipynb`](main.ipynb) | Data exploration and every plot below |
| [`features.ipynb`](features.ipynb) | Builds the 10-feature vector and saves it to `Features/` |
| `Features/` | Feature tables for the training, validation and test engines |
| `figures/` | The plots in this README, exported from `main.ipynb` |
| `TrainingData_Engine.csv` | The 260 training engines as one table, written by [`data1.m`](data1.m) from the `.mat` file |
| `Training_DataSet/`, `Testing_DataSet/` | Original MATLAB data files |
| `References/` | Project brief and background papers |
| `GroundTruth/` | True test RULs, not used anywhere in this work so that test results stay a fair evaluation |

## How to run

Requires Python 3 with pandas, NumPy, SciPy, scikit-learn, matplotlib and Jupyter. Developed with pandas 2.3, NumPy 2.3, SciPy 1.16, scikit-learn 1.7 and matplotlib 3.10.

```bash
pip install pandas numpy scipy scikit-learn matplotlib jupyter
```

Run `features.ipynb` first, then `main.ipynb`. Sections 9–11 of `main.ipynb` read the files that `features.ipynb` saves.

## Data

Each row is one flight, with 24 measurements: three operating settings (altitude, Mach number, throttle) and 21 sensors (Power Setting (TRA), T2 Total temperature at fan inlet °R, T24 Total temperature at LPC outlet °R, T30 Total temperature at HPC outlet °R, T50 Total temperature at LPT outlet °R, P2 Pressure at fan inlet psia, P15 Total pressure in bypass-duct psia, P30 Total pressure at HPC outlet psia, Nf Physical fan speed rpm, Nc Physical core speed rpm, epr Engine pressure ratio (P50/P2) --, Ps30 Static pressure at HPC outlet psia,phi Ratio of fuel flow to Ps30 pps/psi, NRf Corrected fan speed rpm, NRc Corrected core speed rpm, BPR Bypass Ratio --, farB Burner fuel-air ratio --, htBleed (Bleed Enthalpy), Nf_dmd Demanded fan speed rpm, PCNfR_dmd Demanded corrected fan speed rpm, W31 HPT coolant bleed lbm/s, W32 LPT coolant bleed lbm/s, and Regime) covering temperatures, pressures, spool speeds and bleed flows. Every flight is flown in one of six operating regimes.

- **Training:** 260 engines, 53,759 flights, each running from new to retirement after 128–378 flights (median 199).
- **Test:** 100 engines, stopped at 21–367 flights, still in service.

Test engines are younger on average than the training engines: 51 of the 100 have flown no more than the shortest training life of 128 flights.

![Engine life distributions](figures/01_engine_life.png)

## Feature extraction

### Regime normalization

The operating regime dominates every sensor reading. Regime explains over 96% of the variance in every sensor except `farB` (89%), and consecutive flights switch regime 83% of the time, so a raw sensor trace is a sawtooth that hides any wear trend.

![Raw sensor trace and operating regimes](figures/02_regime_effect.png)

Each sensor is therefore z-scored within its regime against a healthy baseline: the mean and standard deviation of each engine's first 30 flights. After normalization, a value of 0 means "typical of a new engine in this regime".

Four sensors (`T2`, `P2`, `Nf_dmd`, `PCNfR_dmd`) are constant within every regime, so they carry no information once the regime is known.

### Sensor trends approaching retirement

Each panel shows one normalized sensor for every training engine, smoothed over 15 flights and aligned at retirement (flight 0). The blue line is the fleet median and the band the middle 50% of engines.

![Normalized sensors versus flights before retirement](figures/03_sensors_vs_rul.png)

The strongest sensors rise smoothly and consistently toward retirement. The speed sensors (`Nf`, `NRf`, `Nc`, `NRc`) fan out widely, and `epr` and `farB` barely move.

## Feature selection

### Fisher criterion

The Fisher score measures how well a sensor separates each engine's first 30 flights (healthy) from its last 30 flights (degraded), relative to the spread within each group. Sensors scoring 2 or more were kept.

![Fisher scores by sensor](figures/04_fisher_scores.png)

### Consistency across engines

A good degradation signal should move the same way on every engine. Each box shows the per-engine Spearman correlation between a sensor and flight number.

![Per-engine trend consistency](figures/05_trend_consistency.png)

The speed sensors are the least consistent: `Nc`, for example, clearly rises on 63% of engines and falls on 17%. They are excluded even though their trend correlates with engine life, because including them in PCA widened the smoothed indicator's spread at retirement from 0.48 to 6.35 (standard deviation across engines).

### Redundancy

The physical and corrected speeds are near duplicates (`Nf`/`NRf` r = 0.96, `Nc`/`NRc` r = 0.94). The six selected sensors form one correlated block.

![Correlation of normalized sensors](figures/06_sensor_correlation.png)

**Selected sensors:** `Ps30`, `T50`, `BPR`, `htBleed`, `T24`, `T30`.

## Dimensionality reduction

PCA on the raw data finds the operating regimes: raw PC1 holds 80% of the variance, 99.9% of which is explained by regime, and its correlation with RUL is only −0.12. On the six normalized sensors, PC1 follows wear instead, with a correlation of −0.77.

![PCA before and after normalization](figures/07_pca.png)

Principal components of the six selected sensors, fitted on the 208 training engines (from `features.ipynb`):

| | Variance | Cumulative | Ps30 | T50 | BPR | htBleed | T24 | T30 |
|---|---|---|---|---|---|---|---|---|
| **PC1** | 71.0% | 71.0% | 0.494 | 0.443 | 0.444 | 0.361 | 0.348 | 0.334 |
| PC2 | 6.8% | 77.8% | −0.043 | −0.050 | −0.088 | −0.454 | −0.130 | 0.874 |
| PC3 | 6.6% | 84.5% | −0.061 | −0.049 | −0.184 | −0.397 | 0.890 | −0.098 |
| PC4 | 6.5% | 90.9% | −0.204 | −0.186 | −0.549 | 0.693 | 0.206 | 0.315 |
| PC5 | 5.0% | 95.9% | 0.142 | 0.753 | −0.601 | −0.140 | −0.147 | −0.105 |
| PC6 | 4.1% | 100.0% | 0.830 | −0.444 | −0.315 | −0.073 | −0.072 | −0.065 |

PC1 is the only component that combines all six sensors, so it is used as the health indicator, oriented so that higher means more degraded. PC2–PC6 each centre on one or two sensors and mostly capture individual sensor variation; their signs are arbitrary.

### PC1 as a health indicator

Smoothed PC1 follows the same accelerating shape as Figure 1 of the project brief. Engines start at quite different levels (−2.26 ± 1.44 at flight 15) but retire at a consistent one (6.05 ± 0.48).

![PC1 health indicator](figures/08_health_indicator.png)

## Feature vector

`features.ipynb` builds one row per flight, from flight 20 onward. Every fitted step (regime baseline, Fisher selection, PCA) uses only the 208 training engines and is then applied unchanged to 52 held-out validation engines and the 100 test engines. Every feature uses only the current and earlier flights, which is checked by truncating engines and confirming their feature rows are unchanged.

| Feature | Definition | Spearman with RUL (validation) |
|---|---|---|
| `Ps30`, `T50`, `BPR`, `htBleed`, `T24`, `T30` | Regime-normalized sensor, averaged over the last 15 flights | −0.76 to −0.77 |
| `health` | PC1 averaged over the last 15 flights | −0.78 |
| `health_delta` | `health` minus the engine's mean PC1 over its first 20 flights | −0.88 |
| `health_slope` | Linear slope of PC1 over the last 20 flights | −0.67 |
| `Cycle` | Flights flown so far | −0.69 |

| Set | Engines | Rows |
|---|---|---|
| Training | 208 | 38,860 |
| Validation | 52 | 9,959 |
| Test | 100 | 100 (each engine's last flight) |

The window lengths are starting choices, not tuned values. The 20-flight windows are set by the shortest test engine, which has 21 flights.

### Features by engine

The shortest, median and longest-lived training engines:

![Features for three training engines](figures/09_features_train_engines.png)

The same for the validation engines, which played no part in any fitting:

![Features for three validation engines](figures/10_features_val_engines.png)

And for the test engines with the fewest, median and most flights so far:

![Features for three test engines](figures/11_features_test_engines.png)

### Derived features across the fleet

`health_delta` removes each engine's starting level, which tightens the fleet considerably compared with `health`. `health_slope` trends upward but is noisy from flight to flight.

![Derived features versus flights before retirement](figures/12_derived_features.png)

### Where the test engines sit

Each test engine at its last flight, against the training and validation fleet at the same flight count. Young test engines cluster near zero, which leaves little to distinguish them; engines at 130–200 flights range from almost new to near retirement.

![Test engines against the fleet](figures/13_test_engine_position.png)

## Notes on the analysis

- Sections 1–8 of `main.ipynb` fit on all 260 training engines as exploration. Results for modeling come from `features.ipynb`, which fits on training engines only.
- The healthy baseline assumes each engine's first 30 flights are undamaged. Normalized sensor medians stay near 0 over flights 1–30 and drift only afterwards, which supports this.

## Next steps

1. Trajectory similarity RUL prediction on the `health` indicator, evaluated on validation engines cut at random points across the test range (21–367 flights).
2. A logistic regression health indicator trained on healthy vs. degraded flights, compared against PC1.
3. An exponential curve fit as a cross-check.
4. An LSTM on the normalized sensor sequences, kept only if it beats trajectory similarity.
