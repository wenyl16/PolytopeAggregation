# Manuscript tables recalculated from retained results

Saved-data recalculation only; no solver or training calls

## Table I: saved training schedule

| Stage | Updates | Equivalent lambda | Learning rate | Opt/feas ratio |
| --- | --- | --- | --- | --- |
| 1 | 1--300 | 0.5 | 0.02 | 1 |
| 2 | 301--500 | 0.99009901 | 0.1 | 0.01 |
| 3 | 501--1000 | 0.999999 | 1 | 1e-06 |

## Table II: mean cost deviations

| Method | USD/day |
| --- | --- |
| Proposed method (PEB initialization) | 8.65 |
| PEB inner | 9.82 |
| PEB outer | -7.22 |
| Federated PEB | 56.08 |
| Proposed method (PB initialization) | 27.08 |
| PB inner | 65.29 |
| Cutting plane (original) | -11.10 |
| Cutting plane (modified) | -5.77 |
| PVE (original) | 16.76 |
| PVE (modified) | 34.37 |

## Table III: 500 continuous devices

| T | Training (s) | Mean feasibility | Max feasibility | Mean optimality | Max optimality |
| --- | --- | --- | --- | --- | --- |
| 24 | 741.34 | 2.66e-13 | 1.33e-09 | 0.2083 | 0.5770 |
| 96 | 3262.06 | 1.62e-08 | 3.19e-05 | 0.2800 | 0.5411 |

## Table IV: sampled nonconvexity gaps

| Portfolio / MT minimum on-time | Mean gap (%) | Max gap (%) |
| --- | --- | --- |
| 7 devices / 4 h | 0.0351 | 1.0849 |
| 21 devices / 4 h | 0.0113 | 0.3671 |
| 21 devices / removed | 0.0001 | 0.0480 |

## Table V: saved training schedule

| Stage | Updates | Equivalent lambda | Learning rate | Opt/feas ratio |
| --- | --- | --- | --- | --- |
| 1 | 1--600 | 0.5 | 0.05 | 1 |
| 2 | 601--1200 | 0.99009901 | 0.1 | 0.01 |
| 3 | 1201--2000 | 0.999999 | 0.2 | 1e-06 |

## Provenance notes

- Cost is each method's saved native daily optimum minus the same day's original physical optimum.
- Table III errors are normalized squared Euclidean distances, not percentage or RMS distances.
- Table IV uses upper certified distances for reported values; lower interval endpoints are retained too.
- The removed-minimum-on-time row uses paired_samples.csv; generic samples.csv is a different experiment.
- Stage lambda is shown as 1/(1+saved_optimality_to_feasibility_ratio); the manuscript rounds these weights.
- The saved T96 run has 1003 updates. Historical counts, timings and array values are preserved.
- Manuscript gradient timings 197.215/195.594 seconds were author supplied; current retained run logs differ.
- Budget-study logs contain 1003/2006 updates and 225.768/461.017 seconds, versus manuscript 1000/2000 and 197.2/390.7.
