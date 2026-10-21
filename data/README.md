# Using the published datasets

`src/data.py` supports two modes.

## 1. Surrogate streams (default)

`data.synthesize()` generates per-node Ornstein-Uhlenbeck soil and
environmental streams on the native 10-minute grid, with marginals matched to
the Stuard tomato deployment and three irrigation regimes. No download needed;
this is what `run_all.py` uses.

## 2. The real files

**Dataset 1 - LoRaWAN tomato cultivation.** Download from
<https://data.mendeley.com/datasets/35wh56287y/1> (doi:10.17632/35wh56287y.1)
and place here:

* `data/raw/stuard_environmental_data.csv`
* `data/raw/stuard_soil_data.csv`

Then:

```python
from data import load_real, stale_view
df = load_real("data/raw/stuard_environmental_data.csv",
               "data/raw/stuard_soil_data.csv")
sv = stale_view(df)
```

Replace the `synthesize(...)` calls in `src/experiments.py` with `load_real`.
Note the soil file reports electrical conductivity in uS/cm and moisture in
%RH, which is what the urgency thresholds in `data.TH` assume.

**Dataset 2 - Azure Functions Trace 2019.** Download
`invocations_per_function_md.anon.d01.csv` from
<https://github.com/Azure/AzurePublicDataset>. `scheduler.arrival_process`
currently reproduces the steady/fluctuating/spiking phase structure reported
for that trace; to drive arrivals directly from it, bin the per-minute
invocation counts of one function and pass the resulting rate sequence.
