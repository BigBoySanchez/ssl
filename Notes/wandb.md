# WandB API Performance Notes

## Challenge: Slow Fetching of Large Run Sets
Fetching runs directly via `api.runs(path)` can be extremely slow when the project contains thousands of runs (e.g., >1200). The default pagination and data hydration (fetching config/summary) result in high latency.

### Symptoms
- Script hangs or takes minutes to "Find runs".
- High network latency due to many small sequential requests.
- Potential timeouts or `KeyboardInterrupt` necessity.

## Solutions

### 1. Optimize `api.runs()` (Minor Improvement)
Increase the page size to reduce round-trips.
```python
# Default is often small (50), increasing to 1000 speeds up initial listing
runs = api.runs(project_name, per_page=1000)
```
*Note: This helps, but if you iterate and access `.config` or `.summary` for every run, it still triggers lazy-loading network calls.*

### 2. Leverage Sweeps (Major Improvement)
If runs are organized into Sweeps, iterate over **Sweeps** instead of flat Runs.
```python
# Fetching 120 sweep objects is much faster than 1200 run objects
sweeps = api.project(project_name).sweeps()

for sweep in sweeps:
    # Access runs attached to this specific sweep
    # Often you only need the best run or a subset
    runs = sweep.runs
    best_run = sweep.best_run() # WandB helper if available, or sort manually
```
**Why this works**: Reduces the search space from $N_{runs}$ to $N_{sweeps}$. You only drill down into runs for relevant sweeps.

### 3. Server-Side Filtering
Always use filters to limit data on the server side before fetching.
```python
filters = {"state": "finished"} # Only get finished runs
runs = api.runs(project_name, filters=filters)
```
