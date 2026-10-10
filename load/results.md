# Load Test Results

> Run `make load-k6` (or `k6 run load/k6_score.js`) against a locally running API (`make api` or dockerized `api`).
> Paste the k6 summary here. Update the p95 badge in `README.md` to match.

## How to run

```bash
# Local API (host)
API_KEYS=dev-api-key-change-in-prod k6 run load/k6_score.js

# Dockerized API
make up   # brings up api on :8000
make load-k6
# or via Docker without local k6:  make load-k6-docker

# Locust alternative
pip install -r load/requirements.txt
locust -f load/locustfile.py --host http://localhost:8000  # open http://localhost:8089
```

## Latest run — _fill in after your first run_

| Metric | Value |
|--------|-------|
| VUs | 50 |
| Duration | 30s |
| Requests | _e.g. 5,214_ |
| p50 latency | _e.g. 6.2ms_ |
| p95 latency | _e.g. 12.1ms_ |
| p99 latency | _e.g. 28ms_ |
| Errors | _e.g. 0.0%_ |
| Throughput | _e.g. 173 rps_ |

Raw k6 output:
```
# paste `k6 run` summary here
```

Notes:
- Numbers are from a single-node local run (MacBook / Docker), not distributed.
- The p95 badge in `README.md` should reflect the steady-state `/score` path without `?explain=true` (SHAP adds ~15ms).
