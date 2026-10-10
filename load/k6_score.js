import http from 'k6/http';
import { check, sleep } from 'k6';

// k6 load profile for POST /score — quantifies the "p95 < 15ms" claim.
// Usage:
//   API_KEYS=dev-api-key-change-in-prod k6 run load/k6_score.js
//   # or: make load-k6
//   # or via Docker: make load-k6-docker
export const options = {
  // Short smoke + steady load: 50 VUs for 30s (~5k requests)
  scenarios: {
    steady: {
      executor: 'constant-vus',
      vus: 50,
      duration: '30s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],          // <1% errors
    http_req_duration: ['p(95)<50', 'p(99)<100'], // p95 target — tighten to 15ms once warm
  },
};

const BASE = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEYS = __ENV.API_KEYS || 'dev-api-key-change-in-prod';

const payloads = [
  { user_id: 'user_0001', amount: 1500, merchant_category: 'electronics', merchant_country: 'NG' },
  { user_id: 'user_0042', amount: 45, merchant_category: 'groceries', merchant_country: 'US' },
  { user_id: 'user_0123', amount: 2500, merchant_category: 'travel', merchant_country: 'RU' },
  { user_id: 'user_0007', amount: 499, merchant_category: 'restaurants', merchant_country: 'US' },
];

export default function () {
  const body = JSON.stringify(payloads[Math.floor(Math.random() * payloads.length)]);
  const res = http.post(`${BASE}/score`, body, {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEYS,
    },
  });
  check(res, {
    'status 200': (r) => r.status === 200,
    'has fraud_probability': (r) => {
      try {
        const j = r.json();
        return typeof j.fraud_probability === 'number';
      } catch (_) {
        return false;
      }
    },
  });
  sleep(0.05);
}
