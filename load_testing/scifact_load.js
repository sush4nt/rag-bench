// k6 load test — SciFact retrieval across all four pipelines.
// Run: k6 run load_testing/scifact_load.js  (override host with BASE_URL env var)

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";

const QUERIES = [
  "0-dimensional biomaterials show inductive properties.",
  "The Mediterranean diet reduces the risk of cardiovascular disease.",
  "Aspirin lowers the risk of colorectal cancer.",
  "Vitamin D supplementation prevents respiratory infections.",
  "Statins reduce LDL cholesterol levels.",
  "Regular exercise improves cardiovascular fitness.",
  "Smoking is a leading cause of lung cancer.",
  "Sleep deprivation impairs immune function.",
  "Antibiotic overuse drives bacterial resistance.",
  "Insulin resistance is associated with type 2 diabetes.",
];

const PIPELINES = ["bm25", "dense", "hybrid", "reranked"];

const latency = new Trend("ragbench_client_latency_ms", true);

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "60s", target: 10 },
        { duration: "60s", target: 50 },
        { duration: "30s", target: 0 },
      ],
      gracefulRampDown: "10s",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{pipeline:bm25}": ["p(95)<500"],
    "http_req_duration{pipeline:dense}": ["p(95)<500"],
    "http_req_duration{pipeline:hybrid}": ["p(95)<800"],
    "http_req_duration{pipeline:reranked}": ["p(95)<1000"],
  },
};

export default function () {
  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const pipeline = PIPELINES[Math.floor(Math.random() * PIPELINES.length)];

  const res = http.post(
    `${BASE_URL}/api/scifact/retrieve`,
    JSON.stringify({ query, pipeline, top_k: 10 }),
    { headers: { "Content-Type": "application/json" }, tags: { pipeline } }
  );

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has results": (r) => {
      try {
        return JSON.parse(r.body).results.length > 0;
      } catch (_e) {
        return false;
      }
    },
  });
  latency.add(res.timings.duration, { pipeline });
  sleep(0.5);
}
