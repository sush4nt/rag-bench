// Serving-performance load test. This is separate from retrieval-quality eval.
// Quality metrics retrieve at depth 100 (see evaluation/runner.py). This script
// hits the online retrieve endpoint at one top_k and one concurrency.
//
//   k6 run load_testing/serving_bench.js
//   k6 run load_testing/serving_bench.js -e DATASET=scifact -e PIPELINE=dense -e TOP_K=5 -e VUS=10 -e DURATION=30s

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const DATASET = __ENV.DATASET || "scifact";
const PIPELINE = __ENV.PIPELINE || "bm25";
const TOP_K = __ENV.TOP_K || "10";
const VUS = Number(__ENV.VUS || 1);
const DURATION = __ENV.DURATION || "30s";

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

const latency = new Trend("ragbench_serving_latency_ms", true);

export const options = {
  scenarios: {
    serving: {
      executor: "constant-vus",
      vus: VUS,
      duration: DURATION,
      tags: { pipeline: PIPELINE, top_k: String(TOP_K), concurrency: String(VUS) },
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
  },
};

export default function () {
  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const res = http.post(
    `${BASE_URL}/api/${DATASET}/retrieve`,
    JSON.stringify({ query, pipeline: PIPELINE, top_k: Number(TOP_K) }),
    {
      headers: { "Content-Type": "application/json" },
      tags: { pipeline: PIPELINE, top_k: String(TOP_K) },
    }
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
  latency.add(res.timings.duration, { pipeline: PIPELINE, top_k: String(TOP_K) });
  sleep(0.2);
}
