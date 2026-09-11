// k6 load test — FiQA retrieval across all four pipelines.
// Run: k6 run load_testing/fiqa_load.js  (override host with BASE_URL env var)
//
// Scenario: ramp 10 -> 50 VUs over ~2 min. Each request picks a random query and
// a random pipeline. Latency thresholds are enforced PER pipeline via tags.

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";

const QUERIES = [
  "What is dollar cost averaging?",
  "How to evaluate REIT dividend yield?",
  "Difference between ETF and index fund?",
  "What are the risks of investing in REITs?",
  "How does dollar cost averaging work?",
  "How to evaluate a company's P/E ratio?",
  "What is a good debt to equity ratio?",
  "How are capital gains taxed?",
  "What is the difference between a Roth and traditional IRA?",
  "How does compound interest grow savings?",
  "Is it better to pay off debt or invest?",
  "What is dollar hedging in forex?",
  "How do bond yields relate to interest rates?",
  "What is a stock buyback and why do companies do it?",
  "How is EBITDA different from net income?",
  "What are index funds and how do they work?",
  "How to diversify an investment portfolio?",
  "What is the wash sale rule?",
  "How do dividends affect stock price?",
  "What is a mutual fund expense ratio?",
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
    `${BASE_URL}/api/fiqa/retrieve`,
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
