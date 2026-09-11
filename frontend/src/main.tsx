import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";
import App from "./App";
import FiQA from "./pages/FiQA";
import MetricsEmbed from "./pages/MetricsEmbed";
import SciFact from "./pages/SciFact";
import "./index.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Navigate to="/fiqa" replace /> },
      { path: "scifact", element: <SciFact /> },
      { path: "fiqa", element: <FiQA /> },
      { path: "metrics", element: <MetricsEmbed /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
