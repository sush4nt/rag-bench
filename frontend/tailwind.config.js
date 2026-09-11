/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        pipeline: {
          bm25: "#38bdf8", // sky
          dense: "#a78bfa", // violet
          hybrid: "#34d399", // emerald
          reranked: "#fbbf24", // amber
        },
      },
    },
  },
  plugins: [],
};
