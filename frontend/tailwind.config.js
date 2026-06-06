/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          700: "#15803d",
        },
        secondary: {
          500: "#22c55e",
        },
      },
      boxShadow: {
        soft: "0 10px 30px rgba(0, 0, 0, 0.35)",
      },
      keyframes: {
        pulseSoft: {
          "0%, 100%": { opacity: "0.45" },
          "50%": { opacity: "1" },
        },
      },
      animation: {
        pulseSoft: "pulseSoft 1.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
