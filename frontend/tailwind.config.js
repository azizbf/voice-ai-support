/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        brand: "#1a2744",
        vantage: "#4f7dff",
        mint: "#3ecf9a",
        ink: "#2a3352",
        soft: "#6b738c",
        primary: {
          DEFAULT: "#1a2744",
          foreground: "#f7f9fc",
        },
        accent: {
          DEFAULT: "#4f7dff",
          foreground: "#f7f9fc",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "Sora", "sans-serif"],
        body: ["var(--font-body)", "Inter", "sans-serif"],
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
      },
      boxShadow: {
        glass: "0 24px 60px -24px rgba(26, 39, 68, 0.35)",
      },
    },
  },
  plugins: [],
};
