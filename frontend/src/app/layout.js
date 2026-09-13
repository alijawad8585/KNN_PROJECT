import "./globals.css";

export const metadata = {
  title: "KNN Loan Approval Predictor",
  description: "Next.js frontend for real-time KNN loan approval prediction",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
