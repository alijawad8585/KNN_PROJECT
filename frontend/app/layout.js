import "./globals.css";

export const metadata = {
  title: "Loan Approval Predictor",
  description: "KNN-powered loan approval prediction web app",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
