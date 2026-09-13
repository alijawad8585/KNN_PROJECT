"use client";

import { useMemo, useState } from "react";
import styles from "./page.module.css";

const numericFields = [
  { name: "ApplicantIncome", label: "Applicant Income", step: "0.01" },
  { name: "CoapplicantIncome", label: "Coapplicant Income", step: "0.01" },
  { name: "LoanAmount", label: "Loan Amount", step: "0.01" },
  { name: "Loan_Amount_Term", label: "Loan Amount Term", step: "1" },
  { name: "Credit_History", label: "Credit History (0 or 1)", step: "0.1" },
];

const initialState = {
  Gender: "Male",
  Married: "Yes",
  Dependents: "0",
  Education: "Graduate",
  Self_Employed: "No",
  ApplicantIncome: "5000",
  CoapplicantIncome: "0",
  LoanAmount: "128",
  Loan_Amount_Term: "360",
  Credit_History: "1",
  Property_Area: "Urban",
};

export default function Home() {
  const [form, setForm] = useState(initialState);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const apiBaseUrl = useMemo(
    () => process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000",
    []
  );

  const handleChange = (name, value) => {
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setResult("");
    setError("");

    const payload = {
      ...form,
      ApplicantIncome: Number(form.ApplicantIncome),
      CoapplicantIncome: Number(form.CoapplicantIncome),
      LoanAmount: Number(form.LoanAmount),
      Loan_Amount_Term: Number(form.Loan_Amount_Term),
      Credit_History: Number(form.Credit_History),
    };

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Prediction failed");
      }

      const data = await response.json();
      setResult(data.result);
    } catch (submissionError) {
      setError(submissionError.message || "Could not reach backend service.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <h1>Loan Approval Prediction (KNN)</h1>
        <p className={styles.subtitle}>
          Enter applicant details to get a real-time approval prediction.
        </p>

        <form className={styles.form} onSubmit={handleSubmit}>
          <label>
            Gender
            <select value={form.Gender} onChange={(e) => handleChange("Gender", e.target.value)}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </label>

          <label>
            Married
            <select value={form.Married} onChange={(e) => handleChange("Married", e.target.value)}>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </label>

          <label>
            Dependents
            <select
              value={form.Dependents}
              onChange={(e) => handleChange("Dependents", e.target.value)}
            >
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3+">3+</option>
            </select>
          </label>

          <label>
            Education
            <select
              value={form.Education}
              onChange={(e) => handleChange("Education", e.target.value)}
            >
              <option value="Graduate">Graduate</option>
              <option value="Not Graduate">Not Graduate</option>
            </select>
          </label>

          <label>
            Self Employed
            <select
              value={form.Self_Employed}
              onChange={(e) => handleChange("Self_Employed", e.target.value)}
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </label>

          <label>
            Property Area
            <select
              value={form.Property_Area}
              onChange={(e) => handleChange("Property_Area", e.target.value)}
            >
              <option value="Urban">Urban</option>
              <option value="Semiurban">Semiurban</option>
              <option value="Rural">Rural</option>
            </select>
          </label>

          {numericFields.map((field) => (
            <label key={field.name}>
              {field.label}
              <input
                type="number"
                min="0"
                step={field.step}
                value={form[field.name]}
                onChange={(e) => handleChange(field.name, e.target.value)}
                required
              />
            </label>
          ))}

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Loan Status"}
          </button>
        </form>

        {result && <p className={styles.result}>Prediction: {result}</p>}
        {error && <p className={styles.error}>Error: {error}</p>}
      </main>
    </div>
  );
}
