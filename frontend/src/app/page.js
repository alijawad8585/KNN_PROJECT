"use client";

import { useMemo, useState } from "react";
import styles from "./page.module.css";

const INITIAL_FORM = {
  Gender: "Male",
  Married: "Yes",
  Dependents: "0",
  Education: "Graduate",
  Self_Employed: "No",
  ApplicantIncome: "5000",
  CoapplicantIncome: "0",
  LoanAmount: "120",
  Loan_Amount_Term: "360",
  Credit_History: "1",
  Property_Area: "Urban",
};

export default function Home() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");

  const backendUrl = useMemo(
    () => process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000",
    []
  );

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((previous) => ({ ...previous, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction("");

    try {
      const response = await fetch(`${backendUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || "Prediction failed.");
      }

      setPrediction(result.prediction);
    } catch (submissionError) {
      setError(submissionError.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <h1>KNN Loan Approval Predictor</h1>
        <p>Enter loan applicant details and get a real-time prediction.</p>

        <form className={styles.form} onSubmit={handleSubmit}>
          <label>
            Gender
            <select name="Gender" value={formData.Gender} onChange={handleChange}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </label>

          <label>
            Married
            <select name="Married" value={formData.Married} onChange={handleChange}>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </label>

          <label>
            Dependents
            <select name="Dependents" value={formData.Dependents} onChange={handleChange}>
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3+">3+</option>
            </select>
          </label>

          <label>
            Education
            <select
              name="Education"
              value={formData.Education}
              onChange={handleChange}
            >
              <option value="Graduate">Graduate</option>
              <option value="Not Graduate">Not Graduate</option>
            </select>
          </label>

          <label>
            Self Employed
            <select
              name="Self_Employed"
              value={formData.Self_Employed}
              onChange={handleChange}
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </label>

          <label>
            Applicant Income
            <input
              type="number"
              name="ApplicantIncome"
              value={formData.ApplicantIncome}
              onChange={handleChange}
              min="0"
              step="1"
              required
            />
          </label>

          <label>
            Coapplicant Income
            <input
              type="number"
              name="CoapplicantIncome"
              value={formData.CoapplicantIncome}
              onChange={handleChange}
              min="0"
              step="1"
              required
            />
          </label>

          <label>
            Loan Amount
            <input
              type="number"
              name="LoanAmount"
              value={formData.LoanAmount}
              onChange={handleChange}
              min="0"
              step="1"
              required
            />
          </label>

          <label>
            Loan Amount Term
            <input
              type="number"
              name="Loan_Amount_Term"
              value={formData.Loan_Amount_Term}
              onChange={handleChange}
              min="1"
              step="1"
              required
            />
          </label>

          <label>
            Credit History
            <select
              name="Credit_History"
              value={formData.Credit_History}
              onChange={handleChange}
            >
              <option value="1">1</option>
              <option value="0">0</option>
            </select>
          </label>

          <label>
            Property Area
            <select
              name="Property_Area"
              value={formData.Property_Area}
              onChange={handleChange}
            >
              <option value="Urban">Urban</option>
              <option value="Semiurban">Semiurban</option>
              <option value="Rural">Rural</option>
            </select>
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Loan Status"}
          </button>
        </form>

        {prediction && (
          <section className={styles.result}>
            Prediction: <strong>{prediction}</strong>
          </section>
        )}
        {error && <section className={styles.error}>Error: {error}</section>}
      </main>
    </div>
  );
}
