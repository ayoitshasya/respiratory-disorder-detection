import "./PredictionPanel.css";

const SEVERITY = {
  Healthy:              { color: "#22c55e", risk: "No disease detected" },
  COPD:                 { color: "#ef4444", risk: "High risk" },
  Pneumonia:            { color: "#f97316", risk: "High risk" },
  Bronchitis:           { color: "#f59e0b", risk: "Moderate risk" },
  Asthma:               { color: "#a78bfa", risk: "Moderate risk" },
  "Pulmonary Fibrosis": { color: "#ef4444", risk: "High risk" },
};

export default function PredictionPanel({ prediction, loading }) {
  // ── Loading: phase 1 done, waiting for phase 2 ────────────────────────
  if (loading) {
    return (
      <div className="pred-panel">
        <PendingShell
          icon="⏳"
          title="Running inference…"
          message="Visualizations are ready. Waiting for model prediction."
          pulse
        />
      </div>
    );
  }

  // ── Model not yet plugged in ───────────────────────────────────────────
  if (!prediction || prediction.model_pending) {
    return (
      <div className="pred-panel">
        <PendingShell
          icon="🔬"
          title="Model in Training"
          message={prediction?.message || "Your model is still being developed. Upload your trained model file and set MODEL_READY = True in app.py to enable predictions."}
          steps={[
            "Train your model and export it (.pt or .h5)",
            "Implement run_model() in backend/app.py",
            "Set MODEL_READY = True at the top of app.py",
            "Restart Flask — predictions will appear here automatically",
          ]}
        />
      </div>
    );
  }

  // ── Error from model ───────────────────────────────────────────────────
  if (!prediction.success) {
    return (
      <div className="pred-panel">
        <PendingShell
          icon="⚠"
          title="Prediction Error"
          message={prediction.error || "Model inference failed. Check the Flask console for details."}
          isError
        />
      </div>
    );
  }

  // ── Full prediction results ────────────────────────────────────────────
  const { predictions, top_prediction: top } = prediction;
  const { color, risk } = SEVERITY[top.disease] || { color: "#3b82f6", risk: "" };
  const pct = (top.probability * 100).toFixed(1);
  const circumference = 2 * Math.PI * 50;

  return (
    <div className="pred-panel">
      <div className="pred-top" style={{ "--accent-color": color }}>
        <div className="pred-top-left">
          <p className="pred-label">Primary Diagnosis</p>
          <h2 className="pred-disease">{top.disease}</h2>
          <span className="pred-risk">{risk}</span>
        </div>
        <svg viewBox="0 0 120 120" width="120" height="120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--bg-raised)" strokeWidth="10" />
          <circle
            cx="60" cy="60" r="50" fill="none" stroke={color} strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - top.probability)}
            transform="rotate(-90 60 60)"
            style={{ transition: "stroke-dashoffset 1s ease" }}
          />
          <text x="60" y="56" textAnchor="middle" fill={color} fontSize="22" fontWeight="700" fontFamily="DM Sans, system-ui">
            {pct}%
          </text>
          <text x="60" y="72" textAnchor="middle" fill="var(--text-3)" fontSize="10" fontFamily="DM Sans, system-ui">
            confidence
          </text>
        </svg>
      </div>

      <div className="pred-all">
        <p className="pred-all-title">All disease probabilities</p>
        <div className="pred-bars">
          {predictions.map((p) => {
            const c = (SEVERITY[p.disease] || {}).color || "#3b82f6";
            const w = (p.probability * 100).toFixed(1);
            const isTop = p.disease === top.disease;
            return (
              <div key={p.disease} className={`pred-bar-row ${isTop ? "is-top" : ""}`}>
                <span className="pred-bar-label">{p.disease}</span>
                <div className="pred-bar-track">
                  <div className="pred-bar-fill" style={{ width: `${w}%`, background: c, opacity: isTop ? 1 : 0.45 }} />
                </div>
                <span className="pred-bar-pct" style={{ color: isTop ? c : "var(--text-3)" }}>{w}%</span>
              </div>
            );
          })}
        </div>
      </div>

      <p className="pred-disclaimer">⚠ For research use only. Not a substitute for clinical diagnosis.</p>
    </div>
  );
}

function PendingShell({ icon, title, message, steps, pulse, isError }) {
  return (
    <div className={`pending-shell ${isError ? "is-error" : ""}`}>
      <div className={`pending-icon ${pulse ? "pulse" : ""}`}>{icon}</div>
      <div className="pending-body">
        <h3 className="pending-title">{title}</h3>
        <p className="pending-message">{message}</p>
        {steps && (
          <ol className="pending-steps">
            {steps.map((s, i) => (
              <li key={i}><span className="step-num">{i + 1}</span>{s}</li>
            ))}
          </ol>
        )}
      </div>
      <div className="pending-badge">
        {pulse ? "Processing…" : isError ? "Error" : "Pending"}
      </div>
    </div>
  );
}

