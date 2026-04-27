import "./PredictionPanel.css";

// Sound classification labels (primary model head)
const SEVERITY = {
  Normal:  { color: "#22c55e", risk: "No abnormal sounds detected" },
  Crackle: { color: "#f97316", risk: "Crackles detected — possible fluid / fibrosis" },
  Wheeze:  { color: "#f59e0b", risk: "Wheeze detected — possible obstruction / asthma" },
  Both:    { color: "#ef4444", risk: "Crackles & wheeze detected — further review advised" },
};

export default function PredictionPanel({ prediction, loading }) {

  // ── Loading ──────────────────────────────────────────────────────────
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

  // ── Model not ready ──────────────────────────────────────────────────
  if (!prediction || prediction.model_pending) {
    return (
      <div className="pred-panel">
        <PendingShell
          icon="🔬"
          title="Model in Training"
          message={
            prediction?.message ||
            "Your model is still being developed. Upload your trained model file to enable predictions."
          }
        />
      </div>
    );
  }

  // ── Error ───────────────────────────────────────────────────────────
  if (!prediction.success) {
    return (
      <div className="pred-panel">
        <PendingShell
          icon="⚠"
          title="Prediction Error"
          message={prediction.error || "Model inference failed."}
          isError
        />
      </div>
    );
  }

  // ── Extract data ─────────────────────────────────────────────────────
  const {
    predictions = [],
    top_prediction: topSound,
    top_diagnosis: topDiag,
    diagnosis_predictions: diagPreds
  } = prediction;

  const { color, risk } =
    SEVERITY[topSound?.disease] || { color: "#3b82f6", risk: "" };

  const pct = topSound?.probability ? (topSound.probability * 100).toFixed(1) : "0.0";
  const circumference = 2 * Math.PI * 50;

  return (
    <div className="pred-panel">

      {/* ───────────── SOUND ANALYSIS ───────────── */}
      <div className="pred-top" style={{ "--accent-color": color }}>
        <div className="pred-top-left">
          <p className="pred-label">Lung Sound Analysis</p>
          <h2 className="pred-disease">{topSound?.disease || "N/A"}</h2>
          <span className="pred-risk">{risk}</span>
        </div>

        <svg viewBox="0 0 120 120" width="120" height="120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--bg-raised)" strokeWidth="10" />
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - (topSound?.probability || 0))}
            transform="rotate(-90 60 60)"
            style={{ transition: "stroke-dashoffset 1s ease" }}
          />
          <text x="60" y="56" textAnchor="middle" fill={color} fontSize="22" fontWeight="700">
            {pct}%
          </text>
          <text x="60" y="72" textAnchor="middle" fill="var(--text-3)" fontSize="10">
            confidence
          </text>
        </svg>
      </div>

      {/* ───────────── DIAGNOSIS (NEW) ───────────── */}
      {topDiag && (
        <div className="pred-top diagnosis-block">
          <div className="pred-top-left">
            <p className="pred-label">Predicted Condition</p>
            <h2 className="pred-disease">{topDiag.disease}</h2>
            <span className="pred-risk">
              Based on learned lung sound patterns
            </span>
          </div>

          <div className="diag-confidence">
            {(topDiag.probability * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* ───────────── SOUND PROBABILITIES ───────────── */}
      <div className="pred-all">
        <p className="pred-all-title">All sound probabilities</p>
        <div className="pred-bars">
          {predictions.map((p) => {
            const c = (SEVERITY[p.disease] || {}).color || "#3b82f6";
            const w = (p.probability * 100).toFixed(1);
            const isTop = p.disease === topSound?.disease;

            return (
              <div key={p.disease} className={`pred-bar-row ${isTop ? "is-top" : ""}`}>
                <span className="pred-bar-label">{p.disease}</span>
                <div className="pred-bar-track">
                  <div
                    className="pred-bar-fill"
                    style={{
                      width: `${w}%`,
                      background: c,
                      opacity: isTop ? 1 : 0.45,
                    }}
                  />
                </div>
                <span
                  className="pred-bar-pct"
                  style={{ color: isTop ? c : "var(--text-3)" }}
                >
                  {w}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ───────────── DIAGNOSIS PROBABILITIES (NEW) ───────────── */}
      {diagPreds && (
        <div className="pred-all">
          <p className="pred-all-title">All diagnosis probabilities</p>
          <div className="pred-bars">
            {diagPreds.map((p) => {
              const w = (p.probability * 100).toFixed(1);
              const isTop = p.disease === topDiag?.disease;

              return (
                <div key={p.disease} className={`pred-bar-row ${isTop ? "is-top" : ""}`}>
                  <span className="pred-bar-label">{p.disease}</span>
                  <div className="pred-bar-track">
                    <div
                      className="pred-bar-fill"
                      style={{
                        width: `${w}%`,
                        background: "#8b5cf6",
                        opacity: isTop ? 1 : 0.45,
                      }}
                    />
                  </div>
                  <span className="pred-bar-pct">{w}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <p className="pred-disclaimer">
        ⚠ For research use only. Not a substitute for clinical diagnosis.
      </p>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────────
function PendingShell({ icon, title, message, pulse, isError }) {
  return (
    <div className={`pending-shell ${isError ? "is-error" : ""}`}>
      <div className={`pending-icon ${pulse ? "pulse" : ""}`}>{icon}</div>
      <div className="pending-body">
        <h3 className="pending-title">{title}</h3>
        <p className="pending-message">{message}</p>
      </div>
      <div className="pending-badge">
        {pulse ? "Processing…" : isError ? "Error" : "Pending"}
      </div>
    </div>
  );
}