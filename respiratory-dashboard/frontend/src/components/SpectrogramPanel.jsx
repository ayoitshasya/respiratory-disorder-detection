import { useState } from "react";
import "./SpectrogramPanel.css";

const TABS = [
  {
    key: "mel",
    label: "Mel Spectrogram",
    color: "#ef6c00",
    desc: "128 Mel-frequency bands · inferno colormap · primary model input",
  },
  {
    key: "mfcc",
    label: "MFCC",
    color: "#3b82f6",
    desc: "13 Mel-frequency cepstral coefficients · coolwarm colormap",
  },
];

export default function SpectrogramPanel({ spectrograms }) {
  const [active, setActive] = useState("mel");
  const tab = TABS.find(t => t.key === active);

  return (
    <div className="sp-card">
      <div className="sp-header">
        <h2 className="sp-section-title">Spectrograms</h2>
        <p className="sp-desc">{tab.desc}</p>
      </div>

      <div className="sp-tabs">
        {TABS.map(t => (
          <button
            key={t.key}
            className={`sp-tab ${active === t.key ? "active" : ""}`}
            style={active === t.key ? { "--tab-color": t.color } : {}}
            onClick={() => setActive(t.key)}
          >
            <span className="sp-tab-dot" style={{ background: t.color }} />
            {t.label}
          </button>
        ))}
      </div>

      <div className="sp-image-wrap">
        {spectrograms?.[active]
          ? <img
              key={active}
              src={`data:image/png;base64,${spectrograms[active]}`}
              alt={`${tab.label} spectrogram`}
              className="sp-image"
            />
          : <div className="sp-placeholder">Generating…</div>
        }
      </div>

      <div className="sp-legend">
        {active === "mel"   && <MelLegend />}
        {active === "mfcc"  && <MfccLegend />}
      </div>
    </div>
  );
}

function Chip({ label }) {
  return <span className="sp-chip">{label}</span>;
}

function MelLegend() {
  return (
    <>
      <Chip label="128 Mel bands" />
      <Chip label="fmax 8 kHz" />
      <Chip label="dB scale" />
      <Chip label="Model input" />
    </>
  );
}

function MfccLegend() {
  return (
    <>
      <Chip label="13 coefficients" />
      <Chip label="Captures timbre" />
      <Chip label="Blue = low · Red = high" />
    </>
  );
}

