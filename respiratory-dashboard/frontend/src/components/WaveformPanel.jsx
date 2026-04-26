import { useState, useEffect, useRef } from "react";
import "./WaveformPanel.css";

const TABS = [
  { key: "raw",               label: "Raw Waveform",        color: "#3b82f6", desc: "Time-domain amplitude signal" },
  { key: "rms_envelope",      label: "RMS Envelope",         color: "#22c55e", desc: "Energy envelope — breathing rhythm" },
  { key: "spectral_centroid", label: "Spectral Centroid",    color: "#f59e0b", desc: "Brightness over time — wheeze indicator" },
  { key: "zcr",               label: "Zero-Crossing Rate",   color: "#a78bfa", desc: "Crackle & fricative detector" },
  { key: "mfcc_mean",         label: "MFCC Coefficients",    color: "#14b8a6", desc: "Mean spectral shape fingerprint" },
];

export default function WaveformPanel({ waveforms }) {
  const [active, setActive] = useState("raw");
  const canvasRef = useRef();

  const tab = TABS.find(t => t.key === active);

  useEffect(() => {
    if (!canvasRef.current || !waveforms) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (active === "mfcc_mean") {
      drawMfccBar(ctx, w, h, waveforms.mfcc_mean, tab.color);
    } else {
      drawLine(ctx, w, h, waveforms[active], active, tab.color);
    }
  }, [active, waveforms]);

  return (
    <div className="wp-card">
      <div className="wp-header">
        <h2 className="wp-section-title">Waveform Analysis</h2>
        <p className="wp-desc">{tab.desc}</p>
      </div>

      <div className="wp-tabs">
        {TABS.map(t => (
          <button
            key={t.key}
            className={`wp-tab ${active === t.key ? "active" : ""}`}
            style={active === t.key ? { "--tab-color": t.color } : {}}
            onClick={() => setActive(t.key)}
          >
            <span className="wp-tab-dot" style={{ background: t.color }} />
            {t.label}
          </button>
        ))}
      </div>

      <div className="wp-canvas-wrap">
        <canvas ref={canvasRef} className="wp-canvas" />
      </div>

      <WaveformStats waveforms={waveforms} active={active} color={tab.color} />
    </div>
  );
}

// ── Canvas drawing helpers ────────────────────────────────────────────────────

function drawGrid(ctx, w, h) {
  ctx.strokeStyle = "rgba(34,45,66,0.7)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = (i / 5) * h;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let i = 0; i <= 8; i++) {
    const x = (i / 8) * w;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
}

function drawLine(ctx, w, h, data, type, color) {
  drawGrid(ctx, w, h);

  let times, values;
  if (type === "raw") {
    times = data.times; values = data.amplitudes;
  } else if (type === "spectral_centroid") {
    times = data.times; values = data.values;
  } else {
    times = data.times; values = data.values;
  }

  const isCentered = type === "raw";
  const max = Math.max(...values.map(Math.abs)) || 1;
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const range = maxV - minV || 1;

  const toY = (v) => isCentered
    ? h / 2 - (v / max) * (h / 2 - 8)
    : h - 8 - ((v - minV) / range) * (h - 16);

  // Centre line for raw waveform
  if (isCentered) {
    ctx.strokeStyle = `${color}22`;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  }

  // Gradient stroke
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, color + "cc");
  grad.addColorStop(0.5, color);
  grad.addColorStop(1, color + "cc");

  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = toY(v);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = grad;
  ctx.lineWidth = 1.8;
  ctx.lineJoin = "round";
  ctx.stroke();

  // Fill
  const baseline = isCentered ? h / 2 : h - 8;
  ctx.lineTo(w, baseline);
  ctx.lineTo(0, baseline);
  ctx.closePath();
  const fill = ctx.createLinearGradient(0, 0, 0, h);
  fill.addColorStop(0, color + "28");
  fill.addColorStop(1, color + "03");
  ctx.fillStyle = fill;
  ctx.fill();

  // X-axis labels
  ctx.fillStyle = "rgba(77,96,128,0.85)";
  ctx.font = "10px DM Sans, system-ui";
  if (times?.length) {
    ctx.fillText("0s", 4, h - 4);
    ctx.fillText(`${times[times.length - 1]?.toFixed(1)}s`, w - 28, h - 4);
  }
}

function drawMfccBar(ctx, w, h, mfccData, color) {
  drawGrid(ctx, w, h);
  if (!mfccData) return;

  const { coefficients, means, stds } = mfccData;
  const n = coefficients.length;
  const barW = (w / n) * 0.65;
  const gap = (w / n) * 0.35;
  const maxAbs = Math.max(...means.map(Math.abs)) || 1;

  means.forEach((mean, i) => {
    const x = (i / n) * w + gap / 2;
    const barH = Math.abs(mean / maxAbs) * (h / 2 - 16);
    const y = mean >= 0 ? h / 2 - barH : h / 2;
    const std = stds[i];
    const stdH = (std / maxAbs) * (h / 2 - 16);

    // Error bar
    ctx.strokeStyle = color + "55";
    ctx.lineWidth = 1;
    const cx = x + barW / 2;
    const topY = (mean >= 0 ? h / 2 - barH : h / 2) - stdH;
    const botY = (mean >= 0 ? h / 2 - barH : h / 2 + barH) + stdH;
    ctx.beginPath(); ctx.moveTo(cx, topY); ctx.lineTo(cx, botY); ctx.stroke();

    // Bar
    ctx.fillStyle = mean >= 0 ? color + "cc" : color + "66";
    ctx.fillRect(x, y, barW, barH);
  });

  // Centre line
  ctx.strokeStyle = `${color}33`;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

  // Labels
  ctx.fillStyle = "rgba(77,96,128,0.85)";
  ctx.font = "9px DM Sans, system-ui";
  ctx.textAlign = "center";
  coefficients.forEach((c, i) => {
    const x = (i / n) * w + gap / 2 + barW / 2;
    ctx.fillText(`M${c}`, x, h - 3);
  });
  ctx.textAlign = "left";
}

// ── Stats strip ───────────────────────────────────────────────────────────────

function stat(label, value) {
  return (
    <div key={label} className="wp-stat">
      <span className="wp-stat-label">{label}</span>
      <span className="wp-stat-val">{value}</span>
    </div>
  );
}

function WaveformStats({ waveforms, active, color }) {
  if (!waveforms) return null;

  let items = [];

  if (active === "raw") {
    const a = waveforms.raw.amplitudes;
    const max = Math.max(...a.map(Math.abs));
    const rms = Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
    items = [
      stat("Peak amplitude", max.toFixed(4)),
      stat("RMS", rms.toFixed(4)),
      stat("Duration", `${waveforms.raw.times.at(-1)?.toFixed(2)}s`),
      stat("Samples shown", waveforms.raw.amplitudes.length),
    ];
  } else if (active === "rms_envelope") {
    const v = waveforms.rms_envelope.values;
    items = [
      stat("Mean RMS", (v.reduce((a, b) => a + b, 0) / v.length).toFixed(4)),
      stat("Peak RMS", Math.max(...v).toFixed(4)),
      stat("Min RMS", Math.min(...v).toFixed(4)),
    ];
  } else if (active === "spectral_centroid") {
    const hz = waveforms.spectral_centroid.values_hz;
    items = [
      stat("Mean centroid", `${(hz.reduce((a,b)=>a+b,0)/hz.length).toFixed(0)} Hz`),
      stat("Peak centroid", `${Math.max(...hz).toFixed(0)} Hz`),
      stat("Min centroid", `${Math.min(...hz).toFixed(0)} Hz`),
    ];
  } else if (active === "zcr") {
    const v = waveforms.zcr.values;
    items = [
      stat("Mean ZCR", (v.reduce((a,b)=>a+b,0)/v.length).toFixed(4)),
      stat("Peak ZCR", Math.max(...v).toFixed(4)),
    ];
  } else if (active === "mfcc_mean") {
    const m = waveforms.mfcc_mean.means;
    items = [
      stat("MFCC 1 (energy)", m[0]?.toFixed(2)),
      stat("MFCC 2", m[1]?.toFixed(2)),
      stat("MFCC 3", m[2]?.toFixed(2)),
      stat("MFCC 4", m[3]?.toFixed(2)),
    ];
  }

  return (
    <div className="wp-stats" style={{ "--stat-color": color }}>
      {items}
    </div>
  );
}
