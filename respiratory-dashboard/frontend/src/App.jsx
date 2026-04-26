import { useState } from "react";
import UploadZone from "./components/UploadZone";
import WaveformPanel from "./components/WaveformPanel";
import SpectrogramPanel from "./components/SpectrogramPanel";
import PredictionPanel from "./components/PredictionPanel";
import MetadataBar from "./components/MetadataBar";
import StatusBadge from "./components/StatusBadge";
import "./App.css";

const API = "";

export default function App() {
  // "idle" | "processing" | "viz_ready" | "done" | "error"
  const [status, setStatus]         = useState("idle");
  const [vizData, setVizData]       = useState(null);   // waveforms + spectrograms
  const [prediction, setPrediction] = useState(null);   // null | { model_pending } | { predictions }
  const [errorMsg, setErrorMsg]     = useState("");
  const [currentFile, setCurrentFile] = useState(null);

  async function handleFile(file) {
    setStatus("processing");
    setVizData(null);
    setPrediction(null);
    setErrorMsg("");
    setCurrentFile(file);

    // ── Phase 1: always fetch visualizations ──────────────────────────────
    try {
      const form1 = new FormData();
      form1.append("file", file);
      const res1 = await fetch(`${API}/api/visualize`, { method: "POST", body: form1 });
      const d1 = await res1.json();
      if (!res1.ok || !d1.success) throw new Error(d1.error || "Visualization failed");
      setVizData(d1);
      setStatus("viz_ready");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
      return;
    }

    // ── Phase 2: try model prediction (non-blocking — always resolves) ─────
    try {
      const form2 = new FormData();
      form2.append("file", file);
      const res2 = await fetch(`${API}/api/predict`, { method: "POST", body: form2 });
      const d2 = await res2.json();
      setPrediction(d2);
    } catch {
      setPrediction({ model_pending: true, message: "Prediction unavailable." });
    } finally {
      setStatus("done");
    }
  }

  const showViz = vizData && (status === "viz_ready" || status === "done");

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <div className="logo-mark">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="13" stroke="var(--accent)" strokeWidth="1.5" />
              <path d="M7 14 Q10 8 14 14 Q18 20 21 14" stroke="var(--accent)" strokeWidth="1.8" fill="none" strokeLinecap="round" />
              <circle cx="14" cy="14" r="2" fill="var(--accent)" />
            </svg>
          </div>
          <div>
            <h1 className="app-title">PulmoScan</h1>
            <p className="app-subtitle">Respiratory Disease Detection</p>
          </div>
        </div>
        <StatusBadge status={status} />
      </header>

      <main className="app-main">
        <section className="upload-section">
          <UploadZone onFile={handleFile} status={status} />
          {status === "error" && (
            <div className="error-banner">
              <span>⚠</span> {errorMsg}
            </div>
          )}
        </section>

        {showViz && (
          <>
            <MetadataBar meta={vizData.metadata} />
            <WaveformPanel waveforms={vizData.waveforms} />
            <SpectrogramPanel spectrograms={vizData.spectrograms} />
            <PredictionPanel prediction={prediction} loading={status === "viz_ready"} />
          </>
        )}

        {status === "idle" && (
          <div className="idle-hint">
            Upload a <strong>.wav</strong> lung auscultation recording to begin analysis
          </div>
        )}
      </main>
    </div>
  );
}
