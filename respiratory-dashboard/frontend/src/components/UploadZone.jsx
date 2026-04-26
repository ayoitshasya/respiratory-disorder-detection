import { useRef, useState } from "react";
import "./UploadZone.css";

export default function UploadZone({ onFile, status }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const busy = status === "processing" || status === "viz_ready";

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (busy) return;
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith(".wav")) onFile(file);
  }

  function handleChange(e) {
    const file = e.target.files[0];
    if (file) onFile(file);
    e.target.value = "";
  }

  return (
    <div
      className={`upload-zone ${dragging ? "dragging" : ""} ${busy ? "busy" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !busy && inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".wav"
        hidden
        onChange={handleChange}
      />

      {busy ? (
        <div className="upload-busy">
          <div className="pulse-ring" />
          <div className="upload-busy-text">
            {status === "uploading" ? "Uploading…" : "Processing audio & running inference…"}
          </div>
          <div className="upload-steps">
            <Step done label="Upload" />
            <div className="step-line" />
            <Step active={status === "processing"} done={status === "viz_ready" || status === "done"} label="Waveforms & spectrograms" />
            <div className="step-line" />
            <Step active={status === "viz_ready"} done={status === "done"} label="Model inference" />
          </div>
        </div>
      ) : (
        <div className="upload-idle">
          <div className="upload-icon">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <circle cx="20" cy="20" r="19" stroke="var(--accent)" strokeWidth="1" strokeDasharray="4 3" />
              <path d="M20 26V14M14 20l6-6 6 6" stroke="var(--accent)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <p className="upload-primary">Drop a <span>.wav</span> file here</p>
          <p className="upload-secondary">or click to browse — lung auscultation recordings</p>
        </div>
      )}
    </div>
  );
}

function Step({ done, active, label }) {
  return (
    <div className={`step ${done ? "done" : ""} ${active ? "active" : ""}`}>
      <div className="step-dot" />
      <span>{label}</span>
    </div>
  );
}
