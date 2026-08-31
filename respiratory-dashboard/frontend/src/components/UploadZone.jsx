import { useRef, useState, useEffect } from "react";
import "./UploadZone.css";

// Encode a mono AudioBuffer to 16-bit PCM WAV blob
function encodeWav(audioBuffer) {
  const samples = audioBuffer.getChannelData(0);
  const sr = audioBuffer.sampleRate;
  const dataLen = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + dataLen, true);
  str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true);     // chunk size
  v.setUint16(20, 1, true);      // PCM
  v.setUint16(22, 1, true);      // mono
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); // byte rate
  v.setUint16(32, 2, true);      // block align
  v.setUint16(34, 16, true);     // bits per sample
  str(36, "data"); v.setUint32(40, dataLen, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

const DURATIONS = [
  { label: "10 s", s: 10 },
  { label: "15 s", s: 15 },
  { label: "30 s", s: 30 },
  { label: "Free", s: 0 },
];

const fmt = (s) =>
  `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

export default function UploadZone({ onFile, status }) {
  const [dragging, setDragging] = useState(false);
  const [mode, setMode] = useState("upload"); // "upload" | "record"
  const [recState, setRecState] = useState("idle"); // "idle" | "recording" | "processing" | "recorded"
  const [duration, setDuration] = useState(15);
  const [elapsed, setElapsed] = useState(0);
  const [audioUrl, setAudioUrlState] = useState(null);
  const [pendingFile, setPendingFile] = useState(null);
  const [micErr, setMicErr] = useState("");

  const inputRef = useRef();
  const streamRef = useRef(null);
  const mediaRecRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const rafRef = useRef(null);
  const analyserRef = useRef(null);
  const audioCtxRef = useRef(null);
  const canvasRef = useRef(null);
  const audioUrlRef = useRef(null);
  const alive = useRef(true);

  const busy = status === "processing" || status === "viz_ready";

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      cancelAnimationFrame(rafRef.current);
      clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close().catch(() => {});
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  function setAudioUrl(url) {
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    audioUrlRef.current = url;
    setAudioUrlState(url);
  }

  function stopCapture() {
    clearInterval(timerRef.current);
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close().catch(() => {});
    streamRef.current = null;
    analyserRef.current = null;
    audioCtxRef.current = null;
  }

  // ── Upload mode ────────────────────────────────────────────────────────
  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (busy) return;
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith(".wav")) onFile(f);
  }
  function handleChange(e) {
    const f = e.target.files[0];
    if (f) onFile(f);
    e.target.value = "";
  }

  // ── Record mode ────────────────────────────────────────────────────────
  async function startRecording() {
    setMicErr("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!alive.current) { stream.getTracks().forEach((t) => t.stop()); return; }
      streamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 64;
      src.connect(analyser);
      analyserRef.current = analyser;

      const rec = new MediaRecorder(stream);
      mediaRecRef.current = rec;
      chunksRef.current = [];
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = processRecording;
      rec.start();

      setRecState("recording");
      setElapsed(0);

      const t0 = Date.now();
      timerRef.current = setInterval(() => {
        if (!alive.current) return;
        const s = Math.floor((Date.now() - t0) / 1000);
        setElapsed(s);
        if (duration > 0 && s >= duration) stopRecording();
      }, 200);

      const data = new Uint8Array(analyser.frequencyBinCount);
      function draw() {
        if (!analyserRef.current || !canvasRef.current) return;
        analyser.getByteFrequencyData(data);
        const cv = canvasRef.current;
        const gc = cv.getContext("2d");
        const W = cv.width, H = cv.height;
        gc.clearRect(0, 0, W, H);
        const N = 14, gap = 3;
        const bw = (W - gap * (N - 1)) / N;
        for (let i = 0; i < N; i++) {
          const val = data[Math.floor((i * data.length) / N)];
          const h = Math.max(3, (val / 255) * H);
          gc.fillStyle = "rgba(0,212,255,0.82)";
          gc.fillRect(i * (bw + gap), H - h, bw, h);
        }
        rafRef.current = requestAnimationFrame(draw);
      }
      rafRef.current = requestAnimationFrame(draw);
    } catch (err) {
      setMicErr(
        err.name === "NotAllowedError"
          ? "Microphone access denied — allow it in browser settings."
          : "Could not open microphone. Close other apps using it."
      );
    }
  }

  function stopRecording() {
    stopCapture();
    if (mediaRecRef.current?.state !== "inactive") mediaRecRef.current.stop();
  }

  async function processRecording() {
    if (!alive.current) return;
    setRecState("processing");
    const blob = new Blob(chunksRef.current, {
      type: chunksRef.current[0]?.type || "audio/webm",
    });
    setAudioUrl(URL.createObjectURL(blob));
    try {
      const ab = await blob.arrayBuffer();
      const decCtx = new AudioContext();
      const decoded = await decCtx.decodeAudioData(ab);
      await decCtx.close();
      // Resample to 16 kHz mono (matches backend SR=16000)
      const TARGET_SR = 16000;
      const offCtx = new OfflineAudioContext(
        1,
        Math.ceil(decoded.duration * TARGET_SR),
        TARGET_SR
      );
      const bufSrc = offCtx.createBufferSource();
      bufSrc.buffer = decoded;
      bufSrc.connect(offCtx.destination);
      bufSrc.start(0);
      const resampled = await offCtx.startRendering();
      const wavBlob = encodeWav(resampled);
      const file = new File([wavBlob], "recording.wav", { type: "audio/wav" });
      if (alive.current) { setPendingFile(file); setRecState("recorded"); }
    } catch {
      if (alive.current) setMicErr("Failed to encode audio. Please try again.");
    }
  }

  function handleAnalyze() { if (pendingFile) onFile(pendingFile); }

  function handleReRecord() {
    setRecState("idle");
    setPendingFile(null);
    setAudioUrl(null);
    setMicErr("");
  }

  return (
    <div
      className={`upload-zone${dragging ? " dragging" : ""}${busy ? " busy" : ""}${mode === "record" ? " record-mode" : ""}`}
      onDragOver={mode === "upload" ? (e) => { e.preventDefault(); setDragging(true); } : undefined}
      onDragLeave={mode === "upload" ? () => setDragging(false) : undefined}
      onDrop={mode === "upload" ? handleDrop : undefined}
      onClick={mode === "upload" && !busy ? () => inputRef.current.click() : undefined}
    >
      <input ref={inputRef} type="file" accept=".wav" hidden onChange={handleChange} />

      {busy ? (
        <div className="upload-busy">
          <div className="pulse-ring" />
          <div className="upload-busy-text">Processing audio & running inference…</div>
          <div className="upload-steps">
            <Step done label="Upload" />
            <div className="step-line" />
            <Step
              active={status === "processing"}
              done={status === "viz_ready" || status === "done"}
              label="Waveforms & spectrograms"
            />
            <div className="step-line" />
            <Step active={status === "viz_ready"} done={status === "done"} label="Model inference" />
          </div>
        </div>
      ) : (
        <div className="upload-content">
          {/* Mode tabs */}
          <div className="mode-tabs" onClick={(e) => e.stopPropagation()}>
            <button
              className={`mode-tab${mode === "upload" ? " active" : ""}`}
              onClick={() => setMode("upload")}
            >
              <UploadSvg /> Upload File
            </button>
            <button
              className={`mode-tab${mode === "record" ? " active" : ""}`}
              onClick={() => { setMode("record"); setMicErr(""); }}
            >
              <MicSvg /> Record Audio
            </button>
          </div>

          {/* Upload mode */}
          {mode === "upload" && (
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

          {/* Record mode */}
          {mode === "record" && (
            <div className="record-zone" onClick={(e) => e.stopPropagation()}>
              {recState === "idle" && (
                <>
                  <div className="record-icon-wrap"><MicSvg size={34} /></div>
                  <p className="upload-primary">Record breathing sounds</p>
                  <p className="upload-secondary">Place mic near your chest · breathe normally</p>
                  <div className="duration-row">
                    {DURATIONS.map((d) => (
                      <button
                        key={d.s}
                        className={`dur-pill${duration === d.s ? " active" : ""}`}
                        onClick={() => setDuration(d.s)}
                      >
                        {d.label}
                      </button>
                    ))}
                  </div>
                  {micErr && <p className="mic-error">{micErr}</p>}
                  <button className="rec-btn start" onClick={startRecording}>
                    <span className="rec-dot-sm" /> Start Recording
                  </button>
                </>
              )}

              {recState === "recording" && (
                <>
                  <div className="rec-live-row">
                    <span className="rec-dot-sm blink" />
                    <span className="rec-label">REC</span>
                    <span className="rec-timer">
                      {fmt(elapsed)}{duration > 0 ? ` / ${fmt(duration)}` : ""}
                    </span>
                  </div>
                  <canvas ref={canvasRef} className="bars-canvas" width="200" height="48" />
                  <button className="rec-btn stop" onClick={stopRecording}>
                    <span className="stop-sq" /> Stop
                  </button>
                </>
              )}

              {recState === "processing" && (
                <p className="upload-secondary">Encoding audio…</p>
              )}

              {recState === "recorded" && (
                <>
                  <p className="upload-primary">Recording ready</p>
                  {audioUrl && <audio className="rec-audio" src={audioUrl} controls />}
                  {micErr && <p className="mic-error">{micErr}</p>}
                  <div className="rec-action-row">
                    <button className="rec-btn analyze" onClick={handleAnalyze}>Analyze</button>
                    <button className="rec-btn ghost" onClick={handleReRecord}>Re-record</button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UploadSvg() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function MicSvg({ size = 13 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function Step({ done, active, label }) {
  return (
    <div className={`step${done ? " done" : ""}${active ? " active" : ""}`}>
      <div className="step-dot" />
      <span>{label}</span>
    </div>
  );
}
