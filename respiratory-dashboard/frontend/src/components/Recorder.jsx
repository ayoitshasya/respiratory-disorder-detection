import { useEffect, useRef, useState } from "react";
import './Recorder.css';

// Simple recorder: records up to `maxSeconds` then auto-stops.
// Produces a WAV File and calls `onComplete(file)`.
export default function Recorder({ onComplete, onClose, maxSeconds = 10 }) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [previewFile, setPreviewFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [waveformData, setWaveformData] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const streamRef = useRef(null);
  const cancelRef = useRef(false);
  const waveformRef = useRef(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      clearInterval(timerRef.current);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!waveformData || !waveformRef.current) return;

    const canvas = waveformRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 2;
    ctx.beginPath();

    waveformData.forEach((value, index) => {
      const x = (index / Math.max(1, waveformData.length - 1)) * width;
      const y = height / 2 - value * (height / 2 - 10);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();
  }, [waveformData]);

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      cancelRef.current = false;
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      setPreviewFile(null);
      setWaveformData(null);

      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        if (cancelRef.current) {
          cancelRef.current = false;
          return;
        }

        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const arrayBuffer = await blob.arrayBuffer();
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await ac.decodeAudioData(arrayBuffer);
        const wavBlob = encodeWAV(audioBuffer);
        const file = new File([wavBlob], `recording.wav`, { type: 'audio/wav' });

        setPreviewFile(file);
        setPreviewUrl(URL.createObjectURL(wavBlob));
        setWaveformData(extractWaveform(audioBuffer));
        ac.close();
      };

      mr.start();
      setRecording(true);
      setSeconds(0);

      timerRef.current = setInterval(() => {
        setSeconds((s) => {
          const ns = s + 1;
          if (ns >= maxSeconds) stop();
          return ns;
        });
      }, 1000);
    } catch (err) {
      console.error("Could not start recording", err);
      onClose && onClose();
    }
  }

  function stop() {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    clearInterval(timerRef.current);
    setRecording(false);
  }

  function extractWaveform(audioBuffer) {
    const channels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const raw = new Float32Array(length);

    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (let ch = 0; ch < channels; ch++) {
        sum += Math.abs(audioBuffer.getChannelData(ch)[i]);
      }
      raw[i] = sum / channels;
    }

    const step = Math.max(1, Math.floor(length / 220));
    const samples = [];
    for (let i = 0; i < length; i += step) {
      samples.push(raw[i]);
    }
    return samples;
  }

  function discardRecording() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setPreviewFile(null);
    setWaveformData(null);
    setSeconds(0);
  }

  function closeDialog() {
    if (recording) {
      cancelRef.current = true;
    }
    stop();
    onClose && onClose();
  }

  function useRecording() {
    if (!previewFile) return;
    onComplete && onComplete(previewFile);
    onClose && onClose();
  }

  // Minimal WAV encoder (interleaves channels, 16-bit PCM)
  function encodeWAV(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const samples = audioBuffer.length;

    let interleaved;
    if (numChannels === 1) {
      interleaved = audioBuffer.getChannelData(0);
    } else {
      const channelData = [];
      for (let i = 0; i < numChannels; i++) channelData.push(audioBuffer.getChannelData(i));
      interleaved = new Float32Array(samples * numChannels);
      let idx = 0;
      for (let i = 0; i < samples; i++) {
        for (let ch = 0; ch < numChannels; ch++) {
          interleaved[idx++] = channelData[ch][i];
        }
      }
    }

    const buffer = new ArrayBuffer(44 + interleaved.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + interleaved.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, interleaved.length * 2, true);

    let offset = 44;
    for (let i = 0; i < interleaved.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, interleaved[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  const pct = Math.min(100, Math.round((seconds / maxSeconds) * 100));

  return (
    <div className="recorder-modal">
      <div className="recorder-panel">
        <div className="recorder-header">
          <div>
            <h3>Record Audio</h3>
            <p className="recorder-subtitle">Record up to {maxSeconds} seconds, preview waveform, then confirm or discard.</p>
          </div>
          <div className="recorder-duration">{seconds}s / {maxSeconds}s</div>
        </div>

        <div className="recorder-body">
          <div className={`rec-indicator ${recording ? 'live' : ''}`}>{recording ? 'Recording…' : 'Ready to record'}</div>
          <div className="rec-progress">
            <div className="rec-progress-bar" style={{ width: `${pct}%` }} />
          </div>

          {previewUrl && (
            <div className="recorder-preview">
              <div className="preview-label">Preview recording</div>
              <audio controls src={previewUrl} className="preview-audio" />
              <canvas ref={waveformRef} className="rec-waveform" width={380} height={90} />
            </div>
          )}

          <div className="rec-controls">
            {!recording && !previewUrl && <button className="rec-start" onClick={start}>Start</button>}
            {recording && <button className="rec-stop" onClick={stop}>Stop</button>}
            {previewUrl && <button className="rec-confirm" onClick={useRecording}>Use recording</button>}
            {previewUrl && <button className="rec-discard" onClick={discardRecording}>Discard</button>}
            <button className="rec-close" onClick={closeDialog}>Close</button>
          </div>
        </div>
      </div>
    </div>
  );
}

