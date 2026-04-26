export default function MetadataBar({ meta }) {
  const items = [
    { label: "File",         value: meta.filename },
    { label: "Duration",     value: `${meta.duration_s}s` },
    { label: "Sample rate",  value: `${(meta.sample_rate / 1000).toFixed(1)} kHz` },
    { label: "RMS energy",   value: meta.rms_energy },
    { label: "ZCR",          value: meta.zero_crossing_rate },
    { label: "Sp. centroid", value: `${meta.spectral_centroid_hz} Hz` },
    { label: "Sp. rolloff",  value: `${meta.spectral_rolloff_hz} Hz` },
  ];

  return (
    <div style={{
      display: "flex",
      flexWrap: "wrap",
      background: "var(--bg-surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius)",
      overflow: "hidden",
    }}>
      {items.map((item, i) => (
        <div
          key={i}
          style={{
            padding: "10px 16px",
            borderRight: i < items.length - 1 ? "1px solid var(--border)" : "none",
            flex: "1 1 100px",
          }}
        >
          <p style={{
            fontSize: 10, color: "var(--text-3)",
            textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2,
          }}>
            {item.label}
          </p>
          <p style={{
            fontSize: 13, color: "var(--text-1)", fontWeight: 500,
            fontVariantNumeric: "tabular-nums",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {item.value}
          </p>
        </div>
      ))}
    </div>
  );
}
