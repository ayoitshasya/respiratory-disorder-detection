const CONFIG = {
  idle:       { label: "Ready",           color: "var(--text-3)",  dot: "#4d6080" },
  processing: { label: "Processing",      color: "var(--amber)",   dot: "#f59e0b", pulse: true },
  viz_ready:  { label: "Visualizing…",    color: "var(--accent)",  dot: "#3b82f6", pulse: true },
  done:       { label: "Complete",        color: "var(--green)",   dot: "#22c55e" },
  error:      { label: "Error",           color: "var(--red)",     dot: "#ef4444" },
};

export default function StatusBadge({ status }) {
  const { label, color, dot, pulse } = CONFIG[status] || CONFIG.idle;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 7,
      fontSize: 12, color, padding: "5px 12px",
      border: `1px solid ${color}33`,
      borderRadius: 99,
      background: `${color}0d`,
    }}>
      <span style={{
        width: 7, height: 7, borderRadius: "50%",
        background: dot,
        animation: pulse ? "statusPulse 1.2s ease-in-out infinite" : "none",
        display: "inline-block",
      }} />
      {label}
      <style>{`@keyframes statusPulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
