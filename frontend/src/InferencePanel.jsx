export default function InferencePanel({
	bbox,
	onRun,
	result,
	loading,
	error,
}) {
	return (
		<div style={styles.panel}>
			<div style={styles.header}>
				<div style={styles.logo}>📡</div>
				<div>
					<h1 style={styles.title}>SAR Flood Detector</h1>
					<p style={styles.subtitle}>UNet · Sentinel-1 · Val IoU 0.807</p>
				</div>
			</div>

			<div style={styles.section}>
				<h2 style={styles.sectionTitle}>How to use</h2>
				<ol style={styles.steps}>
					<li>Dashed boxes show available coverage areas</li>
					<li>Click and drag on the map to draw an AOI</li>
					<li>Click Run Inference to see flood detection</li>
				</ol>
			</div>

			<button
				onClick={onRun}
				disabled={!bbox || loading}
				style={{ ...styles.runBtn, opacity: !bbox || loading ? 0.5 : 1 }}>
				{loading ? "⏳ Matching tile..." : "▶ Run Inference"}
			</button>

			{error && <div style={styles.errorCard}>⚠ {error}</div>}

			{result && (
				<div style={styles.resultCard}>
					<div style={styles.resultRow}>
						<span style={styles.label}>Event</span>
						<span
							style={{
								maxWidth: 160,
								overflow: "hidden",
								textOverflow: "ellipsis",
								whiteSpace: "nowrap",
								fontSize: 11,
							}}>
							{result.event || "Unknown"}
						</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>Tile match</span>
						<span>{(result.overlap_score * 100).toFixed(0)}% overlap</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>Tile IoU</span>
						<span>{result.tile_iou ?? "—"}</span>
					</div>

					<div style={styles.legend}>
						<div style={styles.legendItem}>
							<div
								style={{
									...styles.legendDot,
									background: "rgba(30,120,255,0.7)",
								}}
							/>
							Flood detected
						</div>
						<div style={styles.legendItem}>
							<div
								style={{
									...styles.legendDot,
									background: "transparent",
									border: "1px solid #475569",
								}}
							/>
							No flood
						</div>
					</div>
				</div>
			)}

			<div style={styles.note}>
				Pre-computed masks · Live pipeline would be: AOI → Sentinel Hub → UNet →
				mask
			</div>
		</div>
	);
}

const styles = {
	panel: {
		width: 300,
		minWidth: 300,
		height: "100%",
		background: "#1e2330",
		borderRight: "1px solid #2d3748",
		padding: "20px 16px",
		display: "flex",
		flexDirection: "column",
		gap: 20,
		overflowY: "auto",
	},
	header: { display: "flex", alignItems: "center", gap: 12 },
	logo: { fontSize: 32 },
	title: { fontSize: 16, fontWeight: 700, color: "#f1f5f9" },
	subtitle: { fontSize: 12, color: "#64748b", marginTop: 2 },
	section: { background: "#0f1117", borderRadius: 8, padding: "12px 14px" },
	sectionTitle: {
		fontSize: 12,
		fontWeight: 600,
		color: "#94a3b8",
		textTransform: "uppercase",
		letterSpacing: "0.05em",
		marginBottom: 10,
	},
	steps: { fontSize: 13, color: "#94a3b8", paddingLeft: 16, lineHeight: 1.8 },
	runBtn: {
		background: "#3b82f6",
		color: "#fff",
		border: "none",
		borderRadius: 8,
		padding: "12px 0",
		fontSize: 14,
		fontWeight: 600,
		cursor: "pointer",
		transition: "all 0.2s",
	},
	errorCard: {
		background: "#450a0a",
		border: "1px solid #7f1d1d",
		borderRadius: 8,
		padding: "12px 14px",
		fontSize: 13,
		color: "#fca5a5",
	},
	resultCard: {
		background: "#0f1117",
		border: "1px solid #1e3a5f",
		borderRadius: 8,
		padding: "14px",
		display: "flex",
		flexDirection: "column",
		gap: 10,
	},
	resultRow: {
		display: "flex",
		justifyContent: "space-between",
		fontSize: 13,
		color: "#94a3b8",
	},
	label: { color: "#64748b" },
	legend: {
		borderTop: "1px solid #1e293b",
		paddingTop: 10,
		display: "flex",
		flexDirection: "column",
		gap: 6,
	},
	legendItem: {
		display: "flex",
		alignItems: "center",
		gap: 8,
		fontSize: 12,
		color: "#94a3b8",
	},
	legendDot: { width: 16, height: 16, borderRadius: 3, flexShrink: 0 },
	note: { marginTop: "auto", fontSize: 11, color: "#334155", lineHeight: 1.6 },
};
