function formatMetric(value) {
	if (value === null || value === undefined) return "—";
	return Number(value).toFixed(3);
}

function buildSubtitle(metrics) {
	const threshold = metrics?.model?.best_threshold;
	const testIoU = metrics?.splits?.test?.mean_iou;
	const valIoU = metrics?.splits?.val?.mean_iou;

	if (threshold !== undefined && testIoU) {
		return `UNet · tuned threshold ${threshold.toFixed(2)} · test IoU ${testIoU.toFixed(3)}`;
	}
	if (threshold !== undefined && valIoU) {
		return `UNet · tuned threshold ${threshold.toFixed(2)} · val IoU ${valIoU.toFixed(3)}`;
	}
	return "UNet · Sentinel-1 flood segmentation";
}

export default function InferencePanel({
	bbox,
	onRun,
	result,
	metrics,
	loading,
	error,
}) {
	return (
		<div style={styles.panel}>
			<div style={styles.header}>
				<div style={styles.logo}>📡</div>
				<div>
					<h1 style={styles.title}>SAR Flood Detector</h1>
					<p style={styles.subtitle}>{buildSubtitle(metrics)}</p>
				</div>
			</div>

			<div style={styles.section}>
				<h2 style={styles.sectionTitle}>Workflow</h2>
				<ol style={styles.steps}>
					<li>Coverage rectangles show train, validation, and holdout test tiles.</li>
					<li>Draw an AOI on top of a coverage rectangle.</li>
					<li>Run inference to preview the closest exported prediction tile.</li>
				</ol>
			</div>

			<button
				onClick={onRun}
				disabled={!bbox || loading}
				style={{ ...styles.runBtn, opacity: !bbox || loading ? 0.5 : 1 }}>
				{loading ? "Matching tile..." : "Run Inference"}
			</button>

			{error && <div style={styles.errorCard}>{error}</div>}

			{result && (
				<div style={styles.resultCard}>
					<div style={styles.resultRow}>
						<span style={styles.label}>Event</span>
						<span style={styles.value}>{result.event || "Unknown"}</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>Split</span>
						<span style={styles.value}>{result.split || "unknown"}</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>Tile match</span>
						<span style={styles.value}>{(result.overlap_score * 100).toFixed(0)}% overlap</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>UNet IoU</span>
						<span style={styles.value}>{formatMetric(result.unet_tile_iou)}</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>RF IoU</span>
						<span style={styles.value}>{formatMetric(result.rf_tile_iou)}</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>Threshold</span>
						<span style={styles.value}>{result.threshold ? result.threshold.toFixed(2) : "—"}</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>UNet flood</span>
						<span style={styles.value}>
							{result.unet_flood_pct !== null && result.unet_flood_pct !== undefined
								? `${Number(result.unet_flood_pct).toFixed(1)}%`
								: "—"}
						</span>
					</div>
					<div style={styles.resultRow}>
						<span style={styles.label}>RF flood</span>
						<span style={styles.value}>
							{result.rf_flood_pct !== null && result.rf_flood_pct !== undefined
								? `${Number(result.rf_flood_pct).toFixed(1)}%`
								: "—"}
						</span>
					</div>

					<div style={styles.legend}>
						<div style={styles.legendItem}>
							<div
								style={{
									...styles.legendDot,
									background: "rgba(30, 120, 255, 0.85)",
								}}
							/>
							Predicted flood
						</div>
						<div style={styles.legendItem}>
							<div
								style={{
									...styles.legendDot,
									background: "#10b981",
								}}
							/>
							Holdout test tile
						</div>
						<div style={styles.legendItem}>Modal opens with raw tile, UNet, and RF overlays.</div>
					</div>
				</div>
			)}

			<div style={styles.metricsCard}>
				<div style={styles.sectionTitle}>Saved Metrics</div>
				<div style={styles.resultRow}>
					<span style={styles.label}>Val IoU</span>
					<span style={styles.value}>{formatMetric(metrics?.splits?.val?.mean_iou)}</span>
				</div>
				<div style={styles.resultRow}>
					<span style={styles.label}>Test IoU</span>
					<span style={styles.value}>{formatMetric(metrics?.splits?.test?.mean_iou)}</span>
				</div>
				<div style={styles.resultRow}>
					<span style={styles.label}>Val F1</span>
					<span style={styles.value}>{formatMetric(metrics?.splits?.val?.mean_f1)}</span>
				</div>
				<div style={styles.resultRow}>
					<span style={styles.label}>Test F1</span>
					<span style={styles.value}>{formatMetric(metrics?.splits?.test?.mean_f1)}</span>
				</div>
			</div>

			<div style={styles.note}>
				Metrics and threshold come from <code>checkspots/metrics.json</code>. Demo overlays
				are exported from the holdout test event after evaluation.
			</div>
		</div>
	);
}

const styles = {
	panel: {
		width: 320,
		minWidth: 320,
		height: "100%",
		background: "#111827",
		borderRight: "1px solid #1f2937",
		padding: "20px 16px",
		display: "flex",
		flexDirection: "column",
		gap: 16,
		overflowY: "auto",
	},
	header: { display: "flex", alignItems: "center", gap: 12 },
	logo: { fontSize: 30 },
	title: { fontSize: 16, fontWeight: 700, color: "#f8fafc", margin: 0 },
	subtitle: { fontSize: 12, color: "#94a3b8", marginTop: 4, lineHeight: 1.5 },
	section: { background: "#0f172a", borderRadius: 8, padding: "12px 14px" },
	sectionTitle: {
		fontSize: 12,
		fontWeight: 600,
		color: "#94a3b8",
		textTransform: "uppercase",
		letterSpacing: "0.04em",
		marginBottom: 10,
	},
	steps: { fontSize: 13, color: "#cbd5e1", paddingLeft: 18, lineHeight: 1.8, margin: 0 },
	runBtn: {
		background: "#2563eb",
		color: "#fff",
		border: "none",
		borderRadius: 8,
		padding: "12px 0",
		fontSize: 14,
		fontWeight: 600,
		cursor: "pointer",
	},
	errorCard: {
		background: "#450a0a",
		border: "1px solid #7f1d1d",
		borderRadius: 8,
		padding: "12px 14px",
		fontSize: 13,
		color: "#fecaca",
	},
	resultCard: {
		background: "#0f172a",
		border: "1px solid #1d4ed8",
		borderRadius: 8,
		padding: "14px",
		display: "flex",
		flexDirection: "column",
		gap: 10,
	},
	metricsCard: {
		background: "#0f172a",
		border: "1px solid #1f2937",
		borderRadius: 8,
		padding: "14px",
		display: "flex",
		flexDirection: "column",
		gap: 10,
	},
	resultRow: {
		display: "flex",
		justifyContent: "space-between",
		alignItems: "baseline",
		gap: 12,
		fontSize: 13,
	},
	label: { color: "#94a3b8" },
	value: {
		color: "#e2e8f0",
		maxWidth: 150,
		overflow: "hidden",
		textOverflow: "ellipsis",
		whiteSpace: "nowrap",
		textAlign: "right",
	},
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
		color: "#cbd5e1",
	},
	legendDot: { width: 14, height: 14, borderRadius: 3, flexShrink: 0 },
	note: { marginTop: "auto", fontSize: 11, color: "#64748b", lineHeight: 1.6 },
};
