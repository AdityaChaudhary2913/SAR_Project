import { API_BASE_URL } from "./api";

function formatMetric(value) {
	if (value === null || value === undefined) return "—";
	return Number(value).toFixed(3);
}

function TilePanel({ title, baseUrl, overlayUrl, metricRows }) {
	return (
		<div style={styles.tilePanel}>
			<div style={styles.tileHeader}>{title}</div>
			<div style={styles.imageFrame}>
				{baseUrl ? (
					<img src={`${API_BASE_URL}${baseUrl}`} alt={title} style={styles.baseImage} />
				) : (
					<div style={styles.emptyState}>Re-export demo assets to include the base tile preview.</div>
				)}
				{baseUrl && overlayUrl && (
					<img
						src={`${API_BASE_URL}${overlayUrl}`}
						alt={`${title} overlay`}
						style={styles.overlayImage}
					/>
				)}
			</div>
			<div style={styles.metricList}>
				{metricRows.map(([label, value]) => (
					<div key={label} style={styles.metricRow}>
						<span style={styles.metricLabel}>{label}</span>
						<span style={styles.metricValue}>{value}</span>
					</div>
				))}
			</div>
		</div>
	);
}

export default function ComparisonModal({ result, open, onClose }) {
	if (!open || !result) return null;

	return (
		<div style={styles.backdrop} onClick={onClose}>
			<div style={styles.modal} onClick={(e) => e.stopPropagation()}>
				<div style={styles.topRow}>
					<div>
						<h2 style={styles.title}>Matched Tile Comparison</h2>
						<p style={styles.subtitle}>
							{result.event} · {result.split} · {(result.overlap_score * 100).toFixed(0)}% AOI overlap
						</p>
					</div>
					<button onClick={onClose} style={styles.closeBtn}>
						Close
					</button>
				</div>

				<div style={styles.grid}>
					<TilePanel
						title="Raw Tile"
						baseUrl={result.base_tile_url}
						overlayUrl={null}
						metricRows={[
							["Event", result.event || "—"],
							["Source", result.source_event_id || result.source || "—"],
							["Threshold", result.threshold !== undefined ? result.threshold.toFixed(2) : "—"],
						]}
					/>
					<TilePanel
						title="UNet Overlay"
						baseUrl={result.base_tile_url}
						overlayUrl={result.unet_overlay_url}
						metricRows={[
							["IoU", formatMetric(result.unet_tile_iou)],
							["F1", formatMetric(result.unet_tile_f1)],
							["Precision", formatMetric(result.unet_precision)],
							["Recall", formatMetric(result.unet_recall)],
							["Flood %", result.unet_flood_pct !== undefined && result.unet_flood_pct !== null ? `${Number(result.unet_flood_pct).toFixed(1)}%` : "—"],
						]}
					/>
					<TilePanel
						title="RF Overlay"
						baseUrl={result.base_tile_url}
						overlayUrl={result.rf_overlay_url}
						metricRows={[
							["IoU", formatMetric(result.rf_tile_iou)],
							["F1", formatMetric(result.rf_tile_f1)],
							["Precision", formatMetric(result.rf_precision)],
							["Recall", formatMetric(result.rf_recall)],
							["Flood %", result.rf_flood_pct !== undefined && result.rf_flood_pct !== null ? `${Number(result.rf_flood_pct).toFixed(1)}%` : "—"],
						]}
					/>
				</div>
			</div>
		</div>
	);
}

const styles = {
	backdrop: {
		position: "fixed",
		inset: 0,
		background: "rgba(2, 6, 23, 0.72)",
		zIndex: 3000,
		display: "flex",
		alignItems: "center",
		justifyContent: "center",
		padding: 20,
	},
	modal: {
		width: "min(1200px, 100%)",
		maxHeight: "90vh",
		overflowY: "auto",
		background: "#0f172a",
		border: "1px solid #1e293b",
		borderRadius: 8,
		padding: 18,
		boxShadow: "0 24px 80px rgba(0,0,0,0.45)",
	},
	topRow: {
		display: "flex",
		justifyContent: "space-between",
		alignItems: "flex-start",
		gap: 16,
		marginBottom: 16,
	},
	title: { margin: 0, color: "#f8fafc", fontSize: 18 },
	subtitle: { margin: "4px 0 0", color: "#94a3b8", fontSize: 13 },
	closeBtn: {
		border: "1px solid #334155",
		background: "#111827",
		color: "#e2e8f0",
		borderRadius: 8,
		padding: "8px 12px",
		fontSize: 13,
		fontWeight: 600,
		cursor: "pointer",
	},
	grid: {
		display: "grid",
		gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
		gap: 16,
	},
	tilePanel: {
		background: "#111827",
		border: "1px solid #1e293b",
		borderRadius: 8,
		padding: 12,
		display: "flex",
		flexDirection: "column",
		gap: 12,
	},
	tileHeader: { color: "#e2e8f0", fontSize: 14, fontWeight: 600 },
	imageFrame: {
		position: "relative",
		aspectRatio: "1 / 1",
		borderRadius: 6,
		overflow: "hidden",
		background: "#020617",
		border: "1px solid #1f2937",
	},
	baseImage: { width: "100%", height: "100%", objectFit: "cover", display: "block" },
	overlayImage: {
		position: "absolute",
		inset: 0,
		width: "100%",
		height: "100%",
		objectFit: "cover",
	},
	emptyState: {
		width: "100%",
		height: "100%",
		display: "flex",
		alignItems: "center",
		justifyContent: "center",
		padding: 20,
		textAlign: "center",
		fontSize: 13,
		color: "#94a3b8",
		lineHeight: 1.6,
	},
	metricList: { display: "flex", flexDirection: "column", gap: 8 },
	metricRow: { display: "flex", justifyContent: "space-between", gap: 12, fontSize: 13 },
	metricLabel: { color: "#94a3b8" },
	metricValue: { color: "#e2e8f0", textAlign: "right" },
};
