import { useEffect, useState } from "react";
import {
	MapContainer,
	TileLayer,
	ImageOverlay,
	Rectangle,
	useMap,
	useMapEvents,
} from "react-leaflet";
import { API_BASE_URL, getTilesCoverage } from "./api";

const REGION_CENTER = [13.845, 0.95];
const REGION_ZOOM = 6;

const SPLIT_STYLES = {
	train: { color: "#38bdf8", fillColor: "#38bdf8" },
	val: { color: "#f59e0b", fillColor: "#f59e0b" },
	test: { color: "#10b981", fillColor: "#10b981" },
	unknown: { color: "#94a3b8", fillColor: "#94a3b8" },
};

function DrawHandler({ onBBoxSelect, drawMode }) {
	const [start, setStart] = useState(null);
	const [current, setCurrent] = useState(null);
	const map = useMap();

	useMapEvents({
		mousedown(e) {
			if (!drawMode) return;
			map.dragging.disable();
			setStart(e.latlng);
			setCurrent(e.latlng);
		},
		mousemove(e) {
			if (!drawMode || !start) return;
			setCurrent(e.latlng);
		},
		mouseup(e) {
			if (!drawMode) return;
			map.dragging.enable();
			if (!start) return;

			const minLat = Math.min(start.lat, e.latlng.lat);
			const maxLat = Math.max(start.lat, e.latlng.lat);
			const minLng = Math.min(start.lng, e.latlng.lng);
			const maxLng = Math.max(start.lng, e.latlng.lng);

			setStart(null);
			setCurrent(null);

			if (Math.abs(maxLat - minLat) < 0.001 || Math.abs(maxLng - minLng) < 0.001) return;

			onBBoxSelect({
				bounds: [
					[minLat, minLng],
					[maxLat, maxLng],
				],
				bbox: [minLng, minLat, maxLng, maxLat],
			});
		},
	});

	if (start && current && drawMode) {
		const minLat = Math.min(start.lat, current.lat);
		const maxLat = Math.max(start.lat, current.lat);
		const minLng = Math.min(start.lng, current.lng);
		const maxLng = Math.max(start.lng, current.lng);
		return (
			<Rectangle
				bounds={[
					[minLat, minLng],
					[maxLat, maxLng],
				]}
				pathOptions={{
					color: "#f59e0b",
					weight: 2,
					fillOpacity: 0.15,
					dashArray: "4",
				}}
			/>
		);
	}
	return null;
}

export default function MapView({ onBBoxSelect, result }) {
	const [coverage, setCoverage] = useState([]);
	const [drawnBox, setDrawnBox] = useState(null);
	const [drawMode, setDrawMode] = useState(false);

	useEffect(() => {
		getTilesCoverage().then(setCoverage).catch(console.error);
	}, []);

	function handleSelect(selection) {
		setDrawnBox(selection.bounds);
		onBBoxSelect(selection);
		setDrawMode(false);
	}

	return (
		<div style={{ flex: 1, height: "100%", position: "relative" }}>
			<button
				onClick={() => setDrawMode((mode) => !mode)}
				style={{
					position: "absolute",
					top: 12,
					right: 12,
					zIndex: 1000,
					padding: "8px 14px",
					borderRadius: 8,
					border: "none",
					cursor: "pointer",
					fontWeight: 600,
					fontSize: 13,
					background: drawMode ? "#f59e0b" : "#1e2330",
					color: drawMode ? "#000" : "#e2e8f0",
					boxShadow: "0 2px 8px rgba(0,0,0,0.35)",
				}}>
				{drawMode ? "Drawing AOI" : "Draw AOI"}
			</button>

			<div
				style={{
					position: "absolute",
					top: 56,
					right: 12,
					zIndex: 1000,
					background: "rgba(15, 23, 42, 0.92)",
					padding: "10px 12px",
					borderRadius: 8,
					color: "#cbd5e1",
					fontSize: 12,
					display: "flex",
					flexDirection: "column",
					gap: 6,
				}}>
				{[
					["train", "Train coverage"],
					["val", "Validation coverage"],
					["test", "Holdout test coverage"],
				].map(([split, label]) => (
					<div key={split} style={{ display: "flex", alignItems: "center", gap: 8 }}>
						<span
							style={{
								width: 12,
								height: 12,
								borderRadius: 2,
								background: SPLIT_STYLES[split].fillColor,
								opacity: 0.85,
							}}
						/>
						{label}
					</div>
				))}
			</div>

			<MapContainer
				center={REGION_CENTER}
				zoom={REGION_ZOOM}
				style={{ width: "100%", height: "100%" }}>
				<TileLayer
					url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
					attribution='© <a href="https://openstreetmap.org">OpenStreetMap</a>'
				/>

				{coverage.map((tile) => {
					const style = SPLIT_STYLES[tile.split] ?? SPLIT_STYLES.unknown;
					return (
						<Rectangle
							key={tile.id}
							bounds={[
								[tile.bbox[1], tile.bbox[0]],
								[tile.bbox[3], tile.bbox[2]],
							]}
							pathOptions={{
								color: style.color,
								weight: 1.8,
								fillOpacity: 0.1,
								fillColor: style.fillColor,
								dashArray: tile.split === "test" ? null : "6 4",
							}}
						/>
					);
				})}

				{drawnBox && (
					<Rectangle
						bounds={drawnBox}
						pathOptions={{ color: "#f59e0b", weight: 2, fillOpacity: 0.1 }}
					/>
				)}

				{result && (
					<ImageOverlay
						url={`${API_BASE_URL}${result.mask_url}`}
						bounds={[
							[result.bbox[1], result.bbox[0]],
							[result.bbox[3], result.bbox[2]],
						]}
						opacity={0.85}
					/>
				)}

				<DrawHandler onBBoxSelect={handleSelect} drawMode={drawMode} />
			</MapContainer>
		</div>
	);
}
