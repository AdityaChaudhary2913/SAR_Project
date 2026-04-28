import { useEffect, useState } from "react";
import {
	MapContainer,
	TileLayer,
	ImageOverlay,
	Rectangle,
	useMap,
	useMapEvents,
} from "react-leaflet";
import { getTilesCoverage } from "./api";

const REGION_CENTER = [13.845, 0.95];
const REGION_ZOOM = 9;

function DrawHandler({ onBBoxSelect, drawMode }) {
	const [start, setStart] = useState(null);
	const [current, setCurrent] = useState(null);
	const map = useMap();

	useMapEvents({
		mousedown(e) {
			if (!drawMode) return; // ignore if not in draw mode
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

			if (Math.abs(maxLat - minLat) < 0.001) return;

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
		setDrawMode(false); // auto-exit draw mode after drawing
	}

	return (
		<div style={{ flex: 1, height: "100%", position: "relative" }}>
			{/* Draw Mode Toggle Button — overlaid on map */}
			<button
				onClick={() => setDrawMode((m) => !m)}
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
					color: drawMode ? "#000" : "#94a3b8",
					boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
					transition: "all 0.2s",
				}}>
				{drawMode ? "✏️ Drawing — click & drag" : "✏️ Draw AOI"}
			</button>

			<MapContainer
				center={REGION_CENTER}
				zoom={REGION_ZOOM}
				style={{ width: "100%", height: "100%" }}
				cursor={drawMode ? "crosshair" : "grab"}>
				<TileLayer
					url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
					attribution='© <a href="https://openstreetmap.org">OpenStreetMap</a>'
				/>

				{coverage.map((tile) => (
					<Rectangle
						key={tile.id}
						bounds={[
							[tile.bbox[1], tile.bbox[0]],
							[tile.bbox[3], tile.bbox[2]],
						]}
						pathOptions={{
							color: "#38bdf8",
							weight: 1.5,
							fillOpacity: 0.08,
							fillColor: "#38bdf8",
							dashArray: "6 4",
						}}
					/>
				))}

				{drawnBox && (
					<Rectangle
						bounds={drawnBox}
						pathOptions={{ color: "#f59e0b", weight: 2, fillOpacity: 0.1 }}
					/>
				)}

				{result && (
					<ImageOverlay
						url={`http://localhost:8000${result.mask_url}`}
						bounds={[
							[result.bbox[1], result.bbox[0]],
							[result.bbox[3], result.bbox[2]],
						]}
						opacity={0.8}
					/>
				)}

				<DrawHandler onBBoxSelect={handleSelect} drawMode={drawMode} />
			</MapContainer>
		</div>
	);
}
