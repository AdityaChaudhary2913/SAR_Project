import { useEffect, useState } from "react";
import MapView from "./MapView";
import InferencePanel from "./InferencePanel";
import ComparisonModal from "./ComparisonModal";
import { getMetrics, runInference } from "./api";

export default function App() {
	const [selection, setSelection] = useState(null);
	const [result, setResult] = useState(null);
	const [metrics, setMetrics] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	const [modalOpen, setModalOpen] = useState(false);

	useEffect(() => {
		getMetrics().then(setMetrics).catch(() => null);
	}, []);

	async function handleRun() {
		if (!selection) return;
		setLoading(true);
		setError(null);
		setResult(null);
		try {
			const data = await runInference(selection.bbox);
			setResult(data);
			setModalOpen(true);
		} catch (e) {
			setError(e.message);
		} finally {
			setLoading(false);
		}
	}

	return (
		<>
			<InferencePanel
				bbox={selection?.bbox}
				onRun={handleRun}
				result={result}
				metrics={metrics}
				loading={loading}
				error={error}
			/>
			<MapView onBBoxSelect={setSelection} result={result} />
			<ComparisonModal result={result} open={modalOpen} onClose={() => setModalOpen(false)} />
		</>
	);
}
