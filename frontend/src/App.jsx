import { useState } from "react";
import MapView from "./MapView";
import InferencePanel from "./InferencePanel";
import { runInference } from "./api";

export default function App() {
	const [selection, setSelection] = useState(null); // { bbox, bounds }
	const [result, setResult] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);

	async function handleRun() {
		if (!selection) return;
		setLoading(true);
		setError(null);
		setResult(null);
		try {
			const data = await runInference(selection.bbox);
			setResult(data);
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
				loading={loading}
				error={error}
			/>
			<MapView onBBoxSelect={setSelection} result={result} />
		</>
	);
}
