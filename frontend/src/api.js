export const API_BASE_URL =
	import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function parseResponse(res, fallbackMessage) {
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw new Error(err.detail || err.message || fallbackMessage);
	}
	return res.json();
}

export async function runInference(bbox) {
	const res = await fetch(`${API_BASE_URL}/predict`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ bbox }),
	});
	return parseResponse(res, "Inference failed");
}

export async function getTilesCoverage() {
	const res = await fetch(`${API_BASE_URL}/tiles_list`);
	return parseResponse(res, "Could not load coverage tiles");
}

export async function getMetrics() {
	const res = await fetch(`${API_BASE_URL}/metrics`);
	return parseResponse(res, "Could not load metrics");
}
