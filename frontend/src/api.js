const BASE = "http://localhost:8000";

export async function runInference(bbox) {
	const res = await fetch(`${BASE}/predict`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ bbox }),
	});
	if (!res.ok) {
		const err = await res.json();
		throw new Error(err.detail || "Inference failed");
	}
	return res.json();
}

export async function getTilesCoverage() {
	const res = await fetch(`${BASE}/tiles_list`);
	if (!res.ok) throw new Error("Could not load coverage tiles");
	return res.json();
}
