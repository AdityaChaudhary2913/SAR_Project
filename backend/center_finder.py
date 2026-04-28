import json

with open("./tile_registry.json") as f:
    reg = json.load(f)

bboxes = [v["bbox"] for v in reg.values() if v.get("bbox")]
center_lat = sum((b[1] + b[3]) / 2 for b in bboxes) / len(bboxes)
center_lon = sum((b[0] + b[2]) / 2 for b in bboxes) / len(bboxes)

# Also get bounds to set a good zoom
all_lats = [b[1] for b in bboxes] + [b[3] for b in bboxes]
all_lons = [b[0] for b in bboxes] + [b[2] for b in bboxes]
print(f"REGION_CENTER = [{center_lat:.4f}, {center_lon:.4f}]")
print(f"Lat range: {min(all_lats):.2f} → {max(all_lats):.2f}")
print(f"Lon range: {min(all_lons):.2f} → {max(all_lons):.2f}")
