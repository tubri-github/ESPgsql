# -*- coding: utf-8 -*-
"""
One-time backfill: tag each georeferenced ES record with the US HUC4
(4-digit hydrologic subregion) its coordinate falls in.

Adds two fields to docs whose Latitude/Longitude fall inside a US HUC4 polygon:
  - huc4       : the 4-digit code (e.g. "0809")
  - huc4_name  : the drainage name  (e.g. "Lower Mississippi-New Orleans")

Records outside the US (no matching polygon) are left untouched.

This is independent of the ETL re-harvest; when the re-harvest happens it will
re-tag naturally. Safe to re-run (idempotent updates).

Usage:
  python backfill_huc4.py --dry-run --max-docs 50000   # test: no writes, report match rate
  python backfill_huc4.py                              # full run over all georeferenced docs
"""
import argparse
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, streaming_bulk

from config import settings

# Boundaries ship with the repo (gis/) so this runs anywhere after `git pull`.
# Override with the HUC4_GEOJSON env var if you keep the file elsewhere.
HUC4_GEOJSON = os.getenv(
    "HUC4_GEOJSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "gis", "huc4_us.geojson"),
)
BATCH = 20000          # docs pulled + sjoined + bulk-updated per chunk

# US continental+territories bounding box — cheap pre-filter so we never even
# sjoin points that obviously cannot be in a US drainage (most of the 4.4M).
US_BBOX = (-179.9, 17.0, -64.0, 72.0)   # minlon, minlat, maxlon, maxlat


def get_es():
    if settings.ES_USER and settings.ES_PASSWORD:
        return Elasticsearch(settings.ES_URL,
                             basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
                             verify_certs=False, request_timeout=120)
    return Elasticsearch(settings.ES_URL, verify_certs=False, request_timeout=120)


def ensure_mapping(es, idx):
    """Additively declare huc4 / huc4_name as text+keyword (same shape as
    CountryCode) so `huc4_name.keyword` exists for filtering/aggregation."""
    body = {"properties": {
        "huc4":      {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "huc4_name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
    }}
    es.indices.put_mapping(index=idx, body=body)
    print("mapping ensured: huc4, huc4_name (text+keyword)")


def load_huc4():
    gdf = gpd.read_file(HUC4_GEOJSON)
    # normalize field names (arcgis geojson lowercases them)
    cols = {c.lower(): c for c in gdf.columns}
    gdf = gdf.rename(columns={cols.get("huc4", "huc4"): "huc4",
                              cols.get("name", "name"): "huc4_name"})
    gdf = gdf[["huc4", "huc4_name", "geometry"]].copy()
    # repair any self-intersections introduced by simplification
    gdf["geometry"] = gdf["geometry"].buffer(0)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    print(f"loaded {len(gdf)} HUC4 polygons")
    return gdf


def sjoin_batch(rows, huc4_gdf):
    """rows: list of (id, lat, lon). Returns dict id -> (huc4, huc4_name)."""
    minlon, minlat, maxlon, maxlat = US_BBOX
    pts_id, pts_lat, pts_lon = [], [], []
    for _id, lat, lon in rows:
        if lat is None or lon is None:
            continue
        if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
            continue   # outside US bbox — cannot be a US drainage
        pts_id.append(_id); pts_lat.append(lat); pts_lon.append(lon)
    if not pts_id:
        return {}
    pts = gpd.GeoDataFrame(
        {"_id": pts_id},
        geometry=gpd.points_from_xy(pts_lon, pts_lat),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, huc4_gdf, how="inner", predicate="within")
    out = {}
    for _id, huc4, name in zip(joined["_id"], joined["huc4"], joined["huc4_name"]):
        out[_id] = (huc4, name)   # first match wins (polygons don't overlap)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="sjoin + report, no ES writes")
    ap.add_argument("--max-docs", type=int, default=0, help="stop after N docs (0 = all)")
    args = ap.parse_args()

    es = get_es()
    idx = settings.ES_INDEX
    print("index:", idx, "| dry-run:", args.dry_run, "| max-docs:", args.max_docs or "ALL")

    huc4_gdf = load_huc4()
    if not args.dry_run:
        ensure_mapping(es, idx)

    query = {"query": {"bool": {"must": [
        {"exists": {"field": "Latitude"}},
        {"exists": {"field": "Longitude"}},
    ]}}}

    scanner = scan(es, index=idx, query=query,
                   _source=["Latitude", "Longitude"],
                   size=BATCH, scroll="10m", preserve_order=False)

    seen = matched = written = 0
    batch = []

    def flush(batch):
        nonlocal matched, written
        assign = sjoin_batch(batch, huc4_gdf)
        matched += len(assign)
        if args.dry_run or not assign:
            return
        actions = ({"_op_type": "update", "_index": idx, "_id": _id,
                    "doc": {"huc4": h, "huc4_name": n}}
                   for _id, (h, n) in assign.items())
        ok = 0
        for success, _ in streaming_bulk(es, actions, chunk_size=2000,
                                         raise_on_error=False, request_timeout=180):
            ok += 1 if success else 0
        written += ok

    for hit in scanner:
        src = hit.get("_source", {})
        batch.append((hit["_id"], src.get("Latitude"), src.get("Longitude")))
        seen += 1
        if len(batch) >= BATCH:
            flush(batch); batch = []
            print(f"  scanned {seen:>9,} | matched {matched:>9,} | written {written:>9,}")
        if args.max_docs and seen >= args.max_docs:
            break
    if batch:
        flush(batch)

    print("-" * 60)
    print(f"DONE: scanned {seen:,} | matched {matched:,} "
          f"({100*matched/seen:.1f}% of scanned) | written {written:,}")


if __name__ == "__main__":
    main()
