from datetime import datetime
from collections import defaultdict
import math
import time
import hashlib
import json

from fastapi import FastAPI, Query, HTTPException, Depends


# ============== Wildcard Search Support ==============

def has_wildcard(term: str) -> bool:
    """Check if search term contains wildcard characters (* or ?)"""
    return '*' in term or '?' in term


def build_wildcard_query(term: str, fields: list) -> dict:
    """
    Build wildcard query
    Supports * (match any characters) and ? (match single character)
    """
    # Convert to lowercase for case-insensitive search
    term_lower = term.lower()

    # Build should query for multiple fields
    should_clauses = []
    for field in fields:
        should_clauses.append({
            "wildcard": {
                field: {
                    "value": term_lower,
                    "case_insensitive": True
                }
            }
        })

    return {
        "bool": {
            "should": should_clauses,
            "minimum_should_match": 1
        }
    }


# Target fields for wildcard search (species name related)
WILDCARD_SEARCH_FIELDS = [
    "ScientificName", "ValidName", "Genus", "Family"
]


# ============== Document Field Normalization ==============

# Standard field list for frontend table (ensures consistent document structure)
STANDARD_DOCUMENT_FIELDS = [
    "ScientificName", "ValidName", "Genus", "Family",
    "InstitutionCode", "CollectionCode", "CatalogNumber",
    "Country", "StateProvince", "County", "Locality",
    "Latitude", "Longitude", "YearCollected", "MonthCollected", "DayCollected",
    "BasisOfRecord", "RecordedBy", "Remarks"
]


def normalize_document(doc: dict, fields: list = None) -> dict:
    """
    Normalize document fields, ensuring all specified fields exist (fill missing with None).
    This allows frontend to maintain consistent column structure.
    """
    if fields is None:
        fields = STANDARD_DOCUMENT_FIELDS

    normalized = {}
    for field in fields:
        normalized[field] = doc.get(field, None)

    # Preserve other fields in the document (e.g., id, score, etc.)
    for key, value in doc.items():
        if key not in normalized:
            normalized[key] = value

    return normalized


# ============== Geo-spatial Utility Functions ==============

def point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """
    Ray casting algorithm to determine if point is inside polygon.
    polygon: [[lon, lat], [lon, lat], ...] polygon vertices (must be closed, first == last)
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]  # lon, lat
        xj, yj = polygon[j][0], polygon[j][1]

        # Check if point is within edge's y range and calculate intersection
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def point_in_circle(lat: float, lon: float, center_lat: float, center_lon: float, radius_meters: float) -> bool:
    """
    Determine if point is within circular area.
    Uses Haversine formula to calculate distance.
    """
    R = 6371000  # Earth radius in meters

    lat1_rad = math.radians(center_lat)
    lat2_rad = math.radians(lat)
    delta_lat = math.radians(lat - center_lat)
    delta_lon = math.radians(lon - center_lon)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance <= radius_meters


def calculate_bounding_box(geo_filter) -> dict:
    """
    Calculate bounding box from geo_filter.
    Returns {"min_lat": ..., "max_lat": ..., "min_lon": ..., "max_lon": ...}
    """
    if geo_filter.type == "circle":
        # Circle: calculate approximate bounding box from center and radius
        center_lon, center_lat = geo_filter.coordinates[0]
        radius_meters = geo_filter.radius or 10000  # Default 10km

        # 1 degree latitude ~= 111km, 1 degree longitude ~= 111km * cos(lat)
        lat_offset = radius_meters / 111000
        lon_offset = radius_meters / (111000 * math.cos(math.radians(center_lat)))

        return {
            "min_lat": center_lat - lat_offset,
            "max_lat": center_lat + lat_offset,
            "min_lon": center_lon - lon_offset,
            "max_lon": center_lon + lon_offset
        }
    else:
        # Polygon or rectangle: get min/max from all vertices
        lons = [coord[0] for coord in geo_filter.coordinates]
        lats = [coord[1] for coord in geo_filter.coordinates]

        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        }


def filter_by_geo(hits: list, geo_filter) -> list:
    """
    Perform precise geo filtering on search results.
    hits: list of _source from ES response
    """
    if not geo_filter:
        return hits

    filtered = []
    for hit in hits:
        lat = hit.get("Latitude")
        lon = hit.get("Longitude")

        # Skip records without coordinates
        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            continue

        # Check based on geo filter type
        if geo_filter.type == "circle":
            center_lon, center_lat = geo_filter.coordinates[0]
            radius = geo_filter.radius or 10000
            if point_in_circle(lat, lon, center_lat, center_lon, radius):
                filtered.append(hit)
        else:
            # polygon or rectangle
            polygon = geo_filter.coordinates
            # Ensure polygon is closed
            if polygon[0] != polygon[-1]:
                polygon = polygon + [polygon[0]]
            if point_in_polygon(lat, lon, polygon):
                filtered.append(hit)

    return filtered

# ============== End of Geo-spatial Utility Functions ==============
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, func, text, desc, asc
from typing import List, Optional, Dict

# Normalization mappings for common field variations
COUNTRY_NORMALIZATION = {
    # USA variations
    "u.s.a": "USA", "u.s.a.": "USA", "united states": "USA",
    "united states of america": "USA", "us": "USA", "u.s.": "USA",
    "america": "USA", "estados unidos": "USA", "usa": "USA",
    # Canada variations
    "can": "Canada", "can.": "Canada", "ca": "Canada", "canada": "Canada",
    # Mexico variations
    "mex": "Mexico", "mex.": "Mexico", "mexique": "Mexico",
    "mx": "Mexico", "estados unidos mexicanos": "Mexico", "mexico": "Mexico",
    # UK variations
    "u.k.": "UK", "u.k": "UK", "united kingdom": "UK",
    "great britain": "UK", "britain": "UK", "england": "UK",
    # Other common variations
    "p.r. china": "China", "p.r.china": "China", "peoples republic of china": "China",
    "russian federation": "Russia", "brasil": "Brazil",
    "republic of south africa": "South Africa", "rsa": "South Africa",
    "federal republic of germany": "Germany", "west germany": "Germany",
    "republic of korea": "South Korea", "korea, republic of": "South Korea",
    "democratic peoples republic of korea": "North Korea",
}

STATE_NORMALIZATION = {
    # US state abbreviations to full names (or vice versa - pick one standard)
    "calif.": "California", "calif": "California", "ca": "California",
    "fla.": "Florida", "fla": "Florida", "fl": "Florida",
    "tex.": "Texas", "tex": "Texas", "tx": "Texas",
    "n.y.": "New York", "ny": "New York",
    "la": "Louisiana", "la.": "Louisiana",
    "ill.": "Illinois", "ill": "Illinois", "il": "Illinois",
    "mich.": "Michigan", "mich": "Michigan", "mi": "Michigan",
    "penn.": "Pennsylvania", "penn": "Pennsylvania", "pa": "Pennsylvania",
    "mass.": "Massachusetts", "mass": "Massachusetts", "ma": "Massachusetts",
}

def normalize_value(value: str, field_name: str) -> str:
    """Normalize field values to canonical form"""
    if not value or not isinstance(value, str):
        return value

    value_lower = value.lower().strip()

    if field_name == "Country":
        return COUNTRY_NORMALIZATION.get(value_lower, value)
    elif field_name == "StateProvince":
        return STATE_NORMALIZATION.get(value_lower, value)

    return value

def merge_normalized_buckets(buckets: list, field_name: str) -> list:
    """Merge buckets that normalize to the same value"""
    merged = defaultdict(int)
    original_keys = {}  # Keep track of the most common original key

    for bucket in buckets:
        key = bucket.get("key", "")
        if not key:
            continue

        normalized = normalize_value(key, field_name)
        merged[normalized] += bucket.get("doc_count", 0)

        # Keep the original key with highest count as display name
        if normalized not in original_keys or bucket.get("doc_count", 0) > original_keys[normalized][1]:
            original_keys[normalized] = (key, bucket.get("doc_count", 0))

    # Build result list
    result = []
    for normalized_key, count in merged.items():
        result.append({
            "key": normalized_key,
            "doc_count": count
        })

    # Sort by count descending
    result.sort(key=lambda x: x["doc_count"], reverse=True)
    return result

# Build reverse mapping: normalized value -> list of original values
def get_all_variations(normalized_value: str, field_name: str) -> list:
    """Get all original variations that map to a normalized value"""
    variations = [normalized_value]  # Include the normalized value itself

    if field_name == "Country":
        mapping = COUNTRY_NORMALIZATION
    elif field_name == "StateProvince":
        mapping = STATE_NORMALIZATION
    else:
        return variations

    # Find all keys that map to this normalized value
    for original, normalized in mapping.items():
        if normalized == normalized_value:
            variations.append(original)
            # Also add common case variations
            variations.append(original.upper())
            variations.append(original.title())

    return list(set(variations))  # Remove duplicates
from config import settings
from elasticsearch import Elasticsearch
from auth_client import (
    init_auth, AuthConfig, get_current_user, get_current_user_optional,
    UserInfo, require_permissions
)
from models.models import (
    HarvestedRecord, FamilyResponse, GenusResponse, SpeciesResponse,
    InstitutionResponse, RecordResponse, TaxonomyStatsResponse,
    InstitutionStatsResponse, PaginatedResponse,
    TaxonomyFilterParams, InstitutionFilterParams, RecordFilterParams
)

# Simple TTL cache for expensive taxon queries (data only changes during sync)
_cache = {}
_CACHE_TTL = 3600  # 1 hour


def cache_get(key):
    """Get value from cache if not expired"""
    entry = _cache.get(key)
    if entry and time.time() - entry["t"] < _CACHE_TTL:
        return entry["v"]
    return None


def cache_set(key, value):
    """Store value in cache"""
    _cache[key] = {"v": value, "t": time.time()}


def cache_key(*args):
    """Build a cache key from arguments"""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_clear():
    """Clear all cache (call after data sync)"""
    _cache.clear()


app = FastAPI(
    title="FishNet 2 API",
    description="This API is a comprehensive backend API used to access the Fishnet2 original database, Fishnet2 database, and Elasticsearch API. Currently, it provides functionality for basic search and aggregation as well as advanced search aggregation; access to the Elasticsearch API for creating, editing, and rebuilding Elasticsearch indices; and basic querying of occurrence data.",
    version="Beta 1.2",
    docs_url="/docs",
    redoc_url="/redoc",)

# Initialize SSO authentication
init_auth(AuthConfig(
    auth_center_url="http://localhost:8010",
    project_code="FN2",
    current_domain="http://localhost:8000"
))

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]


@app.post("/cache/clear", tags=["Admin"])
async def clear_cache():
    """Clear server-side query cache. Call after data sync."""
    cache_clear()
    return {"message": "Cache cleared"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["Content-Disposition"],
)


from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize Elasticsearch client
# Use basic_auth if ES_USER and ES_PASSWORD are configured
if settings.ES_USER and settings.ES_PASSWORD:
    es = Elasticsearch(
        settings.ES_URL,
        basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
        verify_certs=False
    )
else:
    es = Elasticsearch(settings.ES_URL, verify_certs=False)

# Define the response model
class SearchResult(BaseModel):
    total_results: int
    results: List[dict]
    aggregations: dict


class OrCondition(BaseModel):
    operator: str
    value: str

class Condition(BaseModel):
    field: str
    operator: str
    value: str
    orConditions: List[OrCondition] = []

class FilterValue(BaseModel):
    """Value type supporting multi-select and range filtering"""
    values: Optional[List[str]] = None  # Multi-select value list
    range_min: Optional[float] = None   # Range minimum value
    range_max: Optional[float] = None   # Range maximum value

class GeoFilter(BaseModel):
    """Geo filtering conditions"""
    type: str = "polygon"  # polygon, rectangle, circle
    coordinates: List[List[float]]  # [[lon, lat], [lon, lat], ...] polygon vertices
    # For circle type: coordinates = [[center_lon, center_lat]], radius required
    radius: Optional[float] = None  # Circle radius (meters), only for circle type

class SearchPayload(BaseModel):
    term: Optional[str] = None  # Simple search keyword, optional
    conditions: Optional[List[Condition]] = None  # Advanced search conditions, optional
    filter_conditions: Optional[Dict[str, str]] = None  # Legacy: single-select filter
    multi_filters: Optional[Dict[str, List[str]]] = None  # Multi-select filter conditions
    range_filters: Optional[Dict[str, Dict[str, float]]] = None  # Range filters {"YearCollected": {"min": 1900, "max": 2024}}
    geo_filter: Optional[GeoFilter] = None  # Geo filter conditions
    page: int = 0
    page_size: int = 10

def apply_sort(query, sort_by: str, model_class):
    """Apply sorting"""
    if sort_by == "records_desc":
        return query.order_by(desc("record_count"))
    elif sort_by == "records_asc":
        return query.order_by("record_count")
    elif sort_by == "name_asc":
        return query.order_by("scientificname" if hasattr(model_class, 'scientificname') else "family")
    elif sort_by == "name_desc":
        return query.order_by(desc("scientificname" if hasattr(model_class, 'scientificname') else "family"))
    elif sort_by == "species_desc":
        return query.order_by(desc("species_count"))
    elif sort_by == "date_desc":
        return query.order_by(desc("last_updated"))
    else:
        return query.order_by(desc("record_count"))

def apply_record_count_filter(query, record_count: str, count_field: str = "record_count"):
    """Apply record count filter"""
    if record_count == "high":
        return query.having(text(f"{count_field} > 1000"))
    elif record_count == "medium":
        return query.having(text(f"{count_field} BETWEEN 100 AND 1000"))
    elif record_count == "low":
        return query.having(text(f"{count_field} < 100"))
    return query

def build_es_query(conditions: List[Condition]) -> dict:
    """
    Build Elasticsearch query DSL from user conditions.
    """
    must_clauses = []

    for condition in conditions:
        # Build query for main condition
        main_query = build_field_query(condition.field, condition.operator, condition.value)

        # If OR sub-conditions exist, build should query
        if condition.orConditions:
            should_clauses = [
                build_field_query(condition.field, or_cond.operator, or_cond.value)
                for or_cond in condition.orConditions
            ]
            main_query = {
                "bool": {
                    "should": [main_query, *should_clauses],
                    "minimum_should_match": 1,  # At least one condition must match
                }
            }

        # Add main condition to must clause
        must_clauses.append(main_query)

    return {"query": {"bool": {"must": must_clauses}}}


def get_field_types(es, index_name):
    """Fetch field types from Elasticsearch index mapping."""
    try:
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings']['properties']
        field_types = {field: properties[field].get('type', 'unknown') for field in properties}
        return field_types
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch field types: {str(e)}")



def build_field_query(field: str, operator: Optional[str], value: str) -> dict:
    """
    Generate Elasticsearch query based on field and operator.
    - Numeric fields support range queries (>, <, >=, <=, etc.).
    - String fields support exact match, fuzzy match, wildcard, prefix, etc.
    - Enum fields support multi-select.
    """
    # If no operator provided, set default based on field type
    if not operator: # none and ''
        # Default logic: determine operator based on value type
        if isinstance(value, str):
            operator = "contains"  # Default to fuzzy match for strings
        elif isinstance(value, bool):
            operator = "="  # Default to exact match for booleans
        elif isinstance(value, list):
            operator = "in"  # Default to multi-select for lists (enums)
        else:
            raise ValueError(f"Unsupported type for field {field} with value {value}")

    # Generate corresponding query based on operator
    if operator == "=":
        # Exact match (using keyword field)
        return {"term": {f"{field}.keyword": value}}
    elif operator == "!=":
        # Exclude match
        return {"bool": {"must_not": {"term": {f"{field}.keyword": value}}}}
    elif operator in [">", "<", ">=", "<="]:
        # Range query, for numeric types
        range_operators = {
            ">": "gt",
            "<": "lt",
            ">=": "gte",
            "<=": "lte",
        }
        return {"range": {field: {range_operators[operator]: value}}}
    elif operator == "contains":
        # Fuzzy match (tokenized), for string types
        return {"match": {field: value}}
    elif operator == "phrase":
        # Phrase match (exact phrase)
        return {"match_phrase": {field: value}}
    elif operator == "prefix":
        # Prefix match (starts with...)
        return {"prefix": {f"{field}.keyword": {"value": value, "case_insensitive": True}}}
    elif operator == "wildcard":
        # Wildcard match (* and ?)
        # Auto-add wildcards if user didn't provide them
        wildcard_value = value if '*' in value or '?' in value else f"*{value}*"
        return {"wildcard": {f"{field}.keyword": {"value": wildcard_value, "case_insensitive": True}}}
    elif operator == "fuzzy":
        # Fuzzy match (spelling correction)
        return {"fuzzy": {f"{field}.keyword": {"value": value, "fuzziness": "AUTO"}}}
    elif operator == "regexp":
        # Regular expression match
        return {"regexp": {f"{field}.keyword": {"value": value, "case_insensitive": True}}}
    elif operator == "in":
        # Multi-select (for enum or multi-value fields)
        return {"terms": {f"{field}.keyword": value if isinstance(value, list) else [value]}}
    else:
        raise ValueError(f"Unsupported operator: {operator}")


@app.post("/aggregation",tags=["Search"], summary="Simple Aggregation")
async def aggregation(payload: Optional[SearchPayload] = None, term: Optional[str] = None):
    """
    Aggregated Statistics API: Supports Simple Search (term)
    """
    try:
        # Dynamically build query
        es_query = {
            "size": 0,
            "query": {},
            "aggs": {
                "ScientificName": {"terms": {"field": "ScientificName.keyword", "size": 10}},
                "Family": {"terms": {"field": "Family.keyword", "size": 10}},
                "Occurrence": {
                    "filters": {
                        "filters": {
                            "occurrence_match": {
                                "multi_match": {
                                    "query": term or "",
                                    "fields": ["ScientificName", "Family", "InstitutionCode", "CatalogNumber", "CollectionCode"]
                                }
                            }
                        }
                    }
                }
            }
        }

        if term:
            # Check if wildcard search
            if has_wildcard(term):
                es_query["query"] = build_wildcard_query(term, WILDCARD_SEARCH_FIELDS)
            else:
                # Smart search: supports synonym (ValidName), fuzzy match, phonetic search
                es_query["query"] = {
                    "bool": {
                        "should": [
                            # Exact match (highest weight) - includes synonym field
                            {"multi_match": {
                                "query": term,
                                "fields": ["ScientificName^3", "ValidName^3", "Genus^2", "Family^2",
                                          "InstitutionCode", "CollectionCode", "CatalogNumber",
                                          "Country", "StateProvince", "Locality"],
                                "boost": 3
                            }},
                            # Fuzzy match (allows spelling errors) - text fields, excluding CatalogNumber
                            {"multi_match": {
                                "query": term,
                                "fields": ["ScientificName", "ValidName", "Genus", "Family",
                                          "InstitutionCode", "CollectionCode",
                                          "Country", "StateProvince", "Locality"],
                                "fuzziness": "AUTO",
                                "boost": 2
                            }},
                            # Phonetic match (similar pronunciation) - species name fields only
                            {"multi_match": {
                                "query": term,
                                "fields": ["ScientificName.phonetic", "ValidName.phonetic",
                                          "Genus.phonetic", "Family.phonetic"],
                                "boost": 1
                            }}
                        ],
                        "minimum_should_match": 1
                    }
                }
        elif payload and payload.conditions:
            es_query["query"] = build_es_query(payload.conditions)["query"]
        else:
            # If no term or conditions, return empty aggregation
            es_query["query"] = {"match_all": {}}

        # Execute query
        response = es.search(index=settings.ES_INDEX, body=es_query)

        # Parse results
        aggregations = response.get("aggregations", {})
        scientificname_buckets = aggregations.get("ScientificName", {}).get("buckets", [])
        family_buckets = aggregations.get("Family", {}).get("buckets", [])
        occurrence_count = aggregations.get("Occurrence", {}).get("buckets", {}).get("occurrence_match", {}).get("doc_count", 0)

        return {
            "aggregations": {
                "ScientificName": [{"key": bucket["key"], "doc_count": bucket["doc_count"]} for bucket in scientificname_buckets],
                "Family": [{"key": bucket["key"], "doc_count": bucket["doc_count"]} for bucket in family_buckets],
                "Occurrence": occurrence_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adaggregation",tags=["Search"],summary="Advanced Aggregation")
async def aggregation(payload: Optional[SearchPayload] = None):
    """
    Aggregated Statistics API: Supports Advanced Search (conditions)
    """
    try:
        # Build query
        es_query = {
            "size": 0,
            "query": {},
            "aggs": {
                "Taxon": {
                    "aggs": {
                        "ScientificName": {
                            "terms": {"field": "ScientificName.keyword", "size": 10, "missing": ""}
                        },
                        "Family": {
                            "terms": {"field": "Family.keyword", "size": 10, "missing": ""}
                        }
                    },
                    "filter": {"bool": {"must_not": {"term": {"ScientificName.keyword": ""}}}}
                },
                "Occurrence": {
                    "aggs": {
                        "InstitutionCode": {
                            "terms": {"field": "InstitutionCode.keyword", "size": 10, "missing": ""}
                        },
                        "CollectionCode": {
                            "terms": {"field": "CollectionCode.keyword", "size": 10, "missing": ""}
                        },
                    },
                    "filter": {"bool": {"must_not": {"term": {"InstitutionCode.keyword": ""}}}}
                },
                "Location": {
                    "aggs": {
                        "Country": {
                            "terms": {"field": "Country.keyword", "size": 10, "missing": ""}
                        },
                        "StateProvince": {
                            "terms": {"field": "StateProvince.keyword", "size": 10, "missing": ""}
                        },
                        "County": {
                            "terms": {"field": "County.keyword", "size": 10, "missing": ""}
                        }
                    },
                    "filter": {"bool": {"must_not": {"term": {"Country.keyword": ""}}}}
                }
            }
        }

        # Dynamically build query based on simple or advanced search conditions
        if payload.term:
            # Check if wildcard search
            if has_wildcard(payload.term):
                es_query["query"] = build_wildcard_query(payload.term, WILDCARD_SEARCH_FIELDS)
            else:
                # Smart search: supports synonym (ValidName), fuzzy match, phonetic search
                es_query["query"] = {
                    "bool": {
                        "should": [
                            # Exact match (highest weight) - includes synonym field
                            {"multi_match": {
                                "query": payload.term,
                                "fields": ["ScientificName^3", "ValidName^3", "Genus^2", "Family^2",
                                          "InstitutionCode", "CollectionCode", "CatalogNumber",
                                          "Country", "StateProvince", "County", "Locality"],
                                "boost": 3
                            }},
                            # Fuzzy match (allows spelling errors) - text fields, excluding CatalogNumber
                            {"multi_match": {
                                "query": payload.term,
                                "fields": ["ScientificName", "ValidName", "Genus", "Family",
                                          "InstitutionCode", "CollectionCode",
                                          "Country", "StateProvince", "County", "Locality"],
                                "fuzziness": "AUTO",
                                "boost": 2
                            }},
                            # Phonetic match (similar pronunciation) - species name fields only
                            {"multi_match": {
                                "query": payload.term,
                                "fields": ["ScientificName.phonetic", "ValidName.phonetic",
                                          "Genus.phonetic", "Family.phonetic"],
                                "boost": 1
                            }}
                        ],
                        "minimum_should_match": 1
                    }
                }
        elif payload and payload.conditions:
            es_query["query"] = build_es_query(payload.conditions)["query"]
        else:
            es_query["query"] = {"match_all": {}}

        # Execute query
        # response = es.search(index=settings.ES_INDEX, body=es_query)
        response = es.search(index=settings.ES_INDEX, body=es_query)

        # Parse results
        aggs = response.get("aggregations", {})
        taxon = aggs.get("Taxon", {})
        occurrence = aggs.get("Occurrence", {})
        location = aggs.get("Location", {})

        def parse_buckets(bucket_aggregation, field_name=None):
            buckets = [
                {"key": bucket["key"], "doc_count": bucket["doc_count"]}
                for bucket in bucket_aggregation.get("buckets", [])
                if bucket["key"] != ""  # Filter empty values
            ]
            # Apply normalization for fields that need it
            if field_name in ["Country", "StateProvince"]:
                return merge_normalized_buckets(buckets, field_name)
            return buckets

        return {
            "aggregations": {
                "Taxon": {
                    "ScientificName": parse_buckets(taxon.get("ScientificName", {})),
                    "Family": parse_buckets(taxon.get("Family", {})),
                    "TotalCount": taxon.get("doc_count", 0)
                },
                "Occurrence": {
                    "InstitutionCode": parse_buckets(occurrence.get("InstitutionCode", {})),
                    "CollectionCode": parse_buckets(occurrence.get("CollectionCode", {})),
                    "IndividualCount": parse_buckets(occurrence.get("IndividualCount", {})),
                    "TotalCount": occurrence.get("doc_count", 0)
                },
                "Location": {
                    "Country": parse_buckets(location.get("Country", {}), "Country"),
                    "StateProvince": parse_buckets(location.get("StateProvince", {}), "StateProvince"),
                    "County": parse_buckets(location.get("County", {})),
                    "TotalCount": location.get("doc_count", 0)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search",tags=["Search"],summary="Simple Search")
async def search(
    term: str,
    page: int = Query(0, ge=0),        # Page number, default 0
    page_size: int = Query(10, gt=0),  # Page size, default 10
    fuzzy_threshold: int = Query(5)    # Auto-enable fuzzy search when exact results < this
):
    """
    Smart search:
    1. Supports wildcard search (* matches any chars, ? matches single char)
    2. First try exact search (includes synonym ValidName)
    3. If results < fuzzy_threshold, auto-enable fuzzy + phonetic search
    4. Results sorted by relevance, exact matches first
    """

    # Check if wildcard search
    if has_wildcard(term):
        wildcard_query = build_wildcard_query(term, WILDCARD_SEARCH_FIELDS)

        aggregations_config = {
            "scientificname_count": {"terms": {"field": "ScientificName.keyword", "size": 10}},
            "family_count": {"terms": {"field": "Family.keyword", "size": 10}}
        }

        source_fields = ["ScientificName", "ValidName", "Family", "InstitutionCode", "CatalogNumber", "CollectionCode"]

        try:
            query = {
                "query": wildcard_query,
                "size": page_size,
                "from": page * page_size,
                "track_total_hits": True,
                "_source": source_fields,
                "aggregations": aggregations_config
            }

            response = es.search(index=settings.ES_INDEX, body=query)

            hits = response["hits"]["hits"]
            documents = [
                normalize_document({
                    "id": hit["_id"],
                    "score": hit.get("_score", 0),
                    **hit["_source"]
                })
                for hit in hits
            ]

            aggregations = response.get("aggregations", {})
            scientificname_buckets = aggregations.get("scientificname_count", {}).get("buckets", [])
            family_buckets = aggregations.get("family_count", {}).get("buckets", [])
            occurrence_count = response["hits"]["total"]["value"]

            return {
                "search_mode": "wildcard",
                "documents": documents,
                "aggregations": {
                    "ScientificName": [{"key": bucket["key"], "doc_count": bucket["doc_count"]} for bucket in scientificname_buckets],
                    "Family": [{"key": bucket["key"], "doc_count": bucket["doc_count"]} for bucket in family_buckets],
                    "Occurrence": occurrence_count,
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Exact search query body (includes synonyms)
    exact_query_body = {
        "multi_match": {
            "query": term,
            "fields": ["ScientificName", "ValidName", "Genus", "Family",
                      "InstitutionCode", "CollectionCode", "CatalogNumber",
                      "Country", "StateProvince", "Locality"]
        }
    }

    # Fuzzy + phonetic search query body
    fuzzy_query_body = {
        "bool": {
            "should": [
                # Exact match (highest weight)
                {"multi_match": {
                    "query": term,
                    "fields": ["ScientificName^3", "ValidName^3", "Genus^2", "Family^2",
                              "InstitutionCode", "CollectionCode", "CatalogNumber",
                              "Country", "StateProvince", "Locality"],
                    "boost": 3
                }},
                # Fuzzy match (allows spelling errors) - text fields, excluding CatalogNumber
                {"multi_match": {
                    "query": term,
                    "fields": ["ScientificName", "ValidName", "Genus", "Family",
                              "InstitutionCode", "CollectionCode",
                              "Country", "StateProvince", "Locality"],
                    "fuzziness": "AUTO",
                    "boost": 2
                }},
                # Phonetic match (similar pronunciation) - species name fields only
                {"multi_match": {
                    "query": term,
                    "fields": ["ScientificName.phonetic", "ValidName.phonetic",
                              "Genus.phonetic", "Family.phonetic"],
                    "boost": 1
                }}
            ],
            "minimum_should_match": 1
        }
    }

    aggregations_config = {
        "scientificname_count": {
            "terms": {"field": "ScientificName.keyword", "size": 10}
        },
        "family_count": {
            "terms": {"field": "Family.keyword", "size": 10}
        }
    }

    source_fields = ["ScientificName", "ValidName", "Family", "InstitutionCode", "CatalogNumber", "CollectionCode"]

    try:
        # Step 1: Exact search
        exact_query = {
            "query": exact_query_body,
            "size": page_size,
            "from": page * page_size,
            "track_total_hits": True,
            "_source": source_fields,
            "aggregations": aggregations_config
        }

        response = es.search(index=settings.ES_INDEX, body=exact_query)
        total_hits = response["hits"]["total"]["value"]
        search_mode = "exact"

        # Step 2: If exact results too few, auto fallback to fuzzy search
        if total_hits < fuzzy_threshold:
            fuzzy_query = {
                "query": fuzzy_query_body,
                "size": page_size,
                "from": page * page_size,
                "track_total_hits": True,
                "_source": source_fields,
                "aggregations": aggregations_config
            }
            response = es.search(index=settings.ES_INDEX, body=fuzzy_query)
            search_mode = "fuzzy"

        # Parse results, normalize fields for consistency
        hits = response["hits"]["hits"]
        documents = [
            normalize_document({
                "id": hit["_id"],
                "score": hit.get("_score", 0),
                **hit["_source"]
            })
            for hit in hits
        ]

        aggregations = response.get("aggregations", {})
        scientificname_buckets = aggregations.get("scientificname_count", {}).get("buckets", [])
        family_buckets = aggregations.get("family_count", {}).get("buckets", [])
        occurrence_count = response["hits"]["total"]["value"]

        return {
            "search_mode": search_mode,  # Inform frontend which search mode was used
            "documents": documents,
            "aggregations": {
                "ScientificName": [
                    {"key": bucket["key"], "doc_count": bucket["doc_count"]}
                    for bucket in scientificname_buckets
                ],
                "Family": [
                    {"key": bucket["key"], "doc_count": bucket["doc_count"]}
                    for bucket in family_buckets
                ],
                "Occurrence": occurrence_count,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adsearch",tags=["Search"],summary="Advanced Search")
async def adsearch(payload: SearchPayload):
    try:
        filter_conditions = payload.filter_conditions or {}

        field_types = get_field_types(es, settings.ES_INDEX)

        # Separate fields by type
        text_fields = [field for field, ftype in field_types.items() if ftype in ["text", "keyword"]]
        numeric_fields = [field for field, ftype in field_types.items() if
                          ftype in ["integer", "long", "float", "double"]]
        date_fields = [field for field, ftype in field_types.items() if ftype == "date"]


        es_query = {
            "size": payload.page_size,
            "from": payload.page * payload.page_size,
            "query": {},
            "track_total_hits": True
        }

        if payload.term:
            # Determine term type and select fields accordingly
            if isinstance(payload.term, str):
                # Check if wildcard search
                if has_wildcard(payload.term):
                    es_query["query"] = build_wildcard_query(payload.term, WILDCARD_SEARCH_FIELDS)
                else:
                    # Smart search: supports synonym (ValidName), fuzzy match, phonetic search
                    es_query["query"] = {
                        "bool": {
                            "should": [
                                # Exact match (highest weight) - includes synonym field
                                {"multi_match": {
                                    "query": payload.term,
                                    "fields": ["ScientificName^3", "ValidName^3", "Genus^2", "Family^2",
                                              "InstitutionCode", "CollectionCode", "CatalogNumber",
                                              "Country", "StateProvince", "Locality"],
                                    "boost": 3
                                }},
                                # Fuzzy match (allows spelling errors) - text fields, excluding CatalogNumber
                                {"multi_match": {
                                    "query": payload.term,
                                    "fields": ["ScientificName", "ValidName", "Genus", "Family",
                                              "InstitutionCode", "CollectionCode",
                                              "Country", "StateProvince", "Locality"],
                                    "fuzziness": "AUTO",
                                    "boost": 2
                                }},
                                # Phonetic match (similar pronunciation) - species name fields only
                                {"multi_match": {
                                    "query": payload.term,
                                    "fields": ["ScientificName.phonetic", "ValidName.phonetic",
                                              "Genus.phonetic", "Family.phonetic"],
                                    "boost": 1
                                }}
                            ],
                            "minimum_should_match": 1
                        }
                    }
            elif isinstance(payload.term, (int, float)):
                fields = numeric_fields
                es_query["query"] = {
                    "multi_match": {
                        "query": payload.term,
                        "fields": fields
                    }
                }
            elif isinstance(payload.term, (datetime, str)):  # Date input
                fields = date_fields
                es_query["query"] = {
                    "multi_match": {
                        "query": payload.term,
                        "fields": fields
                    }
                }
            else:
                raise HTTPException(status_code=400, detail="Unsupported term type.")
        elif payload.conditions:
            # Advanced search logic
            es_query["query"] = build_es_query(payload.conditions)["query"]
        else:
            raise HTTPException(status_code=400, detail="Either 'term' or 'conditions' must be provided.")

        # Execute query with filters (supports legacy single-select, multi-select, and range filters)
        filter_clauses = []

        # Legacy single-select filter conditions
        if filter_conditions:
            for key, value in filter_conditions.items():
                # For normalized fields, search all variations
                if key in ["Country", "StateProvince"]:
                    variations = get_all_variations(value, key)
                    filter_clauses.append({"terms": {f"{key}.keyword": variations}})
                else:
                    filter_clauses.append({"term": {f"{key}.keyword": value}})

        # Multi-select filter conditions
        multi_filters = payload.multi_filters or {}
        for key, values in multi_filters.items():
            if values and len(values) > 0:
                # For normalized fields, expand all variations
                if key in ["Country", "StateProvince"]:
                    all_variations = []
                    for v in values:
                        all_variations.extend(get_all_variations(v, key))
                    filter_clauses.append({"terms": {f"{key}.keyword": list(set(all_variations))}})
                else:
                    filter_clauses.append({"terms": {f"{key}.keyword": values}})

        # Range filter conditions
        range_filters = payload.range_filters or {}
        for key, range_values in range_filters.items():
            range_query = {}
            if "min" in range_values and range_values["min"] is not None:
                range_query["gte"] = range_values["min"]
            if "max" in range_values and range_values["max"] is not None:
                range_query["lte"] = range_values["max"]
            if range_query:
                filter_clauses.append({"range": {key: range_query}})

        # Geo filter conditions - use bounding box for fast initial filtering
        geo_filter = payload.geo_filter
        if geo_filter and geo_filter.coordinates:
            bbox = calculate_bounding_box(geo_filter)
            # Add bounding box range query (fast filtering)
            filter_clauses.append({
                "bool": {
                    "must": [
                        {"exists": {"field": "Latitude"}},
                        {"exists": {"field": "Longitude"}},
                        {"range": {"Latitude": {"gte": bbox["min_lat"], "lte": bbox["max_lat"]}}},
                        {"range": {"Longitude": {"gte": bbox["min_lon"], "lte": bbox["max_lon"]}}}
                    ]
                }
            })

        # Apply all filter clauses
        if filter_clauses:
            # If original query is already a bool query, merge into filter clause
            if "bool" in es_query["query"]:
                es_query["query"]["bool"]["filter"] = filter_clauses
            else:
                # Convert existing query to bool query and add filter
                es_query["query"] = {
                    "bool": {
                        "must": es_query["query"],
                        "filter": filter_clauses
                    }
                }

        # Sort: push records with empty ScientificName to the bottom
        es_query["sort"] = [
            {
                "_script": {
                    "type": "number",
                    "script": {
                        "source": "doc.containsKey('ScientificName.keyword') && doc['ScientificName.keyword'].size() > 0 && doc['ScientificName.keyword'].value.length() > 0 ? 1 : 0",
                        "lang": "painless"
                    },
                    "order": "desc"
                }
            },
            "_score"
        ]

        response = es.search(index=settings.ES_INDEX, body=es_query)
        hits = response["hits"]["hits"]
        total = response["hits"]["total"]["value"]

        # Get statistics using aggregations
        # Field names: Latitude, Longitude (not DecimalLatitude/DecimalLongitude)
        stats_query = {
            "size": 0,
            "query": es_query["query"],
            "aggs": {
                "georeferenced": {
                    "filter": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "Latitude"}},
                                {"exists": {"field": "Longitude"}}
                            ],
                            "must_not": [
                                {"bool": {
                                    "must": [
                                        {"term": {"Latitude": 0}},
                                        {"term": {"Longitude": 0}}
                                    ]
                                }}
                            ]
                        }
                    }
                },
                "unique_species": {
                    "cardinality": {"field": "ScientificName.keyword"}
                },
                "unique_institutions": {
                    "cardinality": {"field": "InstitutionCode.keyword"}
                },
                "unique_countries": {
                    "cardinality": {"field": "CountryCode.keyword"}
                }
            }
        }

        stats_response = es.search(index=settings.ES_INDEX, body=stats_query)
        aggs = stats_response["aggregations"]

        georef_count = aggs["georeferenced"]["doc_count"]
        georef_percent = round((georef_count / total * 100), 1) if total > 0 else 0

        stats = {
            "total": total,
            "georeferenced": georef_count,
            "georefPercent": georef_percent,
            "speciesCount": aggs["unique_species"]["value"],
            "institutionCount": aggs["unique_institutions"]["value"],
            "countryCount": aggs["unique_countries"]["value"]
        }

        # Get chart data aggregations
        chart_query = {
            "size": 0,
            "query": es_query["query"],
            "aggs": {
                # Timeline: by year
                "timeline": {
                    "terms": {
                        "field": "YearCollected",
                        "size": 100,
                        "order": {"_key": "asc"}
                    }
                },
                # Diversity: by family
                "families": {
                    "terms": {
                        "field": "Family.keyword",
                        "size": 20
                    }
                },
                # Map: get georeferenced records with coordinates
                "map_points": {
                    "filter": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "Latitude"}},
                                {"exists": {"field": "Longitude"}}
                            ]
                        }
                    },
                    "aggs": {
                        "by_country": {
                            "terms": {
                                "field": "Country.keyword",
                                "size": 50
                            },
                            "aggs": {
                                "avg_lat": {"avg": {"field": "Latitude"}},
                                "avg_lng": {"avg": {"field": "Longitude"}}
                            }
                        }
                    }
                }
            }
        }

        chart_response = es.search(index=settings.ES_INDEX, body=chart_query)
        chart_aggs = chart_response["aggregations"]

        # Process timeline data
        timeline_data = [
            {"year": bucket["key"], "count": bucket["doc_count"]}
            for bucket in chart_aggs["timeline"]["buckets"]
            if bucket["key"] and bucket["key"] > 1700  # Filter out invalid years
        ]

        # Process diversity data
        diversity_data = [
            {"key": bucket["key"], "doc_count": bucket["doc_count"]}
            for bucket in chart_aggs["families"]["buckets"]
        ]

        # Process map data (country centroids with counts)
        map_points = [
            {
                "lat": bucket["avg_lat"]["value"],
                "lng": bucket["avg_lng"]["value"],
                "count": bucket["doc_count"],
                "country": bucket["key"],
                "location": bucket["key"]
            }
            for bucket in chart_aggs["map_points"]["by_country"]["buckets"]
            if bucket["avg_lat"]["value"] and bucket["avg_lng"]["value"]
        ]

        # Extract hits data, normalize fields for consistency
        hits_data = [normalize_document({
            "id": hit["_id"],
            "score": hit.get("_score", 0),
            **hit["_source"]
        }) for hit in hits]

        # If geo_filter exists, perform precise polygon/circle filtering
        if geo_filter and geo_filter.coordinates:
            hits_data = filter_by_geo(hits_data, geo_filter)

        return {
            "hits": hits_data,
            "total": total,  # Note: This is the count after bounding box filter, may be slightly less after precise filtering
            "stats": stats,
            "charts": {
                "timeline": timeline_data,
                "diversity": diversity_data,
                "mapPoints": map_points
            },
            "geo_filtered": geo_filter is not None  # Flag whether geo filtering was applied
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CustomAggregationPayload(BaseModel):
    field: str  # Field name to aggregate
    size: int = 10  # Number of buckets to return
    term: Optional[str] = None  # Optional search term
    conditions: Optional[List[Condition]] = None  # Optional advanced search conditions


@app.post("/custom-aggregation", tags=["Search"], summary="Custom Field Aggregation")
async def custom_aggregation(payload: CustomAggregationPayload):
    """
    Custom field aggregation API: Generate statistical charts for user-selected fields.
    Supported fields: ScientificName, Family, Genus, Order, Class, Country, StateProvince,
    County, InstitutionCode, CollectionCode, YearCollected, BasisOfRecord, RecordedBy
    """
    try:
        # Validate field name
        allowed_fields = [
            'ScientificName', 'Family', 'Genus', 'Order', 'Class',
            'Country', 'StateProvince', 'County',
            'InstitutionCode', 'CollectionCode', 'YearCollected',
            'BasisOfRecord', 'RecordedBy'
        ]

        if payload.field not in allowed_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid field. Allowed fields: {', '.join(allowed_fields)}"
            )

        # Build base query
        base_query = {}
        if payload.term:
            base_query = {
                "multi_match": {
                    "query": payload.term,
                    "fields": ["ScientificName", "Family", "InstitutionCode", "Country"]
                }
            }
        elif payload.conditions:
            base_query = build_es_query(payload.conditions)["query"]
        else:
            base_query = {"match_all": {}}

        # Determine field type - numeric fields don't need .keyword
        numeric_fields = ['YearCollected']
        field_name = payload.field if payload.field in numeric_fields else f"{payload.field}.keyword"

        # Build aggregation query
        es_query = {
            "size": 0,
            "query": base_query,
            "aggs": {
                "field_stats": {
                    "terms": {
                        "field": field_name,
                        "size": payload.size,
                        "order": {"_count": "desc"}
                    }
                },
                "total_unique": {
                    "cardinality": {
                        "field": field_name
                    }
                }
            }
        }

        # Execute query
        response = es.search(index=settings.ES_INDEX, body=es_query)

        # Parse results
        aggs = response.get("aggregations", {})
        buckets = aggs.get("field_stats", {}).get("buckets", [])
        total_unique = aggs.get("total_unique", {}).get("value", 0)
        total_docs = response.get("hits", {}).get("total", {}).get("value", 0)

        # Filter empty values
        buckets = [
            {"key": bucket["key"], "doc_count": bucket["doc_count"]}
            for bucket in buckets
            if bucket["key"] and str(bucket["key"]).strip() != ""
        ]

        return {
            "field": payload.field,
            "buckets": buckets,
            "total_unique": total_unique,
            "total_docs": total_docs
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occurrence/{id}",tags=["Occurrence"],summary="Get Occurrence by ID")
async def get_occurrence(
    id: int,  # RESTful path parameter id
    db: Session = Depends(get_db)  # Get database session
):
    # SQL query
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # Execute query
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # Convert to dictionary and return
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/occurrences",tags=["Occurrence"],summary="Get Occurrences")
async def get_occurrence(
    id: int,  # RESTful path parameter id
    db: Session = Depends(get_db)  # Get database session
):
    # SQL query
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # Execute query
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # Convert to dictionary and return
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/occurrences/{term}",tags=["Occurrence"],summary="Get Occurrences by user defined terms")
async def get_occurrence(
    id: int,  # RESTful path parameter id
    db: Session = Depends(get_db)  # Get database session
):
    # SQL query
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # Execute query
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # Convert to dictionary and return
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/ESIndexCon",tags=["ElasticSearch"],summary="Get ES connection")
async def get_occurrence(
    id: int,  # RESTful path parameter id
    db: Session = Depends(get_db)  # Get database session
):
    # SQL query
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # Execute query
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # Convert to dictionary and return
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/ESIndex",tags=["ElasticSearch"],summary="Get ES index ")
async def get_occurrence(
    id: int,  # RESTful path parameter id
    db: Session = Depends(get_db)  # Get database session
):
    # SQL query
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # Execute query
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # Convert to dictionary and return
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/ESIndexRebuild",tags=["ElasticSearch"],summary="Rebuild ES Index")
async def get_occurrence(
    id: int,  # RESTful path parameter id
    db: Session = Depends(get_db)  # Get database session
):
    # SQL query
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # Execute query
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # Convert to dictionary and return
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/ESIndexBuild",tags=["ElasticSearch"],summary="Build ES Index")
async def get_occurrence(
    id: int,  # RESTful path parameter id
    db: Session = Depends(get_db)  # Get database session
):
    # SQL query
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # Execute query
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # Convert to dictionary and return
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/families", response_model=PaginatedResponse)
async def get_families(
        params: TaxonomyFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """Get all families"""
    # Build base query
    base_select = """
        SELECT family, "order", genera_count, species_count, record_count, 
               countries_count, institutions_count, georeferencing_quality, 
               date_quality, last_updated
        FROM dbo.family_stats
    """

    # Build WHERE conditions
    conditions = []
    params_dict = {}

    if params.search:
        conditions.append("(family ILIKE :search OR \"order\" ILIKE :search)")
        params_dict["search"] = f"%{params.search}%"

    if params.order:
        conditions.append("\"order\" = :order_filter")
        params_dict["order_filter"] = params.order

    # Record count filter
    if params.record_count:
        if params.record_count == "high":
            conditions.append("record_count > 1000")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 100 AND 1000")
        elif params.record_count == "low":
            conditions.append("record_count < 100")

    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # Sort mapping
    sort_mapping = {
        "records_desc": "record_count DESC",
        "records_asc": "record_count ASC",
        "name_asc": "family ASC",
        "name_desc": "family DESC",
        "genera_desc": "genera_count DESC",
        "species_desc": "species_count DESC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")

    # Build complete data query
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")

    # Build count query
    count_query = text(f"SELECT COUNT(*) FROM dbo.family_stats{where_clause}")

    # Execute query
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # Get total count
    total = db.execute(count_query, {k: v for k, v in params_dict.items() if k not in ['limit', 'offset']}).scalar()

    # Get data
    result = db.execute(data_query, params_dict).fetchall()

    # Convert to frontend expected response format - using camelCase
    families = []
    for row in result:
        families.append({
            "family": row[0],
            "order": row[1],
            "generaCount": row[2] or 0,
            "speciesCount": row[3] or 0,
            "recordCount": row[4] or 0,
            "countriesCount": row[5] or 0,
            "institutionsCount": row[6] or 0,
            "geoReferencingQuality": float(row[7]) if row[7] else 0.0,
            "dateQuality": float(row[8]) if row[8] else 0.0,
            "lastUpdated": row[9] if row[9] else None
            # "lastUpdated": row[9].isoformat() if row[9] else None
        })

    return PaginatedResponse(
        data=families,
        page=params.page,
        per_page=params.per_page,
        total=total,
        pages=(total + params.per_page - 1) // params.per_page
    )


@app.get("/families/{family_name}")
async def get_family_detail(family_name: str, db: Session = Depends(get_db)):
    """Get family details"""
    query = text("""
        SELECT family, "order", genera_count, species_count, record_count, 
               countries_count, institutions_count, georeferencing_quality, 
               date_quality, last_updated
        FROM dbo.family_stats
        WHERE family = :family_name
    """)

    result = db.execute(query, {"family_name": family_name}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Family not found")

    return {
        "family": result[0],
        "order": result[1],
        "generaCount": result[2],
        "speciesCount": result[3],
        "recordCount": result[4],
        "countriesCount": result[5],
        "institutionsCount": result[6],
        "geoReferencingQuality": result[7],
        "dateQuality": result[8],
        "lastUpdated": result[9],
        "taxonomicStatus": "accepted",
        "nomenclaturalCode": "ICZN"
    }


@app.get("/genera", response_model=PaginatedResponse)
async def get_genera(
        params: TaxonomyFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """Get all genera"""
    base_select = """
        SELECT genus, family, "order", species_count, record_count, 
               countries_count, institutions_count, georeferencing_quality, 
               date_quality, last_updated
        FROM dbo.genus_stats
    """

    conditions = []
    params_dict = {}

    # Add search conditions
    if params.search:
        conditions.append("(genus ILIKE :search OR family ILIKE :search)")
        params_dict["search"] = f"%{params.search}%"

    if params.family:
        conditions.append("family = :family_filter")
        params_dict["family_filter"] = params.family

    # Record count filter
    if params.record_count:
        if params.record_count == "high":
            conditions.append("record_count > 500")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 50 AND 500")
        elif params.record_count == "low":
            conditions.append("record_count < 50")

    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # Sort
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "name_asc": "genus ASC",
        "family_asc": "family ASC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")

    # Build query
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.genus_stats{where_clause}")

    # Execute query
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # Get total count and data
    total = db.execute(count_query, {k: v for k, v in params_dict.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(data_query, params_dict).fetchall()

    genera = []
    for row in result:
        genera.append({
            "genus": row[0],
            "family": row[1],
            "order": row[2],
            "speciesCount": row[3],
            "recordCount": row[4],
            "countriesCount": row[5],
            "institutionsCount": row[6],
            "geoReferencingQuality": row[7],
            "dateQuality": row[8],
            "lastUpdated": row[9]
        })

    return PaginatedResponse(
        data=genera,
        page=params.page,
        per_page=params.per_page,
        total=total,
        pages=(total + params.per_page - 1) // params.per_page
    )


@app.get("/genera/{genus_name}")
async def get_genus_detail(genus_name: str, db: Session = Depends(get_db)):
    """Get genus details"""
    query = text("""
        SELECT genus, family, "order", species_count, record_count, 
               countries_count, institutions_count, georeferencing_quality, 
               date_quality, last_updated
        FROM dbo.genus_stats
        WHERE genus = :genus_name
    """)

    result = db.execute(query, {"genus_name": genus_name}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Genus not found")

    return {
        "genus": result[0],
        "family": result[1],
        "order": result[2],
        "speciesCount": result[3],
        "recordCount": result[4],
        "countriesCount": result[5],
        "institutionsCount": result[6],
        "geoReferencingQuality": result[7],
        "dateQuality": result[8],
        "lastUpdated": result[9],
        "taxonomicStatus": "accepted",
        "nomenclaturalCode": "ICZN"
    }


@app.get("/species", response_model=PaginatedResponse)
async def get_species(
        params: TaxonomyFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """Get all species"""
    base_query = """
        SELECT CONCAT(genus, ' ', specificepithet) as scientific_name, scientificnameauthorship, vernacularname, 
               genus, family, "order", record_count, countries_count, 
               institutions_count, georeferencing_quality, date_quality, 
               last_record, last_updated
        FROM dbo.species_stats
    """

    conditions = []
    params_dict = {}

    # Search conditions
    if params.search:
        conditions.append("""
            (specificepithet ILIKE :search OR vernacularname ILIKE :search 
             OR family ILIKE :search OR genus ILIKE :search)
        """)
        params_dict["search"] = f"%{params.search}%"

    if params.family:
        conditions.append("family = :family_filter")
        params_dict["family_filter"] = params.family

    if params.genus:
        conditions.append("genus = :genus_filter")
        params_dict["genus_filter"] = params.genus

    # Record count filter
    if params.record_count:
        if params.record_count == "high":
            conditions.append("record_count > 1000")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 100 AND 1000")
        elif params.record_count == "low":
            conditions.append("record_count < 100")

    # Data quality filter
    if params.data_quality:
        if params.data_quality == "excellent":
            conditions.append("georeferencing_quality > 95")
        elif params.data_quality == "good":
            conditions.append("georeferencing_quality BETWEEN 80 AND 95")
        elif params.data_quality == "poor":
            conditions.append("georeferencing_quality < 80")

    # Build query
    if conditions:
        query = text(base_query + " WHERE " + " AND ".join(conditions))
    else:
        query = text(base_query)

    # Sort
    sort_mapping = {
        "records_desc": "record_count DESC",
        "name_asc": "scientific_name ASC",
        "recent_desc": "last_record DESC",
        "quality_desc": "georeferencing_quality DESC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")
    query = text(str(query) + f" ORDER BY {order_by}")

    # Total count
    count_base = "SELECT COUNT(*) FROM dbo.species_stats"
    if conditions:
        count_query = text(count_base + " WHERE " + " AND ".join(conditions))
    else:
        count_query = text(count_base)
    total = db.execute(count_query, params_dict).scalar()

    # Pagination
    query = text(str(query) + " LIMIT :limit OFFSET :offset")
    params_dict["limit"] = params.per_page
    params_dict["offset"] = (params.page - 1) * params.per_page

    result = db.execute(query, params_dict).fetchall()

    species_list = []
    for row in result:
        species_list.append({
            "scientificName": row[0],
            "authority": row[1],
            "vernacularName": row[2],
            "genus": row[3],
            "family": row[4],
            "order": row[5],
            "recordCount": row[6],
            "countriesCount": row[7],
            "institutionsCount": row[8],
            "geoReferencingQuality": row[9],
            "dateQuality": row[10],
            "lastRecord": row[11],
            "lastUpdated": row[12]
        })

    return PaginatedResponse(
        data=species_list,
        page=params.page,
        per_page=params.per_page,
        total=total,
        pages=(total + params.per_page - 1) // params.per_page
    )

@app.get("/species/{scientific_name}")
async def get_species_detail(scientific_name: str, db: Session = Depends(get_db)):
    """Get species details - uses species_stats cache table"""

    # Parse scientific name: supports "Brycon henni" or "henni"
    parts = scientific_name.strip().split(' ')
    if len(parts) >= 2:
        genus_name = parts[0]
        specific_epithet = parts[1]
        use_full_name = True
    else:
        specific_epithet = scientific_name
        genus_name = None
        use_full_name = False

    # Query from species_stats cache table
    if use_full_name:
        stats_query = text("""
            SELECT genus, specificepithet, scientificnameauthorship, vernacularname,
                   family, "order", record_count, countries_count, institutions_count,
                   georeferencing_quality, date_quality, last_record, last_updated
            FROM dbo.species_stats
            WHERE genus = :genus_name AND specificepithet = :specific_epithet
        """)
        params = {"genus_name": genus_name, "specific_epithet": specific_epithet}
    else:
        stats_query = text("""
            SELECT genus, specificepithet, scientificnameauthorship, vernacularname,
                   family, "order", record_count, countries_count, institutions_count,
                   georeferencing_quality, date_quality, last_record, last_updated
            FROM dbo.species_stats
            WHERE specificepithet = :specific_epithet
        """)
        params = {"specific_epithet": specific_epithet}

    result = db.execute(stats_query, params).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Species not found")

    scientific_name_full = f"{result[0]} {result[1]}"

    return {
        "scientificName": scientific_name_full,
        "authority": result[2],
        "vernacularName": result[3],
        "genus": result[0],
        "family": result[4],
        "order": result[5],
        "recordCount": result[6] or 0,
        "countriesCount": result[7] or 0,
        "institutionsCount": result[8] or 0,
        "geoReferencingQuality": float(result[9]) if result[9] else 0.0,
        "dateQuality": float(result[10]) if result[10] else 0.0,
        "lastRecord": result[11],
        "lastUpdated": result[12],
        "taxonomicStatus": "accepted",
        "nomenclaturalCode": "ICZN"
    }


@app.get("/institutions", response_model=PaginatedResponse)
async def get_institutions(
        params: InstitutionFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """Get all institutions - fixed version, each institution shown only once"""
    base_select = """
        SELECT institutioncode, institution_name, ownerinstitutioncode, 
               country, region, institution_type, record_count, species_count, 
               families_count, countries_count, georeferencing_quality, 
               date_quality, taxonomic_quality, overall_quality, collection_codes, 
               first_record, latest_record, last_updated
        FROM dbo.institution_stats
    """

    conditions = []
    params_dict = {}

    # Search conditions
    if params.search:
        conditions.append("""
            (institutioncode ILIKE :search OR institution_name ILIKE :search 
             OR country ILIKE :search)
        """)
        params_dict["search"] = f"%{params.search}%"

    if params.region:
        conditions.append("region = :region_filter")
        params_dict["region_filter"] = params.region

    if params.institution_type:
        conditions.append("institution_type = :type_filter")
        params_dict["type_filter"] = params.institution_type

    # Record count filter
    if params.record_count:
        if params.record_count == "major":
            conditions.append("record_count > 10000")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 1000 AND 10000")
        elif params.record_count == "small":
            conditions.append("record_count < 1000")

    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # Sort
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "quality_desc": "overall_quality DESC",
        "name_asc": "institution_name ASC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")

    # Build query
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.institution_stats{where_clause}")

    # Execute query
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # Get total count and data
    total = db.execute(count_query, {k: v for k, v in params_dict.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(data_query, params_dict).fetchall()

    institutions = []
    for row in result:
        institutions.append({
            "institutionCode": row[0],
            "institutionName": row[1],
            "ownerInstitutionCode": row[2],
            "country": row[3],
            "region": row[4],
            "institutionType": row[5],
            "recordCount": row[6] or 0,
            "speciesCount": row[7] or 0,
            "familiesCount": row[8] or 0,
            "countriesCount": row[9] or 0,
            "geoReferencingQuality": float(row[10]) if row[10] else 0.0,
            "dateQuality": float(row[11]) if row[11] else 0.0,
            "taxonomicQuality": float(row[12]) if row[12] else 0.0,
            "overallQuality": float(row[13]) if row[13] else 0.0,
            "collectionCodes": row[14] if row[14] else [],
            "firstRecord": row[15],
            "latestRecord": row[16],
            "lastUpdated": row[17] if row[17] else None
        })

    return PaginatedResponse(
        data=institutions,
        page=params.page,
        per_page=params.per_page,
        total=total,
        pages=(total + params.per_page - 1) // params.per_page
    )


# ============================================================
# Institution V2 API - With Survey Details
# ============================================================

@app.get("/institutions/v2", response_model=PaginatedResponse)
async def get_institutions_v2(
        params: InstitutionFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """Get institutions with survey details (v2 test endpoint)"""
    base_select = """
        SELECT
            s.institutioncode, s.institution_name, s.ownerinstitutioncode,
            s.country as stats_country, s.region, s.institution_type,
            s.record_count, s.species_count, s.families_count, s.countries_count,
            s.georeferencing_quality, s.date_quality, s.taxonomic_quality,
            s.overall_quality, s.first_record, s.latest_record,
            d.official_name, d.alternate_name, d.abbreviation_name,
            d.address, d.city, d.state, d.country as detail_country, d.zip,
            d.phone, d.email, d.website,
            d.latitude, d.longitude,
            d.sns_twitter, d.sns_facebook, d.sns_instagram,
            d.environment, d.specimens_amount, d.data_url,
            d.source
        FROM dbo.institution_stats s
        LEFT JOIN dbo.institution_details d ON s.institutioncode = d.institution_code
    """

    conditions = []
    params_dict = {}

    if params.search:
        conditions.append("""
            (s.institutioncode ILIKE :search
             OR s.institution_name ILIKE :search
             OR s.country ILIKE :search
             OR d.official_name ILIKE :search
             OR d.city ILIKE :search)
        """)
        params_dict["search"] = f"%{params.search}%"

    if params.region:
        conditions.append("s.region = :region_filter")
        params_dict["region_filter"] = params.region

    if params.institution_type:
        conditions.append("s.institution_type = :type_filter")
        params_dict["type_filter"] = params.institution_type

    if params.record_count:
        if params.record_count == "major":
            conditions.append("s.record_count > 10000")
        elif params.record_count == "medium":
            conditions.append("s.record_count BETWEEN 1000 AND 10000")
        elif params.record_count == "small":
            conditions.append("s.record_count < 1000")

    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    sort_mapping = {
        "records_desc": "s.record_count DESC",
        "species_desc": "s.species_count DESC",
        "quality_desc": "s.overall_quality DESC",
        "name_asc": "COALESCE(d.official_name, s.institution_name) ASC"
    }
    order_by = sort_mapping.get(params.sort_by, "s.record_count DESC")

    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"""
        SELECT COUNT(*) FROM dbo.institution_stats s
        LEFT JOIN dbo.institution_details d ON s.institutioncode = d.institution_code
        {where_clause}
    """)

    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    total = db.execute(count_query, {k: v for k, v in params_dict.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(data_query, params_dict).fetchall()

    institutions = []
    for row in result:
        institutions.append({
            # Stats data
            "institutionCode": row[0],
            "institutionName": row[16] or row[1],  # Prefer official_name from details
            "ownerInstitutionCode": row[2],
            "statsCountry": row[3],  # Country from stats (specimen locations)
            "region": row[4],
            "institutionType": row[5],
            "recordCount": row[6] or 0,
            "speciesCount": row[7] or 0,
            "familiesCount": row[8] or 0,
            "countriesCount": row[9] or 0,
            "geoReferencingQuality": float(row[10]) if row[10] else 0.0,
            "dateQuality": float(row[11]) if row[11] else 0.0,
            "taxonomicQuality": float(row[12]) if row[12] else 0.0,
            "overallQuality": float(row[13]) if row[13] else 0.0,
            "firstRecord": row[14],
            "latestRecord": row[15],
            # Details data
            "officialName": row[16],
            "alternateName": row[17],
            "abbreviationName": row[18],
            "address": row[19],
            "city": row[20],
            "state": row[21],
            "country": row[22],  # Institution physical location
            "zip": row[23],
            "phone": row[24],
            "email": row[25],
            "website": row[26],
            "latitude": float(row[27]) if row[27] else None,
            "longitude": float(row[28]) if row[28] else None,
            "twitter": row[29],
            "facebook": row[30],
            "instagram": row[31],
            "environment": row[32],
            "specimensAmount": row[33],
            "dataUrl": row[34],
            "source": row[35]
        })

    return PaginatedResponse(
        data=institutions,
        page=params.page,
        per_page=params.per_page,
        total=total,
        pages=(total + params.per_page - 1) // params.per_page
    )


@app.get("/institutions/v2/{institution_code}")
async def get_institution_detail_v2(institution_code: str, db: Session = Depends(get_db)):
    """Get institution details with survey data and contacts (v2 test endpoint)"""
    query = text("""
        SELECT
            s.institutioncode, s.institution_name, s.ownerinstitutioncode,
            s.country as stats_country, s.region, s.institution_type,
            s.record_count, s.species_count, s.families_count, s.countries_count,
            s.georeferencing_quality, s.date_quality, s.taxonomic_quality,
            s.overall_quality, s.first_record, s.latest_record,
            d.official_name, d.alternate_name, d.abbreviation_name,
            d.address, d.city, d.state, d.country as detail_country, d.zip,
            d.phone, d.email, d.website,
            d.latitude, d.longitude,
            d.sns_twitter, d.sns_facebook, d.sns_instagram, d.sns_other,
            d.preparation_type, d.environment, d.specimens_amount, d.establish_time,
            d.data_url, d.data_format,
            d.primary_type_lots, d.secondary_type_lots, d.genetic_resources,
            d.source, d.survey_collection_id, d.gbif_link
        FROM dbo.institution_stats s
        LEFT JOIN dbo.institution_details d ON s.institutioncode = d.institution_code
        WHERE s.institutioncode = :institution_code
    """)

    result = db.execute(query, {"institution_code": institution_code}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Institution not found")

    # Fetch contacts for this institution
    contacts_query = text("""
        SELECT first_name, last_name, title, contact_type,
               phone, email, address, city, state, country, zip
        FROM dbo.institution_contacts
        WHERE institution_code = :institution_code
        ORDER BY id
    """)
    contacts_result = db.execute(contacts_query, {"institution_code": institution_code}).fetchall()

    contacts = []
    for contact in contacts_result:
        contacts.append({
            "firstName": contact[0],
            "lastName": contact[1],
            "title": contact[2],
            "contactType": contact[3],
            "phone": contact[4],
            "email": contact[5],
            "address": contact[6],
            "city": contact[7],
            "state": contact[8],
            "country": contact[9],
            "zip": contact[10]
        })

    return {
        # Stats data
        "institutionCode": result[0],
        "institutionName": result[16] or result[1],
        "ownerInstitutionCode": result[2],
        "statsCountry": result[3],
        "region": result[4],
        "institutionType": result[5],
        "recordCount": result[6] or 0,
        "speciesCount": result[7] or 0,
        "familiesCount": result[8] or 0,
        "countriesCount": result[9] or 0,
        "geoReferencingQuality": float(result[10]) if result[10] else 0.0,
        "dateQuality": float(result[11]) if result[11] else 0.0,
        "taxonomicQuality": float(result[12]) if result[12] else 0.0,
        "overallQuality": float(result[13]) if result[13] else 0.0,
        "firstRecord": result[14],
        "latestRecord": result[15],
        # Details data
        "officialName": result[16],
        "alternateName": result[17],
        "abbreviationName": result[18],
        "address": result[19],
        "city": result[20],
        "state": result[21],
        "country": result[22],
        "zip": result[23],
        "phone": result[24],
        "email": result[25],
        "website": result[26],
        "latitude": float(result[27]) if result[27] else None,
        "longitude": float(result[28]) if result[28] else None,
        "twitter": result[29],
        "facebook": result[30],
        "instagram": result[31],
        "otherSocial": result[32],
        "preparationType": result[33],
        "environment": result[34],
        "specimensAmount": result[35],
        "establishTime": result[36],
        "dataUrl": result[37],
        "dataFormat": result[38],
        "primaryTypeLots": result[39],
        "secondaryTypeLots": result[40],
        "geneticResources": result[41],
        "source": result[42],
        "surveyCollectionId": result[43],
        "gbifLink": result[44],
        # Contacts
        "contacts": contacts
    }


@app.get("/institutions/{institution_code}")
async def get_institution_detail(institution_code: str, db: Session = Depends(get_db)):
    """Get institution details"""
    query = text("""
        SELECT institutioncode, institution_name, ownerinstitutioncode, 
               region, institution_type, record_count, species_count, 
               families_count, countries_count, georeferencing_quality, 
               date_quality, overall_quality, collection_codes, 
               first_record, latest_record, last_updated
        FROM dbo.institution_stats
        WHERE institutioncode = :institution_code
    """)

    result = db.execute(query, {"institution_code": institution_code}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Institution not found")

    return {
        "institutionCode": result[0],
        "institutionName": result[1],
        "ownerInstitutionCode": result[2],
        "region": result[3],
        "institutionType": result[4],
        "recordCount": result[5],
        "speciesCount": result[6],
        "familiesCount": result[7],
        "countriesCount": result[8],
        "geoReferencingQuality": result[9],
        "dateQuality": result[10],
        "overallQuality": result[11],
        "collectionCodes": result[12] if result[12] else [],
        "firstRecord": result[13],
        "latestRecord": result[14],
        "lastUpdated": result[15]
    }


@app.get("/records", response_model=PaginatedResponse)
async def get_records(
        params: RecordFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """Get records"""
    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country,
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, geo_quality, date_quality
        FROM dbo.record_details
    """

    conditions = []
    params_dict = {}

    # Search conditions
    if params.search:
        conditions.append("""
            (scientificname ILIKE :search OR recordedby ILIKE :search 
             OR locality ILIKE :search OR catalognumber::text ILIKE :search)
        """)
        params_dict["search"] = f"%{params.search}%"

    if params.year:
        conditions.append("\"year\" = :year_filter")
        params_dict["year_filter"] = params.year

    if params.country:
        conditions.append("country = :country_filter")
        params_dict["country_filter"] = params.country

    if params.institution_code:
        conditions.append("institutioncode = :institution_filter")
        params_dict["institution_filter"] = params.institution_code

    if params.family:
        conditions.append("family = :family_filter")
        params_dict["family_filter"] = params.family

    if params.genus:
        conditions.append("genus = :genus_filter")
        params_dict["genus_filter"] = params.genus

    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # Sort
    sort_mapping = {
        "date_desc": "eventdate DESC NULLS LAST",
        "date_asc": "eventdate ASC NULLS LAST",
        "collector_asc": "recordedby ASC NULLS LAST",
        "locality_asc": "locality ASC NULLS LAST"
    }
    order_by = sort_mapping.get(params.sort_by, "eventdate DESC NULLS LAST")

    # Build query
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.record_details{where_clause}")

    # Execute query
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # Get total count and data
    total = db.execute(count_query, {k: v for k, v in params_dict.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(data_query, params_dict).fetchall()

    records = []
    for row in result:
        records.append({
            "id": row[0],
            "catalogNumber": row[0],
            "scientificName": row[1],
            "vernacularName": row[2],
            "family": row[3],
            "genus": row[4],
            "authority": row[5],
            "recordedBy": row[6],
            "eventDate": row[7],
            "country": row[8],
            "locality": row[9],
            "decimalLatitude": row[10],
            "decimalLongitude": row[11],
            "institutionCode": row[12],
            "collectionCode": row[13],
            "basisOfRecord": row[14],
            "geoQuality": row[15],
            "dateQuality": row[16]
        })

    return PaginatedResponse(
        data=records,
        page=params.page,
        per_page=params.per_page,
        total=total,
        pages=(total + params.per_page - 1) // params.per_page
    )


@app.get("/taxonomy/stats")
async def get_taxonomy_stats(db: Session = Depends(get_db)):
    """Get taxonomy statistics - uses cache tables for performance, raw table only for quality metrics"""

    # Query 1: Family-level stats from cache table (fast - only hundreds of rows)
    family_stats_query = text("""
        SELECT
            COUNT(*) as total_families,
            COALESCE(SUM(record_count), 0) as total_records,
            ROUND(SUM(georeferencing_quality * record_count) / NULLIF(SUM(record_count), 0), 1) as avg_georeferencing,
            ROUND(SUM(date_quality * record_count) / NULLIF(SUM(record_count), 0), 1) as avg_date_quality,
            COUNT(CASE WHEN genera_count > 50 THEN 1 END) as high_diversity_families,
            COUNT(CASE WHEN record_count > 1000 THEN 1 END) as well_sampled_families,
            COUNT(CASE WHEN countries_count > 20 THEN 1 END) as global_coverage_families,
            COUNT(CASE WHEN record_count > 10000 THEN 1 END) as major_families,
            COUNT(CASE WHEN record_count BETWEEN 1000 AND 10000 THEN 1 END) as active_families,
            COUNT(CASE WHEN record_count < 100 THEN 1 END) as underrepresented_families
        FROM dbo.family_stats
    """)

    # Query 2: Genus-level stats from cache table (fast)
    genus_stats_query = text("""
        SELECT
            COUNT(*) as total_genera,
            COUNT(CASE WHEN species_count > 20 THEN 1 END) as species_rich_genera,
            COUNT(CASE WHEN record_count > 500 THEN 1 END) as well_sampled_genera,
            COUNT(CASE WHEN countries_count > 15 THEN 1 END) as global_coverage_genera
        FROM dbo.genus_stats
    """)

    # Query 3: Species/institution counts from cache tables (fast)
    counts_query = text("""
        SELECT
            (SELECT COUNT(*) FROM dbo.species_stats) as total_species,
            (SELECT COUNT(*) FROM dbo.institution_stats) as total_institutions
    """)

    # Query 4: Global metrics from cache table (fast - single row)
    global_stats_query = text("""
        SELECT
            total_countries,
            taxonomy_completeness,
            institution_coverage,
            recently_active_families
        FROM dbo.global_stats
        LIMIT 1
    """)

    family_result = db.execute(family_stats_query).fetchone()
    genus_result = db.execute(genus_stats_query).fetchone()
    counts_result = db.execute(counts_query).fetchone()
    global_result = db.execute(global_stats_query).fetchone()

    return {
        # Basic statistics
        "totalFamilies": family_result[0] or 0,
        "totalGenera": genus_result[0] or 0,
        "totalSpecies": counts_result[0] or 0,
        "totalRecords": family_result[1] or 0,
        "totalInstitutions": counts_result[1] or 0,
        "totalCountries": global_result[0] or 0,

        # Diversity metrics
        "highDiversityFamilies": family_result[4] or 0,
        "wellSampledFamilies": family_result[5] or 0,
        "globalCoverageFamilies": family_result[6] or 0,

        # Genus-level statistics
        "speciesRichGenera": genus_result[1] or 0,
        "wellSampledGenera": genus_result[2] or 0,
        "globalCoverageGenera": genus_result[3] or 0,

        # Quality metrics (weighted average from cache, raw metrics from single scan)
        "avgGeoreferencing": float(family_result[2]) if family_result[2] else 0.0,
        "avgDateQuality": float(family_result[3]) if family_result[3] else 0.0,
        "taxonomyCompleteness": float(global_result[1]) if global_result[1] else 0.0,
        "institutionCoverage": float(global_result[2]) if global_result[2] else 0.0,

        # Activity metrics
        "recentlyActiveFamilies": global_result[3] or 0,

        # Contribution distribution
        "majorContributorFamilies": family_result[7] or 0,
        "activeContributorFamilies": family_result[8] or 0,
        "underrepresentedFamilies": family_result[9] or 0,

        # Calculation timestamp
        "calculatedAt": datetime.now().isoformat()
    }


# Institution records endpoints

@app.get("/institutions/{institution_code}/records", response_model=PaginatedResponse)
async def get_institution_records(
        institution_code: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        search: Optional[str] = None,
        year: Optional[int] = None,
        country: Optional[str] = None,
        family: Optional[str] = None,
        genus: Optional[str] = None,
        sort_by: str = "date_desc",
        db: Session = Depends(get_db)
):
    """Get all records for a specific institution"""

    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country, 
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, "year", "month"
        FROM dbo.harvestedfn2_fin
        WHERE institutioncode = :institution_code
    """

    conditions = []
    params = {"institution_code": institution_code}

    # Search conditions
    if search:
        conditions.append("""
            (scientificname ILIKE :search OR recordedby ILIKE :search 
             OR locality ILIKE :search OR catalognumber ILIKE :search
             OR family ILIKE :search OR genus ILIKE :search)
        """)
        params["search"] = f"%{search}%"

    if year:
        conditions.append('"year" = :year_filter')
        params["year_filter"] = year

    if country:
        conditions.append("country = :country_filter")
        params["country_filter"] = country

    if family:
        conditions.append("family = :family_filter")
        params["family_filter"] = family

    if genus:
        conditions.append("genus = :genus_filter")
        params["genus_filter"] = genus

    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    # Sort
    sort_mapping = {
        "date_desc": "eventdate DESC NULLS LAST",
        "date_asc": "eventdate ASC NULLS LAST",
        "collector_asc": "recordedby ASC NULLS LAST",
        "locality_asc": "locality ASC NULLS LAST",
        "species_asc": "scientificname ASC NULLS LAST",
        "family_asc": "family ASC NULLS LAST"
    }
    order_by = sort_mapping.get(sort_by, "eventdate DESC NULLS LAST")

    # Build query
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.harvestedfn2_fin WHERE institutioncode = :institution_code{where_clause}")

    # Execute query
    params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })

    # Get total count and data
    total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(data_query, params).fetchall()

    records = []
    for row in result:
        records.append({
            "id": f"{row[0]}_{row[12]}",  # catalogNumber_institutionCode
            "catalogNumber": row[0],
            "scientificName": row[1],
            "vernacularName": row[2],
            "family": row[3],
            "genus": row[4],
            "authority": row[5],
            "recordedBy": row[6],
            "eventDate": row[7],
            "country": row[8],
            "locality": row[9],
            "decimalLatitude": float(row[10]) if row[10] else None,
            "decimalLongitude": float(row[11]) if row[11] else None,
            "institutionCode": row[12],
            "collectionCode": row[13],
            "basisOfRecord": row[14],
            "year": row[15],
            "month": row[16]
        })

    return PaginatedResponse(
        data=records,
        page=page,
        per_page=per_page,
        total=total,
        pages=(total + per_page - 1) // per_page
    )


@app.get("/institutions/{institution_code}/records/summary")
async def get_institution_records_summary(
        institution_code: str,
        db: Session = Depends(get_db)
):
    """Get summary statistics for institution records"""

    summary_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT COALESCE(validname, scientificname)) as unique_species,
            COUNT(DISTINCT family) as unique_families,
            COUNT(DISTINCT genus) as unique_genera,
            COUNT(DISTINCT countrycode) as unique_countries,
            COUNT(DISTINCT collectioncode) as unique_collections,
            COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) as georeferenced_records,
            COUNT(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 1 END) as dated_records,
            MIN(CASE WHEN "year" IS NOT NULL AND "year" > 1800 THEN "year" END) as earliest_year,
            MAX(CASE WHEN "year" IS NOT NULL AND "year" > 1800 THEN "year" END) as latest_year,
            COUNT(CASE WHEN "year" >= 2020 THEN 1 END) as recent_records
        FROM dbo.harvestedfn2_fin 
        WHERE institutioncode = :institution_code
    """)

    result = db.execute(summary_query, {"institution_code": institution_code}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Institution not found or has no records")

    # Calculate percentages
    total = result[0]
    georef_percentage = round((result[6] / total) * 100, 1) if total > 0 else 0
    date_percentage = round((result[7] / total) * 100, 1) if total > 0 else 0
    recent_percentage = round((result[10] / total) * 100, 1) if total > 0 else 0

    return {
        "totalRecords": result[0],
        "uniqueSpecies": result[1],
        "uniqueFamilies": result[2],
        "uniqueGenera": result[3],
        "uniqueCountries": result[4],
        "uniqueCollections": result[5],
        "georeferencedRecords": result[6],
        "georeferencedPercentage": georef_percentage,
        "datedRecords": result[7],
        "datedPercentage": date_percentage,
        "earliestYear": result[8],
        "latestYear": result[9],
        "recentRecords": result[10],
        "recentPercentage": recent_percentage,
        "collectionSpan": (result[9] - result[8]) if result[8] and result[9] else 0
    }


@app.get("/institutions/{institution_code}/records/filters")
async def get_institution_records_filters(
        institution_code: str,
        db: Session = Depends(get_db)
):
    """Get filter options for institution records"""

    # Get available years
    years_query = text("""
        SELECT DISTINCT "year"
        FROM dbo.harvestedfn2_fin 
        WHERE institutioncode = :institution_code 
          AND "year" IS NOT NULL AND "year" > 1800
        ORDER BY "year" DESC
    """)

    # Get available countries
    countries_query = text("""
        SELECT DISTINCT country, COUNT(*) as record_count
        FROM dbo.harvestedfn2_fin 
        WHERE institutioncode = :institution_code 
          AND country IS NOT NULL AND country != ''
        GROUP BY country
        ORDER BY record_count DESC
    """)

    # Get available families
    families_query = text("""
        SELECT DISTINCT family, COUNT(*) as record_count
        FROM dbo.harvestedfn2_fin 
        WHERE institutioncode = :institution_code 
          AND family IS NOT NULL AND family != ''
        GROUP BY family
        ORDER BY record_count DESC
    """)

    # Get available genera
    genera_query = text("""
        SELECT DISTINCT genus, COUNT(*) as record_count
        FROM dbo.harvestedfn2_fin 
        WHERE institutioncode = :institution_code 
          AND genus IS NOT NULL AND genus != ''
        GROUP BY genus
        ORDER BY record_count DESC
        LIMIT 100
    """)

    # Get available collection codes
    collections_query = text("""
        SELECT DISTINCT collectioncode, COUNT(*) as record_count
        FROM dbo.harvestedfn2_fin 
        WHERE institutioncode = :institution_code 
          AND collectioncode IS NOT NULL AND collectioncode != ''
        GROUP BY collectioncode
        ORDER BY record_count DESC
    """)

    params = {"institution_code": institution_code}

    years_result = db.execute(years_query, params).fetchall()
    countries_result = db.execute(countries_query, params).fetchall()
    families_result = db.execute(families_query, params).fetchall()
    genera_result = db.execute(genera_query, params).fetchall()
    collections_result = db.execute(collections_query, params).fetchall()

    return {
        "years": [row[0] for row in years_result],
        "countries": [{"name": row[0], "count": row[1]} for row in countries_result],
        "families": [{"name": row[0], "count": row[1]} for row in families_result],
        "genera": [{"name": row[0], "count": row[1]} for row in genera_result],
        "collections": [{"name": row[0], "count": row[1]} for row in collections_result]
    }


@app.get("/institutions/{institution_code}/records/export")
async def export_institution_records(
        institution_code: str,
        format: str = Query("csv", regex="^(csv|json|dwc)$"),
        search: Optional[str] = None,
        year: Optional[int] = None,
        country: Optional[str] = None,
        family: Optional[str] = None,
        genus: Optional[str] = None,
        limit: int = Query(10000, le=50000),
        db: Session = Depends(get_db)
):
    """Export institution records data"""

    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country, 
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, "year", "month", day,
               county, stateprovince, kingdom, phylum, "class", "order"
        FROM dbo.harvestedfn2_fin
        WHERE institutioncode = :institution_code
    """

    conditions = []
    params = {"institution_code": institution_code}

    # Apply filter conditions
    if search:
        conditions.append("""
            (scientificname ILIKE :search OR recordedby ILIKE :search 
             OR locality ILIKE :search OR catalognumber ILIKE :search)
        """)
        params["search"] = f"%{search}%"

    if year:
        conditions.append('"year" = :year_filter')
        params["year_filter"] = year

    if country:
        conditions.append("country = :country_filter")
        params["country_filter"] = country

    if family:
        conditions.append("family = :family_filter")
        params["family_filter"] = family

    if genus:
        conditions.append("genus = :genus_filter")
        params["genus_filter"] = genus

    # Build query
    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    query = text(base_select + where_clause + f" ORDER BY eventdate DESC NULLS LAST LIMIT :limit")
    params["limit"] = limit

    result = db.execute(query, params).fetchall()

    if format == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        headers = [
            "catalogNumber", "scientificName", "vernacularName", "family", "genus",
            "scientificNameAuthorship", "recordedBy", "eventDate", "country",
            "locality", "decimalLatitude", "decimalLongitude", "institutionCode",
            "collectionCode", "basisOfRecord", "year", "month", "day",
            "county", "stateProvince", "kingdom", "phylum", "class", "order"
        ]
        writer.writerow(headers)

        # Write data
        for row in result:
            writer.writerow(row)

        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={institution_code}_records.csv"}
        )

    elif format == "json":
        from fastapi.responses import JSONResponse

        records = []
        for row in result:
            record = {
                "catalogNumber": row[0],
                "scientificName": row[1],
                "vernacularName": row[2],
                "family": row[3],
                "genus": row[4],
                "scientificNameAuthorship": row[5],
                "recordedBy": row[6],
                "eventDate": row[7],
                "country": row[8],
                "locality": row[9],
                "decimalLatitude": row[10],
                "decimalLongitude": row[11],
                "institutionCode": row[12],
                "collectionCode": row[13],
                "basisOfRecord": row[14],
                "year": row[15],
                "month": row[16],
                "day": row[17],
                "county": row[18],
                "stateProvince": row[19],
                "kingdom": row[20],
                "phylum": row[21],
                "class": row[22],
                "order": row[23]
            }
            records.append(record)

        return JSONResponse(
            content={"records": records, "count": len(records)},
            headers={"Content-Disposition": f"attachment; filename={institution_code}_records.json"}
        )

    elif format == "dwc":
        # Darwin Core Archive format (simplified)
        import xml.etree.ElementTree as ET
        from fastapi.responses import Response

        root = ET.Element("SimpleDarwinRecordSet")
        root.set("xmlns", "http://rs.tdwg.org/dwc/xsd/simpledarwincore/")

        for row in result:
            record = ET.SubElement(root, "SimpleDarwinRecord")

            # Map fields to Darwin Core terms
            dwc_mapping = {
                "catalogNumber": row[0],
                "scientificName": row[1],
                "vernacularName": row[2],
                "family": row[3],
                "genus": row[4],
                "scientificNameAuthorship": row[5],
                "recordedBy": row[6],
                "eventDate": row[7],
                "country": row[8],
                "locality": row[9],
                "decimalLatitude": row[10],
                "decimalLongitude": row[11],
                "institutionCode": row[12],
                "collectionCode": row[13],
                "basisOfRecord": row[14],
                "year": row[15],
                "month": row[16],
                "day": row[17]
            }

            for field, value in dwc_mapping.items():
                if value is not None:
                    elem = ET.SubElement(record, field)
                    elem.text = str(value)

        xml_str = ET.tostring(root, encoding='unicode')

        return Response(
            content=xml_str,
            media_type="application/xml",
            headers={"Content-Disposition": f"attachment; filename={institution_code}_records.xml"}
        )

    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")


# RecordsTable component API
@app.get("/records/by-institution/{institution_code}")
async def get_records_by_institution(
        institution_code: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        db: Session = Depends(get_db)
):
    """Simplified institution records fetch for RecordsTable component"""

    query = text("""
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               recordedby, eventdate, country, locality, decimallatitude, 
               decimallongitude, institutioncode, collectioncode
        FROM dbo.harvestedfn2_fin
        WHERE institutioncode = :institution_code
        ORDER BY eventdate DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """)

    count_query = text("""
        SELECT COUNT(*) FROM dbo.harvestedfn2_fin WHERE institutioncode = :institution_code
    """)

    total = db.execute(count_query, {"institution_code": institution_code}).scalar()
    result = db.execute(query, {
        "institution_code": institution_code,
        "limit": per_page,
        "offset": (page - 1) * per_page
    }).fetchall()

    records = []
    for row in result:
        records.append({
            "id": f"{row[0]}_{row[11]}",
            "catalogNumber": row[0],
            "scientificName": row[1],
            "vernacularName": row[2],
            "family": row[3],
            "genus": row[4],
            "recordedBy": row[5],
            "eventDate": row[6],
            "country": row[7],
            "locality": row[8],
            "decimalLatitude": row[9],
            "decimalLongitude": row[10],
            "institutionCode": row[11],
            "collectionCode": row[12]
        })

    return {
        "data": records,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page
    }


# Institution records v2 endpoint

# Geo-coordinate filter parameters for get_records_by_institution_v2

@app.get("/records/institution/{institution_code}", response_model=PaginatedResponse)
async def get_records_by_institution_v2(
        institution_code: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        search: Optional[str] = None,
        year: Optional[int] = None,
        country: Optional[str] = None,
        family: Optional[str] = None,
        genus: Optional[str] = None,
        sort_by: str = "date_desc",
        georeferenced_only: bool = Query(False),  # Only get records with geo-coordinates
        db: Session = Depends(get_db)
):
    """Get records by institution - adapted for frontend API calls, supports geo-coordinate filtering"""

    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country, 
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, "year", "month"
        FROM dbo.harvestedfn2_fin
        WHERE institutioncode = :institution_code
    """

    conditions = []
    params = {"institution_code": institution_code}

    # Geo-coordinate filter
    if georeferenced_only:
        conditions.append("decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL")
        conditions.append("decimallatitude != 0 OR decimallongitude != 0")  # Exclude 0,0 coordinates

    # Other search conditions
    if search:
        conditions.append("""
            (scientificname ILIKE :search OR recordedby ILIKE :search 
             OR locality ILIKE :search OR catalognumber ILIKE :search
             OR family ILIKE :search OR genus ILIKE :search)
        """)
        params["search"] = f"%{search}%"

    if year:
        conditions.append('"year" = :year_filter')
        params["year_filter"] = year

    if country:
        conditions.append("country = :country_filter")
        params["country_filter"] = country

    if family:
        conditions.append("family = :family_filter")
        params["family_filter"] = family

    if genus:
        conditions.append("genus = :genus_filter")
        params["genus_filter"] = genus

    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    # Sort
    sort_mapping = {
        "date_desc": "eventdate DESC NULLS LAST",
        "date_asc": "eventdate ASC NULLS LAST",
        "collector_asc": "recordedby ASC NULLS LAST",
        "locality_asc": "locality ASC NULLS LAST",
        "species_asc": "scientificname ASC NULLS LAST",
        "family_asc": "family ASC NULLS LAST"
    }
    order_by = sort_mapping.get(sort_by, "eventdate DESC NULLS LAST")

    # Build query
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.harvestedfn2_fin WHERE institutioncode = :institution_code{where_clause}")

    # Execute query
    params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })

    # Get total count and data
    total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(data_query, params).fetchall()

    records = []
    for row in result:
        records.append({
            "id": f"{row[0]}_{row[12]}",  # catalogNumber_institutionCode
            "catalogNumber": row[0],
            "scientificName": row[1],
            "vernacularName": row[2],
            "family": row[3],
            "genus": row[4],
            "authority": row[5],
            "recordedBy": row[6],
            "eventDate": row[7],
            "country": row[8],
            "locality": row[9],
            "decimalLatitude": float(row[10]) if row[10] else None,
            "decimalLongitude": float(row[11]) if row[11] else None,
            "institutionCode": row[12],
            "collectionCode": row[13],
            "basisOfRecord": row[14],
            "year": row[15],
            "month": row[16]
        })

    return PaginatedResponse(
        data=records,
        page=page,
        per_page=per_page,
        total=total,
        pages=(total + per_page - 1) // per_page
    )


@app.get("/institutions/{institution_code}/species")
async def get_institution_species(
        institution_code: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        search: Optional[str] = None,
        sort_by: str = "records_desc",
        db: Session = Depends(get_db)
):
    """Get species list for a specific institution with record counts"""

    base_query = """
        SELECT COALESCE(validname, scientificname) as species_name,
               family,
               COUNT(*) as record_count,
               COUNT(DISTINCT countrycode) as countries_count
        FROM dbo.harvestedfn2_fin
        WHERE institutioncode = :institution_code
          AND COALESCE(validname, scientificname) IS NOT NULL
          AND COALESCE(validname, scientificname) != ''
    """

    conditions = []
    params = {"institution_code": institution_code}

    if search:
        conditions.append("""
            (COALESCE(validname, scientificname) ILIKE :search OR family ILIKE :search)
        """)
        params["search"] = f"%{search}%"

    if conditions:
        base_query += " AND " + " AND ".join(conditions)

    base_query += " GROUP BY COALESCE(validname, scientificname), family"

    # Sorting
    sort_mapping = {
        "records_desc": "record_count DESC",
        "records_asc": "record_count ASC",
        "name_asc": "species_name ASC",
        "name_desc": "species_name DESC",
        "family_asc": "family ASC, species_name ASC"
    }
    order_clause = sort_mapping.get(sort_by, "record_count DESC")

    # Count query
    count_query = text(f"""
        SELECT COUNT(*) FROM (
            {base_query}
        ) as species_count
    """)

    # Data query with pagination
    data_query = text(f"""
        {base_query}
        ORDER BY {order_clause}
        LIMIT :limit OFFSET :offset
    """)

    params["limit"] = per_page
    params["offset"] = (page - 1) * per_page

    total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar() or 0
    result = db.execute(data_query, params).fetchall()

    species_list = []
    for row in result:
        species_list.append({
            "scientificName": row[0],
            "family": row[1],
            "recordCount": row[2],
            "countriesCount": row[3]
        })

    return {
        "data": species_list,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page
    }


@app.get("/institution/stats")
async def get_institutions_stats(db: Session = Depends(get_db)):
    """Get institution statistics - fixed version"""

    # 1. Basic statistics - count all institutions
    basic_stats_query = text("""
        SELECT
            COUNT(*) as total_institutions,
            (SELECT SUM(array_length(collection_codes, 1)) FROM dbo.institution_stats WHERE collection_codes IS NOT NULL) as total_collection_codes,
            SUM(record_count) as total_records,
            (SELECT COUNT(DISTINCT country) FROM dbo.institution_country_distribution) as total_countries,
            ROUND(AVG(georeferencing_quality), 1) as avg_georeferencing,
            ROUND(AVG(date_quality), 1) as avg_date_quality
        FROM dbo.institution_stats
    """)

    basic_result = db.execute(basic_stats_query).fetchone()

    # 2. Contributor classification statistics
    contributors_query = text("""
        SELECT 
            COUNT(CASE WHEN record_count > 10000 THEN 1 END) as major_contributors,
            COUNT(CASE WHEN record_count BETWEEN 1000 AND 10000 THEN 1 END) as active_contributors,
            COUNT(CASE WHEN institution_type IN ('museum', 'university') THEN 1 END) as research_collections,
            COUNT(CASE WHEN overall_quality > 90 THEN 1 END) as high_quality_data
        FROM dbo.institution_stats
    """)

    contributors_result = db.execute(contributors_query).fetchone()

    # 3. Geographic distribution statistics - fixed version
    geographic_distribution_query = text("""
        SELECT 
            COUNT(CASE WHEN region = 'North America' THEN 1 END) as north_america,
            COUNT(CASE WHEN region = 'Europe' THEN 1 END) as europe,
            COUNT(CASE WHEN region = 'Asia-Pacific' THEN 1 END) as asia_pacific,
            COUNT(CASE WHEN region = 'South America' THEN 1 END) as south_america,
            COUNT(CASE WHEN region = 'Africa' THEN 1 END) as africa,
            COUNT(CASE WHEN region = 'Other Regions' THEN 1 END) as other_regions
        FROM dbo.institution_stats
    """)

    geo_result = db.execute(geographic_distribution_query).fetchone()

    # # 4. Collection diversity statistics
    # collection_diversity_query = text("""
    #     SELECT
    #         COUNT(CASE WHEN collection_codes IS NOT NULL THEN 1 END) as institutions_with_collections,
    #         (SELECT COUNT(DISTINCT unnest(collection_codes))
    #          FROM dbo.institution_stats
    #          WHERE collection_codes IS NOT NULL) as unique_collection_codes,
    #         ROUND(AVG(array_length(collection_codes, 1)), 1) as avg_collections_per_institution
    #     FROM dbo.institution_stats
    # """)

    # collection_result = db.execute(collection_diversity_query).fetchone()

    # 5. Temporal coverage analysis - safely handle NULL values
    temporal_coverage_query = text("""
        SELECT 
            COALESCE(SUM(CASE WHEN latest_year >= 2020 THEN record_count ELSE 0 END), 0) as recent_records,
            COALESCE(SUM(CASE WHEN latest_year BETWEEN 2010 AND 2019 THEN record_count ELSE 0 END), 0) as decade_records,
            COALESCE(SUM(CASE WHEN latest_year BETWEEN 2000 AND 2009 THEN record_count ELSE 0 END), 0) as millennium_records,
            COALESCE(SUM(CASE WHEN latest_year < 2000 THEN record_count ELSE 0 END), 0) as historical_records
        FROM dbo.institution_stats
        WHERE latest_year IS NOT NULL AND latest_year > 0
    """)

    temporal_result = db.execute(temporal_coverage_query).fetchone()

    return {
        # Basic statistics
        "totalInstitutions": basic_result[0] or 0,
        "totalCollectionCodes": basic_result[1] or 0,
        "totalRecords": basic_result[2] or 0,
        "totalCountries": basic_result[3] or 0,
        "avgGeoreferenced": float(basic_result[4]) if basic_result[4] else 0.0,
        "avgDateQuality": float(basic_result[5]) if basic_result[5] else 0.0,

        # Contributor classification
        "majorContributors": contributors_result[0] or 0,
        "activeContributors": contributors_result[1] or 0,
        "researchCollections": contributors_result[2] or 0,
        "highQualityData": contributors_result[3] or 0,

        # Geographic distribution
        "northAmericaInstitutions": geo_result[0] or 0,
        "europeInstitutions": geo_result[1] or 0,
        "asiaPacificInstitutions": geo_result[2] or 0,
        "southAmericaInstitutions": geo_result[3] or 0,
        "africaInstitutions": geo_result[4] or 0,
        "otherRegionsInstitutions": geo_result[5] or 0,

        # # Collection diversity
        # "institutionsWithCollections": collection_result[0] or 0,
        # "uniqueCollectionCodes": collection_result[1] or 0,
        # "avgCollectionsPerInstitution": float(collection_result[2]) if collection_result[2] else 0.0,

        # Temporal coverage
        "recentRecords": temporal_result[0] or 0,
        "decadeRecords": temporal_result[1] or 0,
        "millenniumRecords": temporal_result[2] or 0,
        "historicalRecords": temporal_result[3] or 0,

        # Calculation timestamp
        "calculatedAt": datetime.now().isoformat()
    }


@app.get("/institutions/{institution_code}/countries")
async def get_institution_countries(institution_code: str, db: Session = Depends(get_db)):
    """Get institution distribution by country"""
    query = text("""
        SELECT country, records_in_country, species_in_country, percentage_of_records
        FROM dbo.institution_country_distribution
        WHERE institutioncode = :institution_code
        ORDER BY records_in_country DESC
    """)

    result = db.execute(query, {"institution_code": institution_code}).fetchall()

    countries = []
    for row in result:
        countries.append({
            "country": row[0],
            "recordCount": row[1],
            "speciesCount": row[2],
            "percentage": float(row[3])
        })

    return {"countries": countries}


@app.get("/orders")
async def get_orders(db: Session = Depends(get_db)):
    """Get all orders"""
    query = text("""
        SELECT "order", COUNT(*) as family_count,
               COALESCE(SUM(species_count), 0) as species_count
        FROM dbo.family_stats
        WHERE "order" IS NOT NULL AND "order" != ''
        GROUP BY "order"
        ORDER BY species_count DESC
    """)

    result = db.execute(query).fetchall()

    return [
        {
            "name": row[0],
            "familyCount": row[1],
            "speciesCount": row[2]
        }
        for row in result
    ]


@app.get("/records/taxon/{taxon_type}/{taxon_name}")
async def get_records_by_taxon(
        taxon_type: str,
        taxon_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=1000),
        has_coordinates: Optional[bool] = Query(None, description="Filter records with coordinates"),
        db: Session = Depends(get_db)
):
    """Get records by taxonomic unit"""
    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    taxon_condition, taxon_params = parse_taxon_filter(taxon_type, taxon_name)

    # Build WHERE clause
    where_conditions = [taxon_condition]

    if has_coordinates is True:
        where_conditions.append("decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL")
    elif has_coordinates is False:
        where_conditions.append("(decimallatitude IS NULL OR decimallongitude IS NULL)")

    where_clause = " AND ".join(where_conditions)

    query = text(f"""
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               recordedby, eventdate, country, locality, decimallatitude,
               decimallongitude, institutioncode, collectioncode
        FROM dbo.harvestedfn2_fin
        WHERE {where_clause}
        ORDER BY eventdate DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """)

    count_query = text(f"""
        SELECT COUNT(*) FROM dbo.harvestedfn2_fin WHERE {where_clause}
    """)

    total = db.execute(count_query, taxon_params).scalar()
    query_params = dict(taxon_params)
    query_params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })
    result = db.execute(query, query_params).fetchall()

    records = []
    for row in result:
        records.append({
            "id": row[0],
            "catalogNumber": row[0],
            "scientificName": row[1],
            "vernacularName": row[2],
            "family": row[3],
            "genus": row[4],
            "recordedBy": row[5],
            "eventDate": row[6],
            "country": row[7],
            "locality": row[8],
            "decimalLatitude": row[9],
            "decimalLongitude": row[10],
            "institutionCode": row[11],
            "collectionCode": row[12]
        })

    return PaginatedResponse(
        data=records,
        page=page,
        per_page=per_page,
        total=total,
        pages=(total + per_page - 1) // per_page
    )


@app.get("/records/taxon/{taxon_type}/{taxon_name}/map-points")
async def get_taxon_map_points(
        taxon_type: str,
        taxon_name: str,
        south: Optional[float] = Query(None, description="South bound latitude"),
        north: Optional[float] = Query(None, description="North bound latitude"),
        west: Optional[float] = Query(None, description="West bound longitude"),
        east: Optional[float] = Query(None, description="East bound longitude"),
        zoom: Optional[int] = Query(None, description="Map zoom level (1-18), controls grid precision"),
        db: Session = Depends(get_db)
):
    """Return coordinate points for heatmap rendering.
    Grid precision adapts to zoom level:
      - zoom 1-3 (world): ~111km grid (0 decimals)
      - zoom 4-6 (continent): ~11km grid (1 decimal)
      - zoom 7+ (region): ~1km grid (2 decimals)
    Returns [[lat, lng, weight], ...] — weight = record count at that grid cell.
    Frontend: L.heatLayer(data.points)  (Leaflet.heat accepts [lat, lng, intensity])
    """
    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    taxon_condition, taxon_params = parse_taxon_filter(taxon_type, taxon_name)
    query_params = dict(taxon_params)

    # Determine grid precision based on zoom level
    if zoom is None or zoom <= 3:
        precision = 0  # ~111km grid
    elif zoom <= 6:
        precision = 1  # ~11km grid
    else:
        precision = 2  # ~1km grid

    # Viewport bounds filter
    bounds_condition = ""
    if south is not None and north is not None and west is not None and east is not None:
        bounds_condition = " AND decimallatitude BETWEEN :south AND :north AND decimallongitude BETWEEN :west AND :east"
        query_params.update({"south": south, "north": north, "west": west, "east": east})

    # Limit points to prevent browser overload (keep highest-weight cells)
    max_points = 50000

    query = text(f"""
        SELECT ROUND(decimallatitude::numeric, {precision}) as lat,
               ROUND(decimallongitude::numeric, {precision}) as lng,
               COUNT(*) as weight
        FROM dbo.harvestedfn2_fin
        WHERE {taxon_condition}
          AND decimallatitude IS NOT NULL
          AND decimallongitude IS NOT NULL
          {bounds_condition}
        GROUP BY ROUND(decimallatitude::numeric, {precision}), ROUND(decimallongitude::numeric, {precision})
        ORDER BY weight DESC
        LIMIT {max_points}
    """)

    result = db.execute(query, query_params).fetchall()

    total_records = 0
    points = []
    for row in result:
        w = int(row[2])
        total_records += w
        points.append([float(row[0]), float(row[1]), w])

    return {
        "points": points,
        "total": len(points),
        "totalRecords": total_records
    }


@app.get("/institutions/{institution_code}/map-points")
async def get_institution_map_points(
        institution_code: str,
        south: Optional[float] = Query(None, description="South bound latitude"),
        north: Optional[float] = Query(None, description="North bound latitude"),
        west: Optional[float] = Query(None, description="West bound longitude"),
        east: Optional[float] = Query(None, description="East bound longitude"),
        zoom: Optional[int] = Query(None, description="Map zoom level (1-18), controls grid precision"),
        db: Session = Depends(get_db)
):
    """Return coordinate points for institution heatmap rendering.
    Grid precision adapts to zoom level.
    Returns [[lat, lng, weight], ...] — weight = record count at that grid cell.
    """
    # Determine grid precision based on zoom level
    if zoom is None or zoom <= 3:
        precision = 0  # ~111km grid
    elif zoom <= 6:
        precision = 1  # ~11km grid
    else:
        precision = 2  # ~1km grid

    query_params = {"institution_code": institution_code}

    # Viewport bounds filter
    bounds_condition = ""
    if south is not None and north is not None and west is not None and east is not None:
        bounds_condition = " AND decimallatitude BETWEEN :south AND :north AND decimallongitude BETWEEN :west AND :east"
        query_params.update({"south": south, "north": north, "west": west, "east": east})

    max_points = 50000

    query = text(f"""
        SELECT ROUND(decimallatitude::numeric, {precision}) as lat,
               ROUND(decimallongitude::numeric, {precision}) as lng,
               COUNT(*) as weight
        FROM dbo.harvestedfn2_fin
        WHERE institutioncode = :institution_code
          AND decimallatitude IS NOT NULL
          AND decimallongitude IS NOT NULL
          {bounds_condition}
        GROUP BY ROUND(decimallatitude::numeric, {precision}), ROUND(decimallongitude::numeric, {precision})
        ORDER BY weight DESC
        LIMIT {max_points}
    """)

    result = db.execute(query, query_params).fetchall()

    total_records = 0
    points = []
    for row in result:
        w = int(row[2])
        total_records += w
        points.append([float(row[0]), float(row[1]), w])

    return {
        "points": points,
        "total": len(points),
        "totalRecords": total_records
    }


# Taxonomy hierarchy endpoints

@app.get("/families/{family_name}/children")
async def get_family_children(
        family_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        search: Optional[str] = None,
        sort_by: str = "records_desc",
        db: Session = Depends(get_db)
):
    """Get all genera under a family - uses genus_stats cache table"""

    # Build query from genus_stats cache table
    base_select = """
        SELECT genus, species_count, record_count, countries_count,
               institutions_count, georeferencing_quality, last_updated
        FROM dbo.genus_stats
        WHERE family = :family_name
    """

    conditions = []
    params = {"family_name": family_name}

    if search:
        conditions.append("genus ILIKE :search")
        params["search"] = f"%{search}%"

    # Sort
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "name_asc": "genus ASC",
        "name_desc": "genus DESC"
    }
    order_clause = f" ORDER BY {sort_mapping.get(sort_by, 'record_count DESC')}"

    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    query = text(base_select + where_clause + order_clause + " LIMIT :limit OFFSET :offset")
    count_query = text(
        f"SELECT COUNT(*) FROM dbo.genus_stats WHERE family = :family_name{where_clause}")

    params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })

    total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(query, params).fetchall()

    genera = []
    for row in result:
        genera.append({
            "name": row[0],
            "speciesCount": row[1] or 0,
            "recordCount": row[2] or 0,
            "countriesCount": row[3] or 0,
            "institutionsCount": row[4] or 0,
            "geoReferencingQuality": float(row[5]) if row[5] else 0.0,
            "lastRecord": row[6]
        })

    return PaginatedResponse(
        data=genera,
        page=page,
        per_page=per_page,
        total=total,
        pages=(total + per_page - 1) // per_page
    )


@app.get("/genera/{genus_name}/children")
async def get_genus_children(
        genus_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        search: Optional[str] = None,
        sort_by: str = "records_desc",
        db: Session = Depends(get_db)
):
    """Get all species under a genus - uses species_stats cache table"""

    base_select = """
        SELECT genus || ' ' || specificepithet as scientificname, vernacularname,
               record_count, countries_count, institutions_count,
               georeferencing_quality, last_record
        FROM dbo.species_stats
        WHERE genus = :genus_name
    """

    conditions = []
    params = {"genus_name": genus_name}

    if search:
        conditions.append("(genus || ' ' || specificepithet ILIKE :search OR vernacularname ILIKE :search)")
        params["search"] = f"%{search}%"

    # Sort
    sort_mapping = {
        "records_desc": "record_count DESC",
        "name_asc": "specificepithet ASC",
        "name_desc": "specificepithet DESC"
    }
    order_clause = f" ORDER BY {sort_mapping.get(sort_by, 'record_count DESC')}"

    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    query = text(base_select + where_clause + order_clause + " LIMIT :limit OFFSET :offset")
    count_query = text(
        f"SELECT COUNT(*) FROM dbo.species_stats WHERE genus = :genus_name{where_clause}")

    params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })

    total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(query, params).fetchall()

    species = []
    for row in result:
        species.append({
            "name": row[0],
            "vernacularName": row[1],
            "recordCount": row[2] or 0,
            "countriesCount": row[3] or 0,
            "institutionsCount": row[4] or 0,
            "geoReferencingQuality": float(row[5]) if row[5] else 0.0,
            "lastRecord": row[6]
        })

    return PaginatedResponse(
        data=species,
        page=page,
        per_page=per_page,
        total=total,
        pages=(total + per_page - 1) // per_page
    )


@app.get("/{taxon_type}/{taxon_name}/diversity")
async def get_taxon_diversity(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """Get diversity statistics for taxonomic unit"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    if taxon_type == "family":
        # Family diversity - from genus_stats cache table
        query = text("""
            SELECT genus, species_count, record_count
            FROM dbo.genus_stats
            WHERE family = :taxon_name
            ORDER BY record_count DESC
            LIMIT 10
        """)
    elif taxon_type == "genus":
        # Genus diversity - from species_stats cache table
        query = text("""
            SELECT genus || ' ' || specificepithet as scientificname,
                   record_count, vernacularname
            FROM dbo.species_stats
            WHERE genus = :taxon_name
            ORDER BY record_count DESC
            LIMIT 10
        """)
    else:
        # Species level doesn't need diversity statistics
        return {"diversityData": []}

    result = db.execute(query, {"taxon_name": taxon_name}).fetchall()

    diversity_data = []
    for row in result:
        if taxon_type == "family":
            diversity_data.append({
                "name": row[0],
                "count": row[1],  # species count
                "records": row[2]
            })
        else:  # genus
            diversity_data.append({
                "name": row[0],
                "count": row[1],  # record count
                "vernacularName": row[2]
            })

    return {"diversityData": diversity_data}


def parse_taxon_filter(taxon_type: str, taxon_name: str):
    """Parse taxon type and name into SQL WHERE condition and params.
    For species, splits 'Genus epithet' into genus + specificepithet conditions.
    """
    if taxon_type == "species":
        parts = taxon_name.strip().split(' ', 1)
        if len(parts) >= 2:
            return ("genus = :genus_name AND specificepithet = :specific_epithet",
                    {"genus_name": parts[0], "specific_epithet": parts[1]})
        else:
            return "specificepithet = :taxon_name", {"taxon_name": taxon_name}
    else:
        field_mapping = {"family": "family", "genus": "genus"}
        field = field_mapping[taxon_type]
        return f"{field} = :taxon_name", {"taxon_name": taxon_name}


@app.get("/{taxon_type}/{taxon_name}/geographic")
async def get_taxon_geographic_distribution(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """Get geographic distribution statistics for taxonomic unit"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    ck = cache_key("geographic", taxon_type, taxon_name)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    taxon_condition, taxon_params = parse_taxon_filter(taxon_type, taxon_name)

    # Use materialized views for family/genus, raw table for species
    if taxon_type == "family":
        combined_query = text("""
            SELECT country, records, species_count, genera_count,
                   ROUND(geo_records * 100.0 / NULLIF(records, 0), 1) as data_quality
            FROM dbo.family_country_stats
            WHERE family = :taxon_name
        """)
        rows = db.execute(combined_query, {"taxon_name": taxon_name}).fetchall()
    elif taxon_type == "genus":
        combined_query = text("""
            SELECT country, records, species_count, genera_count,
                   ROUND(geo_records * 100.0 / NULLIF(records, 0), 1) as data_quality
            FROM dbo.genus_country_stats
            WHERE genus = :taxon_name
        """)
        rows = db.execute(combined_query, {"taxon_name": taxon_name}).fetchall()
    else:
        # species: small data, use raw table
        combined_query = text(f"""
            SELECT country,
                   COUNT(*) as records,
                   COUNT(DISTINCT COALESCE(validname, scientificname)) as species_count,
                   COUNT(DISTINCT genus) as genera_count,
                   ROUND(
                       COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1
                   ) as data_quality
            FROM dbo.harvestedfn2_fin
            WHERE {taxon_condition} AND country IS NOT NULL AND country != ''
            GROUP BY country
        """)
        rows = db.execute(combined_query, taxon_params).fetchall()

    # Parse all rows once
    country_rows = []
    for row in rows:
        country_rows.append({
            "country": row[0], "records": row[1], "species": row[2],
            "genera": row[3], "dataQuality": float(row[4]) if row[4] else 0
        })

    # 1. Continental distribution (aggregate countries → continents)
    continent_map = {}
    for c in ('United States', 'USA', 'US', 'Canada', 'Mexico'):
        continent_map[c] = 'North America'
    for c in ('United Kingdom', 'UK', 'France', 'Germany', 'Spain', 'Italy', 'Netherlands', 'Sweden', 'Norway'):
        continent_map[c] = 'Europe'
    for c in ('China', 'Japan', 'Australia', 'New Zealand', 'India', 'South Korea', 'Thailand', 'Malaysia', 'Singapore'):
        continent_map[c] = 'Asia-Pacific'
    for c in ('Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela'):
        continent_map[c] = 'South America'
    for c in ('South Africa', 'Kenya', 'Tanzania', 'Egypt', 'Morocco', 'Nigeria'):
        continent_map[c] = 'Africa'

    continent_totals = {}
    for cr in country_rows:
        continent = continent_map.get(cr["country"], "Other Regions")
        continent_totals[continent] = continent_totals.get(continent, 0) + cr["records"]

    grand_total = sum(continent_totals.values()) or 1
    continental_distribution = sorted(
        [{"name": k, "records": v, "percentage": round(v * 100.0 / grand_total, 1)}
         for k, v in continent_totals.items()],
        key=lambda x: x["records"], reverse=True
    )

    # 2. Country distribution (top 20 by records)
    sorted_by_records = sorted(country_rows, key=lambda x: x["records"], reverse=True)
    country_distribution = [
        {"name": c["country"], "records": c["records"], "species": c["species"], "dataQuality": c["dataQuality"]}
        for c in sorted_by_records[:20]
    ]

    # 3. Biodiversity hotspots (top 8 by species, records > 100)
    hotspot_candidates = [c for c in country_rows if c["records"] > 100]
    hotspot_candidates.sort(key=lambda x: (x["species"], x["records"]), reverse=True)
    biodiversity_hotspots = [
        {"name": c["country"], "records": c["records"], "species": c["species"],
         "genera": c["genera"], "dataQuality": c["dataQuality"],
         "description": f"Major biodiversity center with {c['species']} species"}
        for c in hotspot_candidates[:8]
    ]

    result = {
        "continentalDistribution": continental_distribution,
        "countryDistribution": country_distribution,
        "biodiversityHotspots": biodiversity_hotspots
    }
    cache_set(ck, result)
    return result


# Fix SQL syntax error in temporal patterns query

@app.get("/{taxon_type}/{taxon_name}/temporal")
async def get_taxon_temporal_patterns(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """Get temporal pattern statistics for taxonomic unit"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    ck = cache_key("temporal", taxon_type, taxon_name)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    taxon_condition, taxon_params = parse_taxon_filter(taxon_type, taxon_name)

    try:
        # Use materialized views for family/genus, raw table for species
        if taxon_type == "family":
            combined_query = text("""
                SELECT year, month, cnt, has_date
                FROM dbo.family_temporal_stats
                WHERE family = :taxon_name
            """)
            rows = db.execute(combined_query, {"taxon_name": taxon_name}).fetchall()
        elif taxon_type == "genus":
            combined_query = text("""
                SELECT year, month, cnt, has_date
                FROM dbo.genus_temporal_stats
                WHERE genus = :taxon_name
            """)
            rows = db.execute(combined_query, {"taxon_name": taxon_name}).fetchall()
        else:
            # species: small data, use raw table
            combined_query = text(f"""
                SELECT year, month,
                    COUNT(*) as cnt,
                    COUNT(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 1 END) as has_date
                FROM dbo.harvestedfn2_fin
                WHERE {taxon_condition}
                GROUP BY year, month
            """)
            rows = db.execute(combined_query, taxon_params).fetchall()

        # Accumulators
        period_counts = {"Pre-1950": 0, "1950-1990": 0, "1991-2010": 0, "2011-Present": 0}
        season_counts = {
            "Spring (Mar-May)": 0, "Summer (Jun-Aug)": 0,
            "Autumn (Sep-Nov)": 0, "Winter (Dec-Feb)": 0
        }
        season_month_map = {
            3: "Spring (Mar-May)", 4: "Spring (Mar-May)", 5: "Spring (Mar-May)",
            6: "Summer (Jun-Aug)", 7: "Summer (Jun-Aug)", 8: "Summer (Jun-Aug)",
            9: "Autumn (Sep-Nov)", 10: "Autumn (Sep-Nov)", 11: "Autumn (Sep-Nov)",
            12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)",
        }
        recent_counts = {}       # year -> count (2020+)
        yearly_counts = {}       # year -> count (1950+)
        monthly_counts = {}      # month -> count
        total_records = 0
        records_with_year = 0
        records_with_month = 0
        records_with_date = 0
        current_year = datetime.now().year

        for yr, mo, cnt, has_date in rows:
            total_records += cnt
            records_with_date += has_date

            if yr is not None:
                records_with_year += cnt
                if 1800 < yr <= current_year:
                    if yr < 1950:
                        period_counts["Pre-1950"] += cnt
                    elif yr <= 1990:
                        period_counts["1950-1990"] += cnt
                    elif yr <= 2010:
                        period_counts["1991-2010"] += cnt
                    else:
                        period_counts["2011-Present"] += cnt

                    if yr >= 1950:
                        yearly_counts[yr] = yearly_counts.get(yr, 0) + cnt
                    if yr >= 2020:
                        recent_counts[yr] = recent_counts.get(yr, 0) + cnt

            if mo is not None:
                records_with_month += cnt
                if 1 <= mo <= 12:
                    monthly_counts[mo] = monthly_counts.get(mo, 0) + cnt
                    season = season_month_map.get(mo)
                    if season:
                        season_counts[season] += cnt

        # Format results
        period_total = sum(period_counts.values()) or 1
        period_order = ["Pre-1950", "1950-1990", "1991-2010", "2011-Present"]
        historical_periods = [
            {"name": p, "records": period_counts[p],
             "percentage": round(period_counts[p] * 100.0 / period_total, 1)}
            for p in period_order if period_counts[p] > 0
        ]

        seasonal_patterns = sorted(
            [{"name": s, "records": c} for s, c in season_counts.items() if c > 0],
            key=lambda x: x["records"], reverse=True
        )

        recent_activity = [
            {"year": str(yr), "records": c}
            for yr, c in sorted(recent_counts.items(), reverse=True)
        ]

        yearly_trend = [
            {"year": yr, "records": c}
            for yr, c in sorted(yearly_counts.items())
        ]

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_distribution = [
            {"month": mo, "monthName": month_names[mo - 1], "records": monthly_counts[mo]}
            for mo in sorted(monthly_counts.keys())
        ]

        data_coverage = {
            "totalRecords": total_records,
            "recordsWithYear": records_with_year,
            "recordsWithMonth": records_with_month,
            "recordsWithFullDate": records_with_date,
            "yearCoverage": round(records_with_year / total_records * 100, 1) if total_records else 0,
            "monthCoverage": round(records_with_month / total_records * 100, 1) if total_records else 0,
            "dateCoverage": round(records_with_date / total_records * 100, 1) if total_records else 0,
        }

        result = {
            "historicalPeriods": historical_periods,
            "seasonalPatterns": seasonal_patterns,
            "recentActivity": recent_activity,
            "yearlyTrend": yearly_trend,
            "monthlyDistribution": monthly_distribution,
            "dataCoverage": data_coverage,
            "summary": {
                "totalRecordsAnalyzed": sum(period["records"] for period in historical_periods),
                "oldestRecord": yearly_trend[0]["year"] if yearly_trend else None,
                "newestRecord": yearly_trend[-1]["year"] if yearly_trend else None,
                "peakCollectionPeriod": max(historical_periods, key=lambda x: x["records"])[
                    "name"] if historical_periods else None,
                "peakCollectionSeason": max(seasonal_patterns, key=lambda x: x["records"])[
                    "name"] if seasonal_patterns else None
            }
        }
        cache_set(ck, result)
        return result

    except Exception as e:
        print(f"Error in temporal query: {str(e)}")
        # Return empty result instead of throwing exception
        return {
            "historicalPeriods": [],
            "seasonalPatterns": [],
            "recentActivity": [],
            "yearlyTrend": [],
            "monthlyDistribution": [],
            "dataCoverage": {"totalRecords": 0, "yearCoverage": 0, "monthCoverage": 0, "dateCoverage": 0},
            "summary": {}
        }


# 3. Institution API modification - ensure map-required data is returned
@app.get("/{taxon_type}/{taxon_name}/institutions")
async def get_taxon_institutions(
        taxon_type: str,
        taxon_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=100),
        search: Optional[str] = None,
        sort_by: str = "records_desc",
        db: Session = Depends(get_db)
):
    """Get contributing institution statistics for taxonomic unit"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    taxon_condition, taxon_params = parse_taxon_filter(taxon_type, taxon_name)

    # Sort
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "name_asc": "institutioncode ASC",
        "countries_desc": "countries_count DESC"
    }
    order_clause = sort_mapping.get(sort_by, "record_count DESC")

    params = {"limit": per_page, "offset": (page - 1) * per_page}

    search_clause = ""
    if search:
        search_clause = " AND institutioncode ILIKE :search"
        params["search"] = f"%{search}%"

    # Use materialized views for family/genus, raw table for species
    if taxon_type == "family":
        params["taxon_name"] = taxon_name
        query = text(f"""
            WITH grouped AS (
                SELECT institutioncode,
                       total_records as record_count,
                       species_count,
                       genus_count as genera_count,
                       country_count as countries_count,
                       georeferencing_quality,
                       collection_codes,
                       latest_record,
                       primary_country
                FROM dbo.family_institution_stats
                WHERE family = :taxon_name {search_clause}
            ),
            totals AS (
                SELECT COUNT(*) as total_count, GREATEST(SUM(record_count), 1) as total_records FROM grouped
            )
            SELECT g.institutioncode, g.record_count, g.species_count, g.genera_count,
                   g.countries_count, g.georeferencing_quality, g.collection_codes,
                   g.latest_record,
                   ROUND(g.record_count * 100.0 / t.total_records, 1) as percentage,
                   g.primary_country,
                   t.total_count
            FROM grouped g, totals t
            ORDER BY {order_clause}
            LIMIT :limit OFFSET :offset
        """)
    elif taxon_type == "genus":
        params["taxon_name"] = taxon_name
        query = text(f"""
            WITH grouped AS (
                SELECT institutioncode,
                       total_records as record_count,
                       species_count,
                       genus_count as genera_count,
                       country_count as countries_count,
                       georeferencing_quality,
                       collection_codes,
                       latest_record,
                       primary_country
                FROM dbo.genus_institution_stats
                WHERE genus = :taxon_name {search_clause}
            ),
            totals AS (
                SELECT COUNT(*) as total_count, GREATEST(SUM(record_count), 1) as total_records FROM grouped
            )
            SELECT g.institutioncode, g.record_count, g.species_count, g.genera_count,
                   g.countries_count, g.georeferencing_quality, g.collection_codes,
                   g.latest_record,
                   ROUND(g.record_count * 100.0 / t.total_records, 1) as percentage,
                   g.primary_country,
                   t.total_count
            FROM grouped g, totals t
            ORDER BY {order_clause}
            LIMIT :limit OFFSET :offset
        """)
    else:
        # species: small data, use raw table
        params.update(taxon_params)
        if search:
            search_clause_raw = " AND institutioncode ILIKE :search"
        else:
            search_clause_raw = ""
        query = text(f"""
            WITH grouped AS (
                SELECT institutioncode,
                       COUNT(*) as record_count,
                       COUNT(DISTINCT COALESCE(validname, scientificname)) as species_count,
                       COUNT(DISTINCT genus) as genera_count,
                       COUNT(DISTINCT countrycode) as countries_count,
                       ROUND(
                           COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1
                       ) as georeferencing_quality,
                       array_agg(DISTINCT collectioncode ORDER BY collectioncode)
                       FILTER (WHERE collectioncode IS NOT NULL AND collectioncode != '') as collection_codes,
                       MAX(eventdate) as latest_record,
                       MIN(country) as primary_country
                FROM dbo.harvestedfn2_fin
                WHERE {taxon_condition} AND institutioncode IS NOT NULL AND institutioncode != ''
                  {search_clause_raw}
                GROUP BY institutioncode
            ),
            totals AS (
                SELECT COUNT(*) as total_count, GREATEST(SUM(record_count), 1) as total_records FROM grouped
            )
            SELECT g.institutioncode, g.record_count, g.species_count, g.genera_count,
                   g.countries_count, g.georeferencing_quality, g.collection_codes,
                   g.latest_record,
                   ROUND(g.record_count * 100.0 / t.total_records, 1) as percentage,
                   g.primary_country,
                   t.total_count
            FROM grouped g, totals t
            ORDER BY {order_clause}
            LIMIT :limit OFFSET :offset
        """)

    result = db.execute(query, params).fetchall()
    total = result[0][10] if result else 0

    institutions = []
    for row in result:
        # Institution name mapping
        institution_name = {
            'NMNH': 'National Museum of Natural History',
            'AMNH': 'American Museum of Natural History',
            'CAS': 'California Academy of Sciences',
            'BMNH': 'Natural History Museum, London',
            'MNHN': 'Muséum National d\'Histoire Naturelle',
            'KU': 'University of Kansas'
        }.get(row[0], f"{row[0]} Institution")

        institutions.append({
            "code": row[0],
            "name": institution_name,
            "recordCount": row[1],
            "speciesCount": row[2] or 0,
            "generaCount": row[3] or 0,
            "countriesCount": row[4] or 0,
            "geoReferencingQuality": float(row[5]) if row[5] else 0.0,
            "collectionCodes": row[6] if row[6] else [],
            "latestRecord": row[7],
            "percentage": float(row[8]) if row[8] else 0.0,
            "country": row[9] or 'Unknown'  # For map display
        })

    return {
        "data": institutions,
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": (total + per_page - 1) // per_page
    }


# 4. Map data endpoint
@app.get("/{taxon_type}/{taxon_name}/map-data")
async def get_taxon_map_data(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """Get institution distribution data for map display"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    ck = cache_key("map-data", taxon_type, taxon_name)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    taxon_condition, taxon_params = parse_taxon_filter(taxon_type, taxon_name)

    # Use materialized views for family/genus, raw table for species
    if taxon_type == "family":
        query = text("""
            SELECT country, institution_count, records as total_records,
                   ROUND(geo_records * 100.0 / NULLIF(records, 0), 1) as avg_georeferencing,
                   avg_lat, avg_lng, species_count
            FROM dbo.family_country_stats
            WHERE family = :taxon_name AND records > 10
            ORDER BY records DESC
            LIMIT 50
        """)
        result = db.execute(query, {"taxon_name": taxon_name}).fetchall()
    elif taxon_type == "genus":
        query = text("""
            SELECT country, institution_count, records as total_records,
                   ROUND(geo_records * 100.0 / NULLIF(records, 0), 1) as avg_georeferencing,
                   avg_lat, avg_lng, species_count
            FROM dbo.genus_country_stats
            WHERE genus = :taxon_name AND records > 10
            ORDER BY records DESC
            LIMIT 50
        """)
        result = db.execute(query, {"taxon_name": taxon_name}).fetchall()
    else:
        # species: small data, use raw table
        query = text(f"""
            SELECT
                country,
                COUNT(DISTINCT institutioncode) as institution_count,
                COUNT(*) as total_records,
                ROUND(AVG(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL
                               THEN 1.0 ELSE 0.0 END) * 100, 1) as avg_georeferencing,
                AVG(decimallatitude) FILTER (WHERE decimallatitude IS NOT NULL) as avg_lat,
                AVG(decimallongitude) FILTER (WHERE decimallongitude IS NOT NULL) as avg_lng,
                COUNT(DISTINCT COALESCE(validname, scientificname)) as species_count
            FROM dbo.harvestedfn2_fin
            WHERE {taxon_condition}
              AND country IS NOT NULL AND country != ''
              AND institutioncode IS NOT NULL AND institutioncode != ''
            GROUP BY country
            HAVING COUNT(*) > 10
            ORDER BY total_records DESC
            LIMIT 50
        """)
        result = db.execute(query, taxon_params).fetchall()

    map_data = []
    for row in result:
        avg_lat = row[4]
        avg_lng = row[5]
        # Skip countries with no coordinate data at all
        if avg_lat is None or avg_lng is None:
            continue

        map_data.append({
            "country": row[0],
            "institutionCount": row[1],
            "recordCount": row[2],
            "avgGeoreferencing": float(row[3]) if row[3] else 0,
            "lat": round(float(avg_lat), 4),
            "lng": round(float(avg_lng), 4),
            "speciesCount": row[6] or 0
        })

    result = {"mapData": map_data}
    cache_set(ck, result)
    return result


@app.get("/{taxon_type}/{taxon_name}/top-species")
async def get_taxon_top_species(
        taxon_type: str,
        taxon_name: str,
        limit: int = Query(8, ge=1, le=20),
        db: Session = Depends(get_db)
):
    """Get species with most records under taxonomic unit"""

    if taxon_type not in ["family", "genus"]:
        raise HTTPException(status_code=400, detail="Only family and genus support top species")

    field_mapping = {"family": "family", "genus": "genus"}
    field = field_mapping[taxon_type]

    query = text(f"""
        SELECT genus || ' ' || specificepithet as scientificname,
               vernacularname, record_count, institutions_count, countries_count
        FROM dbo.species_stats
        WHERE {field} = :taxon_name
        ORDER BY record_count DESC
        LIMIT :limit
    """)

    result = db.execute(query, {"taxon_name": taxon_name, "limit": limit}).fetchall()

    top_species = []
    for i, row in enumerate(result, 1):
        top_species.append({
            "rank": i,
            "name": row[0],
            "vernacularName": row[1] or "No common name",
            "recordCount": row[2] or 0,
            "institutionsCount": row[3] or 0,
            "countriesCount": row[4] or 0
        })

    return {"topSpecies": top_species}


@app.get("/taxonomy/hierarchy/{taxon_type}/{taxon_name}")
async def get_taxon_hierarchy(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """Get hierarchy information for taxonomic unit"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    if taxon_type == "species":
        # For species, we expect full binomial name (genus + species)
        parts = taxon_name.strip().split(' ', 1)
        if len(parts) >= 2:
            query = text("""
                SELECT kingdom, phylum, "class", "order", family, genus, specificepithet, scientificname
                FROM dbo.harvestedfn2_fin
                WHERE genus = :genus AND specificepithet = :epithet
                LIMIT 1
            """)
            result = db.execute(query, {"genus": parts[0], "epithet": parts[1]}).fetchone()
        else:
            query = text("""
                SELECT kingdom, phylum, "class", "order", family, genus, specificepithet, scientificname
                FROM dbo.harvestedfn2_fin
                WHERE scientificname = :taxon_name
                LIMIT 1
            """)
            result = db.execute(query, {"taxon_name": taxon_name}).fetchone()
    else:
        field_mapping = {"family": "family", "genus": "genus"}
        field = field_mapping[taxon_type]
        query = text(f"""
            SELECT kingdom, phylum, "class", "order", family, genus, specificepithet, scientificname
            FROM dbo.harvestedfn2_fin
            WHERE {field} = :taxon_name
            LIMIT 1
        """)
        result = db.execute(query, {"taxon_name": taxon_name}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail=f"{taxon_type.title()} not found")

    return {
        "kingdom": result[0],
        "phylum": result[1],
        "class": result[2],
        "order": result[3],
        "family": result[4],
        "genus": result[5],
        "species": result[6] if taxon_type == "species" else None
    }


# Taxonomy hierarchy endpoints

@app.get("/{taxon_type}/{taxon_name}/institution-coverage")
async def get_taxon_institution_coverage(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """Get institution coverage analysis for taxonomic unit - dynamically calculated"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    ck = cache_key("institution-coverage", taxon_type, taxon_name)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    taxon_condition, taxon_params = parse_taxon_filter(taxon_type, taxon_name)

    try:
        # Use materialized views for family/genus, raw table for species
        if taxon_type == "family":
            combined_query = text("""
                SELECT institutioncode, total_records, country_count,
                       1 as family_count, genus_count, species_count,
                       geo_records, date_records, taxonomy_records, collection_records,
                       earliest_year, latest_year, recent_records, decade_records
                FROM dbo.family_institution_stats
                WHERE family = :taxon_name
            """)
            rows = db.execute(combined_query, {"taxon_name": taxon_name}).fetchall()
        elif taxon_type == "genus":
            combined_query = text("""
                SELECT institutioncode, total_records, country_count,
                       family_count, genus_count, species_count,
                       geo_records, date_records, taxonomy_records, collection_records,
                       earliest_year, latest_year, recent_records, decade_records
                FROM dbo.genus_institution_stats
                WHERE genus = :taxon_name
            """)
            rows = db.execute(combined_query, {"taxon_name": taxon_name}).fetchall()
        else:
            # species: small data, use raw table
            combined_query = text(f"""
                SELECT institutioncode,
                       COUNT(*) as total_records,
                       COUNT(DISTINCT countrycode) as country_count,
                       COUNT(DISTINCT family) as family_count,
                       COUNT(DISTINCT genus) as genus_count,
                       COUNT(DISTINCT COALESCE(validname, scientificname)) as species_count,
                       COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) as geo_records,
                       COUNT(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 1 END) as date_records,
                       COUNT(CASE WHEN scientificname IS NOT NULL AND family IS NOT NULL AND genus IS NOT NULL THEN 1 END) as taxonomy_records,
                       COUNT(CASE WHEN collectioncode IS NOT NULL AND collectioncode != '' THEN 1 END) as collection_records,
                       MIN(CASE WHEN year IS NOT NULL AND year > 1800 THEN year END) as earliest_year,
                       MAX(CASE WHEN year IS NOT NULL AND year > 1800 THEN year END) as latest_year,
                       COUNT(CASE WHEN year >= 2020 THEN 1 END) as recent_records,
                       COUNT(CASE WHEN year BETWEEN 2010 AND 2019 THEN 1 END) as decade_records
                FROM dbo.harvestedfn2_fin
                WHERE {taxon_condition}
                  AND institutioncode IS NOT NULL AND institutioncode != ''
                GROUP BY institutioncode
            """)
            rows = db.execute(combined_query, taxon_params).fetchall()

        # Accumulators
        geo_global = 0; geo_regional = 0; geo_local = 0
        tax_family_spec = 0; tax_genus_spec = 0; tax_regional = 0
        q_high = 0; q_good = 0; q_improving = 0; q_scores = []
        sc_major = 0; sc_substantial = 0; sc_moderate = 0; sc_small = 0; sc_total_records = 0
        type_agg = {}  # type -> {count, total_records, records_list}
        ta_recent = 0; ta_historical = 0; ta_inactive = 0; ta_spans = []
        total_institutions = len(rows)

        def _classify_type(code):
            cu = code.upper()
            if any(p in cu for p in ('MNH', 'MUSEUM', 'MUS')):
                return 'museum'
            if any(p in cu for p in ('UNIV', 'UNI', 'COLL')):
                return 'university'
            if any(p in cu for p in ('GOV', 'USGS', 'NOAA')):
                return 'government'
            if any(p in cu for p in ('ACAD', 'INST')):
                return 'research'
            return 'private'

        for row in rows:
            code, total, countries, families, genera, species, \
                geo_rec, date_rec, tax_rec, coll_rec, \
                earliest, latest, recent, decade = row

            # 1. Geographic coverage
            if countries >= 20:
                geo_global += 1
            elif countries >= 5:
                geo_regional += 1
            else:
                geo_local += 1

            # 2. Taxonomic specialization
            if total >= 1000:
                tax_family_spec += 1
            if genera <= 5 and total >= 100:
                tax_genus_spec += 1
            if species >= 10 and genera >= 3:
                tax_regional += 1

            # 3. Data quality
            if total > 0:
                geo_q = geo_rec * 100.0 / total
                date_q = date_rec * 100.0 / total
                tax_q = tax_rec * 100.0 / total
                coll_q = coll_rec * 100.0 / total
                overall = (geo_q + date_q + tax_q + coll_q) / 4
                q_scores.append(overall)
                if overall > 95:
                    q_high += 1
                elif overall >= 85:
                    q_good += 1
                else:
                    q_improving += 1

            # 4. Collection scale
            sc_total_records += total
            if total > 10000:
                sc_major += 1
            elif total >= 1000:
                sc_substantial += 1
            elif total >= 100:
                sc_moderate += 1
            else:
                sc_small += 1

            # 5. Institution type
            itype = _classify_type(code)
            if itype not in type_agg:
                type_agg[itype] = {"count": 0, "total_records": 0, "records_list": []}
            type_agg[itype]["count"] += 1
            type_agg[itype]["total_records"] += total
            type_agg[itype]["records_list"].append(total)

            # 6. Temporal activity
            if recent > 0:
                ta_recent += 1
            elif decade > 0:
                ta_historical += 1
            else:
                ta_inactive += 1
            if latest is not None and earliest is not None:
                ta_spans.append(latest - earliest)

        # Build institution types breakdown
        institution_types_breakdown = {}
        for itype, data in type_agg.items():
            institution_types_breakdown[itype] = {
                "count": data["count"],
                "totalRecords": data["total_records"],
                "avgRecordsPerInstitution": round(data["total_records"] / data["count"]) if data["count"] else 0
            }

        avg_quality = round(sum(q_scores) / len(q_scores), 1) if q_scores else 0.0
        avg_span = round(sum(ta_spans) / len(ta_spans), 1) if ta_spans else 0.0

        result = {
            "geographicCoverage": {
                "globalCoverage": geo_global,
                "regionalSpecialists": geo_regional,
                "localCollections": geo_local,
                "totalInstitutions": total_institutions
            },
            "taxonomicSpecialization": {
                "familySpecialists": tax_family_spec,
                "genusSpecialists": tax_genus_spec,
                "regionalFaunaFocus": tax_regional,
                "totalInstitutions": total_institutions
            },
            "dataQualityLeaders": {
                "highQuality": q_high,
                "goodQuality": q_good,
                "improvingQuality": q_improving,
                "averageQuality": avg_quality,
                "totalInstitutions": total_institutions
            },
            "collectionScale": {
                "majorCollections": sc_major,
                "substantialCollections": sc_substantial,
                "moderateCollections": sc_moderate,
                "smallCollections": sc_small,
                "totalRecords": sc_total_records,
                "totalInstitutions": total_institutions
            },
            "institutionTypes": institution_types_breakdown,
            "temporalActivity": {
                "recentlyActive": ta_recent,
                "historicallyActive": ta_historical,
                "inactive": ta_inactive,
                "avgCollectionSpan": avg_span,
                "totalInstitutions": total_institutions
            }
        }
        cache_set(ck, result)
        return result

    except Exception as e:
        print(f"Error in institution coverage analysis: {str(e)}")
        # Return default values instead of throwing exception
        return {
            "geographicCoverage": {
                "globalCoverage": 0,
                "regionalSpecialists": 0,
                "localCollections": 0,
                "totalInstitutions": 0
            },
            "taxonomicSpecialization": {
                "familySpecialists": 0,
                "genusSpecialists": 0,
                "regionalFaunaFocus": 0,
                "totalInstitutions": 0
            },
            "dataQualityLeaders": {
                "highQuality": 0,
                "goodQuality": 0,
                "improvingQuality": 0,
                "averageQuality": 0.0,
                "totalInstitutions": 0
            },
            "collectionScale": {
                "majorCollections": 0,
                "substantialCollections": 0,
                "moderateCollections": 0,
                "smallCollections": 0,
                "totalRecords": 0,
                "totalInstitutions": 0
            },
            "institutionTypes": {},
            "temporalActivity": {
                "recentlyActive": 0,
                "historicallyActive": 0,
                "inactive": 0,
                "avgCollectionSpan": 0.0,
                "totalInstitutions": 0
            }
        }


# ============== Record Flags ==============

class RecordFlagCreate(BaseModel):
    """Request model for creating a record flag"""
    es_document_id: Optional[str] = None
    catalog_number: Optional[str] = None
    institution_code: str
    scientific_name: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=2000)

class RecordFlagProviderReply(BaseModel):
    """Request model for provider responding to a flag"""
    response: str = Field(..., min_length=1, max_length=2000)
    new_status: str = Field(default="resolved")

class RecordFlagReplyCreate(BaseModel):
    """Request model for posting a reply in a flag conversation"""
    message: str = Field(..., min_length=1, max_length=2000)


@app.post("/flags/record", tags=["Record Flags"], summary="Create a record flag")
async def create_record_flag(
    flag_data: RecordFlagCreate,
    db: Session = Depends(get_db),
    user: UserInfo = Depends(get_current_user)
):
    """User flags a search result record and sends a message to the data provider."""
    try:
        # Check for duplicate active flag
        dup_query = text("""
            SELECT id FROM record_flags
            WHERE user_id = :user_id
              AND es_document_id = :es_document_id
              AND institution_code = :institution_code
              AND status = 'open'
            LIMIT 1
        """)
        existing = db.execute(dup_query, {
            "user_id": user.user_id,
            "es_document_id": flag_data.es_document_id or "",
            "institution_code": flag_data.institution_code
        }).fetchone()

        if existing:
            raise HTTPException(
                status_code=409,
                detail="You already have an open flag for this record"
            )

        insert_query = text("""
            INSERT INTO record_flags
                (user_id, username, user_email, es_document_id, catalog_number,
                 institution_code, scientific_name, message)
            VALUES
                (:user_id, :username, :user_email, :es_document_id, :catalog_number,
                 :institution_code, :scientific_name, :message)
            RETURNING id, user_id, username, user_email, es_document_id, catalog_number,
                      institution_code, scientific_name, message, provider_response,
                      provider_user_id, provider_username, responded_at, status,
                      created_at, updated_at
        """)
        result = db.execute(insert_query, {
            "user_id": user.user_id,
            "username": user.username,
            "user_email": user.email,
            "es_document_id": flag_data.es_document_id,
            "catalog_number": flag_data.catalog_number,
            "institution_code": flag_data.institution_code,
            "scientific_name": flag_data.scientific_name,
            "message": flag_data.message
        })
        db.commit()
        row = result.fetchone()
        return dict(row._mapping)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create flag: {str(e)}")


@app.get("/flags/record/user", tags=["Record Flags"], summary="Get user's own flags")
async def get_user_record_flags(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: UserInfo = Depends(get_current_user)
):
    """Get the current user's record flags (for User Dashboard)."""
    try:
        where = "WHERE user_id = :user_id"
        params = {"user_id": user.user_id, "limit": limit, "offset": offset}

        if status:
            where += " AND status = :status"
            params["status"] = status

        count_query = text(f"SELECT COUNT(*) FROM record_flags {where}")
        total = db.execute(count_query, params).scalar()

        data_query = text(f"""
            SELECT id, user_id, username, user_email, es_document_id, catalog_number,
                   institution_code, scientific_name, message, provider_response,
                   provider_user_id, provider_username, responded_at, status,
                   created_at, updated_at
            FROM record_flags {where}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        rows = db.execute(data_query, params).fetchall()
        flags = [dict(row._mapping) for row in rows]

        return {"total": total, "flags": flags}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch flags: {str(e)}")


@app.get("/flags/record/provider", tags=["Record Flags"], summary="Get flags for a provider")
async def get_provider_record_flags(
    institution_code: str = Query(..., description="Institution code"),
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: UserInfo = Depends(get_current_user)
):
    """Get record flags for a specific institution (for Provider Dashboard)."""
    try:
        where = "WHERE institution_code = :institution_code"
        params = {"institution_code": institution_code, "limit": limit, "offset": offset}

        if status:
            where += " AND status = :status"
            params["status"] = status

        count_query = text(f"SELECT COUNT(*) FROM record_flags {where}")
        total = db.execute(count_query, params).scalar()

        data_query = text(f"""
            SELECT id, user_id, username, user_email, es_document_id, catalog_number,
                   institution_code, scientific_name, message, provider_response,
                   provider_user_id, provider_username, responded_at, status,
                   created_at, updated_at
            FROM record_flags {where}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        rows = db.execute(data_query, params).fetchall()
        flags = [dict(row._mapping) for row in rows]

        return {"total": total, "flags": flags}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch flags: {str(e)}")


@app.put("/flags/record/{flag_id}/respond", tags=["Record Flags"], summary="Provider responds to a flag")
async def respond_to_record_flag(
    flag_id: int,
    reply: RecordFlagProviderReply,
    db: Session = Depends(get_db),
    user: UserInfo = Depends(get_current_user)
):
    """Provider responds to a record flag with a message and status update."""
    try:
        # Verify flag exists
        check_query = text("SELECT id, institution_code FROM record_flags WHERE id = :id")
        flag = db.execute(check_query, {"id": flag_id}).fetchone()

        if not flag:
            raise HTTPException(status_code=404, detail="Flag not found")

        update_query = text("""
            UPDATE record_flags
            SET provider_response = :response,
                provider_user_id = :provider_user_id,
                provider_username = :provider_username,
                responded_at = CURRENT_TIMESTAMP,
                status = :new_status
            WHERE id = :id
            RETURNING id, user_id, username, user_email, es_document_id, catalog_number,
                      institution_code, scientific_name, message, provider_response,
                      provider_user_id, provider_username, responded_at, status,
                      created_at, updated_at
        """)
        result = db.execute(update_query, {
            "id": flag_id,
            "response": reply.response,
            "provider_user_id": user.user_id,
            "provider_username": user.username,
            "new_status": reply.new_status
        })
        db.commit()
        row = result.fetchone()
        return dict(row._mapping)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to respond to flag: {str(e)}")


@app.delete("/flags/record/{flag_id}", tags=["Record Flags"], summary="Delete a record flag")
async def delete_record_flag(
    flag_id: int,
    db: Session = Depends(get_db),
    user: UserInfo = Depends(get_current_user)
):
    """User deletes their own open flag."""
    try:
        check_query = text("SELECT id, user_id, status FROM record_flags WHERE id = :id")
        flag = db.execute(check_query, {"id": flag_id}).fetchone()

        if not flag:
            raise HTTPException(status_code=404, detail="Flag not found")

        flag_dict = dict(flag._mapping)
        if flag_dict["user_id"] != user.user_id:
            raise HTTPException(status_code=403, detail="You can only delete your own flags")

        if flag_dict["status"] != "open":
            raise HTTPException(status_code=400, detail="Can only delete flags with 'open' status")

        delete_query = text("DELETE FROM record_flags WHERE id = :id")
        db.execute(delete_query, {"id": flag_id})
        db.commit()

        return {"success": True, "message": "Flag deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete flag: {str(e)}")


@app.get("/flags/record/{flag_id}/replies", tags=["Record Flags"], summary="Get replies for a flag")
async def get_flag_replies(
    flag_id: int,
    db: Session = Depends(get_db),
    user: UserInfo = Depends(get_current_user)
):
    """Get all replies in a flag conversation thread."""
    try:
        # Verify flag exists and user has access
        flag_query = text("SELECT id, user_id, institution_code FROM record_flags WHERE id = :id")
        flag = db.execute(flag_query, {"id": flag_id}).fetchone()
        if not flag:
            raise HTTPException(status_code=404, detail="Flag not found")

        replies_query = text("""
            SELECT id, flag_id, user_id, username, user_role, message, created_at
            FROM record_flag_replies
            WHERE flag_id = :flag_id
            ORDER BY created_at ASC
        """)
        rows = db.execute(replies_query, {"flag_id": flag_id}).fetchall()
        replies = [dict(row._mapping) for row in rows]
        return {"replies": replies}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch replies: {str(e)}")


@app.post("/flags/record/{flag_id}/replies", tags=["Record Flags"], summary="Post a reply to a flag")
async def create_flag_reply(
    flag_id: int,
    reply_data: RecordFlagReplyCreate,
    db: Session = Depends(get_db),
    user: UserInfo = Depends(get_current_user)
):
    """Both user and provider can post replies to a flag conversation."""
    try:
        # Verify flag exists
        flag_query = text("SELECT id, user_id, institution_code, status FROM record_flags WHERE id = :id")
        flag = db.execute(flag_query, {"id": flag_id}).fetchone()
        if not flag:
            raise HTTPException(status_code=404, detail="Flag not found")

        flag_dict = dict(flag._mapping)

        # Determine role: flag creator = 'user', anyone else = 'provider'
        user_role = "user" if flag_dict["user_id"] == user.user_id else "provider"

        insert_query = text("""
            INSERT INTO record_flag_replies (flag_id, user_id, username, user_role, message)
            VALUES (:flag_id, :user_id, :username, :user_role, :message)
            RETURNING id, flag_id, user_id, username, user_role, message, created_at
        """)
        result = db.execute(insert_query, {
            "flag_id": flag_id,
            "user_id": user.user_id,
            "username": user.username,
            "user_role": user_role,
            "message": reply_data.message
        })

        # Update flag status to in_progress if it was open
        if flag_dict["status"] == "open":
            db.execute(
                text("UPDATE record_flags SET status = 'in_progress' WHERE id = :id"),
                {"id": flag_id}
            )

        db.commit()
        row = result.fetchone()
        return dict(row._mapping)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create reply: {str(e)}")