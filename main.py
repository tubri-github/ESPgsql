from datetime import datetime
from collections import defaultdict
import math

from fastapi import FastAPI, Query, HTTPException, Depends


# ============== 地理空间工具函数 ==============

def point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """
    射线法判断点是否在多边形内
    polygon: [[lon, lat], [lon, lat], ...] 多边形顶点坐标（需要闭合，首尾相同）
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]  # lon, lat
        xj, yj = polygon[j][0], polygon[j][1]

        # 检查点是否在边的y范围内，并计算交点
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def point_in_circle(lat: float, lon: float, center_lat: float, center_lon: float, radius_meters: float) -> bool:
    """
    判断点是否在圆形范围内
    使用 Haversine 公式计算距离
    """
    R = 6371000  # 地球半径（米）

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
    从 geo_filter 计算 bounding box
    返回 {"min_lat": ..., "max_lat": ..., "min_lon": ..., "max_lon": ...}
    """
    if geo_filter.type == "circle":
        # 圆形：从圆心和半径计算近似的 bounding box
        center_lon, center_lat = geo_filter.coordinates[0]
        radius_meters = geo_filter.radius or 10000  # 默认10km

        # 1度纬度约111km，1度经度约111km * cos(lat)
        lat_offset = radius_meters / 111000
        lon_offset = radius_meters / (111000 * math.cos(math.radians(center_lat)))

        return {
            "min_lat": center_lat - lat_offset,
            "max_lat": center_lat + lat_offset,
            "min_lon": center_lon - lon_offset,
            "max_lon": center_lon + lon_offset
        }
    else:
        # 多边形或矩形：取所有顶点的极值
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
    对搜索结果进行精确的地理筛选
    hits: ES返回的 _source 列表
    """
    if not geo_filter:
        return hits

    filtered = []
    for hit in hits:
        lat = hit.get("Latitude")
        lon = hit.get("Longitude")

        # 跳过没有坐标的记录
        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            continue

        # 根据类型判断
        if geo_filter.type == "circle":
            center_lon, center_lat = geo_filter.coordinates[0]
            radius = geo_filter.radius or 10000
            if point_in_circle(lat, lon, center_lat, center_lon, radius):
                filtered.append(hit)
        else:
            # polygon 或 rectangle
            polygon = geo_filter.coordinates
            # 确保多边形闭合
            if polygon[0] != polygon[-1]:
                polygon = polygon + [polygon[0]]
            if point_in_polygon(lat, lon, polygon):
                filtered.append(hit)

    return filtered

# ============== 地理空间工具函数结束 ==============
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
from models.models import (
    HarvestedRecord, FamilyResponse, GenusResponse, SpeciesResponse,
    InstitutionResponse, RecordResponse, TaxonomyStatsResponse,
    InstitutionStatsResponse, PaginatedResponse,
    TaxonomyFilterParams, InstitutionFilterParams, RecordFilterParams
)

app = FastAPI(
    title="FishNet 2 API",
    description="This API is a comprehensive backend API used to access the Fishnet2 original database, Fishnet2 database, and Elasticsearch API. Currently, it provides functionality for basic search and aggregation as well as advanced search aggregation; access to the Elasticsearch API for creating, editing, and rebuilding Elasticsearch indices; and basic querying of occurrence data.",
    version="Beta 1.2",
    docs_url="/docs",
    redoc_url="/redoc",)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    # allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
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
es = Elasticsearch(settings.ES_URL)

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
    """支持多选和范围筛选的值类型"""
    values: Optional[List[str]] = None  # 多选值列表
    range_min: Optional[float] = None   # 范围最小值
    range_max: Optional[float] = None   # 范围最大值

class GeoFilter(BaseModel):
    """地理筛选条件"""
    type: str = "polygon"  # polygon, rectangle, circle
    coordinates: List[List[float]]  # [[lon, lat], [lon, lat], ...] 多边形顶点坐标
    # circle 类型时: coordinates = [[center_lon, center_lat]], radius 必填
    radius: Optional[float] = None  # 圆形半径（米），仅 circle 类型使用

class SearchPayload(BaseModel):
    term: Optional[str] = None  # term 为简单搜索关键词，可选
    conditions: Optional[List[Condition]] = None  # conditions 为高级搜索条件，可选
    filter_conditions: Optional[Dict[str, str]] = None  # 旧版：单选筛选条件
    multi_filters: Optional[Dict[str, List[str]]] = None  # 新增：多选筛选条件
    range_filters: Optional[Dict[str, Dict[str, float]]] = None  # 新增：范围筛选条件 {"YearCollected": {"min": 1900, "max": 2024}}
    geo_filter: Optional[GeoFilter] = None  # 新增：地理筛选条件
    page: int = 0
    page_size: int = 10

def apply_sort(query, sort_by: str, model_class):
    """应用排序"""
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
    """应用记录数过滤"""
    if record_count == "high":
        return query.having(text(f"{count_field} > 1000"))
    elif record_count == "medium":
        return query.having(text(f"{count_field} BETWEEN 100 AND 1000"))
    elif record_count == "low":
        return query.having(text(f"{count_field} < 100"))
    return query

def build_es_query(conditions: List[Condition]) -> dict:
    """
    根据用户条件构建 Elasticsearch 查询 DSL。
    """
    must_clauses = []

    for condition in conditions:
        # 构建主条件的查询
        main_query = build_field_query(condition.field, condition.operator, condition.value)

        # 如果存在 OR 子条件，构建 should 查询
        if condition.orConditions:
            should_clauses = [
                build_field_query(condition.field, or_cond.operator, or_cond.value)
                for or_cond in condition.orConditions
            ]
            main_query = {
                "bool": {
                    "should": [main_query, *should_clauses],
                    "minimum_should_match": 1,  # 至少满足一个条件
                }
            }

        # 添加主条件到 must 子句
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
    根据字段和操作符生成 Elasticsearch 查询。
    - 数字字段支持范围查询（>, <, >=, <= 等）。
    - 字符串字段支持精确匹配、模糊匹配、通配符、前缀等。
    - 枚举字段支持多选。
    """
    # 如果未提供操作符，根据字段类型设置默认操作符
    if not operator: # none and ''
        # 默认逻辑：根据值类型判断操作符
        if isinstance(value, str):
            operator = "contains"  # 字符串默认为模糊匹配
        elif isinstance(value, bool):
            operator = "="  # 布尔值默认为精确匹配
        elif isinstance(value, list):
            operator = "in"  # 列表（枚举值）默认为多选
        else:
            raise ValueError(f"Unsupported type for field {field} with value {value}")

    # 根据操作符生成对应的查询
    if operator == "=":
        # 精确匹配 (使用 keyword 字段)
        return {"term": {f"{field}.keyword": value}}
    elif operator == "!=":
        # 排除匹配
        return {"bool": {"must_not": {"term": {f"{field}.keyword": value}}}}
    elif operator in [">", "<", ">=", "<="]:
        # 范围查询，适用于数字类型
        range_operators = {
            ">": "gt",
            "<": "lt",
            ">=": "gte",
            "<=": "lte",
        }
        return {"range": {field: {range_operators[operator]: value}}}
    elif operator == "contains":
        # 模糊匹配 (分词匹配)，适用于字符串类型
        return {"match": {field: value}}
    elif operator == "phrase":
        # 短语匹配 (精确短语)
        return {"match_phrase": {field: value}}
    elif operator == "prefix":
        # 前缀匹配 (以...开头)
        return {"prefix": {f"{field}.keyword": {"value": value, "case_insensitive": True}}}
    elif operator == "wildcard":
        # 通配符匹配 (* 和 ?)
        # 自动添加通配符如果用户没有提供
        wildcard_value = value if '*' in value or '?' in value else f"*{value}*"
        return {"wildcard": {f"{field}.keyword": {"value": wildcard_value, "case_insensitive": True}}}
    elif operator == "fuzzy":
        # 模糊匹配 (拼写纠错)
        return {"fuzzy": {f"{field}.keyword": {"value": value, "fuzziness": "AUTO"}}}
    elif operator == "regexp":
        # 正则表达式匹配
        return {"regexp": {f"{field}.keyword": {"value": value, "case_insensitive": True}}}
    elif operator == "in":
        # 多选（适用于枚举或多值字段）
        return {"terms": {f"{field}.keyword": value if isinstance(value, list) else [value]}}
    else:
        raise ValueError(f"Unsupported operator: {operator}")


@app.post("/aggregation",tags=["Search"], summary="Simple Aggregation")
async def aggregation(payload: Optional[SearchPayload] = None, term: Optional[str] = None):
    """
    Aggregated Statistics API: Supports Simple Search (term)
    """
    try:
        # 动态构建查询
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
            es_query["query"] = {
                "multi_match": {
                    "query": term,
                    "fields": ["ScientificName", "Family", "InstitutionCode", "CatalogNumber", "CollectionCode"]
                }
            }
        elif payload and payload.conditions:
            es_query["query"] = build_es_query(payload.conditions)["query"]
        else:
            # 如果没有 term 或 conditions，则返回空聚合
            es_query["query"] = {"match_all": {}}

        # 执行查询
        response = es.search(index="mrservice_index", body=es_query)

        # 解析结果
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
        # 构建查询
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

        # 根据简单搜索或高级搜索条件动态构建查询部分
        if payload.term:
            es_query["query"] = {
                "multi_match": {
                    "query": payload.term,
                    "fields": [
                         "ScientificName", "Family",  # Taxon 相关字段
                        "InstitutionCode", "CollectionCode",  # Occurrence 相关字段
                        "Country", "StateProvince", "County"  # Location 相关字段
                    ]
                }
            }
        elif payload and payload.conditions:
            es_query["query"] = build_es_query(payload.conditions)["query"]
        else:
            es_query["query"] = {"match_all": {}}

        # 执行查询
        response = es.search(index="mrservice_full_index", body=es_query)

        # 解析结果
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
    page: int = Query(0, ge=0),        # 页码，默认为 0
    page_size: int = Query(10, gt=0)  # 每页条数，默认为 10
):
    # Elasticsearch 查询主体
    query = {
        "query": {
            "multi_match": {
                "query": term,
                "fields": ["ScientificName", "Family", "InstitutionCode", "CatalogNumber", "CollectionCode"]
            }
        },
        "size": page_size,          # 分页大小
        "from": page * page_size,   # 分页起始位置
        "track_total_hits": True,
        "_source": ["ScientificName", "Family", "InstitutionCode", "CatalogNumber", "CollectionCode"],  # 返回字段
        "aggregations": {           # 聚合统计
            "scientificname_count": {
                "terms": {
                    "field": "ScientificName.keyword",
                    "size": 10
                }
            },
            "family_count": {
                "terms": {
                    "field": "Family.keyword",
                    "size": 10
                }
            }
        }
    }

    try:
        # 查询 Elasticsearch
        response = es.search(index="mrservice_full_index", body=query)

        # 解析分页结果
        hits = response["hits"]["hits"]
        documents = [
            {
                "id": hit["_id"],  # Elasticsearch 文档 ID
                **hit["_source"]   # 文档内容
            }
            for hit in hits
        ]

        # 解析聚合数据
        aggregations = response.get("aggregations", {})
        scientificname_buckets = aggregations.get("scientificname_count", {}).get("buckets", [])
        family_buckets = aggregations.get("family_count", {}).get("buckets", [])

        # 获取 occurrence 统计
        occurrence_count = response["hits"]["total"]["value"]

        # 返回结果
        return {
            "documents": documents,                    # 当前分页的文档信息
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

        field_types = get_field_types(es, "mrservice_full_index")

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
                fields = text_fields
            elif isinstance(payload.term, (int, float)):
                fields = numeric_fields
            elif isinstance(payload.term, (datetime, str)):  # Date input
                fields = date_fields
            else:
                raise HTTPException(status_code=400, detail="Unsupported term type.")
            # Simple search logic
            es_query["query"] = {
                "multi_match": {
                    "query": payload.term,
                    "fields": fields
                }
            }
        elif payload.conditions:
            # 高级搜索逻辑
            es_query["query"] = build_es_query(payload.conditions)["query"]
        else:
            raise HTTPException(status_code=400, detail="Either 'term' or 'conditions' must be provided.")

        # Execute query with filters (支持旧版单选、新版多选和范围筛选)
        filter_clauses = []

        # 旧版单选筛选条件
        if filter_conditions:
            for key, value in filter_conditions.items():
                # For normalized fields, search all variations
                if key in ["Country", "StateProvince"]:
                    variations = get_all_variations(value, key)
                    filter_clauses.append({"terms": {f"{key}.keyword": variations}})
                else:
                    filter_clauses.append({"term": {f"{key}.keyword": value}})

        # 新版多选筛选条件
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

        # 范围筛选条件
        range_filters = payload.range_filters or {}
        for key, range_values in range_filters.items():
            range_query = {}
            if "min" in range_values and range_values["min"] is not None:
                range_query["gte"] = range_values["min"]
            if "max" in range_values and range_values["max"] is not None:
                range_query["lte"] = range_values["max"]
            if range_query:
                filter_clauses.append({"range": {key: range_query}})

        # 地理筛选条件 - 先用 bounding box 快速过滤
        geo_filter = payload.geo_filter
        if geo_filter and geo_filter.coordinates:
            bbox = calculate_bounding_box(geo_filter)
            # 添加 bounding box 范围查询（快速筛选）
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
        # response = es.search(index="mrservice_index", body=es_query)
        response = es.search(index="mrservice_full_index", body=es_query)
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
                    "cardinality": {"field": "Country.keyword"}
                }
            }
        }

        stats_response = es.search(index="mrservice_full_index", body=stats_query)
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

        chart_response = es.search(index="mrservice_full_index", body=chart_query)
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

        # 提取 hits 数据
        hits_data = [hit["_source"] for hit in hits]

        # 如果有 geo_filter，进行精确的多边形/圆形过滤
        if geo_filter and geo_filter.coordinates:
            hits_data = filter_by_geo(hits_data, geo_filter)

        return {
            "hits": hits_data,
            "total": total,  # 注意：这是 bounding box 筛选后的总数，精确过滤后可能略少
            "stats": stats,
            "charts": {
                "timeline": timeline_data,
                "diversity": diversity_data,
                "mapPoints": map_points
            },
            "geo_filtered": geo_filter is not None  # 标记是否进行了地理筛选
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CustomAggregationPayload(BaseModel):
    field: str  # 要聚合的字段名
    size: int = 10  # 返回的桶数量
    term: Optional[str] = None  # 可选的搜索词
    conditions: Optional[List[Condition]] = None  # 可选的高级搜索条件


@app.post("/custom-aggregation", tags=["Search"], summary="Custom Field Aggregation")
async def custom_aggregation(payload: CustomAggregationPayload):
    """
    自定义字段聚合API：用于生成用户选择字段的统计图表
    支持的字段：ScientificName, Family, Genus, Order, Class, Country, StateProvince,
    County, InstitutionCode, CollectionCode, YearCollected, BasisOfRecord, RecordedBy
    """
    try:
        # 验证字段名
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

        # 构建基础查询
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

        # 确定字段类型 - 数字字段不需要 .keyword
        numeric_fields = ['YearCollected']
        field_name = payload.field if payload.field in numeric_fields else f"{payload.field}.keyword"

        # 构建聚合查询
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

        # 执行查询
        response = es.search(index="mrservice_full_index", body=es_query)

        # 解析结果
        aggs = response.get("aggregations", {})
        buckets = aggs.get("field_stats", {}).get("buckets", [])
        total_unique = aggs.get("total_unique", {}).get("value", 0)
        total_docs = response.get("hits", {}).get("total", {}).get("value", 0)

        # 过滤空值
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
    id: int,  # 接收 RESTful 路径参数 id
    db: Session = Depends(get_db)  # 获取数据库会话
):
    # SQL 查询
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # 执行查询
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # 转换为字典返回
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/occurrences",tags=["Occurrence"],summary="Get Occurrences")
async def get_occurrence(
    id: int,  # 接收 RESTful 路径参数 id
    db: Session = Depends(get_db)  # 获取数据库会话
):
    # SQL 查询
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # 执行查询
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # 转换为字典返回
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/occurrences/{term}",tags=["Occurrence"],summary="Get Occurrences by user defined terms")
async def get_occurrence(
    id: int,  # 接收 RESTful 路径参数 id
    db: Session = Depends(get_db)  # 获取数据库会话
):
    # SQL 查询
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # 执行查询
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # 转换为字典返回
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/ESIndexCon",tags=["ElasticSearch"],summary="Get ES connection")
async def get_occurrence(
    id: int,  # 接收 RESTful 路径参数 id
    db: Session = Depends(get_db)  # 获取数据库会话
):
    # SQL 查询
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # 执行查询
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # 转换为字典返回
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/ESIndex",tags=["ElasticSearch"],summary="Get ES index ")
async def get_occurrence(
    id: int,  # 接收 RESTful 路径参数 id
    db: Session = Depends(get_db)  # 获取数据库会话
):
    # SQL 查询
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # 执行查询
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # 转换为字典返回
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/ESIndexRebuild",tags=["ElasticSearch"],summary="Rebuild ES Index")
async def get_occurrence(
    id: int,  # 接收 RESTful 路径参数 id
    db: Session = Depends(get_db)  # 获取数据库会话
):
    # SQL 查询
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # 执行查询
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # 转换为字典返回
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.get("/ESIndexBuild",tags=["ElasticSearch"],summary="Build ES Index")
async def get_occurrence(
    id: int,  # 接收 RESTful 路径参数 id
    db: Session = Depends(get_db)  # 获取数据库会话
):
    # SQL 查询
    query = text("""
        SELECT * FROM  dbo."MRService"
        WHERE "ID" = :id
    """)

    try:
        # 执行查询
        result = db.execute(query, {"id": id}).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail=f"Occurrence with id {id} not found")

        # 转换为字典返回
        occurrence = result[0]._asdict()

        return {"occurrence": occurrence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/api/families", response_model=PaginatedResponse)
async def get_families(
        params: TaxonomyFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """获取所有科"""
    # 构建基础查询
    base_select = """
        SELECT family, "order", genera_count, species_count, record_count, 
               countries_count, institutions_count, georeferencing_quality, 
               date_quality, last_updated
        FROM dbo.family_stats
    """

    # 构建WHERE条件
    conditions = []
    params_dict = {}

    if params.search:
        conditions.append("(family ILIKE :search OR \"order\" ILIKE :search)")
        params_dict["search"] = f"%{params.search}%"

    if params.order:
        conditions.append("\"order\" = :order_filter")
        params_dict["order_filter"] = params.order

    # 记录数过滤
    if params.record_count:
        if params.record_count == "high":
            conditions.append("record_count > 1000")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 100 AND 1000")
        elif params.record_count == "low":
            conditions.append("record_count < 100")

    # 构建WHERE子句
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # 排序映射
    sort_mapping = {
        "records_desc": "record_count DESC",
        "records_asc": "record_count ASC",
        "name_asc": "family ASC",
        "name_desc": "family DESC",
        "genera_desc": "genera_count DESC",
        "species_desc": "species_count DESC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")

    # 构建完整的数据查询
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")

    # 构建计数查询
    count_query = text(f"SELECT COUNT(*) FROM dbo.family_stats{where_clause}")

    # 执行查询
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # 获取总数
    total = db.execute(count_query, {k: v for k, v in params_dict.items() if k not in ['limit', 'offset']}).scalar()

    # 获取数据
    result = db.execute(data_query, params_dict).fetchall()

    # 转换为前端期望的响应格式 - 使用camelCase
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


@app.get("/api/families/{family_name}")
async def get_family_detail(family_name: str, db: Session = Depends(get_db)):
    """获取科详情"""
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


@app.get("/api/genera", response_model=PaginatedResponse)
async def get_genera(
        params: TaxonomyFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """获取所有属"""
    base_select = """
        SELECT genus, family, "order", species_count, record_count, 
               countries_count, institutions_count, georeferencing_quality, 
               date_quality, last_updated
        FROM dbo.genus_stats
    """

    conditions = []
    params_dict = {}

    # 添加搜索条件
    if params.search:
        conditions.append("(genus ILIKE :search OR family ILIKE :search)")
        params_dict["search"] = f"%{params.search}%"

    if params.family:
        conditions.append("family = :family_filter")
        params_dict["family_filter"] = params.family

    # 记录数过滤
    if params.record_count:
        if params.record_count == "high":
            conditions.append("record_count > 500")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 50 AND 500")
        elif params.record_count == "low":
            conditions.append("record_count < 50")

    # 构建WHERE子句
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # 排序
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "name_asc": "genus ASC",
        "family_asc": "family ASC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")

    # 构建查询
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.genus_stats{where_clause}")

    # 执行查询
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # 获取总数和数据
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


@app.get("/api/genera/{genus_name}")
async def get_genus_detail(genus_name: str, db: Session = Depends(get_db)):
    """获取属详情"""
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


@app.get("/api/species", response_model=PaginatedResponse)
async def get_species(
        params: TaxonomyFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """获取所有物种"""
    base_query = """
        SELECT CONCAT(genus, ' ', specificepithet) as scientific_name, scientificnameauthorship, vernacularname, 
               genus, family, "order", record_count, countries_count, 
               institutions_count, georeferencing_quality, date_quality, 
               last_record, last_updated
        FROM dbo.species_stats
    """

    conditions = []
    params_dict = {}

    # 搜索条件
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

    # 记录数过滤
    if params.record_count:
        if params.record_count == "high":
            conditions.append("record_count > 1000")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 100 AND 1000")
        elif params.record_count == "low":
            conditions.append("record_count < 100")

    # 数据质量过滤
    if params.data_quality:
        if params.data_quality == "excellent":
            conditions.append("georeferencing_quality > 95")
        elif params.data_quality == "good":
            conditions.append("georeferencing_quality BETWEEN 80 AND 95")
        elif params.data_quality == "poor":
            conditions.append("georeferencing_quality < 80")

    # 构建查询
    if conditions:
        query = text(base_query + " WHERE " + " AND ".join(conditions))
    else:
        query = text(base_query)

    # 排序
    sort_mapping = {
        "records_desc": "record_count DESC",
        "name_asc": "scientific_name ASC",
        "recent_desc": "last_record DESC",
        "quality_desc": "georeferencing_quality DESC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")
    query = text(str(query) + f" ORDER BY {order_by}")

    # 总数
    count_base = "SELECT COUNT(*) FROM dbo.species_stats"
    if conditions:
        count_query = text(count_base + " WHERE " + " AND ".join(conditions))
    else:
        count_query = text(count_base)
    total = db.execute(count_query, params_dict).scalar()

    # 分页
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

@app.get("/api/species/{scientific_name}")
async def get_species_detail(scientific_name: str, db: Session = Depends(get_db)):
    """获取物种详情 - 使用实时计算确保数据一致性"""
    
    # 解析学名：支持 "Brycon henni" 或 "henni"
    parts = scientific_name.strip().split(' ')
    if len(parts) >= 2:
        genus_name = parts[0]
        specific_epithet = parts[1]
        use_full_name = True
    else:
        specific_epithet = scientific_name
        genus_name = None
        use_full_name = False
    
    # 实时统计查询
    stats_query = text("""
        SELECT 
            genus,
            specificepithet,
            scientificnameauthorship,
            vernacularname,
            family,
            "order",
            COUNT(*) as record_count,
            COUNT(DISTINCT country) as countries_count,
            COUNT(DISTINCT institutioncode) as institutions_count,
            ROUND(
                COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1
            ) as georeferencing_quality,
            ROUND(
                COUNT(CASE WHEN eventdate IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1
            ) as date_quality,
            MAX(eventdate) as last_record,
            NOW() as last_updated
        FROM dbo.harvestedfn2 
        WHERE specificepithet = :specific_epithet
        """ + (" AND genus = :genus_name" if use_full_name else "") + """
          AND institutioncode IS NOT NULL 
          AND institutioncode != ''
        GROUP BY genus, specificepithet, scientificnameauthorship, vernacularname, family, "order"
    """)
    
    params = {"specific_epithet": specific_epithet}
    if use_full_name:
        params["genus_name"] = genus_name

    result = db.execute(stats_query, params).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Species not found")

    # 构建完整学名
    scientific_name_full = f"{result[0]} {result[1]}"

    return {
        "scientificName": scientific_name_full,
        "authority": result[2],
        "vernacularName": result[3],
        "genus": result[0],
        "family": result[4],
        "order": result[5],
        "recordCount": result[6],
        "countriesCount": result[7],
        "institutionsCount": result[8],
        "geoReferencingQuality": float(result[9]) if result[9] else 0.0,
        "dateQuality": float(result[10]) if result[10] else 0.0,
        "lastRecord": result[11],
        "lastUpdated": result[12],
        "taxonomicStatus": "accepted",
        "nomenclaturalCode": "ICZN"
    }


@app.get("/api/institutions", response_model=PaginatedResponse)
async def get_institutions(
        params: InstitutionFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """获取所有机构 - 修复版本，每个机构只显示一次"""
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

    # 搜索条件
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

    # 记录数过滤
    if params.record_count:
        if params.record_count == "major":
            conditions.append("record_count > 10000")
        elif params.record_count == "medium":
            conditions.append("record_count BETWEEN 1000 AND 10000")
        elif params.record_count == "small":
            conditions.append("record_count < 1000")

    # 构建WHERE子句
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # 排序
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "quality_desc": "overall_quality DESC",
        "name_asc": "institution_name ASC"
    }
    order_by = sort_mapping.get(params.sort_by, "record_count DESC")

    # 构建查询
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.institution_stats{where_clause}")

    # 执行查询
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # 获取总数和数据
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

@app.get("/api/institutions/{institution_code}")
async def get_institution_detail(institution_code: str, db: Session = Depends(get_db)):
    """获取机构详情"""
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


@app.get("/api/records", response_model=PaginatedResponse)
async def get_records(
        params: RecordFilterParams = Depends(),
        db: Session = Depends(get_db)
):
    """获取记录"""
    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country, 
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, geo_quality, date_quality
        FROM dbo.record_details
    """

    conditions = []
    params_dict = {}

    # 搜索条件
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

    # 构建WHERE子句
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # 排序
    sort_mapping = {
        "date_desc": "eventdate DESC NULLS LAST",
        "date_asc": "eventdate ASC NULLS LAST",
        "collector_asc": "recordedby ASC NULLS LAST",
        "locality_asc": "locality ASC NULLS LAST"
    }
    order_by = sort_mapping.get(params.sort_by, "eventdate DESC NULLS LAST")

    # 构建查询
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.record_details{where_clause}")

    # 执行查询
    params_dict.update({
        "limit": params.per_page,
        "offset": (params.page - 1) * params.per_page
    })

    # 获取总数和数据
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


@app.get("/api/taxonomy/stats")
async def get_taxonomy_stats(db: Session = Depends(get_db)):
    """获取分类统计 - 包含动态计算的指标"""

    # 1. 基础统计
    basic_stats_query = text("""
        SELECT 
            COUNT(DISTINCT family) as total_families,
            COUNT(DISTINCT genus) as total_genera,
            COUNT(DISTINCT  trim(specificepithet)) as total_species,
            COUNT(*) as total_records,
            COUNT(DISTINCT institutioncode) as total_institutions,
            COUNT(DISTINCT countrycode) as total_countries,
            ROUND(AVG(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 100.0 ELSE 0.0 END), 1) as avg_georeferencing,
            ROUND(AVG(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 100.0 ELSE 0.0 END), 1) as avg_date_quality
        FROM dbo.harvestedfn2
        WHERE family IS NOT NULL AND family != ''
    """)

    basic_result = db.execute(basic_stats_query).fetchone()

    # 2. 高多样性家族计算 (>50个属的家族)
    high_diversity_families_query = text("""
        SELECT COUNT(*) 
        FROM (
            SELECT family, COUNT(DISTINCT genus) as genera_count
            FROM dbo.harvestedfn2 
            WHERE family IS NOT NULL AND family != '' AND genus IS NOT NULL
            GROUP BY family
            HAVING COUNT(DISTINCT genus) > 50
        ) high_diversity
    """)

    high_diversity_result = db.execute(high_diversity_families_query).scalar()

    # 3. 良好采样家族计算 (>1000条记录的家族)
    well_sampled_families_query = text("""
        SELECT COUNT(*) 
        FROM (
            SELECT family, COUNT(*) as record_count
            FROM dbo.harvestedfn2 
            WHERE family IS NOT NULL AND family != ''
            GROUP BY family
            HAVING COUNT(*) > 1000
        ) well_sampled
    """)

    well_sampled_result = db.execute(well_sampled_families_query).scalar()

    # 4. 全球覆盖家族计算 (在>20个国家有记录的家族)
    global_coverage_families_query = text("""
        SELECT COUNT(*) 
        FROM (
            SELECT family, COUNT(DISTINCT country) as country_count
            FROM dbo.harvestedfn2 
            WHERE family IS NOT NULL AND family != '' AND country IS NOT NULL AND country != ''
            GROUP BY family
            HAVING COUNT(DISTINCT country) > 20
        ) global_coverage
    """)

    global_coverage_result = db.execute(global_coverage_families_query).scalar()

    # 5. 物种丰富属计算 (>20个物种的属)
    species_rich_genera_query = text("""
        SELECT COUNT(*) 
        FROM (
            SELECT genus, COUNT(DISTINCT scientificname) as species_count
            FROM dbo.harvestedfn2 
            WHERE genus IS NOT NULL AND genus != '' AND scientificname IS NOT NULL
            GROUP BY genus
            HAVING COUNT(DISTINCT scientificname) > 20
        ) species_rich
    """)

    species_rich_genera_result = db.execute(species_rich_genera_query).scalar()

    # 6. 良好采样属计算 (>500条记录的属)
    well_sampled_genera_query = text("""
        SELECT COUNT(*) 
        FROM (
            SELECT genus, COUNT(*) as record_count
            FROM dbo.harvestedfn2 
            WHERE genus IS NOT NULL AND genus != ''
            GROUP BY genus
            HAVING COUNT(*) > 500
        ) well_sampled_genera
    """)

    well_sampled_genera_result = db.execute(well_sampled_genera_query).scalar()

    # 7. 全球覆盖属计算 (在>15个国家有记录的属)
    global_coverage_genera_query = text("""
        SELECT COUNT(*) 
        FROM (
            SELECT genus, COUNT(DISTINCT country) as country_count
            FROM dbo.harvestedfn2 
            WHERE genus IS NOT NULL AND genus != '' AND country IS NOT NULL AND country != ''
            GROUP BY genus
            HAVING COUNT(DISTINCT country) > 15
        ) global_coverage_genera
    """)

    global_coverage_genera_result = db.execute(global_coverage_genera_query).scalar()

    # 8. 数据质量指标计算
    quality_stats_query = text("""
        SELECT 
            -- 地理参考质量分布
            COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as georef_percentage,
            -- 时间数据质量
            COUNT(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 1 END) * 100.0 / COUNT(*) as date_percentage,
            -- 分类学完整性
            COUNT(CASE WHEN specificepithet IS NOT NULL AND genus IS NOT NULL AND family IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as taxonomy_completeness,
            -- 机构覆盖度
            COUNT(CASE WHEN institutioncode IS NOT NULL AND institutioncode != '' THEN 1 END) * 100.0 / COUNT(*) as institution_coverage
        FROM dbo.harvestedfn2
    """)

    quality_result = db.execute(quality_stats_query).fetchone()

    # 9. 近期活跃度 (最近5年有记录的家族数)
    recent_activity_query = text("""
        SELECT COUNT(DISTINCT family)
        FROM dbo.harvestedfn2 
        WHERE family IS NOT NULL AND family != ''
        AND (
            "year" >= EXTRACT(YEAR FROM CURRENT_DATE) - 5
            OR eventdate >= (CURRENT_DATE - INTERVAL '5 years')::text
        )
    """)

    recent_families_result = db.execute(recent_activity_query).scalar()

    # 10. 顶级贡献者统计
    top_contributors_query = text("""
        SELECT 
            COUNT(CASE WHEN record_count > 10000 THEN 1 END) as major_families,
            COUNT(CASE WHEN record_count BETWEEN 1000 AND 10000 THEN 1 END) as active_families,
            COUNT(CASE WHEN record_count < 100 THEN 1 END) as underrepresented_families
        FROM (
            SELECT family, COUNT(*) as record_count
            FROM dbo.harvestedfn2 
            WHERE family IS NOT NULL AND family != ''
            GROUP BY family
        ) family_records
    """)

    contributors_result = db.execute(top_contributors_query).fetchone()

    # 组装完整的响应
    return {
        # 基础统计
        "totalFamilies": basic_result[0] or 0,
        "totalGenera": basic_result[1] or 0,
        "totalSpecies": basic_result[2] or 0,
        "totalRecords": basic_result[3] or 0,
        "totalInstitutions": basic_result[4] or 0,
        "totalCountries": basic_result[5] or 0,

        # 多样性指标
        "highDiversityFamilies": high_diversity_result or 0,
        "wellSampledFamilies": well_sampled_result or 0,
        "globalCoverageFamilies": global_coverage_result or 0,

        # 属级统计
        "speciesRichGenera": species_rich_genera_result or 0,
        "wellSampledGenera": well_sampled_genera_result or 0,
        "globalCoverageGenera": global_coverage_genera_result or 0,

        # 质量指标
        "avgGeoreferencing": float(basic_result[6]) if basic_result[6] else 0.0,
        "avgDateQuality": float(basic_result[7]) if basic_result[7] else 0.0,
        "taxonomyCompleteness": float(quality_result[2]) if quality_result[2] else 0.0,
        "institutionCoverage": float(quality_result[3]) if quality_result[3] else 0.0,

        # 活跃度指标
        "recentlyActiveFamilies": recent_families_result or 0,

        # 贡献度分布
        "majorContributorFamilies": contributors_result[0] or 0,
        "activeContributorFamilies": contributors_result[1] or 0,
        "underrepresentedFamilies": contributors_result[2] or 0,

        # 计算时间戳
        "calculatedAt": datetime.now().isoformat()
    }


# 在你的 main.py 中添加以下端点

@app.get("/api/institutions/{institution_code}/records", response_model=PaginatedResponse)
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
    """获取特定机构的所有记录"""

    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country, 
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, "year", "month"
        FROM dbo.harvestedfn2
        WHERE institutioncode = :institution_code
    """

    conditions = []
    params = {"institution_code": institution_code}

    # 搜索条件
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

    # 构建WHERE子句
    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    # 排序
    sort_mapping = {
        "date_desc": "eventdate DESC NULLS LAST",
        "date_asc": "eventdate ASC NULLS LAST",
        "collector_asc": "recordedby ASC NULLS LAST",
        "locality_asc": "locality ASC NULLS LAST",
        "species_asc": "scientificname ASC NULLS LAST",
        "family_asc": "family ASC NULLS LAST"
    }
    order_by = sort_mapping.get(sort_by, "eventdate DESC NULLS LAST")

    # 构建查询
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.harvestedfn2 WHERE institutioncode = :institution_code{where_clause}")

    # 执行查询
    params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })

    # 获取总数和数据
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


@app.get("/api/institutions/{institution_code}/records/summary")
async def get_institution_records_summary(
        institution_code: str,
        db: Session = Depends(get_db)
):
    """获取机构记录的汇总统计信息"""

    summary_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT scientificname) as unique_species,
            COUNT(DISTINCT family) as unique_families,
            COUNT(DISTINCT genus) as unique_genera,
            COUNT(DISTINCT country) as unique_countries,
            COUNT(DISTINCT collectioncode) as unique_collections,
            COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) as georeferenced_records,
            COUNT(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 1 END) as dated_records,
            MIN(CASE WHEN "year" IS NOT NULL AND "year" > 1800 THEN "year" END) as earliest_year,
            MAX(CASE WHEN "year" IS NOT NULL AND "year" > 1800 THEN "year" END) as latest_year,
            COUNT(CASE WHEN "year" >= 2020 THEN 1 END) as recent_records
        FROM dbo.harvestedfn2 
        WHERE institutioncode = :institution_code
    """)

    result = db.execute(summary_query, {"institution_code": institution_code}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Institution not found or has no records")

    # 计算百分比
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


@app.get("/api/institutions/{institution_code}/records/filters")
async def get_institution_records_filters(
        institution_code: str,
        db: Session = Depends(get_db)
):
    """获取机构记录的过滤器选项"""

    # 获取可用的年份
    years_query = text("""
        SELECT DISTINCT "year"
        FROM dbo.harvestedfn2 
        WHERE institutioncode = :institution_code 
          AND "year" IS NOT NULL AND "year" > 1800
        ORDER BY "year" DESC
    """)

    # 获取可用的国家
    countries_query = text("""
        SELECT DISTINCT country, COUNT(*) as record_count
        FROM dbo.harvestedfn2 
        WHERE institutioncode = :institution_code 
          AND country IS NOT NULL AND country != ''
        GROUP BY country
        ORDER BY record_count DESC
    """)

    # 获取可用的科
    families_query = text("""
        SELECT DISTINCT family, COUNT(*) as record_count
        FROM dbo.harvestedfn2 
        WHERE institutioncode = :institution_code 
          AND family IS NOT NULL AND family != ''
        GROUP BY family
        ORDER BY record_count DESC
    """)

    # 获取可用的属
    genera_query = text("""
        SELECT DISTINCT genus, COUNT(*) as record_count
        FROM dbo.harvestedfn2 
        WHERE institutioncode = :institution_code 
          AND genus IS NOT NULL AND genus != ''
        GROUP BY genus
        ORDER BY record_count DESC
        LIMIT 100
    """)

    # 获取可用的收藏代码
    collections_query = text("""
        SELECT DISTINCT collectioncode, COUNT(*) as record_count
        FROM dbo.harvestedfn2 
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


@app.get("/api/institutions/{institution_code}/records/export")
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
    """导出机构记录数据"""

    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country, 
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, "year", "month", day,
               county, stateprovince, kingdom, phylum, "class", "order"
        FROM dbo.harvestedfn2
        WHERE institutioncode = :institution_code
    """

    conditions = []
    params = {"institution_code": institution_code}

    # 应用过滤条件
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

    # 构建查询
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

        # 写入表头
        headers = [
            "catalogNumber", "scientificName", "vernacularName", "family", "genus",
            "scientificNameAuthorship", "recordedBy", "eventDate", "country",
            "locality", "decimalLatitude", "decimalLongitude", "institutionCode",
            "collectionCode", "basisOfRecord", "year", "month", "day",
            "county", "stateProvince", "kingdom", "phylum", "class", "order"
        ]
        writer.writerow(headers)

        # 写入数据
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


# 另外，修改现有的 RecordsTable 组件使用的 API
@app.get("/api/records/by-institution/{institution_code}")
async def get_records_by_institution(
        institution_code: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        db: Session = Depends(get_db)
):
    """简化版本的机构记录获取，用于 RecordsTable 组件"""

    query = text("""
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               recordedby, eventdate, country, locality, decimallatitude, 
               decimallongitude, institutioncode, collectioncode
        FROM dbo.harvestedfn2
        WHERE institutioncode = :institution_code
        ORDER BY eventdate DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """)

    count_query = text("""
        SELECT COUNT(*) FROM dbo.harvestedfn2 WHERE institutioncode = :institution_code
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


# 在你的 main.py 中添加这个端点

# 在你的 main.py 的 get_records_by_institution_v2 函数中添加地理坐标过滤参数

@app.get("/api/records/institution/{institution_code}", response_model=PaginatedResponse)
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
        georeferenced_only: bool = Query(False),  # 新增：只获取有地理坐标的记录
        db: Session = Depends(get_db)
):
    """按机构获取记录 - 适配前端现有API调用，支持地理坐标过滤"""

    base_select = """
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               scientificnameauthorship, recordedby, eventdate, country, 
               locality, decimallatitude, decimallongitude, institutioncode,
               collectioncode, basisofrecord, "year", "month"
        FROM dbo.harvestedfn2
        WHERE institutioncode = :institution_code
    """

    conditions = []
    params = {"institution_code": institution_code}

    # 地理坐标过滤
    if georeferenced_only:
        conditions.append("decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL")
        conditions.append("decimallatitude != 0 OR decimallongitude != 0")  # 排除 0,0 坐标

    # 其他搜索条件
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

    # 构建WHERE子句
    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    # 排序
    sort_mapping = {
        "date_desc": "eventdate DESC NULLS LAST",
        "date_asc": "eventdate ASC NULLS LAST",
        "collector_asc": "recordedby ASC NULLS LAST",
        "locality_asc": "locality ASC NULLS LAST",
        "species_asc": "scientificname ASC NULLS LAST",
        "family_asc": "family ASC NULLS LAST"
    }
    order_by = sort_mapping.get(sort_by, "eventdate DESC NULLS LAST")

    # 构建查询
    data_query = text(base_select + where_clause + f" ORDER BY {order_by} LIMIT :limit OFFSET :offset")
    count_query = text(f"SELECT COUNT(*) FROM dbo.harvestedfn2 WHERE institutioncode = :institution_code{where_clause}")

    # 执行查询
    params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })

    # 获取总数和数据
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

@app.get("/api/institution/stats")
async def get_institutions_stats(db: Session = Depends(get_db)):
    """获取机构统计 - 修复版本"""

    # 1. 基础统计 - 现在每个机构只计算一次
    basic_stats_query = text("""
        SELECT 
            COUNT(*) as total_institutions,
            SUM(array_length(collection_codes, 1)) as total_collection_codes,
            SUM(record_count) as total_records,
            COUNT(DISTINCT country) as total_countries,
            ROUND(AVG(georeferencing_quality), 1) as avg_georeferencing,
            ROUND(AVG(date_quality), 1) as avg_date_quality
        FROM dbo.institution_stats
        WHERE collection_codes IS NOT NULL
    """)

    basic_result = db.execute(basic_stats_query).fetchone()

    # 2. 贡献者分类统计
    contributors_query = text("""
        SELECT 
            COUNT(CASE WHEN record_count > 10000 THEN 1 END) as major_contributors,
            COUNT(CASE WHEN record_count BETWEEN 1000 AND 10000 THEN 1 END) as active_contributors,
            COUNT(CASE WHEN institution_type IN ('museum', 'university') THEN 1 END) as research_collections,
            COUNT(CASE WHEN overall_quality > 90 THEN 1 END) as high_quality_data
        FROM dbo.institution_stats
    """)

    contributors_result = db.execute(contributors_query).fetchone()

    # 3. 地理分布统计 - 修复版本
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

    # # 4. 收藏多样性统计
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

    # 5. 时间覆盖分析 - 安全处理NULL值
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
        # 基础统计
        "totalInstitutions": basic_result[0] or 0,
        "totalCollectionCodes": basic_result[1] or 0,
        "totalRecords": basic_result[2] or 0,
        "totalCountries": basic_result[3] or 0,
        "avgGeoreferenced": float(basic_result[4]) if basic_result[4] else 0.0,
        "avgDateQuality": float(basic_result[5]) if basic_result[5] else 0.0,

        # 贡献者分类
        "majorContributors": contributors_result[0] or 0,
        "activeContributors": contributors_result[1] or 0,
        "researchCollections": contributors_result[2] or 0,
        "highQualityData": contributors_result[3] or 0,

        # 地理分布
        "northAmericaInstitutions": geo_result[0] or 0,
        "europeInstitutions": geo_result[1] or 0,
        "asiaPacificInstitutions": geo_result[2] or 0,
        "southAmericaInstitutions": geo_result[3] or 0,
        "africaInstitutions": geo_result[4] or 0,
        "otherRegionsInstitutions": geo_result[5] or 0,

        # # 收藏多样性
        # "institutionsWithCollections": collection_result[0] or 0,
        # "uniqueCollectionCodes": collection_result[1] or 0,
        # "avgCollectionsPerInstitution": float(collection_result[2]) if collection_result[2] else 0.0,

        # 时间覆盖
        "recentRecords": temporal_result[0] or 0,
        "decadeRecords": temporal_result[1] or 0,
        "millenniumRecords": temporal_result[2] or 0,
        "historicalRecords": temporal_result[3] or 0,

        # 计算时间戳
        "calculatedAt": datetime.now().isoformat()
    }


@app.get("/api/institutions/{institution_code}/countries")
async def get_institution_countries(institution_code: str, db: Session = Depends(get_db)):
    """获取特定机构在各国的分布情况"""
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


@app.get("/api/orders")
async def get_orders(db: Session = Depends(get_db)):
    """获取所有目"""
    query = text("""
        SELECT "order", COUNT(DISTINCT family) as family_count, 
               COUNT(DISTINCT scientificname) as species_count
        FROM dbo.harvestedfn2 
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


@app.get("/api/records/taxon/{taxon_type}/{taxon_name}")
async def get_records_by_taxon(
        taxon_type: str,
        taxon_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=1000),
        has_coordinates: Optional[bool] = Query(None, description="Filter records with coordinates"),
        db: Session = Depends(get_db)
):
    """根据分类单元获取记录"""
    field_mapping = {
        "family": "family",
        "genus": "genus",
        "species": "specificepithet"
    }

    if taxon_type not in field_mapping:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    field = field_mapping[taxon_type]

    # Build WHERE clause
    where_conditions = [f"{field} = :taxon_name"]
    
    if has_coordinates is True:
        where_conditions.append("decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL")
    elif has_coordinates is False:
        where_conditions.append("(decimallatitude IS NULL OR decimallongitude IS NULL)")
    
    where_clause = " AND ".join(where_conditions)

    query = text(f"""
        SELECT catalognumber, scientificname, vernacularname, family, genus,
               recordedby, eventdate, country, locality, decimallatitude, 
               decimallongitude, institutioncode, collectioncode
        FROM dbo.harvestedfn2
        WHERE {where_clause}
        ORDER BY eventdate DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """)

    count_query = text(f"""
        SELECT COUNT(*) FROM dbo.harvestedfn2 WHERE {where_clause}
    """)

    total = db.execute(count_query, {"taxon_name": taxon_name}).scalar()
    result = db.execute(query, {
        "taxon_name": taxon_name,
        "limit": per_page,
        "offset": (page - 1) * per_page
    }).fetchall()

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


# 在 main.py 中添加以下端点

@app.get("/api/families/{family_name}/children")
async def get_family_children(
        family_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=100),
        search: Optional[str] = None,
        sort_by: str = "records_desc",
        db: Session = Depends(get_db)
):
    """获取科下的所有属"""

    # 构建基础查询
    base_query = """
        SELECT genus, COUNT(DISTINCT specificepithet) as species_count,
               COUNT(*) as record_count, COUNT(DISTINCT country) as countries_count,
               COUNT(DISTINCT institutioncode) as institutions_count,
               ROUND(
                   COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2
               ) as georeferencing_quality,
               MAX(eventdate) as last_record
        FROM dbo.harvestedfn2 
        WHERE family = :family_name AND genus IS NOT NULL AND genus != ''
    """

    conditions = []
    params = {"family_name": family_name}

    if search:
        conditions.append("genus ILIKE :search")
        params["search"] = f"%{search}%"

    group_clause = " GROUP BY genus"

    # 排序
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "name_asc": "genus ASC",
        "name_desc": "genus DESC"
    }
    order_clause = f" ORDER BY {sort_mapping.get(sort_by, 'record_count DESC')}"

    # 构建完整查询
    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    query = text(base_query + where_clause + group_clause + order_clause + " LIMIT :limit OFFSET :offset")
    count_query = text(
        f"SELECT COUNT(DISTINCT genus) FROM dbo.harvestedfn2 WHERE family = :family_name AND genus IS NOT NULL AND genus != ''{where_clause}")

    # 执行查询
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


@app.get("/api/genera/{genus_name}/children")
async def get_genus_children(
        genus_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=100),
        search: Optional[str] = None,
        sort_by: str = "records_desc",
        db: Session = Depends(get_db)
):
    """获取属下的所有物种"""

    base_query = """
        SELECT scientificname, vernacularname, COUNT(*) as record_count,
               COUNT(DISTINCT country) as countries_count,
               COUNT(DISTINCT institutioncode) as institutions_count,
               ROUND(
                   COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2
               ) as georeferencing_quality,
               MAX(eventdate) as last_record
        FROM dbo.harvestedfn2 
        WHERE genus = :genus_name AND scientificname IS NOT NULL AND scientificname != ''
    """

    conditions = []
    params = {"genus_name": genus_name}

    if search:
        conditions.append("(scientificname ILIKE :search OR vernacularname ILIKE :search)")
        params["search"] = f"%{search}%"

    group_clause = " GROUP BY scientificname, vernacularname"

    # 排序
    sort_mapping = {
        "records_desc": "record_count DESC",
        "name_asc": "scientificname ASC",
        "name_desc": "scientificname DESC"
    }
    order_clause = f" ORDER BY {sort_mapping.get(sort_by, 'record_count DESC')}"

    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    query = text(base_query + where_clause + group_clause + order_clause + " LIMIT :limit OFFSET :offset")
    count_query = text(
        f"SELECT COUNT(DISTINCT scientificname) FROM dbo.harvestedfn2 WHERE genus = :genus_name AND scientificname IS NOT NULL AND scientificname != ''{where_clause}")

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


@app.get("/api/{taxon_type}/{taxon_name}/diversity")
async def get_taxon_diversity(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """获取分类单元的多样性统计"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    if taxon_type == "family":
        # 科的多样性 - 按属统计
        query = text("""
            SELECT genus, COUNT(DISTINCT specificepithet) as species_count,
                   COUNT(*) as record_count
            FROM dbo.harvestedfn2 
            WHERE family = :taxon_name AND genus IS NOT NULL AND genus != ''
            GROUP BY genus
            ORDER BY record_count DESC
            LIMIT 10
        """)
    elif taxon_type == "genus":
        # 属的多样性 - 按物种统计
        query = text("""
            SELECT scientificname, COUNT(*) as record_count,
                   MAX(vernacularname) as vernacular_name
            FROM dbo.harvestedfn2 
            WHERE genus = :taxon_name AND scientificname IS NOT NULL AND scientificname != ''
            GROUP BY scientificname
            ORDER BY record_count DESC
            LIMIT 10
        """)
    else:
        # 物种层级不需要多样性统计
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


@app.get("/api/{taxon_type}/{taxon_name}/geographic")
async def get_taxon_geographic_distribution(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """获取分类单元的地理分布统计"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    field_mapping = {"family": "family", "genus": "genus", "species": "specificepithet"}
    field = field_mapping[taxon_type]

    # 1. 大陆分布统计
    continent_query = text(f"""
        SELECT 
            CASE 
                WHEN country IN ('United States', 'USA', 'US', 'Canada', 'Mexico') THEN 'North America'
                WHEN country IN ('United Kingdom', 'UK', 'France', 'Germany', 'Spain', 'Italy', 'Netherlands', 'Sweden', 'Norway') THEN 'Europe'
                WHEN country IN ('China', 'Japan', 'Australia', 'New Zealand', 'India', 'South Korea', 'Thailand', 'Malaysia', 'Singapore') THEN 'Asia-Pacific'
                WHEN country IN ('Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela') THEN 'South America'
                WHEN country IN ('South Africa', 'Kenya', 'Tanzania', 'Egypt', 'Morocco', 'Nigeria') THEN 'Africa'
                ELSE 'Other Regions'
            END as continent,
            COUNT(*) as records,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name AND country IS NOT NULL AND country != ''
        GROUP BY 
            CASE 
                WHEN country IN ('United States', 'USA', 'US', 'Canada', 'Mexico') THEN 'North America'
                WHEN country IN ('United Kingdom', 'UK', 'France', 'Germany', 'Spain', 'Italy', 'Netherlands', 'Sweden', 'Norway') THEN 'Europe'
                WHEN country IN ('China', 'Japan', 'Australia', 'New Zealand', 'India', 'South Korea', 'Thailand', 'Malaysia', 'Singapore') THEN 'Asia-Pacific'
                WHEN country IN ('Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela') THEN 'South America'
                WHEN country IN ('South Africa', 'Kenya', 'Tanzania', 'Egypt', 'Morocco', 'Nigeria') THEN 'Africa'
                ELSE 'Other Regions'
            END
        ORDER BY records DESC
    """)

    continent_result = db.execute(continent_query, {"taxon_name": taxon_name}).fetchall()

    # 2. 国家分布统计
    country_query = text(f"""
        SELECT country, COUNT(*) as records,
               COUNT(DISTINCT CASE WHEN {field} = :taxon_name THEN scientificname END) as species_count,
               ROUND(
                   COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1
               ) as data_quality
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name AND country IS NOT NULL AND country != ''
        GROUP BY country
        ORDER BY records DESC
        LIMIT 20
    """)

    country_result = db.execute(country_query, {"taxon_name": taxon_name}).fetchall()

    # 3. 生物多样性热点
    hotspot_query = text(f"""
        SELECT country, COUNT(*) as records,
               COUNT(DISTINCT scientificname) as species_count,
               COUNT(DISTINCT genus) as genera_count,
               ROUND(
                   COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1
               ) as data_quality
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name AND country IS NOT NULL AND country != ''
        GROUP BY country
        HAVING COUNT(*) > 100  -- 只显示有足够记录的地区
        ORDER BY species_count DESC, records DESC
        LIMIT 8
    """)

    hotspot_result = db.execute(hotspot_query, {"taxon_name": taxon_name}).fetchall()

    continental_distribution = []
    for row in continent_result:
        continental_distribution.append({
            "name": row[0],
            "records": row[1],
            "percentage": row[2]
        })

    country_distribution = []
    for row in country_result:
        country_distribution.append({
            "name": row[0],
            "records": row[1],
            "species": row[2] if len(row) > 2 else 0,
            "dataQuality": row[3] if len(row) > 3 else 0
        })

    biodiversity_hotspots = []
    for row in hotspot_result:
        biodiversity_hotspots.append({
            "name": row[0],
            "records": row[1],
            "species": row[2],
            "genera": row[3] if len(row) > 3 else 0,
            "dataQuality": row[4] if len(row) > 4 else 0,
            "description": f"Major biodiversity center with {row[2]} species"
        })

    return {
        "continentalDistribution": continental_distribution,
        "countryDistribution": country_distribution,
        "biodiversityHotspots": biodiversity_hotspots
    }


# 修复 temporal patterns 查询中的 SQL 语法错误

@app.get("/api/{taxon_type}/{taxon_name}/temporal")
async def get_taxon_temporal_patterns(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """获取分类单元的时间模式统计"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    field_mapping = {"family": "family", "genus": "genus", "species": "specificepithet"}
    field = field_mapping[taxon_type]

    # 1. 历史时期统计
    historical_query = text(f"""
        WITH period_data AS (
            SELECT 
                CASE 
                    WHEN year < 1950 THEN 'Pre-1950'
                    WHEN year BETWEEN 1950 AND 1990 THEN '1950-1990'
                    WHEN year BETWEEN 1991 AND 2010 THEN '1991-2010'
                    WHEN year >= 2011 THEN '2011-Present'
                    ELSE 'Unknown'
                END as period,
                COUNT(*) as records
            FROM dbo.harvestedfn2 
            WHERE {field} = :taxon_name AND year IS NOT NULL AND year > 1800
            GROUP BY 
                CASE 
                    WHEN year < 1950 THEN 'Pre-1950'
                    WHEN year BETWEEN 1950 AND 1990 THEN '1950-1990'
                    WHEN year BETWEEN 1991 AND 2010 THEN '1991-2010'
                    WHEN year >= 2011 THEN '2011-Present'
                    ELSE 'Unknown'
                END
        )
        SELECT 
            period,
            records,
            ROUND(records * 100.0 / SUM(records) OVER (), 1) as percentage
        FROM period_data
        ORDER BY 
            CASE period
                WHEN 'Pre-1950' THEN 1
                WHEN '1950-1990' THEN 2
                WHEN '1991-2010' THEN 3
                WHEN '2011-Present' THEN 4
                ELSE 5
            END
    """)

    # 2. 季节性模式
    seasonal_query = text(f"""
        WITH seasonal_data AS (
            SELECT 
                CASE 
                    WHEN month IN (3, 4, 5) THEN 'Spring (Mar-May)'
                    WHEN month IN (6, 7, 8) THEN 'Summer (Jun-Aug)'
                    WHEN month IN (9, 10, 11) THEN 'Autumn (Sep-Nov)'
                    WHEN month IN (12, 1, 2) THEN 'Winter (Dec-Feb)'
                    ELSE 'Unknown'
                END as season,
                COUNT(*) as records
            FROM dbo.harvestedfn2 
            WHERE {field} = :taxon_name AND month IS NOT NULL
            GROUP BY 
                CASE 
                    WHEN month IN (3, 4, 5) THEN 'Spring (Mar-May)'
                    WHEN month IN (6, 7, 8) THEN 'Summer (Jun-Aug)'
                    WHEN month IN (9, 10, 11) THEN 'Autumn (Sep-Nov)'
                    WHEN month IN (12, 1, 2) THEN 'Winter (Dec-Feb)'
                    ELSE 'Unknown'
                END
        )
        SELECT season, records
        FROM seasonal_data
        ORDER BY records DESC
    """)

    # 3. 近期活动
    recent_query = text(f"""
        SELECT year, COUNT(*) as records
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name AND year >= 2020 AND year IS NOT NULL
        GROUP BY year
        ORDER BY year DESC
    """)

    # 4. 年度趋势数据（用于时间线图表）
    yearly_trend_query = text(f"""
        SELECT year, COUNT(*) as records
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name 
          AND year IS NOT NULL 
          AND year BETWEEN 1950 AND EXTRACT(YEAR FROM CURRENT_DATE)
        GROUP BY year
        ORDER BY year
    """)

    # 5. 月度分布
    monthly_distribution_query = text(f"""
        SELECT month, COUNT(*) as records
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name AND month IS NOT NULL AND month BETWEEN 1 AND 12
        GROUP BY month
        ORDER BY month
    """)

    # 6. 数据覆盖率
    data_coverage_query = text(f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN year IS NOT NULL AND year > 1800 THEN 1 END) as records_with_year,
            COUNT(CASE WHEN month IS NOT NULL THEN 1 END) as records_with_month,
            COUNT(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 1 END) as records_with_date
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name
    """)

    try:
        # 执行查询
        historical_result = db.execute(historical_query, {"taxon_name": taxon_name}).fetchall()
        seasonal_result = db.execute(seasonal_query, {"taxon_name": taxon_name}).fetchall()
        recent_result = db.execute(recent_query, {"taxon_name": taxon_name}).fetchall()
        yearly_trend_result = db.execute(yearly_trend_query, {"taxon_name": taxon_name}).fetchall()
        monthly_result = db.execute(monthly_distribution_query, {"taxon_name": taxon_name}).fetchall()
        coverage_result = db.execute(data_coverage_query, {"taxon_name": taxon_name}).fetchone()

        # 处理结果
        historical_periods = []
        for row in historical_result:
            historical_periods.append({
                "name": row[0],
                "records": row[1],
                "percentage": float(row[2]) if row[2] else 0.0
            })

        seasonal_patterns = []
        for row in seasonal_result:
            seasonal_patterns.append({
                "name": row[0],
                "records": row[1]
            })

        recent_activity = []
        for row in recent_result:
            recent_activity.append({
                "year": str(row[0]),
                "records": row[1]
            })

        # 年度趋势数据（用于图表）
        yearly_trend = []
        for row in yearly_trend_result:
            yearly_trend.append({
                "year": row[0],
                "records": row[1]
            })

        # 月度分布数据
        monthly_distribution = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for row in monthly_result:
            monthly_distribution.append({
                "month": row[0],
                "monthName": month_names[row[0] - 1],
                "records": row[1]
            })

        # 数据覆盖率
        data_coverage = {
            "totalRecords": coverage_result[0] if coverage_result else 0,
            "recordsWithYear": coverage_result[1] if coverage_result else 0,
            "recordsWithMonth": coverage_result[2] if coverage_result else 0,
            "recordsWithFullDate": coverage_result[3] if coverage_result else 0,
            "yearCoverage": round((coverage_result[1] / coverage_result[0]) * 100, 1) if coverage_result and
                                                                                         coverage_result[0] > 0 else 0,
            "monthCoverage": round((coverage_result[2] / coverage_result[0]) * 100, 1) if coverage_result and
                                                                                          coverage_result[0] > 0 else 0,
            "dateCoverage": round((coverage_result[3] / coverage_result[0]) * 100, 1) if coverage_result and
                                                                                         coverage_result[0] > 0 else 0
        }

        return {
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

    except Exception as e:
        print(f"Error in temporal query: {str(e)}")
        # 返回空结果而不是抛出异常
        return {
            "historicalPeriods": [],
            "seasonalPatterns": [],
            "recentActivity": [],
            "yearlyTrend": [],
            "monthlyDistribution": [],
            "dataCoverage": {"totalRecords": 0, "yearCoverage": 0, "monthCoverage": 0, "dateCoverage": 0},
            "summary": {}
        }


# 3. 修改机构API，确保返回地图所需的数据
@app.get("/api/{taxon_type}/{taxon_name}/institutions")
async def get_taxon_institutions(
        taxon_type: str,
        taxon_name: str,
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=100),
        search: Optional[str] = None,
        sort_by: str = "records_desc",
        db: Session = Depends(get_db)
):
    """获取分类单元的贡献机构统计"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    field_mapping = {"family": "family", "genus": "genus", "species": "specificepithet"}
    field = field_mapping[taxon_type]

    base_query = f"""
        SELECT institutioncode,
               COUNT(*) as record_count,
               COUNT(DISTINCT scientificname) as species_count,
               COUNT(DISTINCT genus) as genera_count,
               COUNT(DISTINCT country) as countries_count,
               ROUND(
                   COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1
               ) as georeferencing_quality,
               array_agg(DISTINCT collectioncode ORDER BY collectioncode) 
               FILTER (WHERE collectioncode IS NOT NULL AND collectioncode != '') as collection_codes,
               MAX(eventdate) as latest_record,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage,
               -- 添加国家信息用于地图显示
               (array_agg(DISTINCT country ORDER BY country))[1] as primary_country
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name AND institutioncode IS NOT NULL AND institutioncode != ''
    """

    conditions = []
    params = {"taxon_name": taxon_name}

    if search:
        conditions.append("institutioncode ILIKE :search")
        params["search"] = f"%{search}%"

    group_clause = " GROUP BY institutioncode"

    # 排序
    sort_mapping = {
        "records_desc": "record_count DESC",
        "species_desc": "species_count DESC",
        "name_asc": "institutioncode ASC",
        "countries_desc": "countries_count DESC"
    }
    order_clause = f" ORDER BY {sort_mapping.get(sort_by, 'record_count DESC')}"

    where_clause = ""
    if conditions:
        where_clause = " AND " + " AND ".join(conditions)

    query = text(base_query + where_clause + group_clause + order_clause + " LIMIT :limit OFFSET :offset")
    count_query = text(
        f"SELECT COUNT(DISTINCT institutioncode) FROM dbo.harvestedfn2 WHERE {field} = :taxon_name AND institutioncode IS NOT NULL AND institutioncode != ''{where_clause}")

    params.update({
        "limit": per_page,
        "offset": (page - 1) * per_page
    })

    total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar()
    result = db.execute(query, params).fetchall()

    institutions = []
    for row in result:
        # 机构名称映射
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
            "country": row[9] or 'Unknown'  # 用于地图显示
        })

    return {
        "data": institutions,
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": (total + per_page - 1) // per_page
    }


# 4. 添加一个专门用于地图数据的端点
@app.get("/api/{taxon_type}/{taxon_name}/map-data")
async def get_taxon_map_data(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """获取用于地图显示的机构分布数据"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    field_mapping = {"family": "family", "genus": "genus", "species": "specificepithet"}
    field = field_mapping[taxon_type]

    # 获取按国家分组的机构数据
    query = text(f"""
        SELECT 
            country,
            COUNT(DISTINCT institutioncode) as institution_count,
            COUNT(*) as total_records,
            AVG(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100 as avg_georeferencing
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name 
          AND country IS NOT NULL AND country != ''
          AND institutioncode IS NOT NULL AND institutioncode != ''
        GROUP BY country
        HAVING COUNT(*) > 10  -- 只显示有足够记录的国家
        ORDER BY total_records DESC
        LIMIT 50
    """)

    result = db.execute(query, {"taxon_name": taxon_name}).fetchall()

    # 国家坐标映射（实际应用中可以使用地理编码服务）
    country_coords = {
        'United States': {'lat': 39.8283, 'lng': -98.5795},
        'USA': {'lat': 39.8283, 'lng': -98.5795},
        'US': {'lat': 39.8283, 'lng': -98.5795},
        'United Kingdom': {'lat': 55.3781, 'lng': -3.4360},
        'UK': {'lat': 55.3781, 'lng': -3.4360},
        'Canada': {'lat': 56.1304, 'lng': -106.3468},
        'Australia': {'lat': -25.2744, 'lng': 133.7751},
        'Germany': {'lat': 51.1657, 'lng': 10.4515},
        'France': {'lat': 46.2276, 'lng': 2.2137},
        'Japan': {'lat': 36.2048, 'lng': 138.2529},
        'Brazil': {'lat': -14.2350, 'lng': -51.9253},
        'China': {'lat': 35.8617, 'lng': 104.1954},
        'Russia': {'lat': 61.5240, 'lng': 105.3188},
        'India': {'lat': 20.5937, 'lng': 78.9629},
        'Mexico': {'lat': 23.6345, 'lng': -102.5528},
        'Argentina': {'lat': -38.4161, 'lng': -63.6167},
        'Spain': {'lat': 40.4637, 'lng': -3.7492},
        'Italy': {'lat': 41.8719, 'lng': 12.5674},
        'Sweden': {'lat': 60.1282, 'lng': 18.6435},
        'Norway': {'lat': 60.4720, 'lng': 8.4689},
        'South Africa': {'lat': -30.5595, 'lng': 22.9375},
        'Chile': {'lat': -35.6751, 'lng': -71.5430},
        'Colombia': {'lat': 4.5709, 'lng': -74.2973},
        'Peru': {'lat': -9.1900, 'lng': -75.0152},
        'Venezuela': {'lat': 6.4238, 'lng': -66.5897},
        'Netherlands': {'lat': 52.1326, 'lng': 5.2913},
        'Belgium': {'lat': 50.5039, 'lng': 4.4699},
        'Switzerland': {'lat': 46.8182, 'lng': 8.2275},
        'Austria': {'lat': 47.5162, 'lng': 14.5501},
        'Poland': {'lat': 51.9194, 'lng': 19.1451},
        'Czech Republic': {'lat': 49.8175, 'lng': 15.4730},
        'Denmark': {'lat': 56.2639, 'lng': 9.5018},
        'Finland': {'lat': 61.9241, 'lng': 25.7482},
        'Portugal': {'lat': 39.3999, 'lng': -8.2245},
        'Greece': {'lat': 39.0742, 'lng': 21.8243},
        'Turkey': {'lat': 38.9637, 'lng': 35.2433},
        'Egypt': {'lat': 26.0975, 'lng': 31.2357},
        'Morocco': {'lat': 31.7917, 'lng': -7.0926},
        'Kenya': {'lat': -0.0236, 'lng': 37.9062},
        'Tanzania': {'lat': -6.3690, 'lng': 34.8888},
        'Nigeria': {'lat': 9.0820, 'lng': 8.6753},
        'Ghana': {'lat': 7.9465, 'lng': -1.0232},
        'Thailand': {'lat': 15.8700, 'lng': 100.9925},
        'Malaysia': {'lat': 4.2105, 'lng': 101.9758},
        'Singapore': {'lat': 1.3521, 'lng': 103.8198},
        'Indonesia': {'lat': -0.7893, 'lng': 113.9213},
        'Philippines': {'lat': 12.8797, 'lng': 121.7740},
        'South Korea': {'lat': 35.9078, 'lng': 127.7669},
        'New Zealand': {'lat': -40.9006, 'lng': 174.8860},
        'Costa Rica': {'lat': 9.7489, 'lng': -83.7534},
        'Panama': {'lat': 8.5380, 'lng': -80.7821},
        'Ecuador': {'lat': -1.8312, 'lng': -78.1834},
        'Bolivia': {'lat': -16.2902, 'lng': -63.5887},
        'Uruguay': {'lat': -32.5228, 'lng': -55.7658},
        'Paraguay': {'lat': -23.4425, 'lng': -58.4438}
    }

    map_data = []
    for row in result:
        country = row[0]
        coords = country_coords.get(country)

        if coords:
            # 为每个国家创建一个虚拟的"机构聚合点"
            map_data.append({
                'institutionCode': f"AGG_{country.upper().replace(' ', '_')}",
                'institutionName': f"{country} Institutions",
                'lat': coords['lat'] + (hash(country) % 100 - 50) * 0.01,  # 添加小的随机偏移避免重叠
                'lng': coords['lng'] + (hash(country) % 100 - 50) * 0.01,
                'recordCount': row[2],  # total_records
                'country': country,
                'institutionCount': row[1],  # institution_count
                'avgGeoreferencing': round(row[3], 1) if row[3] else 0
            })

    return {"mapData": map_data}


@app.get("/api/{taxon_type}/{taxon_name}/top-species")
async def get_taxon_top_species(
        taxon_type: str,
        taxon_name: str,
        limit: int = Query(8, ge=1, le=20),
        db: Session = Depends(get_db)
):
    """获取分类单元下记录最多的物种"""

    if taxon_type not in ["family", "genus"]:
        raise HTTPException(status_code=400, detail="Only family and genus support top species")

    field_mapping = {"family": "family", "genus": "genus"}
    field = field_mapping[taxon_type]

    query = text(f"""
        SELECT scientificname, vernacularname, COUNT(*) as record_count,
               COUNT(DISTINCT institutioncode) as institutions_count,
               COUNT(DISTINCT country) as countries_count
        FROM dbo.harvestedfn2 
        WHERE {field} = :taxon_name AND scientificname IS NOT NULL AND scientificname != ''
        GROUP BY scientificname, vernacularname
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
            "recordCount": row[2],
            "institutionsCount": row[3],
            "countriesCount": row[4]
        })

    return {"topSpecies": top_species}


@app.get("/api/taxonomy/hierarchy/{taxon_type}/{taxon_name}")
async def get_taxon_hierarchy(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """获取分类单元的层级信息"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    if taxon_type == "species":
        # For species, we expect full binomial name (genus + species)
        query = text("""
            SELECT DISTINCT kingdom, phylum, "class", "order", family, genus, specificepithet, scientificname
            FROM dbo.harvestedfn2 
            WHERE scientificname = :taxon_name OR CONCAT(genus, ' ', specificepithet) = :taxon_name
            LIMIT 1
        """)
    else:
        field_mapping = {"family": "family", "genus": "genus"}
        field = field_mapping[taxon_type]
        query = text(f"""
            SELECT DISTINCT kingdom, phylum, "class", "order", family, genus, specificepithet, scientificname
            FROM dbo.harvestedfn2 
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


# 在 main.py 中添加以下端点

@app.get("/api/{taxon_type}/{taxon_name}/institution-coverage")
async def get_taxon_institution_coverage(
        taxon_type: str,
        taxon_name: str,
        db: Session = Depends(get_db)
):
    """获取特定分类单元的机构覆盖度分析 - 动态计算"""

    if taxon_type not in ["family", "genus", "species"]:
        raise HTTPException(status_code=400, detail="Invalid taxon type")

    field_mapping = {"family": "family", "genus": "genus", "species": "scientificname"}
    field = field_mapping[taxon_type]

    try:
        # 1. 地理覆盖度分析
        geographic_coverage_query = text(f"""
            WITH institution_countries AS (
                SELECT institutioncode, 
                       COUNT(DISTINCT country) as country_count,
                       COUNT(*) as total_records
                FROM dbo.harvestedfn2 
                WHERE {field} = :taxon_name 
                  AND institutioncode IS NOT NULL AND institutioncode != ''
                  AND country IS NOT NULL AND country != ''
                GROUP BY institutioncode
            )
            SELECT 
                COUNT(CASE WHEN country_count >= 20 THEN 1 END) as global_coverage,
                COUNT(CASE WHEN country_count BETWEEN 5 AND 19 THEN 1 END) as regional_specialists,
                COUNT(CASE WHEN country_count < 5 THEN 1 END) as local_collections,
                COUNT(*) as total_institutions
            FROM institution_countries
        """)

        geographic_result = db.execute(geographic_coverage_query, {"taxon_name": taxon_name}).fetchone()

        # 2. 分类专业化分析
        taxonomic_specialization_query = text(f"""
            WITH institution_taxonomy AS (
                SELECT institutioncode,
                       COUNT(DISTINCT family) as family_count,
                       COUNT(DISTINCT genus) as genus_count,
                       COUNT(DISTINCT scientificname) as species_count,
                       COUNT(*) as total_records
                FROM dbo.harvestedfn2 
                WHERE {field} = :taxon_name 
                  AND institutioncode IS NOT NULL AND institutioncode != ''
                GROUP BY institutioncode
            )
            SELECT 
                COUNT(CASE WHEN total_records >= 1000 THEN 1 END) as family_specialists,
                COUNT(CASE WHEN genus_count <= 5 AND total_records >= 100 THEN 1 END) as genus_specialists,
                COUNT(CASE WHEN species_count >= 10 AND genus_count >= 3 THEN 1 END) as regional_fauna_focus,
                COUNT(*) as total_institutions
            FROM institution_taxonomy
        """)

        taxonomic_result = db.execute(taxonomic_specialization_query, {"taxon_name": taxon_name}).fetchone()

        # 3. 数据质量分析
        data_quality_query = text(f"""
            WITH institution_quality AS (
                SELECT institutioncode,
                       COUNT(*) as total_records,
                       ROUND(
                           COUNT(CASE WHEN decimallatitude IS NOT NULL AND decimallongitude IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2
                       ) as geo_quality,
                       ROUND(
                           COUNT(CASE WHEN eventdate IS NOT NULL AND eventdate != '' THEN 1 END) * 100.0 / COUNT(*), 2
                       ) as date_quality,
                       ROUND(
                           COUNT(CASE WHEN scientificname IS NOT NULL AND family IS NOT NULL AND genus IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2
                       ) as taxonomy_quality,
                       ROUND(
                           COUNT(CASE WHEN collectioncode IS NOT NULL AND collectioncode != '' THEN 1 END) * 100.0 / COUNT(*), 2
                       ) as collection_quality
                FROM dbo.harvestedfn2 
                WHERE {field} = :taxon_name 
                  AND institutioncode IS NOT NULL AND institutioncode != ''
                GROUP BY institutioncode
            ),
            overall_quality AS (
                SELECT institutioncode,
                       total_records,
                       (geo_quality + date_quality + taxonomy_quality + collection_quality) / 4 as overall_score
                FROM institution_quality
            )
            SELECT 
                COUNT(CASE WHEN overall_score > 95 THEN 1 END) as high_quality,
                COUNT(CASE WHEN overall_score BETWEEN 85 AND 95 THEN 1 END) as good_quality,
                COUNT(CASE WHEN overall_score < 85 THEN 1 END) as improving_quality,
                ROUND(AVG(overall_score), 1) as average_quality,
                COUNT(*) as total_institutions
            FROM overall_quality
        """)

        quality_result = db.execute(data_quality_query, {"taxon_name": taxon_name}).fetchone()

        # 4. 收藏规模分析
        collection_scale_query = text(f"""
            WITH institution_scale AS (
                SELECT institutioncode,
                       COUNT(*) as record_count,
                       COUNT(DISTINCT scientificname) as species_count
                FROM dbo.harvestedfn2 
                WHERE {field} = :taxon_name 
                  AND institutioncode IS NOT NULL AND institutioncode != ''
                GROUP BY institutioncode
            )
            SELECT 
                COUNT(CASE WHEN record_count > 10000 THEN 1 END) as major_collections,
                COUNT(CASE WHEN record_count BETWEEN 1000 AND 10000 THEN 1 END) as substantial_collections,
                COUNT(CASE WHEN record_count BETWEEN 100 AND 999 THEN 1 END) as moderate_collections,
                COUNT(CASE WHEN record_count < 100 THEN 1 END) as small_collections,
                SUM(record_count) as total_records,
                COUNT(*) as total_institutions
            FROM institution_scale
        """)

        scale_result = db.execute(collection_scale_query, {"taxon_name": taxon_name}).fetchone()

        # 5. 机构类型分析（基于机构代码模式）
        institution_type_query = text(f"""
            WITH institution_types AS (
                SELECT institutioncode,
                       COUNT(*) as record_count,
                       CASE 
                           WHEN institutioncode ILIKE '%MNH%' OR institutioncode ILIKE '%MUSEUM%' OR institutioncode ILIKE '%MUS%' THEN 'museum'
                           WHEN institutioncode ILIKE '%UNIV%' OR institutioncode ILIKE '%UNI%' OR institutioncode ILIKE '%COLL%' THEN 'university'
                           WHEN institutioncode ILIKE '%GOV%' OR institutioncode ILIKE '%USGS%' OR institutioncode ILIKE '%NOAA%' THEN 'government'
                           WHEN institutioncode ILIKE '%ACAD%' OR institutioncode ILIKE '%INST%' THEN 'research'
                           ELSE 'private'
                       END as institution_type
                FROM dbo.harvestedfn2 
                WHERE {field} = :taxon_name 
                  AND institutioncode IS NOT NULL AND institutioncode != ''
                GROUP BY institutioncode
            )
            SELECT 
                institution_type,
                COUNT(*) as institution_count,
                SUM(record_count) as total_records,
                ROUND(AVG(record_count), 0) as avg_records_per_institution
            FROM institution_types
            GROUP BY institution_type
            ORDER BY institution_count DESC
        """)

        type_result = db.execute(institution_type_query, {"taxon_name": taxon_name}).fetchall()

        # 6. 时间活跃度分析
        temporal_activity_query = text(f"""
            WITH institution_activity AS (
                SELECT institutioncode,
                       COUNT(*) as total_records,
                       MIN(CASE WHEN year IS NOT NULL AND year > 1800 THEN year END) as earliest_year,
                       MAX(CASE WHEN year IS NOT NULL AND year > 1800 THEN year END) as latest_year,
                       COUNT(CASE WHEN year >= 2020 THEN 1 END) as recent_records,
                       COUNT(CASE WHEN year BETWEEN 2010 AND 2019 THEN 1 END) as decade_records
                FROM dbo.harvestedfn2 
                WHERE {field} = :taxon_name 
                  AND institutioncode IS NOT NULL AND institutioncode != ''
                GROUP BY institutioncode
            )
            SELECT 
                COUNT(CASE WHEN recent_records > 0 THEN 1 END) as recently_active,
                COUNT(CASE WHEN decade_records > 0 AND recent_records = 0 THEN 1 END) as historically_active,
                COUNT(CASE WHEN latest_year < 2010 OR latest_year IS NULL THEN 1 END) as inactive,
                ROUND(AVG(CASE WHEN latest_year IS NOT NULL AND earliest_year IS NOT NULL 
                              THEN latest_year - earliest_year ELSE NULL END), 1) as avg_collection_span,
                COUNT(*) as total_institutions
            FROM institution_activity
        """)

        activity_result = db.execute(temporal_activity_query, {"taxon_name": taxon_name}).fetchone()

        # 组装结果
        institution_types_breakdown = {}
        for row in type_result:
            institution_types_breakdown[row[0]] = {
                "count": row[1],
                "totalRecords": row[2],
                "avgRecordsPerInstitution": row[3]
            }

        return {
            "geographicCoverage": {
                "globalCoverage": geographic_result[0] or 0,
                "regionalSpecialists": geographic_result[1] or 0,
                "localCollections": geographic_result[2] or 0,
                "totalInstitutions": geographic_result[3] or 0
            },
            "taxonomicSpecialization": {
                "familySpecialists": taxonomic_result[0] or 0,
                "genusSpecialists": taxonomic_result[1] or 0,
                "regionalFaunaFocus": taxonomic_result[2] or 0,
                "totalInstitutions": taxonomic_result[3] or 0
            },
            "dataQualityLeaders": {
                "highQuality": quality_result[0] or 0,
                "goodQuality": quality_result[1] or 0,
                "improvingQuality": quality_result[2] or 0,
                "averageQuality": float(quality_result[3]) if quality_result[3] else 0.0,
                "totalInstitutions": quality_result[4] or 0
            },
            "collectionScale": {
                "majorCollections": scale_result[0] or 0,
                "substantialCollections": scale_result[1] or 0,
                "moderateCollections": scale_result[2] or 0,
                "smallCollections": scale_result[3] or 0,
                "totalRecords": scale_result[4] or 0,
                "totalInstitutions": scale_result[5] or 0
            },
            "institutionTypes": institution_types_breakdown,
            "temporalActivity": {
                "recentlyActive": activity_result[0] or 0,
                "historicallyActive": activity_result[1] or 0,
                "inactive": activity_result[2] or 0,
                "avgCollectionSpan": float(activity_result[3]) if activity_result[3] else 0.0,
                "totalInstitutions": activity_result[4] or 0
            }
        }

    except Exception as e:
        print(f"Error in institution coverage analysis: {str(e)}")
        # 返回默认值而不是抛出异常
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