"""
ES Index Sync Script (harvestedfn2_fin)
Sync data from PostgreSQL dbo.harvestedfn2_fin table to Elasticsearch
This table already contains pre-computed validname, no TaxonRank lookup needed.

Usage:
    python sync_es_index_fin.py --action=info      # View index info
    python sync_es_index_fin.py --action=create    # Create new index and sync data
    python sync_es_index_fin.py --action=rebuild   # Delete old index and rebuild
    python sync_es_index_fin.py --action=sync      # Sync data only (index must exist)
"""

import argparse
import os
import signal
import sys
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import settings

# Allow Ctrl+C to kill the process immediately
signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted, exiting..."), os._exit(1)))

# ES client
if settings.ES_USER and settings.ES_PASSWORD:
    es = Elasticsearch(
        settings.ES_URL,
        basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
        verify_certs=False
    )
else:
    es = Elasticsearch(settings.ES_URL, verify_certs=False)

# Database connection
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Index name
INDEX_NAME = "mrservice_harvestedfn2_fin_index"

# Field mapping: harvestedfn2_fin (lowercase) -> ES field name
FIELD_MAPPING = {
    # Core fields
    "catalognumber": "CatalogNumber",
    "species": "ScientificName",
    "validname": "ValidName",              # Pre-computed in harvestedfn2_fin
    "family": "Family",
    "basisofrecord": "BasisOfRecord",
    "institutioncode": "InstitutionCode",
    "collectioncode": "CollectionCode",

    # Geographic fields
    "country": "Country",
    "stateprovince": "StateProvince",
    "county": "County",
    "locality": "Locality",
    "island": "Island",
    "islandgroup": "IslandGroup",
    "continent": "ContinentOcean",

    # Coordinate fields
    "decimallatitude": "Latitude",
    "decimallongitude": "Longitude",
    "coordinateuncertaintyinmeters": "CoordinateUncertaintyInMeters",

    # Time fields
    "year": "YearCollected",
    "month": "MonthCollected",
    "day": "DayCollected",

    # Collection info
    "recordedby": "RecordedBy",
    "individualcount": "IndividualCount",

    # Elevation/depth
    "verbatimelevation": "VerbatimElevation",
    "verbatimdepth": "VerbatimDepth",

    # Other fields
    "preparations": "PreparationType",
    "occurrenceremarks": "Remarks",
    "modified": "DateLastModified",
    "georeferenceprotocol": "GeorefMethod",

    # Extended fields
    "genus": "Genus",
    "specificepithet": "SpecificEpithet",
    "scientificnameauthorship": "ScientificNameAuthorship",
    "order": "Order",
    "class": "Class",
    "phylum": "Phylum",
    "kingdom": "Kingdom",
    "taxonrank": "TaxonRank",
    "taxonomicstatus": "TaxonomicStatus",
    "vernacularname": "VernacularName",
    "occurrenceid": "OccurrenceID",
    "recordnumber": "RecordNumber",
    "sex": "Sex",
    "lifestage": "LifeStage",
    "eventdate": "EventDate",
    "countrycode": "CountryCode",
    "municipality": "Municipality",
    "waterbody": "WaterBody",
    "identifiedby": "IdentifiedBy",
    "dateidentified": "DateIdentified",
    "habitat": "Habitat",
    "associatedmedia": "AssociatedMedia",
    "datasetname": "DatasetName",
}

# Only SELECT the columns we actually need
# Quote reserved words (order, class, etc.)
PG_RESERVED = {"order", "class", "year", "month", "day", "references"}
SELECT_COLUMNS = ", ".join(
    f'"{col}"' if col in PG_RESERVED else col
    for col in FIELD_MAPPING.keys()
)


def text_keyword_field():
    return {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}


def text_keyword_phonetic_field():
    return {
        "type": "text",
        "analyzer": "standard",
        "fields": {
            "keyword": {"type": "keyword", "ignore_above": 256},
            "phonetic": {
                "type": "text",
                "analyzer": "phonetic_analyzer"
            }
        }
    }


INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index": {
            "max_result_window": 100000,
            "refresh_interval": "-1"
        },
        "analysis": {
            "filter": {
                "phonetic_filter": {
                    "type": "phonetic",
                    "encoder": "double_metaphone",
                    "replace": False
                }
            },
            "analyzer": {
                "phonetic_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "phonetic_filter"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # Core fields (phonetic-enabled)
            "ScientificName": text_keyword_phonetic_field(),
            "ValidName": text_keyword_phonetic_field(),
            "Family": text_keyword_phonetic_field(),
            "Genus": text_keyword_phonetic_field(),

            # Core fields (text+keyword)
            "BasisOfRecord": text_keyword_field(),
            "CatalogNumber": text_keyword_field(),
            "InstitutionCode": text_keyword_field(),
            "CollectionCode": text_keyword_field(),
            "Country": text_keyword_field(),
            "StateProvince": text_keyword_field(),
            "County": text_keyword_field(),
            "Locality": text_keyword_field(),
            "Island": text_keyword_field(),
            "IslandGroup": text_keyword_field(),
            "ContinentOcean": text_keyword_field(),
            "RecordedBy": text_keyword_field(),
            "VerbatimElevation": text_keyword_field(),
            "VerbatimDepth": text_keyword_field(),
            "PreparationType": text_keyword_field(),
            "Remarks": text_keyword_field(),
            "GeorefMethod": text_keyword_field(),
            "CoordinateUncertaintyInMeters": text_keyword_field(),

            # Numeric fields
            "Latitude": {"type": "float"},
            "Longitude": {"type": "float"},
            "YearCollected": {"type": "long"},
            "MonthCollected": {"type": "long"},
            "DayCollected": {"type": "long"},
            "IndividualCount": {"type": "long"},

            # Date fields
            "DateLastModified": {"type": "date", "ignore_malformed": True},

            # Extended fields
            "SpecificEpithet": text_keyword_field(),
            "ScientificNameAuthorship": text_keyword_field(),
            "Order": text_keyword_field(),
            "Class": text_keyword_field(),
            "Phylum": text_keyword_field(),
            "Kingdom": text_keyword_field(),
            "TaxonRank": text_keyword_field(),
            "TaxonomicStatus": text_keyword_field(),
            "VernacularName": text_keyword_field(),
            "OccurrenceID": text_keyword_field(),
            "RecordNumber": text_keyword_field(),
            "Sex": text_keyword_field(),
            "LifeStage": text_keyword_field(),
            "EventDate": text_keyword_field(),
            "CountryCode": text_keyword_field(),
            "Municipality": text_keyword_field(),
            "WaterBody": text_keyword_field(),
            "IdentifiedBy": text_keyword_field(),
            "DateIdentified": text_keyword_field(),
            "Habitat": text_keyword_field(),
            "AssociatedMedia": text_keyword_field(),
            "DatasetName": text_keyword_field(),
        }
    }
}


def create_index():
    """Create new ES index"""
    if es.indices.exists(index=INDEX_NAME):
        print(f"Index {INDEX_NAME} already exists!")
        return False

    print(f"Creating index: {INDEX_NAME}")
    es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    print("Index created successfully!")
    return True


def transform_record(record):
    """Transform database record to ES document format"""
    doc = {}
    for db_field, es_field in FIELD_MAPPING.items():
        value = record.get(db_field)
        if value is not None:
            if isinstance(value, str) and value.strip() == "":
                continue
            doc[es_field] = value
    return doc


def generate_actions(db_session, batch_size=10000):
    """Generate bulk actions using keyset pagination, only fetching needed columns"""
    count_query = text('SELECT COUNT(*) FROM dbo.harvestedfn2_fin')
    total_count = db_session.execute(count_query).scalar()
    print(f"Total records: {total_count}")

    last_id = ''
    processed = 0

    while True:
        query = text(f'''
            SELECT {SELECT_COLUMNS} FROM dbo.harvestedfn2_fin
            WHERE catalognumber > :last_id
            ORDER BY catalognumber
            LIMIT :batch_size
        ''')

        results = db_session.execute(query, {"batch_size": batch_size, "last_id": last_id}).fetchall()

        if not results:
            break

        for row in results:
            record = dict(row._mapping)
            doc = transform_record(record)

            if doc:
                yield {
                    "_index": INDEX_NAME,
                    "_id": record.get("catalognumber"),
                    "_source": doc
                }
                processed += 1

            last_id = record.get("catalognumber")

        pct = processed * 100 // total_count if total_count else 0
        print(f"Processed: {processed}/{total_count} ({pct}%)")


def sync_data():
    """Sync data to ES"""
    db = SessionLocal()
    start_time = time.time()
    try:
        print("Starting data sync (parallel_bulk, 4 threads)...")

        success = 0
        failed = 0
        for ok, item in parallel_bulk(
            es,
            generate_actions(db),
            chunk_size=5000,
            thread_count=4,
            raise_on_error=False,
            raise_on_exception=False
        ):
            if ok:
                success += 1
            else:
                failed += 1

        elapsed = time.time() - start_time
        print(f"\nSync completed in {elapsed:.1f}s")
        print(f"Success: {success}")
        print(f"Failed: {failed}")

        print("Re-enabling refresh...")
        es.indices.put_settings(
            index=INDEX_NAME,
            body={"index": {"refresh_interval": "1s"}}
        )
        es.indices.refresh(index=INDEX_NAME)

        stats = es.indices.stats(index=INDEX_NAME)
        doc_count = stats["indices"][INDEX_NAME]["primaries"]["docs"]["count"]
        print(f"Total documents in index: {doc_count}")

    finally:
        db.close()


def get_index_info():
    """Get index info"""
    print("=" * 50)
    print("ES Index Info")
    print("=" * 50)

    print(f"\nIndex: {INDEX_NAME}")
    if not es.indices.exists(index=INDEX_NAME):
        print("  Status: Does not exist")
        return

    stats = es.indices.stats(index=INDEX_NAME)
    doc_count = stats["indices"][INDEX_NAME]["primaries"]["docs"]["count"]
    size_bytes = stats["indices"][INDEX_NAME]["primaries"]["store"]["size_in_bytes"]

    print(f"  Status: Exists")
    print(f"  Document count: {doc_count:,}")
    print(f"  Size: {size_bytes / 1024 / 1024:.2f} MB")


def delete_index():
    """Delete index"""
    if es.indices.exists(index=INDEX_NAME):
        print(f"Deleting index: {INDEX_NAME}")
        es.indices.delete(index=INDEX_NAME)
        print("Index deleted")
        return True
    else:
        print(f"Index {INDEX_NAME} does not exist")
        return False


def main():
    parser = argparse.ArgumentParser(description="ES Index Sync Tool (harvestedfn2_fin)")
    parser.add_argument(
        "--action",
        choices=["create", "rebuild", "sync", "info"],
        required=True,
        help="create: Create new index and sync | rebuild: Delete and rebuild | sync: Sync data only | info: View index info"
    )

    args = parser.parse_args()

    if args.action == "info":
        get_index_info()

    elif args.action == "create":
        if create_index():
            sync_data()

    elif args.action == "rebuild":
        print("=" * 50)
        print("Rebuilding index")
        print("=" * 50)
        delete_index()
        if create_index():
            sync_data()

    elif args.action == "sync":
        if not es.indices.exists(index=INDEX_NAME):
            print(f"Index {INDEX_NAME} does not exist, please run --action=create first")
            sys.exit(1)
        sync_data()


if __name__ == "__main__":
    main()
