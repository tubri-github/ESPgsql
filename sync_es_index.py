"""
ES Index Sync Script
Sync data from PostgreSQL dbo.harvestedfn2 table to Elasticsearch
Supports synonym resolution: get valid names via TaxonRank database

Usage:
    python sync_es_index.py --action=info      # View index info
    python sync_es_index.py --action=create    # Create new index and sync data
    python sync_es_index.py --action=rebuild   # Delete old index and rebuild
    python sync_es_index.py --action=sync      # Sync data only (index must exist)

New index name: mrservice_harvestedfn2_index (keeps old index mrservice_full_index)
"""

import argparse
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import settings
print('ES_URL:', settings.ES_URL)
print('ES_USER:', settings.ES_USER)
from elasticsearch import Elasticsearch
from config import settings

es = Elasticsearch(
    settings.ES_URL,
    basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
    verify_certs=False
)

# ES client
# Use basic_auth if ES_USER and ES_PASSWORD are configured
if settings.ES_USER and settings.ES_PASSWORD:
    es = Elasticsearch(
        settings.ES_URL,
        basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
        verify_certs=False
    )
else:
    es = Elasticsearch(settings.ES_URL, verify_certs=False)

# harvestedfn2 database connection
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# TaxonRank database connection (for synonym resolution)
# Set TAXON_DB_URL in .env file, e.g.: TAXON_DB_URL=postgresql://user:pass@localhost:5432/taxonomy_dev
taxon_engine = create_engine(settings.TAXON_DB_URL) if settings.TAXON_DB_URL else None
TaxonSession = sessionmaker(autocommit=False, autoflush=False, bind=taxon_engine) if taxon_engine else None

# Synonym cache {scientific_name: valid_name}
_valid_name_cache = {}
_cache_loaded = False


def load_valid_name_cache():
    """Load all synonym mappings from TaxonRank database into cache"""
    global _valid_name_cache, _cache_loaded

    if _cache_loaded:
        return

    if not TaxonSession:
        print("Warning: TAXON_DB_URL not configured in .env, skipping synonym resolution")
        _cache_loaded = True
        return

    print("Loading synonym cache...")
    taxon_db = TaxonSession()
    try:
        # Query all species and their valid names
        # valid species: valid_name = itself
        # synonym species: valid_name = points to valid name
        query = text('''
            SELECT
                t.scientific_name,
                COALESCE(v.scientific_name, t.scientific_name) as valid_name
            FROM taxa t
            LEFT JOIN taxa v ON t.valid_id = v.id
            WHERE t.rank = 'SPECIES'
        ''')

        results = taxon_db.execute(query).fetchall()

        for row in results:
            name = row[0].strip() if row[0] else None
            valid = row[1].strip() if row[1] else None
            if name and valid:
                _valid_name_cache[name.lower()] = valid

        _cache_loaded = True
        print(f"Synonym cache loaded: {len(_valid_name_cache)} records")

    except Exception as e:
        print(f"Warning: Unable to load synonym cache - {e}")
        print("Will continue running without synonym resolution")
    finally:
        taxon_db.close()


def get_valid_name(scientific_name):
    """Get valid name for species (returns valid name if synonym, otherwise original name)"""
    if not scientific_name:
        return None

    # Ensure cache is loaded
    if not _cache_loaded:
        load_valid_name_cache()

    # Find valid name (case-insensitive)
    name_lower = scientific_name.strip().lower()
    return _valid_name_cache.get(name_lower, scientific_name)


# New index name (keeps old mrservice_full_index)
NEW_INDEX_NAME = "mrservice_harvestedfn2_index"
OLD_INDEX_NAME = "mrservice_full_index"

# Field mapping: harvestedfn2 (lowercase) -> ES (same naming as old index)
# Based on actual mapping structure of old index mrservice_full_index
FIELD_MAPPING = {
    # Core fields - consistent with old index
    "catalognumber": "CatalogNumber",
    "species": "ScientificName",             # harvestedfn2.species -> ES.ScientificName
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
    "continent": "ContinentOcean",           # Old index uses ContinentOcean

    # Coordinate fields
    "decimallatitude": "Latitude",
    "decimallongitude": "Longitude",
    "coordinateuncertaintyinmeters": "CoordinateUncertaintyInMeters",

    # Time fields - old index uses xxxCollected naming
    "year": "YearCollected",
    "month": "MonthCollected",
    "day": "DayCollected",

    # Collection info
    "recordedby": "RecordedBy",              # Uses Darwin Core standard naming
    "individualcount": "IndividualCount",

    # Elevation/depth
    "verbatimelevation": "VerbatimElevation",
    "verbatimdepth": "VerbatimDepth",

    # Other fields
    "preparations": "PreparationType",       # Old index uses PreparationType
    "occurrenceremarks": "Remarks",          # Old index uses Remarks
    "modified": "DateLastModified",          # Old index uses DateLastModified
    "georeferenceprotocol": "GeorefMethod",  # Old index uses GeorefMethod

    # Fields in harvestedfn2 but not in old index (new, extended features)
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

# Generic text+keyword field definition (consistent with old index)
def text_keyword_field():
    return {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}


# text+keyword definition with phonetic subfield (for species name search)
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


# ES Index Mapping definition
# Style consistent with old index: most fields use text+keyword combination
# Species name fields have phonetic subfield for phonetic search
INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index": {
            "max_result_window": 100000
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
            # ========== Core fields from old index ==========
            # ScientificName, Family support phonetic search
            "ScientificName": text_keyword_phonetic_field(),
            "Family": text_keyword_phonetic_field(),
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
            "RecordedBy": text_keyword_field(),           # Uses Darwin Core standard naming
            "VerbatimElevation": text_keyword_field(),
            "VerbatimDepth": text_keyword_field(),
            "PreparationType": text_keyword_field(),
            "Remarks": text_keyword_field(),
            "GeorefMethod": text_keyword_field(),
            "CoordinateUncertaintyInMeters": text_keyword_field(),

            # Numeric fields (consistent with old index)
            "Latitude": {"type": "float"},
            "Longitude": {"type": "float"},
            "YearCollected": {"type": "long"},
            "MonthCollected": {"type": "long"},
            "DayCollected": {"type": "long"},
            "IndividualCount": {"type": "long"},

            # Date fields
            "DateLastModified": {"type": "date", "ignore_malformed": True},

            # ========== Synonym resolution field ==========
            "ValidName": text_keyword_phonetic_field(),  # Valid name (resolved via TaxonRank, supports phonetic)

            # ========== New fields from harvestedfn2 (extended) ==========
            "Genus": text_keyword_phonetic_field(),      # Genus name (supports phonetic)
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
    if es.indices.exists(index=NEW_INDEX_NAME):
        print(f"Index {NEW_INDEX_NAME} already exists!")
        return False

    print(f"Creating new index: {NEW_INDEX_NAME}")
    es.indices.create(index=NEW_INDEX_NAME, body=INDEX_MAPPING)
    print("Index created successfully!")
    return True


def transform_record(record):
    """Transform database record to ES document format"""
    doc = {}
    for db_field, es_field in FIELD_MAPPING.items():
        value = record.get(db_field)
        if value is not None:
            # Skip empty strings
            if isinstance(value, str) and value.strip() == "":
                continue
            doc[es_field] = value

    # Add ValidName field (synonym resolution via TaxonRank)
    scientific_name = doc.get("ScientificName")
    if scientific_name:
        valid_name = get_valid_name(scientific_name)
        if valid_name:
            doc["ValidName"] = valid_name

    return doc


def generate_actions(db_session, batch_size=5000):
    """Generate bulk import actions"""
    # Get total record count
    count_query = text('SELECT COUNT(*) FROM dbo.harvestedfn2')
    total_count = db_session.execute(count_query).scalar()
    print(f"Total records: {total_count}")

    # Batch query
    offset = 0
    processed = 0

    while offset < total_count:
        query = text(f'''
            SELECT * FROM dbo.harvestedfn2
            ORDER BY catalognumber
            OFFSET {offset} ROWS FETCH NEXT {batch_size} ROWS ONLY
        ''')

        results = db_session.execute(query).fetchall()

        for row in results:
            record = dict(row._mapping)
            doc = transform_record(record)

            if doc:
                yield {
                    "_index": NEW_INDEX_NAME,
                    "_id": record.get("catalognumber"),
                    "_source": doc
                }
                processed += 1

        offset += batch_size
        print(f"Processed: {processed}/{total_count} ({processed*100//total_count}%)")


def sync_data():
    """Sync data to ES"""
    db = SessionLocal()
    try:
        print("Starting data sync...")

        success, failed = bulk(
            es,
            generate_actions(db),
            chunk_size=1000,
            raise_on_error=False,
            raise_on_exception=False
        )

        print(f"\nSync completed!")
        print(f"Success: {success}")
        print(f"Failed: {failed}")

        es.indices.refresh(index=NEW_INDEX_NAME)

        stats = es.indices.stats(index=NEW_INDEX_NAME)
        doc_count = stats["indices"][NEW_INDEX_NAME]["primaries"]["docs"]["count"]
        print(f"Total documents in index: {doc_count}")

    finally:
        db.close()


def get_index_info():
    """Get info for all related indexes"""
    print("=" * 50)
    print("ES Index Info")
    print("=" * 50)

    for index_name in [OLD_INDEX_NAME, NEW_INDEX_NAME]:
        print(f"\nIndex: {index_name}")
        if not es.indices.exists(index=index_name):
            print("  Status: Does not exist")
            continue

        stats = es.indices.stats(index=index_name)
        doc_count = stats["indices"][index_name]["primaries"]["docs"]["count"]
        size_bytes = stats["indices"][index_name]["primaries"]["store"]["size_in_bytes"]

        print(f"  Status: Exists")
        print(f"  Document count: {doc_count:,}")
        print(f"  Size: {size_bytes / 1024 / 1024:.2f} MB")


def delete_index():
    """Delete new index"""
    if es.indices.exists(index=NEW_INDEX_NAME):
        print(f"Deleting index: {NEW_INDEX_NAME}")
        es.indices.delete(index=NEW_INDEX_NAME)
        print("Index deleted")
        return True
    else:
        print(f"Index {NEW_INDEX_NAME} does not exist")
        return False


def main():
    parser = argparse.ArgumentParser(description="ES Index Sync Tool")
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
        if not es.indices.exists(index=NEW_INDEX_NAME):
            print(f"Index {NEW_INDEX_NAME} does not exist, please run --action=create first")
            sys.exit(1)
        sync_data()


if __name__ == "__main__":
    main()
