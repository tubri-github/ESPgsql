# Search Features Documentation

This document describes the search features implemented in the ESPgsql API.

## Overview

The search API provides multiple search strategies to find species records in Elasticsearch:
- **Exact Search**: Standard term matching
- **Fuzzy Search**: Handles spelling errors with automatic edit distance
- **Phonetic Search**: Finds phonetically similar names (sounds-like matching)
- **Wildcard Search**: Pattern matching with `*` and `?` operators
- **Synonym Resolution**: Automatically resolves synonyms to valid names via ValidName field

## Search Endpoints

### GET /search

Main search endpoint with smart fallback logic.

**Parameters:**
- `scientificname` (str): Species name to search for
- `family` (str): Family name filter
- `country` (str): Country filter
- `institutioncode` (str): Institution code filter
- `size` (int, default=50): Number of results to return
- `from_` (int, default=0): Offset for pagination

**Search Behavior:**
1. Detects if query contains wildcards (`*` or `?`)
2. If wildcards present: Uses wildcard query
3. Otherwise: Uses smart fallback search (exact -> fuzzy if results < threshold)

### GET /search/phonetic

Phonetic search using double_metaphone algorithm.

**How it works:**
- Uses Elasticsearch's analysis-phonetic plugin
- Searches the `.phonetic` subfield of ScientificName, Family, Genus, and ValidName
- Great for finding species when you know how it sounds but not exact spelling

### GET /search/fuzzy

Fuzzy search with configurable fuzziness.

**Parameters:**
- `fuzziness` (str, default="AUTO"): Fuzziness level
  - "AUTO": Edit distance based on term length (recommended)
  - "0", "1", "2": Fixed edit distance

### GET /search/wildcard

Pattern-based search.

**Wildcard Operators:**
- `*`: Matches any sequence of characters (including none)
- `?`: Matches exactly one character

**Examples:**
- `Gadus*` - Matches "Gadus morhua", "Gadus macrocephalus", etc.
- `Gad?s` - Matches "Gadus", "Gadis", etc.
- `*morhua` - Matches anything ending with "morhua"

## Synonym Resolution

### How it Works

The system uses a `ValidName` field populated during ES index sync:

1. During indexing (`sync_es_index.py`):
   - Each record's ScientificName is looked up in TaxonRank database
   - If it's a synonym, the valid name is retrieved
   - Both ScientificName (original) and ValidName (resolved) are indexed

2. During search:
   - Queries search both ScientificName and ValidName fields
   - Finding either the original name or valid name will return matching records

### Example

If "Gadus callarias" is a synonym of "Gadus morhua":
- Searching "Gadus callarias" finds records with `ScientificName="Gadus callarias"` OR `ValidName="Gadus callarias"`
- The valid name resolution ensures synonymous queries return the same specimens

## Document Normalization

All search responses pass through document normalization to ensure consistent field naming:

**Normalized Fields:**
- `CatalogNumber`: Unique specimen identifier
- `ScientificName`: Species name
- `Family`: Taxonomic family
- `ValidName`: Resolved valid name (if synonym)
- `Country`, `Locality`: Geographic information
- `Latitude`, `Longitude`: Coordinates
- `InstitutionCode`, `CollectionCode`: Source institution
- `YearCollected`, `MonthCollected`, `DayCollected`: Collection date
- And more...

## Technical Implementation

### Elasticsearch Index Mapping

Key field configurations:
- `ScientificName`: text + keyword + phonetic subfield
- `Family`: text + keyword + phonetic subfield
- `ValidName`: text + keyword + phonetic subfield
- `Genus`: text + keyword + phonetic subfield
- Geographic/numeric fields: appropriate types (float, long, keyword)

### Phonetic Analyzer

```json
{
  "analysis": {
    "filter": {
      "phonetic_filter": {
        "type": "phonetic",
        "encoder": "double_metaphone",
        "replace": false
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
}
```

### Smart Fallback Logic

```python
# Pseudocode
def smart_search(query):
    results = exact_search(query)
    if len(results) < threshold:  # default threshold: 5
        fuzzy_results = fuzzy_search(query, fuzziness="AUTO")
        if len(fuzzy_results) > len(results):
            return fuzzy_results
    return results
```

## Requirements

- Elasticsearch 8.x with analysis-phonetic plugin
- PostgreSQL with harvestedfn2 table
- TaxonRank database for synonym resolution

## See Also

- `sync_es_index.py`: ES index sync script with ValidName population
- `main.py`: FastAPI application with search endpoints
- `docs/ES_SYNONYM_UPGRADE_PLAN.md`: Future ES synonym filter upgrade plan
