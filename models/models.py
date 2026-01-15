# models.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

Base = declarative_base()


# SQLAlchemy models
class HarvestedRecord(Base):
    __tablename__ = "harvestedfn2"
    __table_args__ = {"schema": "dbo"}

    # Use catalognumber as primary key, add ID field if not present
    catalognumber = Column(Integer, primary_key=True)

    # Darwin Core fields
    scientificname = Column(String)
    genus = Column(String)
    specificepithet = Column(String)
    infraspecificepithet = Column(String)
    scientificnameauthorship = Column(String)
    family = Column(String)
    order = Column(String, name="order")  # order is a SQL keyword
    class_ = Column(String, name="class")  # class is a Python keyword
    phylum = Column(String)
    kingdom = Column(String)

    # Taxonomic status
    taxonrank = Column(String)
    taxonomicstatus = Column(String)
    acceptednameusage = Column(String)
    parentnameusage = Column(String)
    higherclassification = Column(String)
    vernacularname = Column(String)
    nomenclaturalcode = Column(String)

    # Record information
    basisofrecord = Column(String)
    occurrenceid = Column(String)
    recordnumber = Column(String)
    recordedby = Column(String)
    individualcount = Column(String)
    organismquantity = Column(String)
    organismquantitytype = Column(String)
    sex = Column(String)
    lifestage = Column(String)
    reproductivecondition = Column(String)

    # Event information
    eventdate = Column(String)
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)

    # Geographic information
    continent = Column(String)
    country = Column(String)
    countrycode = Column(String)
    stateprovince = Column(String)
    county = Column(String)
    municipality = Column(String)
    locality = Column(Text)
    waterbody = Column(String)
    island = Column(String)
    islandgroup = Column(String)

    # Coordinate information
    decimallatitude = Column(Float)
    decimallongitude = Column(Float)
    coordinateuncertaintyinmeters = Column(Float)
    coordinateprecision = Column(Float)

    # Elevation and depth
    minimumelevationinmeters = Column(String)
    maximumelevationinmeters = Column(String)
    verbatimelevation = Column(String)
    minimumdepthinmeters = Column(String)
    maximumdepthinmeters = Column(String)
    verbatimdepth = Column(String)

    # Georeference
    georeferenceddate = Column(String)
    georeferenceprotocol = Column(String)
    georeferenceremarks = Column(String)
    georeferenceverificationstatus = Column(String)

    # Identification information
    identifiedby = Column(String)
    dateidentified = Column(String)
    identificationqualifier = Column(String)
    identificationremarks = Column(String)

    # Institution information
    institutioncode = Column(String)
    institutionid = Column(String)
    ownerinstitutioncode = Column(String)
    collectioncode = Column(String)
    collectionid = Column(String)
    datasetname = Column(String)

    # Other fields
    occurrencestatus = Column(String)
    establishmentmeans = Column(String)
    occurrenceremarks = Column(String)
    habitat = Column(String)
    associatedmedia = Column(String)
    preparations = Column(String)
    samplingprotocol = Column(String)
    references = Column(String, name="references")  # references is a Python keyword

    # Data management
    modified = Column(String)
    informationwithheld = Column(String)
    datageneralizations = Column(String)
    source_file = Column(String)
    processing_date = Column(String)


# Pydantic models for API responses
class TaxonBase(BaseModel):
    scientificname: Optional[str] = None
    vernacularname: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    order: Optional[str] = Field(None, alias="order")
    class_: Optional[str] = Field(None, alias="class")
    phylum: Optional[str] = None
    kingdom: Optional[str] = None
    taxonomicstatus: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class FamilyResponse(TaxonBase):
    family: str
    order: Optional[str] = None
    genera_count: int = 0
    species_count: int = 0
    record_count: int = 0
    countries_count: int = 0
    institutions_count: int = 0
    georeferencing_quality: float = 0.0
    date_quality: float = 0.0
    last_updated: Optional[str] = None


class GenusResponse(TaxonBase):
    genus: str
    family: Optional[str] = None
    order: Optional[str] = None
    species_count: int = 0
    record_count: int = 0
    countries_count: int = 0
    institutions_count: int = 0
    georeferencing_quality: float = 0.0
    date_quality: float = 0.0
    last_updated: Optional[str] = None


class SpeciesResponse(TaxonBase):
    scientificname: str
    authority: Optional[str] = Field(None, alias="scientificnameauthorship")
    vernacularname: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    record_count: int = 0
    countries_count: int = 0
    institutions_count: int = 0
    georeferencing_quality: float = 0.0
    date_quality: float = 0.0
    last_record: Optional[str] = None


class InstitutionResponse(BaseModel):
    institution_code: str = Field(alias="institutioncode")
    institution_name: Optional[str] = None  # Needs to be fetched from another table or mapped
    owner_institution_code: Optional[str] = Field(None, alias="ownerinstitutioncode")
    country: Optional[str] = None
    region: Optional[str] = None  # Needs mapping
    institution_type: Optional[str] = None  # Needs mapping
    record_count: int = 0
    species_count: int = 0
    families_count: int = 0
    countries_count: int = 0
    georeferencing_quality: float = 0.0
    date_quality: float = 0.0
    collection_codes: List[str] = []
    last_updated: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class RecordResponse(BaseModel):
    id: int = Field(alias="catalognumber")
    catalog_number: Optional[str] = Field(None, alias="catalognumber")
    scientific_name: Optional[str] = Field(None, alias="scientificname")
    vernacular_name: Optional[str] = Field(None, alias="vernacularname")
    family: Optional[str] = None
    genus: Optional[str] = None
    recorded_by: Optional[str] = Field(None, alias="recordedby")
    event_date: Optional[str] = Field(None, alias="eventdate")
    country: Optional[str] = None
    locality: Optional[str] = None
    decimal_latitude: Optional[float] = Field(None, alias="decimallatitude")
    decimal_longitude: Optional[float] = Field(None, alias="decimallongitude")
    institution_code: Optional[str] = Field(None, alias="institutioncode")
    collection_code: Optional[str] = Field(None, alias="collectioncode")
    basis_of_record: Optional[str] = Field(None, alias="basisofrecord")

    class Config:
        allow_population_by_field_name = True


class TaxonomyStatsResponse(BaseModel):
    total_families: int
    total_genera: int
    total_species: int
    total_records: int
    total_institutions: int
    total_countries: int
    high_diversity_families: int = 0
    well_sampled_families: int = 0
    global_coverage_families: int = 0
    avg_georeferencing: float = 0.0


class InstitutionStatsResponse(BaseModel):
    total_institutions: int
    total_collection_codes: int
    total_records: int
    total_countries: int
    avg_georeferenced: float = 0.0
    avg_date_quality: float = 0.0
    major_contributors: int = 0
    active_contributors: int = 0
    research_collections: int = 0
    high_quality_data: int = 0


class PaginatedResponse(BaseModel):
    data: List[dict]
    page: int
    per_page: int
    total: int
    pages: int


# Query parameter models
class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    per_page: int = Field(50, ge=1, le=200)


class TaxonomyFilterParams(PaginationParams):
    search: Optional[str] = None
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    record_count: Optional[str] = None  # high, medium, low
    data_quality: Optional[str] = None  # excellent, good, poor
    region: Optional[str] = None
    sort_by: str = "records_desc"


class InstitutionFilterParams(PaginationParams):
    search: Optional[str] = None
    region: Optional[str] = None
    record_count: Optional[str] = None
    institution_type: Optional[str] = None
    sort_by: str = "records_desc"


class RecordFilterParams(PaginationParams):
    search: Optional[str] = None
    year: Optional[int] = None
    country: Optional[str] = None
    institution_code: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    sort_by: str = "date_desc"