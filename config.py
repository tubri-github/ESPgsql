# -*- coding:utf-8 -*-
"""
@Des: basic configuration
"""

import os
from dotenv import load_dotenv, find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List


# Load .env file based on environment
# Use .env.production if ENV=production, otherwise use .env
env_file = ".env.production" if os.getenv("ENV") == "production" else ".env"
load_dotenv(find_dotenv(env_file), override=True)


class Config(BaseSettings):
    # debug/prod
    APP_DEBUG: bool = os.getenv('APP_DEBUG', 'true').lower() in ['true', '1', 't']

    # project info
    VERSION: str = "1.0.0"
    PROJECT_NAME: str = "Temp Fishnet2 API"
    DESCRIPTION: str = '<a href="/redoc" target="_blank">redoc</a>'

    # static resources path
    STATIC_DIR: str = os.path.join(os.getcwd(), "static")
    TEMPLATE_DIR: str = os.path.join(STATIC_DIR, "templates")

    # cors request
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ])
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default_factory=lambda: ["*"])

    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/fn2_legacy')

    # Elasticsearch
    ES_URL: str = os.getenv('ES_URL', 'http://localhost:9200')
    ES_USER: str = os.getenv('ES_USER', '')
    ES_PASSWORD: str = os.getenv('ES_PASSWORD', '')
    # ES_INDEX: str = os.getenv('ES_INDEX', 'mrservice_full_index')
    ES_INDEX: str = os.getenv('ES_INDEX', 'mrservice_harvestedfn2_index')

    # TaxonRank database (for synonym resolution)
    TAXON_DB_URL: str = os.getenv('TAXON_DB_URL', '')


settings = Config()