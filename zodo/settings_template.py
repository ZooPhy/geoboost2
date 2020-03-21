'''Template for properties file. Original should NEVER be commited to GitHub.'''

# Create your NCBI key and insert key-string here
API_KEY = ""

# Hostname of server where Geonames search is hosted for queries
GEO_HOST = "localhost"

# port number for access of Geonames search service
GEO_PORT = "8092"

# GenBank accession records can be retrieved in multiples but fails if too many are requested
# the following allows the accessions to be retrieved in batches. Max size seems to be somewhere between 250 or 400.
GB_BATCH_SIZE = 300

# if location information is insufficient we can extract external links to PUBMED or PMC articles i.e. articles citing accessions
# this can be helpful, although it takes some extra time to gather links and texts from all those articles
EXTRACT_LINKS = True

# When extracting text from Open Access articles, there may exist supplementary material that may need to be processed as well
# If the following option is set, the supplementary material is downloaded, text is extracted and processed
# Keep in mind that it takes some extra time to download and process these supplemental articles
EXTRACT_SUPPLEMENTAL = True
SUPPLEMENTAL_DATA_DIR = "resources/SUPPL/"
SUPPLEMENTAL_DATA_FILETYPES = {"doc", "pdf"}

# If links are to be extracted then we can query data in batches. 
GB_PM_LINK_BATCH_SIZE = 150

# the following sections are processed in PubMed abstracts
PM_TYPES = ["title", "abstract"]
# the following sections are processed in PubMedCentral Open Access articles
PMCOA_TYPES = ["front", "paragraph", "fig_caption", "table", "table_caption"]

# ---------REDIS SETTINGS-------------
# We use redis for caching to reduce server calls and computation for repeated requests. By default it is set to True for standalone versions.
# We use redis on the ZoDo server. Hence it is set to True on the server.
USE_REDIS = True

# hostname, password and port number for access of Redis storage
REDIS_HOST = "localhost"
REDIS_PASSWORD = ""
REDIS_PORT = "6379"

# Logical separation of caches
# DB for Local PMC cache storage in Redis
# Only stores raw documents for quicker fetch and processing
REDIS_PMC_CACHE_DB = 1
# DB for storing PMC processed results
REDIS_PMC_PROCESSED_DB = 2
# DB for GB object storage in Redis
REDIS_GB_PROCESSED_DB = 3
