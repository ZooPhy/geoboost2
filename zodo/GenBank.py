
"""
Contains classes for GenBank Request and the individual GenBank Record object
"""

import re
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import List
import copy
import json
import jsonpickle
import redis
import regex as nre
from requests.utils import quote
import logging
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="zodo")
from geopy.point import Point

from zodo.settings import (API_KEY, EXTRACT_LINKS, GB_BATCH_SIZE,
                           GB_PM_LINK_BATCH_SIZE, REDIS_GB_PROCESSED_DB,
                           REDIS_HOST, REDIS_PASSWORD, REDIS_PORT, USE_REDIS)
from zodo.utils import (http_get_query, MODE_STRICT, MODE_EXPANDED, MODE_DESPERATE,
                        lookup_location_pop, GB_COUNTRY_FIELD_REGEX)
from zodo.PubMed import PubMedRequest

INSUFFICIENT = ["CONT", "RGN", "PEN"]

GB_POST_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi?db=nuccore&id=IDSPH"+("&api_key="+API_KEY if API_KEY else "")
GB_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&query_key=KEYPH&WebEnv=WEBENVPH&rettype=gb&retmode=xml"+("&api_key="+API_KEY if API_KEY else "")

GB_PUBMED_LINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=nuccore&db=pubmed&id=IDPH&linkname=nuccore_pubmed,nuccore_pubmed_accn&idtype=acc&retmode=json"+("&api_key="+API_KEY if API_KEY else "")
GB_PMC_LINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=nuccore&db=pmc&id=IDPH&linkname=nuccore_pmc&idtype=acc&retmode=json"+("&api_key="+API_KEY if API_KEY else "")
PM_PMC_LINK_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0?ids=IDSPH&format=json"

QK_REGEX = r"<QueryKey>(.*)</QueryKey>"
WE_REGEX = r"<WebEnv>(.*)</WebEnv>"
GB_PUB_EXTRACT_LINK_REGEX = r"<Link>\s*<Id>([0-9]+)</Id>\s*</Link>"
GB_PUB_LINK_REGEX = r"<Link>\s*<Id>([0-9]+)</Id>\s*</Link>"
PMC_PUB_LINK_REGEX = r"pmcid=\"PMC[0-9]+\" pmid=\"([0-9]+)\""

STRAIN_LOCATION_REGEX = r"(?:\/|\()([A-Z]{2,3}|(?:[A-Z]{1}[a-z]+[ -]?(?:of |de |do |d')?)+|(?:[A-Z]{1}[a-z]+-[A-Z]{2}))\/"
# STRAIN_LOCATION_REGEX = r"([A-Z]{2,3}|(?:[A-Z]{1}[a-z]+[ -]?(?:of |de |do |d')?)+|(?:[A-Z]{1}[a-z]+-[A-Z]{2}))\/"

SOURCE_METADATA_FIELDS = set(["country", "lat_lon", "strain", "organism", "isolate", "host", "db_xref", "collection_date"])

STRAIN_BLACKLIST = set(["unk", "chicken", "pig", "goose", "dog", "cat", "duck", "swine"])

HIGH_PROB_TOKENS_RE = r'|'.join(["collect", "obtain", "isolated", "extract", "sampled"])

COUNTRY_PROBABILITY = 1.0
LATLON_PROBABILITY = 1.0
STRAIN_PROBABILITY = 0.9
KEYWORDS_PRESCENCE_PROBABILITY = 0.9
METADATA_PRESCENCE_PROBABILITY = 0.8
FLU_STRAIN_PROBABILITY = 1.0
FLU_KEYWORDS_PRESCENCE_PROBABILITY = 0.2
FLU_METADATA_PRESCENCE_PROBABILITY = 0.3

LOCATION_MAP = {"Viet Nam":"Vietnam"}

class GenBankRequest(object):
    '''GenBankRecord Request Object'''
    def __init__(self, accessionids: List[str], suff_level: str, max_locs: int):
        """Constructor creates a GenBankRequest for processing requested GenBank Accessions
        Handles processing the accessions and returning serialized objects based on return
        Arguments:
            object {GenBankRequest} -- self
            accessionids {List[str]} -- A list of accession ids formatted as strings
            suff_level {str} -- Sufficiency Level based on GeoNames Code type (Choose among ["ADM1", "ADM2", "ADM3"])
            max_locs {int} -- Number of possible locations to be returned per GenBank accession
        """
        self.accessionids = accessionids
        self.suff_level = suff_level
        self.insufficient_codes = []
        self.max_locs = max_locs
        self.genbank_records = []
        self.errors = []

    def process_genbank_ids(self, extract_pubmed=True):
        '''Get GenBank objects using APIs'''
        # First fetch GenBank record objects from NCBI and create objects
        self.extract_gb_objects()

        # Determine if the extracted locations in metadata are sufficient
        self.check_sufficiency_level()

        # Extract associated PubMed IDs if necessary
        if extract_pubmed:
            self.extract_pubmed_records()

        # extract location of infected host (LOIH) based of heuristics
        for gb_rec in self.genbank_records:
            gb_rec.extract_loih(self.insufficient_codes, self.max_locs)

    def extract_gb_objects(self):
        '''Extract GB objects from the response'''
        self.accessionids = list(OrderedDict((x, True) for x in self.accessionids).keys())
        # get accessions that are not cached
        accessions_for_retrieval, cached_accessions = [], {}
        # if redis is used get the connection
        if USE_REDIS:
            gbred = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_GB_PROCESSED_DB)
            for accid in self.accessionids:
                if gbred.exists(accid):
                    # if found, get it from cache
                    gbr_json = gbred.get(accid)
                    # logging.debug("%s found in GB cache DB", accid)
                    gbr = jsonpickle.decode(gbr_json)
                    cached_accessions[accid] = gbr
                else:
                    # logging.debug("%s not found in GB cache DB", accid)
                    # add ids not found to download list
                    accessions_for_retrieval.append(accid)
            logging.info("Uncached %s/%s accessions", len(cached_accessions), len(self.accessionids))
        else:
            # add everything to download list
            accessions_for_retrieval = self.accessionids[:]

        # Now download the ids listed for retrieval
        downloaded_accessions = {}
        # If there are accession ids
        if accessions_for_retrieval:
            logging.info("Downloading %s accessions", len(accessions_for_retrieval))
            for index in range(0, len(accessions_for_retrieval), GB_BATCH_SIZE):
                batch_accessions = accessions_for_retrieval[index:index + GB_BATCH_SIZE]
                gbidsstr = ",".join(batch_accessions)
                # replace placeholder for accession ids
                url = GB_POST_URL.replace("IDSPH", quote(gbidsstr))
                resp = http_get_query(url, timeout=30, retries=2)
                if not resp or not resp.text:
                    continue
                xml = resp.text
                query_re = re.search(QK_REGEX, xml)
                query_key, web_env = None, None
                if query_re:
                    query_key = query_re.group(1)
                web_re = re.search(WE_REGEX, xml)
                if web_re:
                    web_env = web_re.group(1)
                # replace placeholder for NCBI API Key
                if query_key and web_env:
                    url = GB_FETCH_URL.replace("KEYPH", query_key).replace("WEBENVPH", web_env)
                    resp = http_get_query(url, timeout=30, retries=2)
                    if not resp or not resp.text:
                        continue
                    gb_xml = resp.text
                    try:
                        root = ET.fromstring(gb_xml)
                    except Exception as e:
                        logging.error("Error '%s' parsing XML for url %s", e, url)
                        continue
                    # iterate through the xml records
                    for child in root:
                        source = {} # dictionary object to store source metadata
                        accid, pmid = "", ""
                        # get accession 
                        accid = child.find('GBSeq_primary-accession').text
                        # get pubmed id if that is available
                        pmsearch = child.find('.//GBReference_pubmed')
                        pmid = pmsearch.text if pmsearch is not None else ""
                        # get location from country field
                        for item in child.findall('.//GBQualifier'):
                            if len(item) == 2:
                                x, y = item
                                if x.text in SOURCE_METADATA_FIELDS:
                                    source[x.text] = y.text
                        gbr = GenBankRecord(accid, pmid, source)
                        downloaded_accessions[accid] = gbr
                logging.info("Downloaded %s / %s", len(downloaded_accessions), len(accessions_for_retrieval))

        # now collect all accessions in the order they were requested
        for accid in self.accessionids:
            if accid in cached_accessions:
                self.genbank_records.append(cached_accessions[accid])
            elif accid in downloaded_accessions:
                gbr = downloaded_accessions[accid]
                self.genbank_records.append(gbr)
            else:
                logging.error("ERROR: %s could not be downloaded or processed", accid)

        # Normalize the fields in the genbank metadata
        self.normalize_metadata_fields()

        # If required get the externally linked pubmed records i.e. pubmed records citing accessions
        if EXTRACT_LINKS and self.genbank_records:
            logging.debug("Extracting PubMed Links")
            self.extract_pubmed_links()

        # add to cache if using redis
        if USE_REDIS:
            for accid, gbr in downloaded_accessions.items():
                gbr_json = jsonpickle.encode(gbr)
                gbred.set(accid, gbr_json)
                # logging.debug("Caching %s", accid)

        logging.info("Processed %s / %s", len(self.genbank_records), len(self.accessionids))

    def normalize_metadata_fields(self):
        '''Normalize GenBank metadata'''
        # -- Normalize Country field --
        # map accession ids to locations in the country field for faster processing
        # this is because a request will often contain same geographical location
        loc_map = {}
        for gb_rec in self.genbank_records:
            # If already normalized, then skip
            if gb_rec.country_loc or gb_rec.latlon_loc:
                continue
            source_country = gb_rec.source["country"] if "country" in gb_rec.source else None
            if source_country:
                source_country = LOCATION_MAP[source_country] if source_country in LOCATION_MAP else source_country
                # convert camel case to whitespace delimited text i.e. LaoCai -> Lao Cai
                source_country = re.sub(r"([a-z])([A-Z])",r"\g<1> \g<2>", source_country)
                if source_country not in loc_map:
                    # Resolve country field
                    loc = lookup_location_pop(source_country, MODE_DESPERATE)
                    loc = loc[0] if len(loc)>0 else None
                    loc_map[source_country] = loc
                else:
                    loc = loc_map[source_country]
                gb_rec.country_loc = loc
            
            # now process the latitude longitude if it exists in the metadata and perform a reverse lookup
            source_latlon = gb_rec.source["lat_lon"] if "lat_lon" in gb_rec.source else None
            if source_latlon:
                if source_latlon not in loc_map:
                    try:
                        location = geolocator.reverse(Point(source_latlon), language='en')
                    except Exception as e:
                        logging.warning("Geocoder timed out for %s - %s : Message: %s", source_latlon, gb_rec.accid, e)
                    if location and "address" in location.raw:
                        # Resolve country field
                        address = location.raw["address"]
                        location_str = ", ".join([address[x] for x in ["city", "county", "state", "country"] if x in address])
                        logging.debug("Reverse lookup %s -> %s", source_latlon, location_str)
                        if location_str not in loc_map:
                            loc = lookup_location_pop(location_str, MODE_DESPERATE)
                            loc = loc[0] if len(loc)>0 else None
                            loc_map[location_str] = loc
                        else:
                            loc = loc_map[location_str]
                        loc_map[source_latlon] = loc
                else:
                    loc = loc_map[source_latlon]
                gb_rec.latlon_loc = loc

        # -- Extract Location from Strain/Organism/Isolate --
        # TODO: This can further be improved by considering them separately
        for gb_rec in self.genbank_records:
            # If already normalized, then skip
            if gb_rec.strain_loc:
                continue
            source_country = gb_rec.source["country"] if "country" in gb_rec.source else None
            gb_rec.is_flu = True if "organism" in gb_rec.source and "Influenza" in gb_rec.source["organism"] else False
            source_strain = " | ".join(gb_rec.source[x] for x in ["organism", "isolate", "strain"] if x in gb_rec.source)
            if source_strain:
                query_re = []
                if gb_rec.is_flu:
                    strain = gb_rec.source["strain"] if "strain" in gb_rec.source else gb_rec.source["organism"]
                    strain_parts = strain.split("/")
                    if len(strain_parts) >= 4:
                        query_re = [strain_parts[1]] if len(strain_parts)==4 else [strain_parts[2]]
                        query_re = [x for x in query_re if x.lower() not in STRAIN_BLACKLIST]
                if not query_re:
                    query_re = [x for x in nre.findall(STRAIN_LOCATION_REGEX, source_strain, overlapped=True) if x]
                else:
                    logging.debug("Detected strain location '%s' in '%s'", query_re[0], gb_rec.pmid)
                # if text is found, process it
                query_re = [x for x in query_re if x.lower() not in STRAIN_BLACKLIST]
                if not query_re:
                    continue
                # pick the last one for now
                loc_txt = query_re[-1].strip()
                # convert camel case to whitespace delimited text i.e. LaoCai -> Lao Cai
                loc_txt = re.sub(r"([a-z])([A-Z])",r"\g<1> \g<2>", loc_txt)
                # add excerpt in strain to location only if not already in the country field
                if "-" in loc_txt:
                    parts = [x.strip() for x in loc_txt.split("-") if x.strip()]
                    if source_country:
                       parts = [x for x in parts if x.lower() not in source_country.lower()]
                    loc_txt = ", ".join(parts)
                if source_country:
                    source_country = LOCATION_MAP[source_country] if source_country in LOCATION_MAP else source_country
                    loc_txt = LOCATION_MAP[loc_txt] if loc_txt in LOCATION_MAP else loc_txt
                    # if location in strain field is same as country field, copy the location
                    if loc_txt == source_country:
                        gb_rec.strain_loc = gb_rec.country_loc
                        continue
                    elif len(loc_txt)>3 and loc_txt.lower() in source_country.lower():
                        continue
                    # if there is a finer location in country add excerpt and process again
                    if ":" in source_country:
                        gb_format = re.search(GB_COUNTRY_FIELD_REGEX, source_country)
                        if gb_format:
                            country = gb_format.group(1)
                            finer_loc = ", ".join([x.strip() for x in gb_format.group(2).split("-")])
                            source_country = finer_loc.strip() + ", " + country.strip()
                loc_txt = loc_txt.strip()
                # attach extracted part to Strain
                if loc_txt:
                    gb_rec.metadata_extract = loc_txt
                    loc_txt += ", "+source_country if source_country else ""
                    if loc_txt not in loc_map:
                        loc = lookup_location_pop(loc_txt, MODE_STRICT)
                        loc = loc[0] if len(loc)>0 else None
                        loc_map[loc_txt] = loc
                    else:
                        loc = loc_map[loc_txt]
                    # if the countries resolved are different, it isn't probably correct
                    if loc:
                        # check if the countries match, this is good for most (>98%) records
                        # TODO: make it generic to support other ancestors
                        if gb_rec.country_loc or gb_rec.latlon_loc:
                            for resolved_loc in [gb_rec.country_loc, gb_rec.latlon_loc]:
                                if resolved_loc:
                                    for ancestor_type in ["Country", "Continent"]:
                                        if ancestor_type in resolved_loc and ancestor_type in loc:
                                            if loc[ancestor_type] == resolved_loc[ancestor_type]:
                                                gb_rec.strain_loc = loc
                                                continue
                                            else:
                                                break
                                else:
                                    continue
                        else:
                            # if no locations were found in location specific metadata, then
                            gb_rec.strain_loc = loc
                    else:
                        continue

        # -- see normalization results --
        # for gb_rec in self.genbank_records:
        #     logging.debug("GenBank record %s", gb_rec)

    def check_sufficiency_level(self):
        '''Normalize GenBank metadata'''
        # determine which geoname ID types can be deemed insufficient
        self.insufficient_codes = INSUFFICIENT[:]
        # Check if requesting ADM level, if so, add country
        if self.suff_level[:3] == "ADM":
            self.insufficient_codes += ["PCLI", "PCLH", "PCLD", "PCLF", "PCLS", "PCLIX", "PCL"]
            # for finer locations add states and counties wherever applicable
            if self.suff_level == "ADM3":
                self.insufficient_codes += ["ADM1", "ADM1H", "ADM2"]
            elif self.suff_level == "ADM2":
                self.insufficient_codes += ["ADM1", "ADM1H"]
        self.insufficient_codes = set(self.insufficient_codes)
        logging.info("Insufficient codes %s", self.insufficient_codes)

        for gb_rec in self.genbank_records:
            # we say location is insufficient
            if ((gb_rec.country_loc and gb_rec.country_loc["Code"] not in self.insufficient_codes)
                or (gb_rec.latlon_loc and gb_rec.latlon_loc["Code"] not in self.insufficient_codes)):
                gb_rec.sufficient = True

    def extract_pubmed_links(self):
        '''
        Some records do not have PubMed links inside the metadata.
        For them look if links were added externally i.e. by citing in PubMed or PMC
        For some reason, PMC has more links to GenBank than PubMed and are often different
        '''
        pubmedid_map, pmcid_map = {}, {}
        # process in batches
        for index in range(0, len(self.accessionids), GB_PM_LINK_BATCH_SIZE):
            batch_accessions = self.accessionids[index:index + GB_PM_LINK_BATCH_SIZE]
            # first extract links to PubMed under linknames nuccore_pubmed and nuccore_pubmed_accn
            url = GB_PUBMED_LINK_URL.replace("IDPH", "&id=".join(batch_accessions))
            resp = http_get_query(url, timeout=20, retries=2)
            if resp and resp.text:
                jsonobj = json.loads(resp.text)
                if "linksets" in jsonobj:
                    for linkset in jsonobj["linksets"]:
                        if "linksetdbs" in linkset:
                            if "ids" in linkset and len(linkset["ids"]) > 0:
                                # remove the version info
                                accid = linkset["ids"][0].split('.')[0]
                                pmids = []
                                for linksetdb in linkset["linksetdbs"]:
                                    pmids = linksetdb["links"]
                                    if accid not in pubmedid_map:
                                        pubmedid_map[accid] = pmids
                                    else:
                                        pubmedid_map[accid].extend(pmids)

            # next extract links to PubMedCentral under linknames nuccore_pmc.
            # I'm not sure if there are other linknames. If so, investigate and add if relevant.
            url = GB_PMC_LINK_URL.replace("IDPH", "&id=".join(batch_accessions))
            resp = http_get_query(url, timeout=20, retries=2)
            if resp and resp.text:
                jsonobj = json.loads(resp.text)
                if "linksets" in jsonobj:
                    for linkset in jsonobj["linksets"]:
                        if "linksetdbs" in linkset:
                            if "ids" in linkset and len(linkset["ids"]) > 0:
                                # remove the version info
                                accid = linkset["ids"][0].split('.')[0]
                                pmcids = []
                                for linksetdb in linkset["linksetdbs"]:
                                    pmcids = ["PMC"+x for x in linksetdb["links"]]
                                    if accid not in pmcid_map:
                                        pmcid_map[accid] = pmcids
                                    else:
                                        pmcid_map[accid].extend(pmcids)

        pmcids = list(set([y for _, x in pmcid_map.items() for y in x]))
        # convert to pmids for uniformity
        if pmcids:
            pmc_pm_map = {}
            url = PM_PMC_LINK_URL.replace("IDSPH", ",".join(pmcids))
            resp = http_get_query(url, timeout=20, retries=3)
            if resp and resp.text:
                jsonobj = json.loads(resp.text)
                if "records" in jsonobj:
                    for record in jsonobj["records"]:
                        # we need an additional check as sometimes the key pmid does't exist
                        if "pmid" in record and "pmcid" in record:
                            pmc_pm_map[record["pmcid"]] = record["pmid"]
                for accid, pmcids in pmcid_map.items():
                    pmcid_map[accid] = [pmc_pm_map[pmcid] for pmcid in pmcids if pmcid in pmc_pm_map]

        # map extracted pubmed links to the records
        for gb_rec in self.genbank_records:
            if gb_rec.accid in pubmedid_map:
                gb_rec.pubmedlinks += pubmedid_map[gb_rec.accid]
            if gb_rec.accid in pmcid_map:
                gb_rec.pubmedlinks += pmcid_map[gb_rec.accid]
            if gb_rec.pubmedlinks:
                gb_rec.pubmedlinks = list(set([x for x in gb_rec.pubmedlinks if x != gb_rec.pmid]))
            if gb_rec.pmid or gb_rec.pubmedlinks:
                logging.debug("Accession:'%s' \tDirectly Linked PubMed:'%s' \t Cited by:%s", gb_rec.accid, gb_rec.pmid, gb_rec.pubmedlinks)

        return pubmedid_map

    def extract_pubmed_records(self):
        '''Extract Pubmed and PMC objects for applicable GenBank records'''
        # Get the pubmed texts for records for which locations are not deemed sufficient
        insuff_gbrecs = [x for x in self.genbank_records if not x.sufficient]
        logging.debug("ACCIDS with insufficient information: %s", ",".join([x.accid for x in insuff_gbrecs]))
        # first compile all pubmed ids directly linked in the GenBank metadata
        pubmedids = [x.pmid for x in insuff_gbrecs if x.pmid]
        # if insufficient look for external links between genbank and pubmed
        if EXTRACT_LINKS:
            for rec in insuff_gbrecs:
                pubmedids += rec.pubmedlinks
        pubmedids = list(set(pubmedids))
        logging.info("Linked PMIDs for processing: %s", ",".join(pubmedids))
        # Extract pubmed texts and process them
        pubmed_req = PubMedRequest(pubmedids)
        # retrieve the text, extract locations and normalize them
        pubmed_req.get_pubmed_texts()
        # get rid of locations at insufficient levels
        pubmed_req.filter_locations(self.insufficient_codes)

        # now link pubmed records to their respective genbank records
        for gb_rec in insuff_gbrecs:
            added_pmids = set()
            # first link directly linked pubmed records
            if gb_rec.pmid in pubmed_req.pubmed_records:
                # Assign Pubmed record to the GenBank record
                gb_rec.pmobjs.append(copy.deepcopy(pubmed_req.pubmed_records[gb_rec.pmid]))
                added_pmids.add(gb_rec.pmid)
            # now link externally linked pubmed/pmc records
            for pmid in gb_rec.pubmedlinks:
                if pmid not in added_pmids and pmid in pubmed_req.pubmed_records:
                    gb_rec.pmobjs.append(copy.deepcopy(pubmed_req.pubmed_records[pmid]))
                    added_pmids.add(pmid)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def serialize(self):
        '''returns serialized object for conversion to json'''
        return {'genbankrecords': [x.serialize() for x in self.genbank_records],
                'errors': [x.serialize() for x in self.errors]}

class GenBankRecord(object):
    '''GenBankRecord object which contains necessary fields for location extraction'''
    def __init__(self, accid, pmid, source):
        # placeholders
        self.accid = accid
        self.pmid = pmid
        self.source = source
        self.country_loc = {} # normalized from country name
        self.strain_loc = {} # normalized from strain/organism/isolate which can be often wrong
        self.latlon_loc = {} # normalized from reverse lookup of lat_lon field if available
        self.sufficient = False
        self.metadata_extract = "" # location representation extracted from metadata
        self.is_flu = False
        self.pubmedlinks = []
        self.pmobjs = []
        self.possible_locs = []

    def extract_loih(self, insufficient_codes, max_locs):
        '''
        Extract LOIH based on one or more of the linked PubMed/PMC articles
        '''
        # if location in country field is found to be sufficient, then assign preset prob
        # if lat_lon was also found then assign a probability, then assign preset prob
        if self.sufficient:
            if self.country_loc:
                self.country_loc["Probability"] = COUNTRY_PROBABILITY
                self.country_loc["Sufficient"] = True
                self.possible_locs.append(self.country_loc)
            if self.latlon_loc:
                self.latlon_loc["Probability"] = LATLON_PROBABILITY
                self.latlon_loc["Sufficient"] = True
                self.possible_locs.append(self.latlon_loc)
        else:
            # estimate probabilities based on texts extracted in associated pubmed/pmc articles
            self.estimate_probabilities()

            # if location in strain field could be extracted assign a probability of 1.0
            if self.strain_loc and self.strain_loc["Code"] not in insufficient_codes:
                self.strain_loc["Probability"] = FLU_STRAIN_PROBABILITY if self.is_flu else STRAIN_PROBABILITY
                self.strain_loc["Sufficient"] = True
                self.possible_locs.append(self.strain_loc)

        # if no locations exist at the preferred sufficiency level, then just choose
        # the locations extracted from the metadata and add them to the possible locations
        if not len(self.possible_locs):
            # check if pubmed documents have locations at preferred sufficiency level
            has_poss_locs = False
            for pmobj in self.pmobjs:
                for entity in pmobj.entities:
                    if entity.poss_locs:
                        has_poss_locs = True
                        break
                if has_poss_locs:
                    break
            # if no locations were found then add the metadata locations
            if not has_poss_locs:
                for loc in [self.country_loc, self.latlon_loc, self.strain_loc]:
                    if loc:
                        loc["Probability"] = 1.0
                        loc["Sufficient"] = False
                        self.possible_locs.append(loc)

        # limit possible locations and normalize probabilities totalling to 1
        self.normalize_probabilities(max_locs)

    def estimate_probabilities(self):
        '''
        Estimate probabilities from entities in the associated articles
        '''
        # create regex for searching genbank id and strain and country
        strain = self.source["strain"] if "strain" in self.source else None
        country = self.source["country"] if "country" in self.source else None
        metadata_re = r'|'.join([re.escape(x.strip()) for x in [self.accid, strain, country] if x and x.strip()])
        
        # remove locations if country or ancestor is known
        if self.country_loc:
            ancestor_gid = self.country_loc["GeonameId"]
            logging.debug("Filtering locations based on %s: %s", self.country_loc["GeonameId"], self.country_loc["Name"])
            self.filter_locations(ancestor_gid)

        # if flu narrow down based on strain
        if self.is_flu and self.strain_loc:
            ancestor_gid = self.strain_loc["GeonameId"]
            logging.debug("Filtering locations based on %s: %s", self.strain_loc["GeonameId"], self.strain_loc["Name"])
            self.filter_locations(ancestor_gid)

        # estimate probabilities of locations being LOIH based on rules
        for pmobj in self.pmobjs:
            for entity in pmobj.entities:
                sentence = pmobj.raw_text[entity.span.sent_start:entity.span.sent_end]
                # estimate probabilities at sentence level
                self.assign_entity_prob(entity, sentence, metadata_re)

    def filter_locations(self, ancestor_gid):
        ''' remove all locations that have ancestors other than found in genbank metadata'''
        count, removed = 0, 0
        for pmobjs in self.pmobjs:
            for entity in pmobjs.entities:
                for loc in entity.poss_locs[:]:
                    count += 1
                    # remove if a known ancestor is not found
                    if "AncestorsIds" in loc and ancestor_gid in loc["AncestorsIds"].split(", "):
                        pass
                    else:
                        entity.poss_locs.remove(loc)
                        removed += 1
        if removed > 0:
            logging.debug("%s : Removed %s / %s", self.accid, removed, count)

    def assign_entity_prob(self, entity, sentence, metadata_re):
        '''
        Use sentence patterns to assign probabilities
        1) We collected isolated obtained extract etc.
        TODO:
        Other patterns
        Check if it was part of a table, then apply different rules
        Check if accession id was mentioned in the same sentence
        '''
        # logging.debug("Trying pattern: %s", metadata_re)
        try:
            if any(re.findall(metadata_re, sentence, re.IGNORECASE)):
                # logging.debug("Found very high prob in %s === %s", metadata_re, sentence)
                entity.probability += FLU_METADATA_PRESCENCE_PROBABILITY if self.is_flu else METADATA_PRESCENCE_PROBABILITY
            if any(re.findall(HIGH_PROB_TOKENS_RE, sentence, re.IGNORECASE)):
                # logging.debug("Found high prob in === %s", sentence)
                entity.probability += FLU_KEYWORDS_PRESCENCE_PROBABILITY if self.is_flu else METADATA_PRESCENCE_PROBABILITY
        except Exception as e:
            logging.debug("Error parsing - %s : %s", e, metadata_re)
        return entity

    def normalize_probabilities(self, max_locs, floating_point_precision=2):
        '''
        Normalize probabilities
        '''
        # first get the location if already present, usually from the strain field
        locations = {x["GeonameId"]:copy.deepcopy(x) for x in self.possible_locs}
        for pmobj in self.pmobjs:
            for entity in pmobj.entities:
                for loc in entity.poss_locs:
                    if loc["GeonameId"] not in locations:
                        loc["Sufficient"] = True
                        locations[loc["GeonameId"]] = copy.deepcopy(loc)
                    prob = locations[loc["GeonameId"]]["Probability"]
                    prob += (loc["Probability"] * entity.probability)
                    locations[loc["GeonameId"]]["Probability"] = prob

        # collect probabilities and balance them
        total_prob = sum([x["Probability"] for _, x in locations.items()])
        if total_prob > 0:
            for _, location in locations.items():
                location["Probability"] = round(location["Probability"]/total_prob, floating_point_precision)

        # Give additional probability boost to strain in influenza virus
        if self.is_flu:
            for loc in self.possible_locs:
                locations[loc["GeonameId"]]["Probability"] += loc["Probability"] 

        # if no locations exist at the preferred sufficiency level, then just choose
        # the locations extracted from the metadata and add them to the possible locations
        if len(self.possible_locs) <= 0:
            for loc in [self.strain_loc, self.latlon_loc, self.country_loc]:
                if loc:
                    loc["Probability"] = 1.0
                    loc["Sufficient"] = False
                    self.possible_locs.append(loc)

        # sort locations based on probability
        locations_list = [x for _, x in locations.items()]
        locations_list = sorted(locations_list, key=lambda k: k["Probability"], reverse=True)

        # limit to the number requested
        if len(locations_list) > max_locs:
            locations_list = locations_list[:max_locs]

        # collect probabilities and balance them
        total_prob = sum([x["Probability"] for x in locations_list])
        if total_prob > 0:
            for location in locations_list:
                location["Probability"] = round(location["Probability"]/total_prob, floating_point_precision)
        # assign locations based on 
        self.possible_locs = locations_list

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def serialize(self):
        '''returns serialized object for conversion to json'''
        return {'accid': self.accid, 'pmid': self.pmid, 'source': {x:y for x,y in self.source.items()},
                'country_loc': self.country_loc, 'latlon_loc': self.latlon_loc, 'strain_loc': self.strain_loc, 
                'sufficient': self.sufficient, 'poss_locs': self.possible_locs,
                'linked_pmids': self.pubmedlinks,
                'pmobjs': [x.serialize() for x in self.pmobjs]
                }
