"""Utility functions for loading datasets and computing NER performance"""
import copy
import json
import logging
import re
import shutil
import sys
import tarfile
import time
import urllib.request as request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from contextlib import closing
from typing import List
from os import listdir
from os.path import join

import jsonpickle
import redis
import regex as nre
import requests
import textract
from bioc import biocjson
from requests.utils import quote

from zodo.ner.ner_utils import LOC_ANN_TAG, detect, load_ner
from zodo.settings import (EXTRACT_SUPPLEMENTAL, SUPPLEMENTAL_DATA_DIR, SUPPLEMENTAL_DATA_FILETYPES,
                           GEO_HOST, GEO_PORT)

CLEANUP_REGEX = r"(\s+|\-|\.|\"|\(|\)|\\|\?|\!|\/|\:|\;|\_|\+|\`|\[|\]|\#|\*|\%|\<|\>|\=)"
GB_COUNTRY_FIELD_REGEX = r"(.*):(.*)"
FTP_REGEX = r"format=\"tgz\" .* href=\"(.*)\""

MODE_STRICT = 1
MODE_EXPANDED = 2
MODE_DESPERATE = 3

GEO_STOPWORDS_RE = r'|'.join(["Southcentral", "Interior", "Northwestern", "Northeastern", "Southwestern", "Southeastern",
                              "Central"])

# All url's have PH i.e Placeholders that will be replaced during the query
GEONAMES_URL = "http://"+GEO_HOST+":"+GEO_PORT+"/location?location=LPH&count=CPH&mode=MPH"

PUBMED_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json/PMPH/unicode"
PMCOA_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMCOAPH/unicode"
PMC_OA_FTP_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=IDPH"

FULL_CONFIDENCE_GEOCODES = ["CONT", "PCLI", "PCLH", "RGN", "PEN"]


class NamedEntityObj(object):
    '''Named Entity object which contains offsets, location and possible locations'''
    def __init__(self, span, best_loc=None, poss_locs=[], probability=0.01):
        # placeholders
        self.span = span
        self.best_loc = best_loc
        self.poss_locs = poss_locs
        self.probability = probability

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def serialize(self, minimal=True):
        '''returns serialized object for conversion to json'''
        if minimal:
            return {'span': self.span.serialize(),
                    'best_loc': self.best_loc,
                    'probability': self.probability,
                    }
        else:
            return {'span': self.span.serialize(),
                    'best_loc': self.best_loc,
                    'probability': self.probability,
                    'poss_locs': self.poss_locs
                    }

def load_static_objects(args):
    '''Load all the static objects like the NER Graph'''
    load_ner(args)

def extract_probability(records):
    '''Extract probability based on hueristics'''
    # if the location is a country/continent/region, the first record will have the big codes
    # there's no point in keeping the other locations as possible
    # TODO: need to double check with string similarity as well
    if records and records[0]["Code"] in FULL_CONFIDENCE_GEOCODES:
        # logging.debug("Country/Continent/Region detected, removing alternatives: %s", records[0]["Name"])
        records = records[:1]
    # For population heuristic, set probability based on total population in the retrieved records
    total_pop = sum([int(x["Population"]) for x in records if "Population" in x])
    # TODO: assign probability based on string similarity as well
    for record in records:
        try:
            probability = (int(record["Population"]) + 0.00000001)/(total_pop + 0.000001)
            record["Probability"] = probability
        except Exception as err:
            logging.error("ERROR: %s for %s", err, record)
    return records

def get_formatted_locs(location, mode=MODE_STRICT):
    '''Get all possible locations for better desperate search'''
    all_locs = []
    # Cleanup and format string
    clean_loc, finer_loc = location.strip(), ""
    # If string contains a colon it is in GenBank format, so format it
    gb_format = re.search(GB_COUNTRY_FIELD_REGEX, clean_loc)
    if gb_format:
        country = gb_format.group(1)
        finer_loc = gb_format.group(2)
        if "-" in finer_loc:
            finer_loc = ", ".join([x.strip() for x in gb_format.group(2).split("-")])
        clean_loc = finer_loc.strip() + ", " + country.strip()
    all_locs.append(clean_loc)
    # there are times when the locations are reversed
    if finer_loc and "," in finer_loc:
        finer_loc = ",".join([loc for loc in reversed(finer_loc.split(","))])
        clean_loc = finer_loc.strip() + ", " + country.strip()
        all_locs.append(clean_loc)
    if mode != MODE_STRICT and clean_loc.split(","):
        clean_loc = clean_loc.split(",")
        for loc in clean_loc[:-1]:
            trunc_loc = loc.strip() + ", " + clean_loc[-1].strip()
            if trunc_loc not in all_locs:
                all_locs.append(trunc_loc)
            trunc_loc = re.sub(GEO_STOPWORDS_RE, "", trunc_loc, re.IGNORECASE)
            if trunc_loc not in all_locs:
                all_locs.append(trunc_loc.strip())
            # Custom modifications
            if trunc_loc.startswith("St"):
                trunc_loc.replace("St", "Saint")
            elif trunc_loc.startswith("Saint"):
                trunc_loc.replace("Saint", "St")
            if trunc_loc not in all_locs:
                all_locs.append(trunc_loc)
        if mode == MODE_DESPERATE:
            last_loc = clean_loc[-1]
            if last_loc not in all_locs:
                all_locs.append(last_loc)
    return all_locs

def lookup_location_pop(location, mode=MODE_STRICT, count=1, debug=False):
    '''Get Geoname Record using the location APIs and population heuristic'''
    loc = []
    all_locs = get_formatted_locs(location, mode)
    for clean_loc in all_locs:
        clean_loc = re.sub(CLEANUP_REGEX, ' ', clean_loc)
        clean_loc = clean_loc.strip()
        # search by comma splits, beneficial for country and strain fields
        full = "full" if mode == MODE_DESPERATE else "default"
        if clean_loc:
            url = GEONAMES_URL.replace("LPH", clean_loc).replace("CPH", str(count)).replace("MPH", full)
            resp = http_get_query(url)
            if resp:
                jsondata = resp.json()
                if jsondata and "retrieved" in jsondata and int(jsondata["retrieved"]) > 0:
                    loc = jsondata["records"]
                    break
                elif debug:
                    logging.debug("Search returned no results: '%s' -> '%s'", location, clean_loc)
            else:
                logging.error("Could not lookup location: %s, Check ZooPhy GeoNames API.", clean_loc)
        elif debug:
            logging.error("Invalid String: '%s' -> '%s'", location, clean_loc)
    return loc

def normalize_entities(spans):
    '''
    Normalize entities using just the information available in the text extracted.
    Strategies are currently limited to heuristics for scalability and efficiency.
    1) population i.e. Paris, France over Paris, TX because of its population
    2) cooccurence and proximity i.e. 'Assam and Nagaland states in India'
    3) immediate hierarchy i.e. Springfield, IL or Springfield, MA
    4) ALL THE COOL STUFF I HAVE/HAVEN'T THOUGHT ABOUT BUT UNIMPLEMENTED
    e.g. DEPENDENCY PARSING or RELATIONSHIP EXTRACTION FOR ESTABLISHING CAUSALITY
    TODO: Co-occurence is not implemented yet
    Parameter cache_dict is used to minimize responses
    '''
    doc_ent_objs = []
    if spans:
        for span in spans:
            if span.text.strip() and span.encoding == LOC_ANN_TAG:
                records = lookup_location_pop(span.text, MODE_STRICT, 10)
                poss_locs = extract_probability(records)
                # Assume first location is the best location
                best_loc = poss_locs[0] if len(poss_locs) > 0 else None
                norm_ent_obj = NamedEntityObj(span, best_loc, poss_locs)
                doc_ent_objs.append(norm_ent_obj)
        # Remove entities that couldn't be resolved
        doc_ent_objs = [x for x in doc_ent_objs if x.best_loc is not None]
    return doc_ent_objs

def download_pubmed_record(pmid):
    '''Extract pubmed text from the response'''
    raw_json = ""
    # try to get the pmcoa text from pmid
    url = PMCOA_URL.replace("PMCOAPH", pmid)
    resp = http_get_query(url)
    # if there are no OA texts then pickup pubmed abstract
    try:
        if resp.text:
            if resp.text.find("[Error]", 0, 10) > -1:
                logging.debug("Found in PM Only: %s", pmid)
                url = PUBMED_URL.replace("PMPH", pmid)
                resp = http_get_query(url)
                if resp:
                    if resp.text.find("[Error]", 0, 10) > -1:
                        logging.error("PMID %s not found", pmid)
                    else:
                        raw_json = resp.text
            else:
                logging.debug("Found in PMC OA: %s", pmid)
                raw_json = resp.text
                if EXTRACT_SUPPLEMENTAL:
                    raw_json = download_supplemental(raw_json)
    except Exception as error:
        logging.error("Error retrieving json %s from PubMed/PMC OA server: %s", pmid, error)
    return raw_json

def download_supplemental(raw_json):
    '''Download supplemental files and add content to json'''
    # first retrieve the PMC id as PMID is not supported
    try:
        doc_bioc = json.loads(raw_json)
        pmcid = "PMC"+doc_bioc["documents"][0]["id"] if doc_bioc["source"] == "PMC" else None
        passages = [passage for doc in doc_bioc["documents"] for passage in doc["passages"]]
        # for doc in doc_bioc["documents"]:
        for passage in passages:
            section_type = passage["infons"]["section_type"]
            if section_type == "SUPPL":
                logging.debug("Attempting extraction of supplemental information from %s", pmcid)
                url = PMC_OA_FTP_URL.replace("IDPH", pmcid)
                resp = http_get_query(url)
                xml = resp.text
                query_re = re.search(FTP_REGEX, xml)
                supp_text = passage["text"]
                if query_re:
                    ftp_url = query_re.group(1)
                    pmcdir = ftp_get_query(ftp_url)
                    if pmcdir:
                        supp_text = extract_text_from_files(pmcdir)
                passage["text"] = supp_text
                break
        # extract based on file format
        # doc_bioc = format_supplemental_data(doc_bioc, supp_files)
        raw_json = json.dumps(doc_bioc)
    except Exception as error:
        logging.error("Cant extract json from %s: %s", pmcid, error)
        return raw_json
    return raw_json

def extract_text_from_files(pmcdir):
    supp_files = [x for x in listdir(pmcdir) if x.split(".")[-1] in SUPPLEMENTAL_DATA_FILETYPES]
    logging.debug("Files for extraction in %s : %s", pmcdir, ",".join(supp_files))
    # For now just extract all text the same way using textract
    supp_file_contents = {x:str(textract.process(join(pmcdir, x))).replace("\\n", "\n") for x in supp_files}
    supp_contents = ""
    for suppfile, content in supp_file_contents.items():
        supp_contents += "\n\n*** " + str(suppfile) + " ***\n" + str(content)
        logging.debug("Added '%s' chars from '%s'", len(content), suppfile)
    return supp_contents

def format_supplemental_data(doc_bioc, supp_files):
    '''Format and add data to original json as per filetype 
    This is a TODO item and files should be formatted based on filetype
    Observations in filetype
    1) Tables in .doc and .docx format have pipe character i.e. | to separate columns
    2) Excel format .xls and .xlsx may benefit from using pandas
    3) .pdf is challenging
    '''
    return doc_bioc

def http_get_query(url, debug=False, timeout=5, retries=1):
    '''Generic HTTP get query with error handling'''
    resp = None
    start = time.time()
    while not resp and retries > 0:
        try:
            resp = requests.get(url, timeout=timeout)
        except Exception as e:
            logging.error("Error in GET: %s Error: %s", url, e)
        retries -= 1
    if debug:
        logging.debug("URL: %s Time: %s", url, time.time()-start)
    return resp

def ftp_get_query(url, debug=False, retries=1, unzip=True):
    '''Generic FTP get query with error handling'''
    extdir = None
    start = time.time()
    while not extdir and retries > 0:
        try:
            filename = url.split("/")[-1]
            download_path = SUPPLEMENTAL_DATA_DIR+filename
            with closing(request.urlopen(url)) as r:
                with open(download_path, 'wb') as f:
                    shutil.copyfileobj(r, f)
            if unzip:
                with tarfile.open(download_path) as tarf:
                    tarf.extractall(SUPPLEMENTAL_DATA_DIR)
                extdir = SUPPLEMENTAL_DATA_DIR+filename[:-7]
        except Exception as e:
            logging.error("Error in FTP: %s Error: %s", url, e)
        retries -= 1
    if debug:
        logging.debug("URL: %s Time: %s", url, time.time()-start)
    return extdir
