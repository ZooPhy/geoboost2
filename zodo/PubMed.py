"""
Contains classes for PubMed Request and the individual PubMed Record object
"""
import jsonpickle
import redis
import json
import logging
from zodo.settings import (API_KEY, EXTRACT_LINKS, GB_BATCH_SIZE,
                           GB_PM_LINK_BATCH_SIZE, REDIS_PMC_CACHE_DB, REDIS_PMC_PROCESSED_DB,
                           REDIS_HOST, REDIS_PASSWORD, REDIS_PORT, USE_REDIS)
from zodo.utils import (http_get_query, download_pubmed_record, MODE_STRICT,
                        lookup_location_pop, extract_probability, NamedEntityObj)
from zodo.ner.ner_utils import LOC_ANN_TAG, TABLE_LOC_TAG, TABLE_ACCN_TAG, detect, load_ner


class PubMedRequest(object):
    '''PubMedRecord request object'''
    def __init__(self, pubmedids):
        self.pubmedids = pubmedids
        # we use a dictionary as opposed to a list because
        # multiple accessions often have the same pubmed id
        self.pubmed_records = {}
        self.errors = []

    def get_pubmed_texts(self):
        '''get PubMed texts from a list of pubmed ids and extract entities'''
        # we use a cache for minimizing the number of geonames queries
        cache_dict = {}
        # retrieve the records using the API or get it from cache
        if USE_REDIS:
            proc_red = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_PMC_PROCESSED_DB)
            unpr_red = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_PMC_CACHE_DB)
            for pmid in self.pubmedids:
                # first check if the records exist in the processed pubmed cache
                if proc_red.exists(pmid):
                    pmrec_json = proc_red.get(pmid)
                    logging.debug("%s found in PMC processed cache DB", pmid)
                    pubmed_record = jsonpickle.decode(pmrec_json)
                    self.pubmed_records[pmid] = pubmed_record
                # if not check if it exists in our unprocessed pubmed cache
                else:
                    if unpr_red.exists(pmid):
                        # if exists get from cache
                        raw_json = unpr_red.get(pmid)
                        logging.debug("%s found in PMC cache DB", pmid)
                    else:
                        raw_json = download_pubmed_record(pmid)
                        if not raw_json:
                            continue
                        try:
                            # add to unprocessed cache
                            json.loads(raw_json)
                            unpr_red.set(pmid, raw_json)
                        except Exception as e2:
                            logging.error("Invalid JSON: %s for %s", e2, pmid)
                            continue
                    # create record object and process
                    pubmed_record = PubMedRecord(pmid, raw_json)
                    pubmed_record.extract_entities()
                    cache_dict = pubmed_record.normalize_entities(cache_dict)
                    self.pubmed_records[pmid] = pubmed_record
                    # add to cache if using redis
                    pmrec_json = jsonpickle.encode(pubmed_record)
                    proc_red.set(pmid, pmrec_json)
        else:
            # if not using redis cache, just download the articles and process
            for pmid in self.pubmedids:
                raw_json = download_pubmed_record(pmid)
                if raw_json:
                    pubmed_record = PubMedRecord(pmid, raw_json)
                    self.pubmed_records[pmid] = pubmed_record
                    pubmed_record.extract_entities()
                    cache_dict = pubmed_record.normalize_entities(cache_dict)

    def filter_locations(self, insufficient_codes):
        ''' remove all locations that have codes found to be insufficient '''
        processed = set() # store references to make processing faster
        for _, pubmed_record in self.pubmed_records.items():
            for entity in pubmed_record.entities:
                for loc in entity.poss_locs[:]:
                    # next remove if the location is not at sufficient level
                    if loc["GeonameId"] not in processed:
                        if loc["Code"] in insufficient_codes:
                            entity.poss_locs.remove(loc)
                        else:
                            processed.add(loc["GeonameId"])

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def serialize(self):
        '''returns serialized object for conversion to json'''
        return {'pubmedrecords': [y.serialize() for x, y in self.pubmed_records.items()],
                'errors': [x.serialize() for x in self.errors]}

class PubMedRecord(object):
    '''PubMed object which contains necessary fields for location extraction'''
    def __init__(self, pmid, raw_json):
        # placeholders
        self.pmid = pmid
        self.raw_json = raw_json
        # available after processing
        self.raw_text = ""
        self.open_access = False
        # self.proc_json = "" # to be used for adding annotations in bioc
        self.spans = []
        self.entities = []

    def extract_entities(self):
        '''uses the NER to extract entities'''
        # There are problems with loading biocjson for many records retrieved from API
        # TODO: Investigate the errors and fix them, until then use json
        doc_bioc = json.loads(self.raw_json)
        self.open_access = True if doc_bioc["source"] == "PMC" else False
        self.spans, self.raw_text = detect(doc_bioc, bioc_json=True)
        self.spans = sorted(self.spans, key=lambda k: k.start, reverse=True)

    def normalize_entities(self, cache_dict={}):
        '''
        Normalize entities using just the information available in the text extracted.
        Strategies are currently limited for scalability and efficiency.
        1) population i.e. Paris, France over Paris, TX because of its population (IMPLEMENTED)
        2) immediate hierarchy i.e. Springfield, IL or Springfield, MA (IMPLEMENTED AT NER LEVEL)
        3) cooccurence and proximity i.e. 'Assam and Nagaland states in India' TODO
        4) ALL THE COOL STUFF I HAVE/HAVEN'T THOUGHT ABOUT BUT UNIMPLEMENTED
        e.g. DEPENDENCY PARSING or RELATIONSHIP EXTRACTION FOR ESTABLISHING CAUSALITY
        Parameter cache_dict is used to minimize responses
        '''
        if self.spans:
            doc_ent_objs = []
            for span in self.spans:
                if (span.encoding in [TABLE_LOC_TAG, LOC_ANN_TAG] and 
                    len(span.text) > 1):
                    if span.text not in cache_dict.keys():
                        records = lookup_location_pop(span.text, MODE_STRICT, 10)
                        poss_locs = extract_probability(records)
                        cache_dict[span.text] = poss_locs
                    else:
                        poss_locs = cache_dict[span.text]
                    # Assume first location is the best location
                    best_loc = poss_locs[0] if len(poss_locs) > 0 else None
                    if best_loc:
                        norm_ent_obj = NamedEntityObj(span, best_loc, poss_locs)
                        doc_ent_objs.append(norm_ent_obj)
                elif span.encoding == TABLE_ACCN_TAG:
                    doc_ent_objs.append(NamedEntityObj(span))
            self.entities = doc_ent_objs
        return cache_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def serialize(self):
        '''returns serialized object for conversion to json'''
        return {'pmid': self.pmid,
                # 'raw_json': self.raw_json, # comment to save space
                'raw_text': self.raw_text, 'open_access': self.open_access, 
                'pmlocs': [x.serialize() for x in self.entities]}
