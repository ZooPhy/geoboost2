'''
File containing list of API's offered by the server
'''
import logging
from flask import jsonify
from flask_restful import Resource, reqparse

from zodo.GenBank import GenBankRequest
from zodo.PubMed import PubMedRequest
from zodo.utils import normalize_entities
from zodo.ner.ner_utils import detect

class AccToLocs(Resource):
    '''Resolution API for detecting toponym spans and their respective Geoname IDs'''
    def get(self):
        '''Get method'''
        logging.debug("GET - AccToLocs")     # for debugging
        reqparser = reqparse.RequestParser()
        reqparser.add_argument('text', type=str, required=True, location='args',
                               help="comma separated accession ids")
        reqparser.add_argument('suff', type=str, required=True, location='args',
                               help="sufficiency level")
        reqparser.add_argument('maxlocs', type=int, required=True, location='args',
                               help="maximum locations per record")
        reqargs = reqparser.parse_args()
        logging.debug("Text: '%s'", reqargs['text'])              # for debugging
        text = reqargs['text']
        suff = reqargs['suff']
        maxlocs = reqargs['maxlocs']
        # get genbank request object
        gb_req = GenBankRequest(text.split(','), suff, maxlocs)
        gb_req.process_genbank_ids()
        # serialize
        return jsonify(gb_req.serialize())

class PubMedToLocs(Resource):
    '''Resolution API for detecting toponym spans and their respective Geoname IDs'''
    def get(self):
        '''Get method'''
        logging.debug("Get - PubMed")     # for debugging
        reqparser = reqparse.RequestParser()
        reqparser.add_argument('text', type=str, required=True, location='args',
                               help="comma separated PubMed/PMC ids")
        reqargs = reqparser.parse_args()
        logging.debug("Text: '%s'", reqargs['text'])              # for debugging
        text = reqargs['text']
        response = []
        pmids = [pmid.strip() for pmid in text.split(",")]
        if pmids:
            pm_req = PubMedRequest(pmids)
            pm_req.get_pubmed_texts()
            response = pm_req.serialize()
        return jsonify(response)

class TextToLocs(Resource):
    '''Resolution API for detecting toponym spans and their respective Geoname IDs in Raw Text'''
    def post(self):
        '''POST method'''
        logging.debug("POST - Resolve")     # for debugging
        reqparser = reqparse.RequestParser()
        reqparser.add_argument('text', type=str, required=True, location='json',
                               help="text for extracting entities and concept ids")
        reqargs = reqparser.parse_args()
        text = reqargs['text']
        spans, _ = detect(text)
        entities = normalize_entities(spans)
        logging.debug("Found %s spans and %s locations", len(spans), len(entities))
        locs = [e.serialize(minimal=False) for e in entities]
        logging.debug("\nReturning %s locations", len(locs))
        resp_obj = {'text':text,'locations':locs}
        return jsonify(resp_obj)

class Lookup(Resource):
    '''Lookup API for finding the relevant Geonames ID'''
    def get(self):
        '''GET method'''
        logging.debug("GET - Lookup")         # for debugging
        reqparser = reqparse.RequestParser()
        reqparser.add_argument('text', type=str, required=True, location='args',
                               help="text for finding concept id")
        reqargs = reqparser.parse_args()
        logging.debug("Text: '%s'", reqargs['text'])              # for debugging
        logging.debug(reqargs['text'])              # for debugging
        concepts = []
        text = reqargs['text']
        return jsonify(concepts)

class TestNCBIAPIs(Resource):
    '''Test NCBI APIs'''
    def get(self):
        '''GET method'''
        logging.debug("GET - Lookup")         # for debugging
        reqparser = reqparse.RequestParser()
        reqparser.add_argument('text', type=str, required=True, location='args',
                               help="text for finding concept id")
        reqargs = reqparser.parse_args()
        logging.debug("Text: '%s'", reqargs['text'])              # for debugging
        concepts = []
        text = reqargs['text']
        return jsonify(concepts)

class Root(Resource):
    '''Reply for checking API status'''
    def get(self):
        '''GET method'''
        logging.debug("GET - /")
        return "Root API is up."
