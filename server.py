'''
Program to start a webserver api to identify and normalize toponyms
'''

import argparse
import os

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s',
                    filemode='w', filename='logs/server.log',
                    level=logging.DEBUG)

from flask import Flask
from flask_restful import Api

from zodo.api_lib import (AccToLocs, PubMedToLocs, TextToLocs, Root)
from zodo.utils import load_static_objects
from zodo.ner.models import MODEL_NAMES

def main():
    '''Main method : parse input arguments and run server'''
    parser = argparse.ArgumentParser()
    # Input files
    parser.add_argument('--model', type=str, default='BILSTMP',
                        choices=MODEL_NAMES, help="Model to be used")
    parser.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    parser.add_argument('--save', type=str, default="model/",
                        help="path to saved model to be loaded")
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU number')
    # Word Embeddings
    parser.add_argument('--emb_loc', type=str,
                        # default='resources/wikipedia-pubmed-and-PMC-w2v.bin',
                        default='resources/corpus-embeddings.pkl',
                        help='word2vec embedding location')
    parser.add_argument('--embvocab', type=int, default=-1,
                        help='load top n words for w2v to save memory. -1 to load all.')
    # API SETTINGS
    parser.add_argument('--host', type=str, default='localhost',
                        help="Host name (default: localhost)")
    parser.add_argument('--port', type=int, default='8025', help="Port (default: 8025)")
    parser.add_argument('--path', type=str, default='/topo', help="Path (default: /topo)")
    args = parser.parse_args()

    logging.info("Input Arguments : %s", args)

    # Choose GPU to run
    if args.gpu is not None:
        logging.info("Using GPU resource %s", args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load the graph and embedding objects into memory
    load_static_objects(args)

    logging.info("Starting app . . .")
    app = Flask(__name__)
    api = Api(app)

    @app.errorhandler(404)
    def page_not_found(error):
        '''Error message for page not found'''
        return "page not found : " + str(error)

    @app.errorhandler(500)
    def raise_error(error):
        '''Error message for resource not found'''
        return error

    api.add_resource(AccToLocs, args.path+'/accession')
    api.add_resource(PubMedToLocs, args.path+'/pubmed')
    api.add_resource(TextToLocs, args.path+'/resolve')
    api.add_resource(Root, args.path)
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
