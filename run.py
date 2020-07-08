'''
A standalone program for reading input accessions and producing results
'''

import argparse
from argparse import Namespace
import os
from os import listdir
from os.path import isfile, join
import sys
import codecs
import pandas as pd
import re

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s',
                    filemode='w', filename='logs/run.log',
                    level=logging.DEBUG)

from zodo.GenBank import GenBankRequest
from zodo.PubMed import PubMedRequest
from zodo.utils import load_static_objects
from zodo.ner.models import MODEL_NAMES
from zodo.utils import normalize_entities
from zodo.ner.ner_utils import detect, write_annotations

def write_genbank_run_results(gb_req: GenBankRequest, outdir: str):
    '''Write the results of the Run method to file'''
    # Write the Confidence estimate files
    cols = ["Accession", "Sufficient", "PubMedIDs", "GeonameID", "Location", "Country", "Latitude", "Longitude", "Code", "Confidence"]
    gb_rows = []
    for gb_rec in gb_req.genbank_records:
        if len(gb_rec.pmobjs) >= len(gb_rec.pubmedlinks):
            pmids = ", ".join([(str(x.pmid)+" ("+("OA" if x.open_access else "Abstract")+")") for x in gb_rec.pmobjs])
        else:
            pmids = ", ".join(gb_rec.pubmedlinks)
        for loc in gb_rec.possible_locs:
            gb_rows.append([gb_rec.accid,
                            gb_rec.sufficient,
                            pmids,
                            loc["GeonameId"],
                            re.sub(r" \([^)]*\)", "", loc["FullHierarchy"]),
                            loc["Country"] if "Country" in loc else "",
                            loc["Latitude"],
                            loc["Longitude"],
                            loc["Code"],
                            loc["Probability"]
                            ])
    df = pd.DataFrame(gb_rows, columns=cols)
    outfile = "Confidence.tsv"
    out_dir = join(outdir, outfile)
    df.to_csv(out_dir, sep='\t')

    # Next write the file with best location
    cols = ["Accession", "Sufficient", "PubMedIDs", "GeonameID", "Location", "Country", "Latitude", "Longitude", "Code", "Confidence"]
    gb_rows = []
    for gb_rec in gb_req.genbank_records:
        pmids = ", ".join([(str(x.pmid)+" ("+("OA" if x.open_access else "Abstract")+")") for x in gb_rec.pmobjs])
        for loc in gb_rec.possible_locs:
            gb_rows.append([gb_rec.accid,
                            gb_rec.sufficient,
                            pmids,
                            loc["GeonameId"],
                            re.sub(r" \([^)]*\)", "", loc["FullHierarchy"]),
                            loc["Country"] if "Country" in loc else "",
                            loc["Latitude"],
                            loc["Longitude"],
                            loc["Code"],
                            loc["Probability"],
                            ])
            break
    df = pd.DataFrame(gb_rows, columns=cols)
    outfile = "Locations.tsv"
    out_dir = join(outdir, outfile)
    df.to_csv(out_dir, sep='\t')

def extract_genbank_loih(args: Namespace):
    '''Extract LOIH for list of accessions in a file'''
    # load accession ids from file
    accessions = [x.strip() for x in open(args.acc_file) if x.strip()]
    logging.info("Extracting LOIH for %s accessions", len(accessions))
    gb_req = GenBankRequest(accessions, args.suff, args.maxlocs)
    gb_req.process_genbank_ids()
    dir_name = args.acc_file
    output_dir = join(args.outdir, dir_name)
    logging.info("Finished extraction. Writing to files in :%s", output_dir)
    write_genbank_run_results(gb_req, args.outdir)
    logging.info("Done!")

def write_pubmed_run_results(args, pubmed_req):
    '''Write the results of the Run method to file'''
    # TODO

def extract_pubmed_locations(args: Namespace):
    '''Extract locations from txt files'''
    # load pubmed ids
    pubmedids = [x.strip() for x in open(args.pub_file) if x.strip()]
    logging.info("Extracting locations from %s PubMed articles", len(pubmedids))
    # Extract pubmed texts and process them
    pubmed_req = PubMedRequest(pubmedids)
    # retrieve the text, extract locations and normalize them
    pubmed_req.get_pubmed_texts()
    write_pubmed_run_results(args, pubmed_req)
    logging.info("Done!")

def extract_text_locations(args: Namespace):
    '''Extract locations from txt files'''
    # load text files
    txt_files = [f for f in listdir(args.txt_dir) if isfile(join(args.txt_dir, f)) and f.endswith(".txt")]
    logging.info("Extracting locations from %s files", len(txt_files))
    for txtfile in txt_files:
        with codecs.open(join(args.txt_dir, txtfile), 'r', 'utf-8') as myfile:
            doc_text = myfile.read()
        spans, _ = detect(doc_text)
        logging.debug("Found %s spans ", len(spans))
        entities = normalize_entities(spans)
        write_annotations(args.outdir, entities, txtfile[:-4], True)
    logging.info("Done!")

def main():
    '''Main method : parse input arguments and run appropriate operation'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BILSTMP',
                        choices=MODEL_NAMES,
                        help="Type of DNN model to be used")
    # resource directory paths
    parser.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    parser.add_argument('--save', type=str, default="model/",
                        help="path to saved model to be loaded")
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU number to use (if any) as per CUDA')
    # Word Embeddings
    parser.add_argument('--emb_loc', type=str,
                        # default='resources/wikipedia-pubmed-and-PMC-w2v.bin',
                        default='resources/corpus-embeddings.pkl',
                        help='word2vec embedding location')
    parser.add_argument('--embvocab', type=int, default=-1,
                        help='load top n words for w2v to save memory. -1 to load all.')

    subparsers = parser.add_subparsers(dest='input', required=True, help='Type of input (genbank, pubmed, text) see specific help for more options based on input type chosen e.g. python run.py genbank --help')

    # GenBank
    sp_gen = subparsers.add_parser('genbank', help='Determines LOIH for list of GenBank accession ids in a file.')
    sp_gen.add_argument('acc_file', type=str,
                        help="Path to filename with Accessions")
    sp_gen.add_argument('--suff', type=str, default="ADM1",
                        help='Sufficiency level based on Geonames [ADM1, ADM2, ADM3]')
    sp_gen.add_argument('--maxlocs', type=int, default=10,
                        help='Maximum locations to be extracted per accession')
    sp_gen.add_argument('outdir', type=str,
                        help='Output dir for producing GenBank metadata details and LOIH probabilities.')

    # PubMed
    sp_pub = subparsers.add_parser('pubmed', help='Extracts text and geographic locations from a list of PubMed ids in a file.')
    sp_pub.add_argument('pub_file', type=str,
                        help="Path to filename with PubMed ids")
    sp_pub.add_argument('outdir', type=str,
                        help='Output dir for Geographic Location annotated PubMed texts')

    # Text
    sp_txt = subparsers.add_parser('text', help='Extracts geographic locations from text (.txt) files in a directory.')
    sp_txt.add_argument('txt_dir', type=str,
                        help="Path to directory containing text files (.txt)")
    sp_txt.add_argument('outdir', type=str,
                        help='Output dir for Geographic Location annotated texts')


    args = parser.parse_args()

    logging.info("Input Arguments : %s", args)

    # Choose GPU to run for the tensorflow model
    if args.gpu is not None:
        logging.info("Using GPU resource %s", args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load the graph and embedding objects into memory
    load_static_objects(args)

    # Run
    if args.input == "genbank":
        extract_genbank_loih(args)
    if args.input == "pubmed":
        pass
    if args.input == "text":
        extract_text_locations(args)

if __name__ == '__main__':
    main()
