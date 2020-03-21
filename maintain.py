'''
A standalone program for maintainence tasks on the ZoDo server
'''

import argparse
import os
from os import listdir
from os.path import isfile, join
import sys

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s',
                    filemode='w', filename='logs/maintain.log',
                    level=logging.INFO)

from zodo.ner.train import train
from zodo.ner.gen_iob2_files import process_annotated_ner_data
from zodo.ner.gen_embeddings import create_embeddings
from zodo.db.zoophy import insert_into_db, clear_table
from zodo.utils import load_static_objects
from zodo.ner.models import MODEL_NAMES

def main():
    '''Main method : parse input arguments and run appropriate operation'''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # common arguments

    # Input operation
    subparsers = parser.add_subparsers(dest='op', required=True,
                                       help="Choose the maintenance operation to be run. "+
                                            "See operation specific help for to know more options "+
                                            "e.g. python maintain.py insert --help"
                                       )

    # Subcommand CLEAR
    _ = subparsers.add_parser('clear', help="Clears the 'Possible_Location' table in the ZooPhy database for fresh insertion of records")

    # Sub-command INSERT
    sp_idb = subparsers.add_parser('insert', help='Determines LOIH for a given list of accessions (or for unprocessed accessions in the database) and inserts the records into the Possible_Location table in ZooPhy database.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp_idb.add_argument('--filepath', type=str, default=None,
                        help="Path to filename containing accession ids to be processed"+
                        " and inserted into the database")
    sp_idb.add_argument('--suff', type=str, default="ADM1",
                        help='Sufficiency level based on Geonames [ADM1, ADM2, ADM3]')
    sp_idb.add_argument('--maxlocs', type=int, default=10,
                        help='Maximum locations to be extracted per accession')
    sp_idb.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for processing accessions')
    # NER model to use
    sp_idb.add_argument('--model', type=str, default='BIGRU',
                        choices=MODEL_NAMES, help="Model to be used")
    # resource directory paths
    sp_idb.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    sp_idb.add_argument('--save', type=str, default="model/",
                        help="path to the root directory of the saved models")
    sp_idb.add_argument('--gpu', type=int, default=None,
                        help='GPU number to use (if any) as per CUDA')
    # Word Embeddings
    sp_idb.add_argument('--emb_loc', type=str,
                        default='resources/wikipedia-pubmed-and-PMC-w2v.bin',
                        # default='resources/word-embeddings.pkl',
                        help='word2vec embedding location')
    sp_idb.add_argument('--embvocab', type=int, default=-1,
                        help='load top n words in word emb. -1 for all.')

    # Subcommand TRAIN_NER
    sp_trn = subparsers.add_parser('train_ner', help='Trains the named entity recognizer (NER) used for detecting locations in scientific articles.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    sp_trn.add_argument('--skip_generate', action='store_true', 
                        help="skips generation of training files if they've already been generated")
    sp_trn.add_argument('--train_corpus', type=str, default='data/train/',
                        help='path to dir where training corpus files are stored')
    sp_trn.add_argument('--dev_perc', type=float, default=0.05,
                        help='proportion of training data for validation/development')
    sp_trn.add_argument('--seed', type=int, default=666,
                        help='seed to be used for splitting training and development set randomly')
    # NER model to use
    sp_trn.add_argument('--model', type=str, default='BIGRU',
                        choices=MODEL_NAMES+["ALL"], help="Model to be used")
    # resource directory paths
    sp_trn.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    sp_trn.add_argument('--save', type=str, default="model/",
                        help="path to the root directory of the saved models")
    sp_trn.add_argument('--gpu', type=int, default=None,
                        help='GPU number to use (if any) as per CUDA')
    # Word Embeddings
    sp_trn.add_argument('--emb_loc', type=str,
                        # default='resources/wikipedia-pubmed-and-PMC-w2v.bin',
                        default='resources/corpus-embeddings.pkl',
                        help='word2vec embedding location')
    sp_trn.add_argument('--embvocab', type=int, default=-1,
                        help='load top n words for w2v to save memory. -1 to load all.')
    # Hyperparameters
    sp_trn.add_argument('--num_layers', type=int, default=1,
                        help='number of hidden layers')
    sp_trn.add_argument('--hid_dims', type=str, default="150",
                        help='dimensions of hidden layers (comma separated per layer)')
    sp_trn.add_argument('--lrn_rate', type=float, default=0.001, help='learning rate')
    sp_trn.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    # Settings
    sp_trn.add_argument('--train_epochs', type=int, default=100, help='number of train epochs')
    sp_trn.add_argument('--eval_interval', type=int, default=1, help='evaluate once in _ epochs')
    sp_trn.add_argument('--batch_size', type=int, default=10, help='batch size of training')
    sp_trn.add_argument('--max_len', type=int, default=150,
                        help='Max sentence length when RNN/CRF is used')
    sp_trn.add_argument('--use_crf', type=bool, default=False,
                        help='use CRF on the outputs from the Neural Nets')

    # Insert preferences
    args = parser.parse_args()

    logging.info("Input Arguments : %s", args)

    if args.op == "clear":
        # clear the possible location table in zoophy database
        clear_table()
    else:
        # Choose GPU to run
        if args.gpu is not None:
            logging.info("Using GPU resource %s", args.gpu)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        # train or insert
        if args.op == "train_ner":
            if not args.skip_generate:
                logging.info("Generating NER's IOB2 training files")
                process_annotated_ner_data(args)
                logging.info("Generating NER's embedding pickle files. This may take a while.")
                create_embeddings(args)
            # train the NER now
            logging.info("Training NER")
            train(args)
        elif args.op == "insert":
            # Load the graph and embedding objects into memory
            load_static_objects(args)
            logging.info("Inserting rows into database")
            # now process and insert records into zoophy database
            insert_into_db(args)

if __name__ == '__main__':
    main()
