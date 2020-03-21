'''
This python program accomplishes the following
    1. Generates smaller word embeddings because the original embedding files are huge.
    2. Generates character embeddings from training files
    3. Generates embeddings for numbers and unknown tokens
'''

import sys
import re
from argparse import Namespace, ArgumentParser
import pickle
import random
import logging
from os import listdir, makedirs
from os.path import join, exists
import numpy as np
import codecs
import logging
from zodo.ner.ner_utils import (CHAREMB_FILENAME, CEMB_SIZE, CHAR_EMB_LEN,
                                UNK_FILENAME, NUM_FILENAME, CORPUS_EMB_FILENAME, 
                                IOB2_DIR, TRAIN_FILE, DEV_FILE)
from typing import List
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


def get_character_lists(txt_dir:str) -> List[List[str]]:
    """Retrieves character lists for training using word2vec
    Arguments:
        txt_dir {str} -- Path to directory with training text(.txt) files
    
    Returns:
        List[List[str]] -- List of list of characters
    """
    char_lists = []
    txt_files = [join(txt_dir, f) for f in listdir(txt_dir) if f.endswith(".txt")]
    if not txt_files:
        logging.error("No text (.txt) files found in directory '%s'. Please check directory and try again.", txt_dir)
        return [[]]
    logging.info("Loading %s files for creating character embeddings", len(txt_files))
    for txt_file in txt_files:
        with codecs.open(txt_file, 'r', 'utf-8') as tfile:
            doc_text = tfile.read()
            # Remove Non-ascii characters and new lines
            doc_text = re.sub(r'[^\x00-\x7F]+', '', doc_text)
            doc_text = re.sub(r'[\n]+', ' ', doc_text)
            doc_chars = [char for char in doc_text]
            char_lists.append(doc_chars)
    return char_lists

def get_vocab(iob2_files:List[str]) -> List[str]:
    """Retrieve the vocabulary of the iob2 annotated files
    
    Arguments:
        iob2_files {List[str]} -- List of paths to the iob2 annotated files
    
    Returns:
        List[str] -- Returns the unique list of vocabulary found in the files
    """
    vocab = set()
    for iob2_file in iob2_files:
        logging.info("Loading file %s for creating corpus embeddings", iob2_file)
        for line in open(iob2_file):
            token = line.split("\t")[0]
            vocab.add(token)
    return list(vocab)

def create_embeddings(args: Namespace):
    """ Create mini-word embeddings for quick-load-training, character embeddings,
        embeddings for unknown tokens and number tokens
    
    Arguments:
        args {Namespace} -- Namespace object from argparse
    """
    iob2_files = [join(args.work_dir, join(IOB2_DIR, x)) for x in [TRAIN_FILE, DEV_FILE]]
    dataset_vocab = get_vocab(iob2_files)
    logging.info("Corpus vocabulary : %s", len(dataset_vocab))

    dataset_char_lists = get_character_lists(args.train_corpus)
    logging.info("Corpus character lists : %s", len(dataset_char_lists))

    logging.info("Loading word embeddings from '%s'", args.emb_loc)
    unk_words = set()
    wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True)
    wemb_dict = {}
    for word in dataset_vocab:
        try:
            wemb_dict[word] = wvec[word]
        except KeyError:
            unk_words.add(word)
    logging.info("Number of unknown words: %s", len(unk_words))

    # Load additional words from an external file
    vocab_file = join(args.work_dir, "custom_vocab.txt")
    if exists(vocab_file):
        custom_vocab = [x.strip() for x in open(vocab_file)]
        custom_vocab_added = 0
        for word in custom_vocab:
            try:
                wemb_dict[word] = wvec[word]
                custom_vocab_added += 1
            except KeyError:
                pass
        logging.info("Number custom words added: %s / %s", custom_vocab_added, len(custom_vocab))

    if not exists(args.work_dir):
        makedirs(args.work_dir)
    # Dump dictionary pickle to disk
    logging.info("Dumping training files to '%s'", args.work_dir)
    pickle.dump(wemb_dict, open(join(args.work_dir, CORPUS_EMB_FILENAME), "wb"))

    # Create unk and num word embeddings based on the average of a subsample
    # from the loaded word embeddings and write them to disk
    word_count = 50000  # number of subsample words to process for averaging
    num_count = 10000   # number of sumsample numbers to process for averaging
    # Initialize with a random embeddings for corner cases
    unk = [[random.random() for _ in range(wvec.vector_size)]]
    num = [[random.random() for _ in range(wvec.vector_size)]]
    for word in wvec.vocab:
        if word.isdigit() and num_count > 0:
            num.append(wvec[word])
            num_count -= 1
        elif word_count > 0:
            unk.append(wvec[word])
            word_count -= 1
    wvec = None
    unk = np.average(np.asarray(unk), axis=0)
    num = np.average(np.asarray(num), axis=0)
    pickle.dump(unk, open(join(args.work_dir, UNK_FILENAME), "wb"))
    pickle.dump(num, open(join(args.work_dir, NUM_FILENAME), "wb"))

    # Char embeddings section
    cemb_dict = {}
    cvec = Word2Vec(dataset_char_lists, size=CEMB_SIZE, min_count=1, workers=6)
    logging.info("Creating character embeddings in '%s'", args.work_dir)
    for key, _ in cvec.wv.vocab.items():
        cemb_dict[key] = cvec.wv[key]
    unk = np.array([random.random() for _ in range(CEMB_SIZE)])
    cemb_dict["UNKK"] = unk
    pickle.dump(cemb_dict, open(join(args.work_dir, CHAREMB_FILENAME), "wb"))
    logging.info("Done generating training files!")

def generate_embeddings(args=None):
    '''Main method : parse input arguments and train'''
    parser = ArgumentParser()
    # Input and Output paths
    parser.add_argument('-t', '--train_corpus', type=str, default='data/train/',
                        help='path to dir where training corpus files are stored')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='seed to be used for splitting train and dev randomly')
    parser.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    args = args if args else Namespace()
    args = parser.parse_args(namespace=args)
    logging.info("Input arguments: %s", args)

    create_embeddings(args)


if __name__ == '__main__':
    generate_embeddings()
