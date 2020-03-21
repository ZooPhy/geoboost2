            # print(loc)
'''
Utility methods for loading and using the Named Entity Recognizer (NER)
'''
import codecs
import re
from unidecode import unidecode
import sys
import pickle
import json
import logging
from os import makedirs
from os.path import exists, isfile, join
from segtok.segmenter import split_multi, split_newline, split_single
import logging
import requests
from requests.utils import quote
from gensim.models.keyedvectors import KeyedVectors

import tensorflow as tf
import numpy as np

from zodo.ner.models import BiRNNModel
from zodo.settings import *

UNK_FILENAME = "unk.pkl"
NUM_FILENAME = "num.pkl"
SPLIT_REGEX = r"(\s|\,|\.|\"|\(|\)|\\|\-|\'|\?|\!|\/|\:|\;|\_|\+|\`|\[|\]|\#|\*|\%|\<|\>|\=|\|)"
TABLE_LOC_TAG = "TableLocation"
TABLE_ACCN_TAG = "TableAccession"
LOC_ANN_TAG = "Location"
PRO_ANN_TAG = "Exclude"
TOKEN_B = "B"
TOKEN_I = "I"
TOKEN_O = "O"
PADDING = "<PADDING>"
CHAREMB_FILENAME = "char-embeddings.pkl"
CORPUS_EMB_FILENAME = "corpus-embeddings.pkl"
CEMB_SIZE = 5
CHAR_EMB_LEN = 10
HYPRM_FILE_NAME = "hyperprms.pkl"
MODEL_NAMES = ["BIRNN", "BIUGRNN", "BIGRU", "BILSTM", "BILSTMP"]
TRAIN_FILE = "train.txt"
DEV_FILE = "dev.txt"
IOB2_DIR = "iob2"

GEONAMES_URL = "http://"+GEO_HOST+":"+GEO_PORT+"/location?location="

LOC_COLUMNS = set(["location", "region", "state", "town", "city", "district", "province"])

# A Global for storing the NER model object
NER_MODEL = None

class Token(object):
    '''
    Word object which contains fields for offsets and annotation.
    Sometimes used to indicate Tokens and sometimes multi-token entities.
    '''
    def __init__(self, text, start, end, encoding, sent_start=0, sent_end=0):
        # placeholders
        self.text = text
        self.start = start
        self.end = end
        self.encoding = encoding
        self.sent_start = sent_start
        self.sent_end = sent_end

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def serialize(self):
        return {'text': self.text, 'start': self.start, 'end': self.end,
                'encoding': self.encoding, 
                'sent_start': self.sent_start, 'sent_end': self.sent_end}

class Annotation(object):
    '''Annotation object which contains fields for BRAT offsets and annotation'''
    def __init__(self, text, start, end, atype, geonameid=-1, lat=0, lon=0):
        # placeholders
        self.text = text
        self.start = start
        self.end = end
        self.atype = atype
        self.geonameid = geonameid
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class WordEmb(object):
    """Loads the word2vec model"""
    def __init__(self, args):
        logging.debug("Loading word embeddings from %s", str(args.emb_loc))
        if args.emb_loc.endswith("pkl"):
            self.wvec = pickle.load(open(args.emb_loc, "rb"))
            logging.debug("Word Embedding vocabulary: %s", len(self.wvec))
        else:
            if args.embvocab > 0:
                self.wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True,
                                                              limit=args.embvocab)
            else:
                self.wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True)
        unk_filename = join(args.work_dir, UNK_FILENAME)
        num_filename = join(args.work_dir, NUM_FILENAME)
        if isfile(unk_filename) and isfile(num_filename):
            logging.debug("Loading unk from file")
            self.unk = pickle.load(open(unk_filename, "rb"))
            self.num = pickle.load(open(num_filename, "rb"))
        else:
            logging.error("Can't find unk and num pkl files. Run training steps again.")
            sys.exit(0)
        self.is_case_sensitive = True if (self.wvec['the'] != self.wvec['The']).all() else False
        if not self.is_case_sensitive:
            logging.warning("Warning: dictionary is NOT case-sensitive")
        self.cemb = CharEmb(args)

    def __getitem__(self, word):
        if not self.is_case_sensitive:
            word = word.lower()
        try:
            word_vec = self.wvec[word]
        except KeyError:
            if word.isdigit():
                word_vec = self.num
            else:
                word_vec = self.unk
        word_vec = np.append(word_vec, np.array(case_feature(word)))
        word_vec = np.append(word_vec, np.array(char_feature(word, self.cemb)))
        return word_vec

class CharEmb(object):
    """Loads the character embedding model"""
    def __init__(self, args):
        cemb_path = join(args.work_dir, CHAREMB_FILENAME)
        logging.debug("Loading char embeddings from %s", cemb_path)
        self.cvec = pickle.load(open(cemb_path, "rb"))

    def __getitem__(self, char):
        try:
            return self.cvec[char]
        except KeyError:
            return self.cvec["UNKK"]

def case_feature(word):
    '''returns an basic orthographic feature'''
    all_caps = True
    for char in word:
        try:
            if not ord('A') <= ord(char) <= ord('Z'):
                all_caps = False
                break
        except Exception as e:
            all_caps = False
            logging.error("Exception: '%s' in word '%s' at char '%s'", e, word, char)
    if all_caps:
        return [1, 0, 0]
    else:
        try:
            if ord('A') <= ord(word[0]) <= ord('Z'):
                return [0, 1, 0]
            else:
                return [0, 0, 1]
        except Exception as e:
            all_caps = False
            logging.error("Exception: '%s' in word '%s' at char '%s'", e, word, char)
            return [0, 0, 1]

def char_feature(word, char_emb_model):
    '''returns a char emb feature'''
    features = []
    padding = [0 for _ in range(len(char_emb_model["a"]))]
    for i in range(CHAR_EMB_LEN):
        if i < len(word):
            features.extend(char_emb_model[word[i]])
        else:
            features.extend(padding)
    return features

class NER_Model(object):
    '''NER Model object with loads the graph (session), model, and embedding to memory'''
    def __init__(self, sess, hyp, model, emb):
        # placeholders
        self.sess = sess
        self.hyp = hyp
        self.model = model
        self.emb = emb

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

def load_ner(args):
    '''Method for detection of entities and normalization'''
    global NER_MODEL
    logging.info("\nLoading Word Embeddings. This may take a while.")
    word_emb = WordEmb(args)
    logging.info("Loading the Bi-LSTM model variables")
    # Model checkpoint path
    save_loc = join(args.save, join(args.model, args.model))
    # Load Hyperparams from disk
    hypr_loc = save_loc + "_" + HYPRM_FILE_NAME
    hyperparams = pickle.load(open(hypr_loc, "rb"))
    # Create model
    model = BiRNNModel(hyperparams)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_loc + '.meta')
    saver.restore(sess, save_loc)
    # ner_model = NER_Model(sess, hyperparams, model, word_emb)
    NER_MODEL = NER_Model(sess, hyperparams, model, word_emb)
    logging.info("Done loading model!\n")

def detect(text, bioc_json=False):
    '''
    Method for detection of named entities i.e. spans of interest
    if bioc_json is set to True then the text is processed as a json in bioc format
    '''
    spans = []
    # tokenize based on formats
    if bioc_json:
        doc_sents, doc_text, spans = tokenize_bioc(text)
    else:
        doc_sents = tokenize_text(text)
        doc_text = text
    # if text is empty just return an empty array
    if not doc_text.strip():
        return []
    # get the input representation
    pub_t, pub_v, pub_sl = get_rnn_input(NER_MODEL.hyp.max_len, 
                                         doc_sents,
                                         NER_MODEL.emb)
    pub_m = get_mask(pub_v.shape[0], pub_v.shape[1], pub_sl)
    pred = NER_MODEL.sess.run(NER_MODEL.model.prediction,
                              feed_dict={NER_MODEL.model.input_x: np.asarray(pub_v),
                                         NER_MODEL.model.length: np.asarray(pub_sl),
                                         NER_MODEL.model.dropout: 1.0})
    # spans = get_entity_annotations(args.outdir, pub_t, pred, pub_m, pmid)
    prediction = np.argmax(pred, axis=2)
    # Flatten the values based on sequence mask
    prediction = prediction[pub_m].flatten()
    tokens = [x for y in pub_t for x in y]
    spans = get_pred_anns(tokens, prediction, spans)
    spans = remove_duplicates(spans)
    return spans, doc_text

def tokenize_document(doc_path):
    '''Tokenize the text and preserve offsets'''
    with codecs.open(doc_path, 'r', 'utf-8') as myfile:
        doc_text = myfile.read()
    doc_tokens = tokenize_text(doc_text)
    doc_vocab = set([x.text for y in doc_tokens for x in y])
    return doc_tokens, doc_vocab

def remove_duplicates(spans):
    '''Remove duplicates in list of spans'''
    entity_groups, mod_spans = {}, []
    # group by sentence start spans
    for span in spans:
        if span.sent_start not in entity_groups:
            entity_groups[span.sent_start] = []
        entity_groups[span.sent_start].append(span)
    # remove sentences with just one entity
    for _, group in entity_groups.items():
        if len(group)>1:
            removal = set()
            for i, x in enumerate(group[:-1]):
                for j, y in enumerate(group[i+1:]):
                    if x.start <= y.start < x.end or x.start < y.end <= x.end:
                        # print(i, x, "\n", j, y)
                        # add to removal list
                        temp = i if x.encoding == LOC_ANN_TAG else i+j+1
                        logging.debug("Removing %s", group[temp])
                        removal.add(temp)
            group = [x for i, x in enumerate(group) if i not in removal]
            mod_spans.extend(group)
        else:
            mod_spans.extend(group)
    return mod_spans

def tokenize_text(doc_text):
    '''Tokenize the text and preserve offsets'''
    # Split text into sentences using segtok, then words and create Token objects
    sents = [sent for sent in split_multi(doc_text) if sent.strip() != ""]
    doc_tokens = []
    current_offset = 0
    for sent in sents:
        sent_tokens = []
        words = re.split(SPLIT_REGEX, sent)
        words = [word.strip() for word in words if word.strip() != ""]
        for word in words:
            word_offset = doc_text.index(word, current_offset)
            current_offset = word_offset + len(word)
            word = unidecode(word)
            sent_token = Token(word, word_offset, word_offset+len(word), TOKEN_O)
            sent_tokens.append(sent_token)
        if sent_tokens:
            sent_start = sent_tokens[0].start
            sent_end = sent_tokens[-1].end
        # Update sentence offsets
        for token in sent_tokens:
            token.sent_start = sent_start
            token.sent_end = sent_end
        doc_tokens.append(sent_tokens)
    return doc_tokens

def tokenize_bioc(doc_bioc):
    '''Tokenize the text from bioc and preserve offsets'''
    valid_sections = PMCOA_TYPES if doc_bioc["source"] == "PMC" else PM_TYPES
    # collect all the bioc passages
    passages = [passage for doc in doc_bioc["documents"] for passage in doc["passages"]]
    # initialize empty sentences this will be a list of sentences which will be a list of tokens
    doc_text = "" # stores plain text extracted from bioc
    doc_sents = [] # stores lists of sentences
    doc_ents = [] # stores the entities if discovered manually
    for passage in passages:
        section_type = passage["infons"]["type"]
        # skip based on section and process only if necessary
        if section_type not in valid_sections:
            continue
        passage_text = passage["text"] + "\n"
        # set current offset to the length of the text
        current_offset = len(doc_text)
        doc_text += passage_text
        # split differently based on the section type, this maybe easier if we are processing xml
        if section_type == "table":
            section_id = passage["infons"]["id"] if "id" in passage["infons"] else ""
            # first split by newline into rows and use row as the sentence because often accessions
            # will describe the location of infected host in the same row
            rows = [row for row in passage_text.split('\n') if row.strip() != ""]
            accession_cols, location_cols = set(), set()
            for rownum, row in enumerate(rows):
                row_start = doc_text.index(row, current_offset)
                row_end = row_start + len(row)
                # split into columns and remove empty cells if necessary
                # if entries are to be extracted by columns then do not remove empty cells
                cols = [col for col in row.split('\t') if col.strip() != ""]
                if rownum == 0:
                    # title row
                    for numcol, col in enumerate(cols):
                        if "accession" in col.lower():
                            accession_cols.add(numcol)
                            continue
                        if any([x in col.lower() for x in LOC_COLUMNS]):
                            location_cols.add(numcol)
                    if accession_cols or location_cols:
                        logging.debug("Detected accession column '%s' and location column '%s' in Table '%s'",
                                      ",".join([str(x) for x in accession_cols]),
                                      ",".join([str(x) for x in location_cols]), section_id)
                # add individual column content as a sentence to be processed individually
                for numcol, col in enumerate(cols):
                    # if accession column was found or location column was found, add to annotations
                    if rownum != 0 and numcol in accession_cols:
                        logging.debug("Adding Accession entry '%s' in row '%s' in Table '%s'", col, row, section_id)
                        col_start = doc_text.index(col, current_offset)
                        doc_ents.append(Token(col, col_start, col_start+len(col), TABLE_ACCN_TAG,
                                              sent_start=row_start, sent_end=row_end))
                    elif rownum != 0 and numcol in location_cols:
                        logging.debug("Adding Location entry '%s' in row '%s' in Table '%s'", col, row, section_id)
                        col_start = doc_text.index(col, current_offset)
                        doc_ents.append(Token(col, col_start, col_start+len(col), TABLE_LOC_TAG,
                                              sent_start=row_start, sent_end=row_end))
                    # now tokenize and record spans
                    sent_tokens = []
                    words = re.split(SPLIT_REGEX, col)
                    words = [word.strip() for word in words if word.strip() != ""]
                    for word in words:
                        word_offset = doc_text.index(word, current_offset)
                        current_offset = word_offset + len(word)
                        word = unidecode(word)
                        sent_token = Token(word, word_offset, word_offset+len(word), TOKEN_O,
                                           sent_start=row_start, sent_end=row_end)
                        sent_tokens.append(sent_token)
                    # add sentence tokens to list of sentences
                    if sent_tokens:
                        doc_sents.append(sent_tokens)
        else:
            # Split text into sentences using segtok, then words/tokens and create Token objects
            sents = [sent for sent in split_single(passage_text) if sent.strip() != ""]
            for sent in sents:
                sent_tokens = []
                words = re.split(SPLIT_REGEX, sent)
                words = [word.strip() for word in words if word.strip() != ""]
                for word in words:
                    word_offset = doc_text.index(word, current_offset)
                    current_offset = word_offset + len(word)
                    word = unidecode(word)
                    sent_token = Token(word, word_offset, word_offset+len(word), TOKEN_O)
                    sent_tokens.append(sent_token)
                if sent_tokens:
                    sent_start = sent_tokens[0].start
                    sent_end = sent_tokens[-1].end
                # Update sentence offsets
                for token in sent_tokens:
                    token.sent_start = sent_start
                    token.sent_end = sent_end
                doc_sents.append(sent_tokens)
    # verify if necessary
    # logging.debug("Tokenized %s sentences:\n%s", len(doc_sents), "\n".join([" ".join([token.text + ":" + str(token.start) for token in sent]) for sent in doc_sents]))
    return doc_sents, doc_text, doc_ents

def get_rnn_input(max_len, sentences, word_emb_model):
    '''Pads to max length for RNN models for operational efficiency'''
    # get sentence representation
    instances = [[word_emb_model[token.text] for token in sent] for sent in sentences]
    # get tensor representation and use optimized padding for faster processing
    input_len = len(instances[0][0])
    token_sets = []
    instns_sets = []
    seqlen_sets = []
    s_index = 0
    padding_tok = Token(PADDING, 0, 0, TOKEN_O)
    padding_rep = np.expand_dims(np.zeros(input_len), axis=0)
    while s_index < len(sentences):
        s_tokens = sentences[s_index] + [padding_tok]
        s_instns = np.append(instances[s_index], padding_rep, axis=0)
        if len(s_tokens) <= max_len:
            # Append till close to max length for efficiency
            while (s_index + 1 < len(sentences) and
                   len(s_tokens) + len(sentences[s_index+1]) + 1 <= max_len):
                s_index += 1
                # add the padding token and padding vector between sentences
                s_tokens += sentences[s_index] + [padding_tok]
                s_instns = np.append(s_instns, instances[s_index], axis=0)
                s_instns = np.append(s_instns, padding_rep, axis=0)
            s_index += 1
        elif len(s_tokens) > max_len:
            # If greater than max length, just break it
            sentences[s_index] = s_tokens[max_len:]
            instances[s_index] = s_instns[max_len:]
            s_tokens = s_tokens[:max_len]
            s_instns = s_instns[:max_len]
        assert len(s_tokens) == len(s_instns)
        # Add padding
        for _ in range(max_len - len(s_tokens)):
            s_instns = np.append(s_instns, padding_rep, axis=0)
        token_sets.append(s_tokens)
        instns_sets.append(s_instns)
        seqlen_sets.append(len(s_tokens))
    assert len(token_sets) == len(instns_sets) == len(seqlen_sets)
    return token_sets, np.asarray(instns_sets), seqlen_sets

def get_pred_anns(tokens, prediction, entities=[]):
    '''Get list of named entitiess'''
    assert len(tokens) == len(prediction)
    text = ''
    start = -1
    end = -1
    sent_start = -1
    sent_end = -1
    for i, label in enumerate(prediction):
        if tokens[i].text == PADDING:
            continue
        if label == 1:
            token_str = tokens[i].text
            if text != '':
                # Should happen only for dense entity text
                entity = Token(text, start, end, LOC_ANN_TAG, sent_start, sent_end)
                entities.append(entity)
                text = ''
            start = tokens[i].start
            end = tokens[i].end
            sent_start = tokens[i].sent_start
            sent_end = tokens[i].sent_end
            text = "{}".format(token_str)
        elif label == 2:
            token_str = tokens[i].text
            if text != '':
                end = tokens[i].end
                sent_end = tokens[i].sent_end
                text += " {}".format(token_str)
            else:
                # This should not happen either but here we treat it as B
                start = tokens[i].start
                end = tokens[i].end
                text = "{}".format(token_str)
                sent_start = tokens[i].sent_start
                sent_end = tokens[i].sent_end
        else:
            if text != '':
                entity = Token(text, start, end, LOC_ANN_TAG, sent_start, sent_end)
                entities.append(entity)
                text = ''
    return entities

def get_entity_annotations(outdir, tokens, prediction, mask, pmid, write_tokens=False):
    '''Get entity annotations from predictions and write to file for debugging'''
    prediction = np.argmax(prediction, axis=2)
    # Flatten the values based on sequence mask
    prediction = prediction[mask].flatten()
    tokens = [x for y in tokens for x in y]
    if write_tokens:
        if not exists(outdir):
            makedirs(outdir)
        fname = join(outdir, pmid + '_debug.txt')
        rfile = codecs.open(fname, 'w', 'utf-8')
        for i, label in enumerate(prediction):
            if label == 0:
                label = 'O'
            elif label == 1:
                label = 'B'
            elif label == 2:
                label = 'I'
            try:
                print("{}\t{}".format(tokens[i].text, label), file=rfile)
            except UnicodeError:
                print("{}\t{}".format("ERR-TOKEN", label), file=rfile)
        rfile.close()
    entities = get_pred_anns(tokens, prediction)
    return entities

def get_mask(num_rows, num_cols, col_lens):
    '''Get the mask of a 2-D matrix given the sequence lengths'''
    mask = np.full((num_rows, num_cols), True, dtype=bool)
    for i, _ in enumerate(mask):
        mask[i][col_lens[i]:] = False
    return mask

def get_normalized_entities(entities):
    '''Get entity geoname ids from geoname services project'''
    entities = [Annotation(x.text, x.start, x.end, x.encoding) for x in entities]
    for entity in entities:
        if entity.atype == LOC_ANN_TAG:
            url = GEONAMES_URL+quote(entity.text)
            response = requests.get(url)
            jsondata = response.json()
            if jsondata and int(jsondata["retrieved"]) > 0:
                record = jsondata["records"][0]
                entity.geonameid = record["GeonameId"]
                entity.lat = record["Latitude"]
                entity.lon = record["Longitude"]
    # Remove entities that couldn't be resolved
    entities = [x for x in entities if x.geonameid != -1]
    return entities

def write_annotations(outdir, entities, pmid, normalize=False):
    '''Write annotations to file in BRAT format'''
    if not exists(outdir):
        makedirs(outdir)
    logging.debug("writing results to '%s'", outdir)
    fname = join(outdir, pmid + '.ann')
    rfile = codecs.open(fname, 'w', 'utf-8')
    if normalize:
        entities = [x.span for x in entities]
        entities = get_normalized_entities(entities)
    for index, entity in enumerate(entities):
        if entity.atype == LOC_ANN_TAG:
            # print("{}".format(entity), file=rfile)
            print("T{}\tGeographical {} {}\t{}".format(index, entity.start, entity.end, entity.text),
                  file=rfile)
            print("#{}\tAnnotatorNotes T{}\t<latlng>{},{}</latlng><geoID>{}</geoID>".format(
                index, index, entity.lat, entity.lon, entity.geonameid), file=rfile)
    rfile.close()
    logging.debug("%s entities found in %s ", len(entities), pmid)
