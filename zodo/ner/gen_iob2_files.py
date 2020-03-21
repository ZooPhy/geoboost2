'''Process BRAT Annotated Data into the NER friendly IOB2 format'''
import sys
import re
import argparse
import pickle
import random
import requests
from os import listdir, makedirs
from os.path import join, exists
import numpy as np
from segtok.segmenter import split_multi
import codecs
import logging
from zodo.ner.ner_utils import (SPLIT_REGEX, IOB2_DIR,
                                TOKEN_B, TOKEN_I, TOKEN_O,
                                TRAIN_FILE, DEV_FILE,
                                LOC_ANN_TAG, PRO_ANN_TAG,
                                Token, Annotation)
from zodo.settings import GEO_HOST, GEO_PORT

EXT_GLL_REGEX = r"<latlng>(.*)</latlng>"
EXT_GID_REGEX = r"<geoID>(.*)</geoID>"

GEONAMES_QUERY_URL = "http://"+GEO_HOST+":"+GEO_PORT+"/search?query=GeonameId:GIDPH"

def read_annotations(doc_path):
    '''Read annotations into annotation object'''
    annotations = []
    with open(doc_path, 'r') as myfile:
        logging.debug("Reading '%s'", doc_path)
        doc_lines = myfile.readlines()
        index = 0
        span_map = {}
        while index < len(doc_lines):
            line = doc_lines[index].strip()
            parts = line.split("\t")
            # There can exist more than 2 tab spaces if the text itself has tab spaces
            if len(parts) >= 3:
                if parts[0].startswith("T") or parts[0].startswith("#"):
                    if parts[0].startswith("T")  and parts[1].startswith("Location"):
                        ann_id = parts[0].strip()
                        offset_text = ""
                        # for cases when text has tab spaces
                        for part in parts[2:]:
                            offset_text += part
                        # for cases when text has new lines
                        while index+1 < len(doc_lines) and len(doc_lines[index+1].split("\t")) < 3:
                            offset_text += doc_lines[index+1]
                            index += 1
                        ann_type = LOC_ANN_TAG
                        offset_text = offset_text.strip()
                        offset_start = int(parts[1].strip().split()[1])
                        offset_end = int(parts[1].strip().split()[-1])
                        ann = Annotation(offset_text, offset_start, offset_end, ann_type)
                        span_map[ann_id] = ann
                        index += 1
                    elif parts[0].startswith("#") and parts[1].startswith("AnnotatorNotes"):
                        # Extract geonameid
                        id_parts = doc_lines[index].strip().split("\t")
                        ann_id = (id_parts[1].split(" ")[1]).strip()
                        id_search = re.search(EXT_GID_REGEX, id_parts[2])
                        geonameid = id_search.group(1).strip() if id_search else "-1"
                        geonameid = geonameid if geonameid != "NA" else "-1"
                        if not id_search:
                            logging.error("%s\t%s\tNO GeonameId found", doc_path, ann_id)
                        id_search = re.search(EXT_GLL_REGEX, id_parts[2])
                        if id_search:
                            latlng = id_search.group(1).strip()
                            lat = latlng.split(",")[0] if latlng != "NA" else "0"
                            lon = latlng.split(",")[1] if latlng != "NA" else "0"
                        else:
                            logging.error("ERROR: Could not extract Lat Long parts in '%s' - '%s' in %s",
                                          ann_id, id_parts[2], doc_path)
                        if ann_id in span_map:
                            ann = span_map[ann_id]
                            del span_map[ann_id] #remove from map
                            ann = Annotation(ann.text, ann.start, ann.end, ann.atype,
                                             int(geonameid), float(lat), float(lon))
                            annotations.append(ann)
                        else:
                            logging.error("ERROR: %s not found", ann_id)
                        index += 1
                    elif parts[0].startswith("T") and parts[1].startswith("Protein"):
                        # Extract exclusion section. We will not consider text from this span
                        offset_start = int(parts[1].strip().split()[1])
                        # Check if next line ends with BEGIN
                        if not doc_lines[index+1].strip().endswith("BEGIN"):
                            logging.error("Error 1: invalid protein %s Entity: %s", index+1, parts[1])
                        end_parts = doc_lines[index+2].strip().split("\t")
                        offset_end = int(end_parts[1].strip().split()[-1])
                        offset_text = parts[2] + "-" + end_parts[2]
                        # Check if next line ends with END
                        if not doc_lines[index+3].strip().endswith("END"):
                            logging.error("Error 2: invalid protein %s Entity: %s", index+3, parts[1])
                        ann_type = PRO_ANN_TAG
                        index += 4
                        ann = Annotation(offset_text, offset_start, offset_end, ann_type)
                        annotations.append(ann)
                    else:
                        logging.error("Error 1: invalid line %s Entity: %s", index, parts[1])
                        index += 1
                else:
                    logging.error("Error 2: invalid line %s Entity: %s", index, parts[1])
                    index += 1
            else:
                logging.error("Error 3: invalid line %s Entity: %s", index, len(parts))
                index += 1
        # Now check for annotations without ids and add them
        for ann_id, ann in span_map.items():
            annotations.append(ann)
    return annotations

def tokenize_document(doc_path):
    '''Tokenize the text and preserve offsets'''
    with codecs.open(doc_path, 'r', "utf-8") as myfile:
        doc_text = myfile.read()
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
            sent_token = Token(word, word_offset, word_offset+len(word), TOKEN_O)
            sent_tokens.append(sent_token)
        doc_tokens.append(sent_tokens)
    return doc_tokens, doc_text

def verify_annotation(doc_text, annotations):
    """Verify annotations too check if offsets match"""
    errors = 0
    for ann in annotations:
        if ann.atype == LOC_ANN_TAG:
            text_in_ann = ann.text#.replace("\n", " ")
            text_in_doc = doc_text[ann.start:ann.end]
            text_in_doc = re.sub(r'(\s|\n|\r)+', ' ', text_in_doc)
            if text_in_ann != text_in_doc:
                text_in_doc = re.sub('- ', '-', text_in_doc)
                if text_in_ann != text_in_doc:
                    errors += 1
                    logging.warning("'%s' not the same as '%s'", text_in_ann, text_in_doc)
    if errors:
        logging.warning("Errors in file: %s / %s", errors, len(annotations))

def isAncestor(curr_geonameid, next_geonameid):
    ''''''
    url = GEONAMES_QUERY_URL.replace("GIDPH", str(curr_geonameid))
    response = requests.get(url)
    jsondata = response.json()
    if jsondata and jsondata["retrieved"] and int(jsondata["retrieved"]) > 0:
        loc = jsondata["records"][0]
    else:
        logging.warning("Search returned no results: '%s'->'%s'", curr_geonameid, url)
        return False
    if "AncestorsIds" in loc:
        if next_geonameid in loc["AncestorsIds"].split(", "):
            return True
    else:
        return False

def merge_annotations(doc_text, annotations):
    """Merge annotations too check if offsets match"""
    modified_anns = []
    valid_anns = [x for x in annotations if x.atype == LOC_ANN_TAG]
    invalid_anns = [x for x in annotations if x.atype != LOC_ANN_TAG]
    valid_anns.sort(key=lambda x: x.start)
    i = 0
    while i < len(valid_anns):
        curr_ann = valid_anns[i]
        j = i + 1
        to_be_removed = []
        while j < len(valid_anns):
            next_ann = valid_anns[j]
            text_in_bw = doc_text[curr_ann.end:next_ann.start]
            if text_in_bw.strip() in [",", "in"]:
                # now check if the latter is an ancestor
                if curr_ann.geonameid > 0 and next_ann.geonameid > 0:
                    if isAncestor(str(curr_ann.geonameid), str(next_ann.geonameid)):
                        text = doc_text[curr_ann.start:next_ann.end]
                        start, end = curr_ann.start, next_ann.end
                        ann = Annotation(text, start, end, curr_ann.atype,
                                        curr_ann.geonameid, float(curr_ann.lat), float(curr_ann.lon))
                        curr_ann = ann
                        to_be_removed.append(next_ann)
                    else:
                        break
                else: 
                    break
            else: 
                break
            j += 1
        # remove the latter annotations which were combined
        for x in to_be_removed:
            valid_anns.remove(x)
        modified_anns.append(curr_ann)
        i += 1
    # Merge annotations
    modified_anns += invalid_anns
    modified_anns.sort(key=lambda x: x.start)
    if len(annotations) != len(modified_anns):
        logging.info("Merged annotations: Original %s condensed into %s annotations", len(annotations), len(modified_anns))
    return modified_anns

def convert_to_iob2(args, txt_files, outfile):
    """load corpus data and write to IOB2 formatted files"""
    rfile = codecs.open(outfile, 'w', "utf-8")
    for _, txt_file in enumerate(txt_files):
        logging.info("\tProcessing '%s'", txt_file[:-4])
        doc_sents, doc_text = tokenize_document(txt_file)
        annotations = read_annotations(txt_file[:-3]+"ann")
        # verify if annotations match
        verify_annotation(doc_text, annotations)
        # merge annotations based on simple heuristics
        annotations = merge_annotations(doc_text, annotations)
        # process each sentence
        for doc_sent in doc_sents:
            empty_sent = True
            for token in doc_sent:
                ignore_token = False
                for ann in annotations:
                    if token.start >= ann.start and token.end <= ann.end:
                        # Protein annotations are sections that can be ignored
                        if ann.atype == PRO_ANN_TAG:
                            ignore_token = True
                        # IOB annotations
                        if ann.atype == LOC_ANN_TAG:
                            if token.start == ann.start:
                                token.encoding = TOKEN_B + "-" + LOC_ANN_TAG
                            else:
                                token.encoding = TOKEN_I + "-" + LOC_ANN_TAG
                        break
                if not ignore_token:
                    str_to_write = token.text + "\t" + token.encoding
                    print(str_to_write, file=rfile)
                    empty_sent = False
            # write empty line to indicate end of training instance
            if not empty_sent:
                print("", file=rfile)
    rfile.close()

def process_annotated_ner_data(args):
    '''Load the BRAT annotated data for the NER and output IOB2 formatted file'''
    random.seed(args.seed)
    iob2_dir = join(args.work_dir, IOB2_DIR)
    if not exists(iob2_dir):
        makedirs(iob2_dir)
    # Load training set
    txt_files = [join(args.train_corpus, f) for f in listdir(args.train_corpus) if f.endswith(".txt")]
    # calculate number of validation set files and split
    rand_ind = random.sample(range(len(txt_files)), int(len(txt_files)*args.dev_perc))
    dev_files = [txt_files[x] for x in rand_ind]
    train_files = [x for x in txt_files if x not in dev_files]
    logging.info("Training set files: %s", len(train_files))
    # process training data
    training_iob2 = join(iob2_dir, TRAIN_FILE)
    convert_to_iob2(args, train_files, training_iob2)

    logging.info("Development set files: %s", len(dev_files))
    # process development data
    dev_iob2 = join(iob2_dir, DEV_FILE)
    convert_to_iob2(args, dev_files, dev_iob2)
    logging.info("Done generating IOB2 formatted files!")

def generate_iob2():
    '''Main method : parse input arguments and train'''
    parser = argparse.ArgumentParser()
    # Input and Output paths
    parser.add_argument('-t', '--train_corpus', type=str, default='data/train/',
                        help='path to dir where training corpus files are stored')
    parser.add_argument('-v', '--dev_perc', type=float, default=0.05,
                        help='proportion of training data for validation/development')
    parser.add_argument('-s', '--seed', type=int, default=666,
                        help='seed to be used for splitting train and dev randomly')
    parser.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    args = parser.parse_args(args=[])
    logging.debug(args)
    process_annotated_ner_data(args)

if __name__ == '__main__':
    generate_iob2()
