'''
Utility methods for loading and using the Named Entity Recognizer (NER)
'''
from os import makedirs
from os.path import join, exists
import numpy as np

from zodo.ner.ner_utils import PADDING

def get_input(args, word_emb_model, input_file):
    '''loads input dataset based on model selected'''
    # Encode inputs based on annotations in the file
    tokens, instances, labels, max_len = get_sent_input(word_emb_model,
                                                        input_file)

    # Determine type of max length to use i.e. specified or auto
    max_len = args.max_len if args.max_len else max_len
    print("Using max sentence length:", max_len)
    # Based on max length pad sentences to max length for efficiency
    return get_rnn_input(max_len, tokens, instances, labels)

def get_sent_input(word_emb_model, input_file):
    '''loads input dataset for rnn models'''
    print("processing file: {}".format(input_file))
    tokens, instances, labels = [], [], []
    doc_tokens, doc_instances, doc_labels = [], [], []
    for line in open(input_file):
        if len(line.split()) == 2:
            token = line.split()[0]
            label = line.split()[1]
            doc_tokens.append(token)
            doc_instances.append(word_emb_model[token])
            if label.startswith('O'):
                doc_labels.append(np.array([1, 0, 0]))
            elif label.startswith('B'):
                doc_labels.append(np.array([0, 1, 0]))
            elif label.startswith('I'):
                doc_labels.append(np.array([0, 0, 1]))
            else:
                print("Invalid tag {} found for word {}".format(label, token))
        else:
            assert len(doc_tokens) == len(doc_instances) == len(doc_labels)
            tokens.append(doc_tokens)
            instances.append(np.array(doc_instances))
            labels.append(np.array(doc_labels))
            doc_tokens, doc_instances, doc_labels = [], [], []
    assert len(tokens) == len(instances) == len(labels)
    max_len = max([len(x) for x in tokens])
    print("Max length", max_len)
    return tokens, instances, labels, max_len

def get_rnn_input(max_len, sentences, instances, labels):
    '''Pads to max length for RNN models for operational efficiency'''
    input_len = len(instances[0][0])
    num_classes = 3
    token_sets = []
    instns_sets = []
    label_sets = []
    seqlen_sets = []
    s_index = 0
    padding_tok = PADDING
    padding_rep = np.expand_dims(np.zeros(input_len), axis=0)
    padding_lab = np.expand_dims(np.zeros(num_classes), axis=0)
    while s_index < len(sentences):
        s_tokens = sentences[s_index]
        s_instns = instances[s_index]
        s_labels = labels[s_index]
        if len(s_tokens) <= max_len:
            # Append till close to max length for efficiency
            while (s_index + 1 < len(sentences) and
                   len(s_tokens) + len(sentences[s_index+1]) + 1 <= max_len):
                s_index += 1
                s_tokens += sentences[s_index] + [padding_tok]
                s_instns = np.append(s_instns, instances[s_index], axis=0)
                s_instns = np.append(s_instns, padding_rep, axis=0)
                s_labels = np.append(s_labels, labels[s_index], axis=0)
                s_labels = np.append(s_labels, padding_lab, axis=0)
            s_index += 1
        elif len(s_tokens) > max_len:
            # If greater than max length, just break it at max length
            sentences[s_index] = s_tokens[max_len:]
            instances[s_index] = s_instns[max_len:]
            labels[s_index] = s_labels[max_len:]
            s_tokens = s_tokens[:max_len]
            s_instns = s_instns[:max_len]
            s_labels = s_labels[:max_len]
        assert len(s_tokens) == len(s_instns) == len(s_labels)
        # Add padding when short of length
        for _ in range(max_len - len(s_tokens)):
            s_labels = np.append(s_labels, padding_lab, axis=0)
            s_instns = np.append(s_instns, padding_rep, axis=0)
        assert len(s_instns) == len(s_labels)
        token_sets.append(s_tokens)
        instns_sets.append(s_instns)
        label_sets.append(s_labels)
        seqlen_sets.append(len(s_tokens))
    sumtokens = sum([len(x) for x in token_sets])
    print("Total_tokens:{} Total_sents:{}".format(sumtokens, len(sentences)),
          "Token_T:[{}][?]".format(len(token_sets)),
          "Vector_T:[{}][{}][{}]".format(len(instns_sets), len(instns_sets[0]),
                                         len(instns_sets[0][0])),
          "Label_T:[{}][{}][{}]".format(len(label_sets), len(label_sets[0]),
                                        len(label_sets[0][0])))
    assert len(token_sets) == len(instns_sets) == len(label_sets) == len(seqlen_sets)
    return token_sets, np.asarray(instns_sets), np.asarray(label_sets), seqlen_sets

def strict_f1(tokens, prediction, target, write_err=False):
    '''Compute phrasal F1 score for the results'''
    gold_entities = get_ne_indexes(target)
    pred_entities = get_ne_indexes(prediction)
    # inefficient but easy to understand
    true_pos = [x for x in pred_entities if x in gold_entities]
    false_pos = [x for x in pred_entities if x not in gold_entities]
    false_neg = [x for x in gold_entities if x not in pred_entities]
    precision = 1.0 * len(true_pos)/(len(true_pos) + len(false_pos) + 0.000001)
    recall = 1.0 * len(true_pos)/(len(true_pos) + len(false_neg) + 0.000001)
    f1sc = 2.0 * precision * recall / (precision + recall + 0.000001)
    if write_err:
        if not exists("runs"):
            makedirs("runs")
        filename = "runs/ne_{:.5f}".format(f1sc)+".txt"
        print("Writing summary to", filename)
        write_errors(tokens, true_pos, false_pos, false_neg, filename)
    return precision, recall, f1sc

def overlapping_f1(tokens, prediction, target):
    '''Compute phrasal F1 score for the results'''
    gold_entities = get_ne_indexes(target)
    gold_ind = [y for x in gold_entities for y in x.split("_")]
    pred_entities = get_ne_indexes(prediction)
    pred_ind = [y for x in pred_entities for y in x.split("_")]
    # find TP and FP
    true_pos, false_pos, false_neg = 0, 0, 0
    for pred in pred_entities:
        found = False
        for pred_p in pred.split("_"):
            if pred_p in gold_ind:
                found = True
                break
        if found:
            true_pos += 1
        else:
            false_pos += 1
    # find FN
    for gold in gold_entities:
        found = False
        for gold_p in gold.split("_"):
            if gold_p in pred_ind:
                found = True
                break
        if not found:
            false_neg += 1
    precision = 1.0 * true_pos/(true_pos + false_pos + 0.000001)
    recall = 1.0 * true_pos/(true_pos + false_neg + 0.000001)
    f1sc = 2.0 * precision * recall / (precision + recall + 0.000001)
    print("TP {} FP {} FN {}".format(true_pos, false_pos, false_neg))
    return precision, recall, f1sc

def get_ne_indexes(tags):
    '''Get named entities by indices'''
    entities = []
    entity = ''
    for i, label in enumerate(tags):
        if label == 1:
            if entity != '':
                entities.append(entity)
            entity = "{}".format(i)
        elif label == 2:
            if entity != '':
                entity += "_{}".format(i)
            else:
                entity = "{}".format(i)
        else:
            if entity != '':
                entities.append(entity)
                entity = ''
    return entities

def write_errors(tokens, true_pos, false_pos, false_neg, fname='results.log'):
    '''Write the named entities into a file for error analysis'''
    print("TP {} FP {} FN {}".format(len(true_pos), len(false_pos), len(false_neg)))
    rfile = open(fname, 'w')
    print("TP {} FP {} FN {}".format(len(true_pos), len(false_pos), len(false_neg)), file=rfile)
    print("\n--True Positives--", file=rfile)
    for i, item in enumerate(true_pos):
        en_text = " ".join([tokens[int(index)] for index in item.split('_')])
        print("{}\t{}\t{}".format(i, item, en_text), file=rfile)
    print("\n--False Positives--", file=rfile)
    for i, item in enumerate(false_pos):
        en_text = " ".join([tokens[int(index)] for index in item.split('_')])
        print("{}\t{}\t{}".format(i, item, en_text), file=rfile)
    print("\n--FN--", file=rfile)
    for i, item in enumerate(false_neg):
        en_text = " ".join([tokens[int(index)] for index in item.split('_')])
        print("{}\t{}\t{}".format(i, item, en_text), file=rfile)
    rfile.close()