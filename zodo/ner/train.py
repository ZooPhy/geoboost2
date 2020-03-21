'''
Loading training files and optimizing the NER model for recognizing locations
based on the development corpus
'''
from argparse import ArgumentParser, Namespace
import pickle
import os
from os import makedirs
from os.path import join, exists
import numpy as np
import tensorflow as tf
import logging

from zodo.ner.models import BiRNNModel, ModelHypPrms
from zodo.ner.ner_utils import (WordEmb, get_mask, HYPRM_FILE_NAME, MODEL_NAMES, 
                                IOB2_DIR, TRAIN_FILE, DEV_FILE, CORPUS_EMB_FILENAME)
from zodo.ner.train_utils import overlapping_f1, strict_f1, get_input

def train_step(sess, model, instances, lengths, labels, batch_size):
    '''Train and obtain average cost'''
    lengths = np.asarray(lengths)
    shuffle_indices = np.random.permutation(np.arange(len(instances)))
    instances = instances[shuffle_indices]
    lengths = lengths[shuffle_indices]
    labels = labels[shuffle_indices]
    avg_cost = 0.
    total_batch = int(len(instances)/batch_size)
    for ptr in range(0, len(instances), batch_size):
        # Run backprop and cost during training
        _, epoch_cost = sess.run([model.optimizer, model.cost], feed_dict={
            model.input_x: np.asarray(instances[ptr:ptr + batch_size]),
            model.length: np.asarray(lengths[ptr:ptr + batch_size]),
            model.input_y: np.asarray(labels[ptr:ptr + batch_size]),
            model.dropout: 0.5})
        # Compute average loss across batches
        avg_cost += epoch_cost / total_batch
    return avg_cost

def run_step(sess, model, instances, lengths, use_crf):
    '''Run step and get predictions'''
    predictions = []
    predictions, tr_prms = sess.run([model.prediction, model.tr_prms], feed_dict={
        model.input_x: np.asarray(instances),
        model.length: np.asarray(lengths),
        model.dropout: 1.0})
    if not use_crf:
        y_pred = np.argmax(predictions, axis=2)
    else:
        viterbi_sequences = []
        for logit, _ in zip(predictions, lengths):
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit, tr_prms)
            viterbi_sequences += [viterbi_seq]
        y_pred = np.array(viterbi_sequences)
    return y_pred

def evaluate(sess, model, tokens, instances, lengths, mask, labels, num_classes,
             use_crf, verbose=False, detail=False):
    '''Evaluate f1-score based on num_classes'''
    # Get predictions
    y_pred = run_step(sess, model, instances, lengths, use_crf)
    # Get the label indices for the truth values
    y_true = np.argmax(labels, axis=2)
    # Flatten the values based on sequence mask
    y_pred = y_pred[mask].flatten()
    y_true = y_true[mask].flatten()
    # Flatten tokens
    tokens = [x for y in tokens for x in y]
    # Strict evaluation
    str_p, str_r, str_f1 = strict_f1(tokens, y_pred, y_true, detail)
    if verbose:
        print("Strict: P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(str_p, str_r, str_f1))
    # Overlapping evaluation
    if detail:
        ovlp_p, ovlp_r, ovlp_f1 = overlapping_f1(tokens, y_pred, y_true)
        print("Overlapping: P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(ovlp_p, ovlp_r, ovlp_f1))
    return str_f1

def train_model(args, model_name):
    '''Training method'''
    print("\nTraining model:", model_name)
    print("Arguments\n", args)
    if args.gpu is not None:
        print("Using GPU resource", args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Load Word Embeddings i.e. mini-corpus embeddings
    if exists(join(args.work_dir, CORPUS_EMB_FILENAME)):
        args.emb_loc = join(args.work_dir, CORPUS_EMB_FILENAME)
    word_emb = WordEmb(args)
    # Load training and development set tokens, vector instances (input vector) and labels
    train_iob2 = join(args.work_dir, join(IOB2_DIR, TRAIN_FILE))
    train_t, train_v, train_l, train_sl = get_input(args, word_emb, train_iob2)
    dev_iob2 = join(args.work_dir, join(IOB2_DIR, DEV_FILE))
    dev_t, dev_v, dev_l, dev_sl = get_input(args, word_emb, dev_iob2)
    train_m = get_mask(train_v.shape[0], train_v.shape[1], train_sl)
    dev_m = get_mask(dev_v.shape[0], dev_v.shape[1], dev_sl)
    n_input = len(train_v[0][0])
    print("Input size detected ", n_input)
    num_classes = 3
    hyperparams = ModelHypPrms(model_name, n_input, num_classes, args.hid_dims,
                               args.lrn_rate, args.num_layers, args.use_crf,
                               args.max_len)
    # Create model
    model = BiRNNModel(hyperparams)
    # Model checkpoint path
    if not exists(join(args.save, model_name)):
        makedirs(join(args.save, model_name))
    save_loc = join(args.save, join(model_name, model_name))
    # Save hyperparams to disk
    hypr_loc = save_loc + "_" + HYPRM_FILE_NAME
    pickle.dump(hyperparams, open(hypr_loc, "wb"))
    # Initialize TG variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        # Training cycle
        if args.train_epochs > 0:
            max_trf1 = float("-inf")
            max_vaf1 = float("-inf")
            for epoch in range(args.train_epochs):
                avg_cost = train_step(sess, model, train_v, train_sl, train_l,
                                      args.batch_size)
                trf1 = evaluate(sess, model, train_t, train_v, train_sl, train_m, train_l,
                                num_classes, args.use_crf)
                vaf1 = evaluate(sess, model, dev_t, dev_v, dev_sl, dev_m, dev_l,
                                num_classes, args.use_crf)
                print("Epoch:", '%03d' % (epoch+1), "cost=", "{:.5f}".format(avg_cost),
                      "\tTraining : {:.5f}".format(trf1), "Val : {:.5f}".format(vaf1))
                if vaf1 > max_vaf1 or (trf1 > max_trf1 and vaf1 > max_vaf1-0.01):
                    # Write model checkpoint to disk
                    print("Saving model to {}".format(save_loc))
                    saver.save(sess, save_loc)
                    max_vaf1 = vaf1 if vaf1 > max_vaf1 else max_vaf1
                    max_trf1 = trf1 if trf1 > max_trf1 else max_trf1
            print("Optimization Finished!")
        # Load best model and evaluate model on the test set before applying to production
        print("Loading best model from {}".format(save_loc))
        saver = tf.train.import_meta_graph(save_loc + '.meta')
        saver.restore(sess, save_loc)
        print("Evaluating Training Set")
        evaluate(sess, model, train_t, train_v, train_sl, train_m, train_l,
                 num_classes, args.use_crf)
        print("Evaluating Development Set")
        str_f1 = evaluate(sess, model, dev_t, dev_v, dev_sl, dev_m, dev_l,
                 num_classes, args.use_crf, True, True)
        # Close session and deinitialize variables
        sess.close()
        model = init = saver = sess = hyperparams = None
    # Reset graph and return
    tf.reset_default_graph()
    return str_f1

def train(args:Namespace):
    """Training method for the NER
    
    Arguments:
        args {Namespace} -- Namespace object containing command-line arguments from argparse
    """
    if args.model == "ALL":
        for model_name in MODEL_NAMES:
            strct_f1 = train_model(args, model_name)
            print("--Model Name:{}".format(model_name),
                  "--Strict F1:{:.5f}".format(strct_f1))
    else:
        train_model(args, args.model)
    

def main():
    '''Main method : parse input arguments and train'''
    parser = ArgumentParser(description="Program to train the NER")
    # Input files
    parser.add_argument('--model', type=str, default="BIGRU",
                        choices=MODEL_NAMES + ["ALL"],
                        help="Model to be used")
    parser.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    # Word Embeddings
    parser.add_argument('--emb_loc', type=str, default="resources/word-embeddings.pkl",
                        help='word2vec embedding location')
    # Hyperparameters
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of hidden layers')
    parser.add_argument('--hid_dims', type=str, default="150",
                        help='dimensions of hidden layers')
    parser.add_argument('--lrn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    # Settings
    parser.add_argument('--train_epochs', type=int, default=100, help='number of train epochs')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluate once in _ epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size of training')
    parser.add_argument('--max_len', type=int, default=150,
                        help='Max sentence length when RNN/CRF is used')
    parser.add_argument('--use_crf', type=bool, default=False,
                        help='use CRF on the outputs from the Neural Nets')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU number')
    # Model save and restore paths
    parser.add_argument('--save', type=str, default="model/", help="path to save model")
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
