#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import numpy as np
from utils.dataset_control import DATASET_CONTROLLER
if DATASET_CONTROLLER == 'tacred':
    NO_RELATION = "no_relation"
elif DATASET_CONTROLLER == 'semeval':
    NO_RELATION = "Other"
else:
    raise Exception('dataset error')


# NO_RELATION = "no_relation"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def score(key, prediction, verbose=False, micro=True, print_all=True):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if micro:
        if verbose:
            print("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        if print_all:
            print( "Precision (micro): {:.3%}".format(prec_micro) )
            print( "   Recall (micro): {:.3%}".format(recall_micro) )
            print( "       F1 (micro): {:.3%}".format(f1_micro) )
        finall_prec, finall_recall, finall_f1 = prec_micro, recall_micro, f1_micro

    else:
        relations = guessed_by_relation.keys()
        prec_macro = 0
        recall_macro = 0
        f1_macro = 0
        for rel in relations:
            if rel in correct_by_relation.keys():
                prec = float(correct_by_relation[rel]) / float(guessed_by_relation[rel])
                recall = float(correct_by_relation[rel]) / float(gold_by_relation[rel])
                f1 = 2.0 * prec * recall / (prec + recall)
                prec_macro += prec
                recall_macro += recall
                f1_macro += f1
        if len(relations) == 0 and len(prediction) > 0:
            prec_macro = recall_macro = f1_macro = 0
        else:
            prec_macro /= len(relations)
            recall_macro /= len(relations)
            f1_macro /= len(relations)
        if print_all:
            print("Precision (macro): {:.3%}".format(prec_macro))
            print("   Recall (macro): {:.3%}".format(recall_macro))
            print("       F1 (macro): {:.3%}".format(f1_macro))
        finall_prec, finall_recall, finall_f1 = prec_macro, recall_macro, f1_macro

    return finall_prec, finall_recall, finall_f1


def score_aggcn(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")

    prec = []
    for k in guessed_by_relation.keys():
        prec.append(float(correct_by_relation[k]) / (float(guessed_by_relation[k]) + 0.0001))
    recall = []
    for k in gold_by_relation.keys():
        recall.append(float(correct_by_relation[k]) / (float(gold_by_relation[k]) + 0.0001))
    prec_macro = np.mean(prec)
    recall_macro = np.mean(recall)
    f1_macro = 0.0
    if prec_macro + recall_macro > 0.0:
        f1_macro = 2.0 * prec_macro * recall_macro / (prec_macro + recall_macro)
    print("Precision (macro): {:.3%}".format(prec_macro))
    print("   Recall (macro): {:.3%}".format(recall_macro))
    print("       F1 (macro): {:.3%}".format(f1_macro))
    return prec_macro, recall_macro, f1_macro


if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

