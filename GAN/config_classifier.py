import argparse
from argparse import ArgumentParser


def parsers():

    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('--Model_path', default="./classifier_Model_1")
    ARG_PARSER.add_argument('--stats_path', default="./classifier_Stat_1")
    ARG_PARSER.add_argument('--num_feats', default=2, type=int)
    ARG_PARSER.add_argument('--GPU', default=True, type=bool)
    ARG_PARSER.add_argument('--num_epochs', default=10000, type=int)
    ARG_PARSER.add_argument('--batch_size', default=128, type=int)
    ARG_PARSER.add_argument('--epoch', default=2000000, type=int)
    ARG_PARSER.add_argument('--lr', default=5e-3, type=float)

    ARGS = ARG_PARSER.parse_args()

    return ARGS