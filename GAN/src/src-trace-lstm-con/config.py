import argparse
from argparse import ArgumentParser


def parsers():

    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('--Model_path', default="./Model7")
    ARG_PARSER.add_argument('--data_path', default="./trace-by-gan7/data")
    ARG_PARSER.add_argument('--pics_path', default="./trace-by-gan7/pics")
    ARG_PARSER.add_argument('--stats_path', default="./Statistics7")

    ARG_PARSER.add_argument('--GPU', default=True, type=bool)

    ARG_PARSER.add_argument('--load_g', action='store_true')
    ARG_PARSER.add_argument('--load_d', action='store_true')
    ARG_PARSER.add_argument('--no_save_g', action='store_true')
    ARG_PARSER.add_argument('--no_save_d', action='store_true')
    ARG_PARSER.add_argument('--freeze_g', action='store_true')
    ARG_PARSER.add_argument('--freeze_d', action='store_true')
    ARG_PARSER.add_argument('--freezing', default=False, type=bool)

    ARG_PARSER.add_argument('--num_epochs', default=10000, type=int)
    ARG_PARSER.add_argument('--sample_interval', default=12, type=int)

    # Trajectory Information
    ARG_PARSER.add_argument('--seq_len', default=100, type=int)  # =time_length/trace_sample_interval
    ARG_PARSER.add_argument('--num_feats', default=2, type=int)
    ARG_PARSER.add_argument('--time_length', default=4000, type=int)  # 12 frames in 1s.
    ARG_PARSER.add_argument('--trace_sample_interval', default=40, type=int)  # sample a point every 10 frames

    ARG_PARSER.add_argument('--batch_size', default=32, type=int)
    ARG_PARSER.add_argument('--feature_matching', action='store_true')
    ARG_PARSER.add_argument('--label_smoothing', default=False, type=bool)
    ARG_PARSER.add_argument('--Bidirct_D', default=True, type=bool)
    ARG_PARSER.add_argument('--g_lr', default=0.0001, type=float)
    ARG_PARSER.add_argument('--d_lr', default=0.0002, type=float)
    ARG_PARSER.add_argument('--latent_dim', default=100, type=int)
    ARG_PARSER.add_argument('-m', action='store_true')
    ARG_PARSER.add_argument('--no_pretraining', default=True, type=bool)
    ARG_PARSER.add_argument('--pretraining_epochs', default=3, type=int)

    ARGS = ARG_PARSER.parse_args()

    return ARGS