import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MulSetRank.")

    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')

    parser.add_argument('--dataset', nargs='?', default='null',
                        help='Choose a dataset.')
    
    parser.add_argument('--train_size', type=str, default='70',
                        help='the ratio of training set to the whole dataset.')
    
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')

    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--k', type=int, default=5,
                        help='Size of item set.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50]',
                        help='Output sizes of every layer')
    
    parser.add_argument('--beta', type=float, default=0.025,
                        help='wus beta test')

    parser.add_argument('--group_size', type=int, default=3,
                        help='sampled user group size')
    
    parser.add_argument('--l1', type=bool, default=False,
                        help='use l1 normalization or not.')
    
    return parser.parse_args()
