import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--out_dir', type=str, default='logs/sgan')
parser.add_argument('--data_dir', type=str, default='data/cifar-10-python')
parser.add_argument('--save_interval', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--advloss_weight', type=float, default=1.) # weight of adversarial loss
parser.add_argument('--condloss_weight', type=float, default=1.) # weight of conditional loss
parser.add_argument('--entropyloss_weight', type=float, default=10.) # weight of entropy loss
parser.add_argument('--labloss_weight', type=float, default=1.)
parser.add_argument('--g_lr', type=float, default=0.0001)
parser.add_argument('--d_lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--optimizer', type=str, default='adam')

config = parser.parse_args()
