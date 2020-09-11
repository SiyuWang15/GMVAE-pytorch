import logging
import argparse
import os
import time
from runner.GMVAE_runner import GMVAE_runner


def arg_parser():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--run', type=str, default = 'gmvae', help='The runner to execute')
    parser.add_argument('--dataset', type = str, default='mnist')
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--w_dim', type = int, default=32)
    parser.add_argument('--h_dim', type = int, default=32)
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--verbose', type = str, default = 'info')
    parser.add_argument('--gpu_list', type = str, default = '0,1,2,3')
    parser.add_argument('--batch_size', type = int, default=128)
    parser.add_argument('--n_epochs', type = int, default=10)
    parser.add_argument('--test_freq', type = int, default=100)
    parser.add_argument('--draw_freq', type = int, default = 100)
    parser.add_argument('--save_freq', type = int, default = 1000)
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        args.v_dim = 784
        args.n_classes = 10
    args.run = os.path.join('./run', args.run)
    args.datapath = './datasets/'
    args.log = os.path.join(args.run, time.strftime('%H-%M-%S', time.localtime()))
    args.img_dir = os.path.join(args.log, 'image')
    args.ckpt_dir = args.log
    return args
    

def main():
    args = arg_parser()
    os.makedirs(args.img_dir)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'training.log'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
    logging.info('Using GPU {}'.format(args.gpu_list))
    logging.info('Loging in {}'.format(args.log))

    runner = GMVAE_runner(args)
    runner.train()

    

if __name__ == '__main__':
    main()