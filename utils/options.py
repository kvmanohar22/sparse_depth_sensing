import argparse
import os

class options(object):
    """ Holds the different hyper-parameters """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.user = os.environ.get('USER')

    def initialize(self):
        # Training
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--project_root', type=str, default='', help='Root of the project')
        self.parser.add_argument('--data_root', type=str, default='',help='Path to the dataset')
        self.parser.add_argument('--train', action='store_true', help='Train / test')
        self.parser.add_argument('--max_iters', type=int, default=1000, help='Number of iters to train')
        self.parser.add_argument('--img_h', type=int, default=224, help='Image hieght')
        self.parser.add_argument('--img_w', type=int, default=224, help='Image width')
        self.parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id')
        self.parser.add_argument('--L', type=int, default=5, help='No. of layers in densenet block')
        self.parser.add_argument('--k', type=int, default=12, help='No. of features maps in each block')

        # Debugging
        self.parser.add_argument('--display_frq', type=int, default=200, help='Display log after')
        self.parser.add_argument('--base_lr', type=float, default=3e-4, help='Initial learning rate')

        # Mode
        self.parser.add_argument('--demo', action='store_true', help='Run a demo?')
        self.parser.add_argument('--evaluate', action='store_true', help='Evaluate a model?')
        self.parser.add_argument('--benchmark', action='store_true', help='Benchmark a model?')


    def parse(self, train_mode=False):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()
        args.train = train_mode

        self.opts = vars(args)
        return self.opts
