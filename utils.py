import argparse
from torchvision.models import resnet


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', 't', '1')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset', type=str, required=True,
                        help='Path to dataset folder')
    parser.add_argument('-m','--model', type=str,
                        help='Model <r18> <r34> <r50> <r101> <r152>')
    parser.add_argument('-p', '--pretrained', type=str2bool, default=True,
                        help='Load model with pretrain or not')
    parser.add_argument('-w', '--weights', type=str,
                        help='Pretrained weight path')
    parser.add_argument('-t', '--trainable', type=str2bool, default=True,
                        help='Use gradient for parameters')
    parser.add_argument('-s', '--save_weights', type=str2bool, default=True,
                        help='Save best weights')
    parser.add_argument('-g', '--use_gpu', type=str2bool, default=True,
                        help='Use gpu')
    parser.add_argument('-n', '--num_epochs', type=int, default=25,
                        help='Number of epochs')
    return parser.parse_args()