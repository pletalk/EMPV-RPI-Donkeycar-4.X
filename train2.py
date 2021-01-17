#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help              Show this screen.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train
import matplotlib.pyplot as plt


# model의 학습결과를 저장할 파일이름 경로를 설정
MODEL_PLOT_FILE = '/content/mycar/models.myplot.png'

def train_plot(hist):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    #acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    #acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    #acc_ax.set_ylabel('accuracy')
    #acc_ax.legend(loc='upper left')

    plt.savefig(MODEL_PLOT_FILE)


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    history = train(cfg, tubs, model, model_type)
    
    train_plot(history)
    

if __name__ == "__main__":
    main()
