import os
import torch
import logging
import argparse

from tqdm import tqdm
from datetime import datetime

from configs import CFG
from metric import Metric
from models import build_model
from datas import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device for test')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S-test')),
                        help='path for experiment output files')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'test.log')),
            logging.StreamHandler(),
        ])

    # build dataset
    test_dataset = build_dataset('test')
    NUM_CHANNELS = test_dataset.num_channels
    NUM_CLASSES = test_dataset.num_classes
    # build data loader
    test_dataloader = build_dataloader(test_dataset, 'test')
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(args.device)
    # build metric
    metric = Metric(NUM_CLASSES)

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model']['state_dict'])
    best_pa = checkpoint['metric']['pa']
    logging.info('load checkpoint {} with pa={:.3f}'.format(args.checkpoint, best_pa))

    # test
    model.eval()  # set model to evaluation mode
    metric.reset()  # reset metric
    test_bar = tqdm(test_dataloader, desc='testing', ascii=True)
    with torch.no_grad():  # disable gradient back-propagation
        for batch, (x, label) in enumerate(test_bar):
            x, label = x.to(args.device), label.to(args.device)
            y = model(x)

            if NUM_CLASSES > 2:
                pred = y.data.cpu().numpy().argmax(axis=1)
            else:
                pred = (y.data.cpu().numpy() > 0.5).squeeze(1)
            label = label.data.cpu().numpy()
            metric.add(pred, label)
    pa, mpa, ps, rs, f1s = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s()
    logging.info('test | PA={:.3f} mPA={:.3f}'.format(pa, mpa))
    for c in range(NUM_CLASSES):
        logging.info('test | class=#{} P={:.3f} R={:.3f} F1={:.3f}'.format(c, ps[c], rs[c], f1s[c]))


if __name__ == '__main__':
    main()