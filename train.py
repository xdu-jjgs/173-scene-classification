import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from apex import amp
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from apex.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from configs import CFG
from metric import Metric
from models import build_model
from criterions import build_criterion
from optimizers import build_optimizer
from schedulers import build_scheduler
from datas import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S-train')),
                        help='path for experiment output files')
    parser.add_argument('--no-validate',
                        action='store_true',
                        help='whether not to validate in the training process')

    parser.add_argument('-n',
                        '--nodes',
                        type=int,
                        default=1,
                        help='number of nodes / machines')
    parser.add_argument('-g',
                        '--gpus',
                        type=int,
                        default=1,
                        help='number of GPUs per node / machine')
    parser.add_argument('-r',
                        '--rank-node',
                        type=int,
                        default=0,
                        help='ranking of the current node / machine')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='backend for PyTorch DDP')
    parser.add_argument('--master-ip',
                        type=str,
                        default='localhost',
                        help='network IP of the master node / machine')
    parser.add_argument('--master-port',
                        type=str,
                        default='8888',
                        help='network port of the master process on the master node / machine')
    parser.add_argument('--seed',
                        type=int,
                        default=30,
                        help='random seed')
    parser.add_argument('--opt-level',
                        type=str,
                        default='O0',
                        help='optimization level for nvidia/apex')
    args = parser.parse_args()
    # number of GPUs totally, which equals to the number of processes
    args.world_size = args.nodes * args.gpus
    return args


def worker(rank_gpu, args):
    # create experiment output path if not exists
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())
    # print(CFG)
    # log to file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'train.log')),
            logging.StreamHandler(),
        ])

    # rank of global worker
    rank_process = args.gpus * args.rank_node + rank_gpu
    dist.init_process_group(backend=args.backend,
                            init_method=f'tcp://{args.master_ip}:{args.master_port}',
                            world_size=args.world_size,
                            rank=rank_process)
    # number of workers
    logging.info('train on {} of {} processes'.format(rank_process + 1, dist.get_world_size()))

    # use device cuda:n in the process #n
    torch.cuda.set_device(rank_gpu)
    device = torch.device('cuda', rank_gpu)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # initialize TensorBoard summary writer
    # rank of global worker
    if dist.get_rank() == 0:
        writer = SummaryWriter(logdir=args.path)

    # build dataset
    train_dataset = build_dataset('train')
    val_dataset = build_dataset('val')
    assert train_dataset.num_classes == val_dataset.num_classes
    NUM_CHANNELS = train_dataset.num_channels
    NUM_CLASSES = train_dataset.num_classes
    # build data sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # build data loader
    train_dataloader = build_dataloader(train_dataset, 'train', sampler=train_sampler)
    val_dataloader = build_dataloader(val_dataset, 'val', sampler=None)
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    # print(model)
    # build criterion
    criterion = build_criterion()
    criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer = build_optimizer(model)
    # build scheduler
    scheduler = build_scheduler(optimizer)

    # mixed precision
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    # DDP
    model = DistributedDataParallel(model)

    epoch = 0
    iteration = 0
    best_epoch = 0
    best_pa = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model']['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])
        epoch = checkpoint['optimizer']['epoch']
        iteration = checkpoint['optimizer']['iteration']
        best_pa = checkpoint['metric']['pa']
        best_epoch = checkpoint['optimizer']['best_epoch']
        logging.info('load checkpoint {} with mIoU={:.4f}, epoch={}'.format(args.checkpoint, best_pa, epoch))

    # train - validation loop

    while True:
        epoch += 1
        if epoch > CFG.EPOCHS:
            if dist.get_rank() == 0:
                writer.close()
            return

        train_dataloader.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train
        model.train()  # set model to training mode
        metric.reset()  # reset metric
        train_bar = tqdm(train_dataloader, desc='training', ascii=True)
        train_loss = 0.
        for x, label in train_bar:
            iteration += 1

            x, label = x.to(device), label.to(device)
            y = model(x)
            # print("Y shape: {}, label shape:;{}".format(y.shape, label.shape))

            loss = criterion(y, label)
            train_loss += loss.item()
            if dist.get_rank() == 0:
                writer.add_scalar('train/loss-iteration', loss.item(), iteration)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            pred = y.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
            # metric.add(pred, label)
            # TypeError: can't convert cuda:0 device type tensor to numpy.
            # Use Tensor.cpu() to copy the tensor to host memory first.

            train_bar.set_postfix({
                'epoch': epoch,
                'loss': f'{loss.item():.4f}',
                'P': ','.join([f'{p:.4f}' for p in metric.Ps()]),
                'R': ','.join([f'{r:.4f}' for r in metric.Rs()]),
                'F1s': ','.join([f'{f:.4f}' for f in metric.F1s()]),
                'mP': f'{metric.mPA():.4f}',
                'pa': f'{metric.PA():.4f}'
            })

        train_loss /= len(train_dataloader)
        pa, mpa, ps, rs, f1s = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s()
        if dist.get_rank() == 0:
            writer.add_scalar('train/loss-epoch', train_loss, epoch)
            writer.add_scalar('train/PA-epoch', pa, epoch)
            writer.add_scalar('train/mPA-epoch', mpa, epoch)
        logging.info(
            'train epoch={} | loss={:.3f} PA={:.3f} mPA={:.3f}'.format(epoch, train_loss, pa, mpa))
        for c in range(NUM_CLASSES):
            logging.info(
                'train epoch={} | class=#{} P={:.3f} R={:.3f} F1={:.3f}'.format(epoch, c, ps[c], rs[c], f1s[c]))

        # validate
        if args.no_validate:
            continue
        model.eval()  # set model to evaluation mode
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x, label in val_bar:
                x, label = x.to(device), label.to(device)
                y = model(x)

                loss = criterion(y, label)
                val_loss += loss.item()

                pred = y.argmax(axis=1)
                metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

                val_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{loss.item():.4f}',
                    'P': ','.join([f'{p:.4f}' for p in metric.Ps()]),
                    'R': ','.join([f'{r:.4f}' for r in metric.Rs()]),
                    'F1s': ','.join([f'{f:.4f}' for f in metric.F1s()]),
                    'mP': f'{metric.mPA():.4f}',
                    'pa': f'{metric.PA():.4f}'
                })

        val_loss /= len(val_dataloader)

        pa, mpa, ps, rs, f1s = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s()
        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)
            writer.add_scalar('val/PA-epoch', pa, epoch)
            writer.add_scalar('val/mPA-epoch', mpa, epoch)
        if pa > best_pa:
            best_epoch = epoch

        logging.info('val epoch={} | loss={:.3f} PA={:.3f} mPA={:.3f}'.format(epoch, train_loss, pa, mpa))
        for c in range(NUM_CLASSES):
            logging.info('val epoch={} |  class=#{} P={:.3f} R={:.3f} F1={:.3f}'.format(epoch, c, ps[c], rs[c], f1s[c]))

        # adjust learning rate if specified
        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'model': {
                    'state_dict': model.state_dict(),
                },
                'optimizer': {
                    'state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'metric': {
                    'PA': pa,
                    'mPA': mpa,
                    'Ps': ps,
                    'Rs': rs,
                    'F1s': f1s
                },
            }
            torch.save(checkpoint, os.path.join(args.path, 'last.pth'))
            if pa > best_pa:
                best_pa = pa
                torch.save(checkpoint, os.path.join(args.path, 'best.pth'))

    logging.info("Best epoch:{}".format(best_epoch))


def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()