import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import scipy.io as sio
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score

from network import discriminator

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# ==================auxiliary function==================
def data_regular(data):
    data = np.asarray(data).transpose(2, 0, 1)
    data -= np.min(data)
    data /= np.max(data)
    return data


def one_hot(lable, class_number):
    '''转变标签形式'''
    one_hot_array = np.zeros([len(lable), class_number])
    for i in range(len(lable)):
        one_hot_array[i, lable[i]-1] = 1
    # one_hot_array = one_hot_array.astype(np.int32)
    one_hot_array = one_hot_array.astype(np.float32)
    return one_hot_array


def get_criteria(y_pred, y_real):
    y_pred = torch.argmax(y_pred, dim=1)
    y_real = torch.argmax(y_real, dim=1)
    y_pred = y_pred.cpu().numpy()
    y_real = y_real.cpu().numpy()
    oa = accuracy_score(y_real, y_pred)
    per_class_acc = recall_score(y_real, y_pred,
                                 labels=[0, 1, 2, 3, 4, 5],
                                 # labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 此处根据num_class改
                                 average=None)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_real, y_pred)
    return oa, aa, kappa, per_class_acc
# ========================================================


class TrainDataset(Dataset):
    def __init__(self, mode):
        # mode == 'train' or 'test'
        super(TrainDataset, self).__init__()
        self.mode = mode
        self.sar, self.msi, self.label, self.class_num = self.load_data()
        self.label = one_hot(self.label, self.class_num)

    def load_data(self):
        data = sio.loadmat('data/data_'+self.mode+'.mat')
        sar = data['sar']  # N*32*32*8
        msi = data['msi']  # N*32*32*10
        label = data['label']  # N
        label = np.asarray(label).transpose(1, 0)
        class_num = data['class_num']
        # print(sar.shape, msi.shape, label.shape, type(class_num),class_num,int(class_num))
        # (32768, 32, 32, 8) (32768, 32, 32, 10) (32768, 1) <class 'numpy.ndarray'> [[17]] 17
        return sar, msi, label, int(class_num)

    def __getitem__(self, index):
        # Href = self.HSIref[:, hindex*160:(hindex+1)*160, windex*160:(windex+1)*160]
        # Hup = self.HSIup[:, hindex*160:(hindex+1)*160, windex*160:(windex+1)*160]
        sar = data_regular(self.sar[index, :, :, :].astype(np.float32))
        msi = data_regular(self.msi[index, :, :, :].astype(np.float32))
        label = self.label[index]
        return torch.tensor(sar), torch.tensor(msi), torch.tensor(label)

    def __len__(self):
        return len(self.label)


def train(args, dataset):
    print('===> Loading datasets')
    # dataset = TrainDataset(args.Hup_path, args.train_res)
    # print(dataset[0])   # (Hup, HM_Pan)
    training_data_loader = DataLoader(dataset=dataset, batch_size=args.batchSize,
                                      shuffle=True)

    print('===> Initializing the model')
    model = discriminator(args.num_class)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
        # print(args.gpu_id)
        criterion = criterion.cuda(args.gpu_id[0])
        model = model.cuda(args.gpu_id[0])
    if args.pretrained:
        model_path = os.path.join(args.save_folder + args.checkpoint)
        if os.path.exists(model_path):
            # model= torch.load(model_name, map_location=lambda storage, loc: storage)
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            print('Pre-trained model is loaded.')
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.8)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    print('===> Start training')
    torch.autograd.set_detect_anomaly(True)
    loss_min = args.loss_threshold
    epoch_best = 0
    for epoch in range(args.start_iter, args.Epochs):
        epoch_loss = 0
        model.train()
        for iteration, batch in enumerate(training_data_loader):
            sar, msi, label = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            if args.cuda:
                sar = sar.cuda(args.gpu_id[0])
                msi = msi.cuda(args.gpu_id[0])
                label = label.cuda(args.gpu_id[0])

            t0 = time.time()
            optimizer.zero_grad()
            # t0 = time.time()
            prediction = model(sar, msi)
            # print(prediction, prediction.shape, label, label.shape)
            loss = criterion(prediction, label)
            # t1 = time.time()
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()
            t1 = time.time()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
        # learning rate is decayed by a factor of 10
        if epoch+1 == 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        if (epoch + 1) % args.snapshots == 0:
        # if (epoch_loss / len(training_data_loader)) < loss_min:
        #     loss_min = (epoch_loss / len(training_data_loader))
        #     epoch_best = epoch
            model_out_path = args.save_folder + 'RCNN' + "_epoch_{}.pth".format(epoch)
            torch.save(model.state_dict(), model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))
    # args.checkpoint = 'RCNN' + "_epoch_{}.pth".format(epoch_best)
    return None


def val(args, dataset):
    print('===> Loading datasets')
    # dataset = TrainDataset(args.Hup_path, args.train_res)
    args.batchSize = len(dataset)
    eval_data_loader = DataLoader(dataset=dataset, batch_size=args.batchSize, shuffle=True)

    print('===> Initializing the model')
    model = discriminator(args.num_class)
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
        model = model.cuda(args.gpu_id[0])

    args.pretrained = True
    args.checkpoint = 'RCNN_epoch_99.pth'
    if args.pretrained:
        model_path = os.path.join(args.save_folder + args.checkpoint)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            print('Pre-trained model is loaded:', args.checkpoint)

    model.eval()
    for iteration, batch in enumerate(eval_data_loader):
        with torch.no_grad():
            sar, msi, label = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if args.cuda:
            sar = sar.cuda(args.gpu_id[0])
            msi = msi.cuda(args.gpu_id[0])
            # label = label.cuda(args.gpu_id[0])
        t0 = time.time()
        output = model(sar, msi)
        t1 = time.time()
        # output = output.cpu().data     # tensor (b, c)
        oa_test, aa_test, kappa_test, per_class_acc_test = get_criteria(output, label)
        # avg_metric.append(metric)
        print("===> Processing: patch %s || Timer: %.4f sec. || time per img: %.4f sec."
              % (str(iteration), (t1 - t0), (t1-t0)/args.batchSize))
        print('Finished Testing')
        print("oa:", oa_test)
        print("aa:", aa_test)
        print("kappa:", kappa_test)
        print("per_class_acc:", per_class_acc_test)
    return None


def main():
    parser = argparse.ArgumentParser(description='Training for SAR-MSI classification')
    # parser.add_argument('--Href_path', type=str, default='PSinput/Pavia.mat', help='gt for eval')
    # parser.add_argument('--Hup_path', type=str, default='PSinput/PAVIA_UP.mat', help='the upsampled HSI for training')

    parser.add_argument('--cuda', type=bool, default=True, help='use gpu for training or not')
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--seed', type=int, default=5, help='random seed to use. Default=123')
    parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
    parser.add_argument('--Epochs', type=int, default=100, help='number of epochs to train for')
    # TODO change the num_class according to the dataset
    parser.add_argument('--num_class', type=int, default=6, help='class num in dataset')
    parser.add_argument('--loss_threshold', type=int, default=0.02, help='class num in dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.001')
    parser.add_argument('--batchSize', type=int, default=1024, help='training batch size')  # 32768/1024=32iter

    parser.add_argument('--snapshots', type=int, default=10, help='save the checkpoint every snapshot')
    parser.add_argument('--checkpoint', type=str, default='RCNN_epoch_99.pth', help='sr pretrained base model')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--save_folder', type=str, default='checkpoint/exp1/', help='Location to save checkpoint models')
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.cuda:
        try:
            args.gpu_id = [int(s) for s in args.gpu_id.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    print(args)
    # train_dataset = TrainDataset('train')
    test_dataset = TrainDataset('test')
    # print('train size:', len(train_dataset), 'test_size', len(test_dataset), 'num_class', args.num_class)
    # print('=======================STARTING TRAINING=======================')
    # train(args, train_dataset)
    print('=======================STARTING TESTING=======================')
    val(args, test_dataset)

    return None


if __name__ == '__main__':
    main()


