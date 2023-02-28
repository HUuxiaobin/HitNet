import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import Hitnet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
####
####CUDA_VISIBLE_DEVICES=0 python3 Train.py
####
def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def val(model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        # test_loader = test_dataset(image_root=opt.test_path + 'Imgs/',
        #                           gt_root=opt.test_path + 'GT/',
        #                           testsize=opt.trainsize)
        test_loader = test_dataset(image_root=opt.test_path + '/COD10K/Imgs/',
                            gt_root=opt.test_path + '/COD10K/GT/',
                            testsize=opt.trainsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, res1 = model(image)
            # eval Dice
            res = F.upsample(res[-1] + res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [1]
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # print('this is i',i)
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            # print('this is trainsize',trainsize)
            P1, P2= model(images)
            # ---- loss function ----
            losses = [structure_loss(out, gts) for out in P1]
            loss_p1=0
            gamma=0.2
            # print('iteration num',len(P1))
            for it in range(len(P1)):
                loss_p1 += (gamma * it) * losses[it]


            loss_P2 = structure_loss(P2, gts)

            loss = loss_p1 + loss_P2
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model
    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), save_path + str(epoch) + 'Hitnet-PVT.pth')
    
if __name__ == '__main__':

    ##################model_name#############################
    # model_name = 'Hitnet_pvt_wo_pretrained_fusion'
    model_name = 'Hitnet_pvt_wo_pretrained_fusion_debug'

    ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,default=150, help='epoch number')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,default=704, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--decay_rate', type=float,default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,default='/youtu_action_data/xiaobinhu/dataset_hitnet_cod/TrainDataset',help='path to train dataset')
    parser.add_argument('--test_path', type=str,default='/youtu_action_data/xiaobinhu/dataset_hitnet_cod/TestDataset',help='path to testing dataset')
    parser.add_argument('--save_path', type=str,default='/youtu_action_data/xiaobinhu/dataset_hitnet_cod/checkpoints/'+model_name+'/')
    parser.add_argument('--epoch_save', type=int,default=1, help='every n epochs to save model')
    opt = parser.parse_args()


    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    logging.basicConfig(filename=opt.save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")


    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = Hitnet().cuda()


    if opt.load is not None:
        pretrained_dict=torch.load(opt.load)
        print('!!!!!!Sucefully load model from!!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('model paramters',sum(p.numel() for p in model.parameters() if p.requires_grad))

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    writer = SummaryWriter(opt.save_path + 'summary')

    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1
    best_epoch = 0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.save_path)
        if epoch % opt.epoch_save==0:
            val( model, epoch, opt.save_path, writer)
