import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.pvt import Hitnet
from utils.dataloader import My_test_dataset
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='./model_pth/HitnetPVT_origin/Net_epoch_best.pth')
opt = parser.parse_args()

pth_root='./model_pth/Net_epoch_best.pth/'
pth_list=[pth_root + f for f in os.listdir(pth_root) if '.pth' in f ]
pth_list.sort()
print('pth_list',pth_list)

logging.basicConfig(filename='./model_pth/HitnetPVT_origin/results_log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Network-TEST")


# for _data_name in ['CAMO', 'COD10K', 'CHAMELEON']:
for _data_name in ['COD10K']:
    data_path = './Dataset/TestDataset/{}/'.format(_data_name)

    model = Hitnet()
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root',image_root,gt_root)

    mmae=[]
    for pth_file in pth_list:
        print('current test:',pth_file)
        logging.info('current test:{}'.format(pth_file))
        model.load_state_dict(torch.load(pth_file))
        model.cuda()
        model.eval()
        mae_sum = 0

        test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
        # print('****', test_loader.size)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            # print('***name',name)
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            P1, P2 = model(image)
            res = F.upsample(P1[-1] + P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # print('> {} - {}'.format(_data_name, name))
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        mmae.append(mae)
        print('mas loss with test', mae)
        # logging.info('mas loss with test', mae)
        logging.info('mas loss with test:{}'.format(mae))

    print(mmae)
    # index=np.where(np.min(mmae))[0][0]
    index = mmae.index(min(mmae))
    print('best mae and index', min(mmae), index)
    logging.info('best_mae {} and path index {}'.format(min(mmae), pth_list[index]))
    print('best pth fold', pth_list[index])
    # logging.info('best pth fold', pth_list[index])
    # np.where(np.min(model_mae))
    np.save('model_mae.npy', mmae)
    np.save('model_name.npy', pth_list)

        # misc.imsave(save_path+name, res)
            # If `mics` not works in your environment, please comment it and then use CV2
            # cv2.imwrite(save_path+name,res*255)
