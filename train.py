import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.dataset import SegDataset
from utils.utils import get_yaml
from utils.metric import pixel_eval
from model.model import PPLiteSeg


def train(cfg):
    lr = cfg['lr']
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    data_path = cfg['data_path']
    class_num = cfg['class_num']
    save_path = cfg['checkpoints_path']
    size = cfg['input_size']
    mean = cfg['mean']
    std = cfg['std']

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net = PPLiteSeg(n_classes=class_num)
    net.to(device='cuda')

    loss_func1 = nn.CrossEntropyLoss()
    loss_func2 = nn.CrossEntropyLoss()
    loss_func3 = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params=net.parameters(), lr=lr)

    train_dataset = SegDataset(data_path, size, mean, std, 'train')
    test_dataset = SegDataset(data_path, size, mean, std, 'test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)

    for i in range(1, epochs + 1):
        net.train()
        total_loss = []

        prefix = 'Epoch {}/{}'.format(i, epochs)
        iter = tqdm(train_loader, desc=prefix, leave=True)

        # train
        for img, target in iter:
            opt.zero_grad()

            img = img.cuda()
            with torch.no_grad():
                target = target.cuda()

            outputs = net(img)

            loss1 = loss_func1(outputs[0], target)
            loss2 = loss_func2(outputs[1], target)
            loss3 = loss_func3(outputs[2], target)
            loss = loss1 + loss2 + loss3

            loss.backward()
            opt.step()

            total_loss.append(loss.item())
            scheduler.step()

            postfix = 'loss: {:.4f}'.format(np.mean(total_loss))
            iter.set_postfix(loss=postfix, refresh=True)

        # val
        net.eval()
        eval_miou = []
        eval_acc = []
        for img, target in val_loader:
            with torch.no_grad():
                img = img.cuda()
                target = target.cuda()

                output = net(img)

            output[output < 0.5] = 0
            mask = torch.argmax(output, dim=1)

            acc, miou = miou = pixel_eval(mask, target, class_num)
            eval_acc.append(acc)
            eval_miou.append(miou[0])

        print("acc: {:.2f}, miou: {:.2f}".format(np.mean(acc), np.mean(eval_miou)))

        if i % 5 == 0:
            torch.save(net.state_dict(), "{}/net_{}.pth".format(save_path, i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str)
    args = parser.parse_args()

    if os.path.exists(args.config):
        cfg = get_yaml(args.config)
        train(cfg)
