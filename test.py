import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.model import PPLiteSeg
from utils.utils import prepreocess


if __name__ == '__main__':
    net = PPLiteSeg(n_classes=20)
    net.load_state_dict(torch.load('checkpoints/net_10.pht'))
    net.eval()
    net.to(device='cuda')

    img_path = '/mnt/e/dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_007365_leftImg8bit.png'
    img = cv2.imread(img_path)
    p_img = prepreocess(img, (256, 512))
    p_img = np.expand_dims(p_img, axis=0)
    p_img = torch.from_numpy(p_img).to('cuda').float()

    pred = net(p_img)
    pred[pred < 0.5] = 0
    mask = torch.argmax(pred, dim=1).cpu().numpy()[0]

    plt.subplot(121)
    show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(show_img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
