import os
# import os.path as osp
import torchio as tio
import numpy as np
# from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchio.transforms.augmentation.intensity.random_bias_field import RandomBiasField
from torchio.transforms.augmentation.intensity.random_noise import RandomNoise
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import argparse
# import csv
# import random
# import sys

from datasets.dataset import load_data
import models.models as models
import utils.my_trainer as trainer
import utils.train_result as train_result
from utils.data_class import BrainDataset
import utils.confusion as confusion


CLASS_MAP = {"CN": 0, "AD": 1}
SEED_VALUE = 0


def parser():

    parser = argparse.ArgumentParser(description="example")
    # CNN or CAE or VAE
    parser.add_argument("--model", type=str, default="CNN")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log", type=str, default="output")
    parser.add_argument("--n_train", type=float, default=0.8)
    parser.add_argument("--train_or_loadnet", type=str,
                        default="train")    # train or loadnet

    args = parser.parse_args()
    return args


# TorchIO
class ImageTransformio():
    def __init__(self):
        self.transform = {
            "train": tio.Compose([
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=10, isotropic=True,
                                 center="image", default_pad_value="mean", image_interpolation='linear'),
                tio.transforms.RandomNoise(),
                tio.transforms.RandomBiasField(),
                # tio.ZNormalization(),
                tio.transforms.RescaleIntensity((0, 1))
            ]),
            "val": tio.Compose([
                # tio.ZNormalization(),
                # tio.RescaleIntensity((0, 1))  # , in_min_max=(0.1, 255)),
            ])
        }

    def __call__(self, img, phase="train"):
        img_t = torch.tensor(img)
        return self.transform[phase](img_t)


def load_dataloader(n_train_rate, batch_size):

    data = load_data(kinds=["ADNI2-2"], classes=["CN", "AD"], unique=True, blacklist=True)
    #data = dataset.load_data(kinds=kinds,classes=classes,unique=False)
    pids = []
    for i in range(len(data)):
        pids.append(data[i]["pid"])
    gss = GroupShuffleSplit(test_size=1-n_train_rate, random_state=SEED_VALUE)
    train_idx, val_idx = list(gss.split(data, groups=pids))[0]
    train_data = data[train_idx]
    val_data = data[val_idx]

    #train_datadict, val_datadict = train_test_split(dataset, test_size=1-n_train_rate, shuffle=True, random_state=SEED_VALUE)
    transform = ImageTransformio()
    # transform = None
    train_dataset = BrainDataset(
        data_dict=train_data, transform=transform, phase="train")
    val_dataset = BrainDataset(
        data_dict=val_data, transform=transform, phase="val")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def main():
    # randam.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    args = parser()

    if args.model == "CNN":
        net = models.FujiNet1()
        log_path = "./logs/" + args.log + "_cnn-IIP1-drop+DA3/"
        print("net: CNN")
    elif args.model == "CAE":
        net = models.Cae()
        #net = models.CAE3()
        log_path = "./logs/" + args.log + "_cae/"
        print("net: CAE")
    elif args.model == "VAE":
        net = models.Vae()
        log_path = "./logs/" + args.log + "_vae/"
        print("net: VAE")

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "csv/", exist_ok=True)
    # save args
    with open(log_path + "my_args.txt", "w") as f:
        f.write("{}".format(args))

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and True else "cpu")
    print("device:", device)

    # torch.nn.init.kaiming_normal_(net.encoder.conv1.weight)
    # torch.nn.init.kaiming_normal_(net.encoder.conv2.weight)
    # torch.nn.init.kaiming_normal_(net.encoder.conv3.weight)
    # torch.nn.init.kaiming_normal_(net.encoder.conv4.weight)

    # torch.nn.init.kaiming_normal_(net.decoder.deconv1.weight)
    # torch.nn.init.kaiming_normal_(net.decoder.deconv2.weight)
    # torch.nn.init.kaiming_normal_(net.decoder.deconv3.weight)
    # torch.nn.init.kaiming_normal_(net.decoder.deconv4.weight)

    train_loader, val_loader = load_dataloader(args.n_train, args.batch_size)
    # loadnet or train
    if args.train_or_loadnet == "loadnet":
        net.load_state_dict(torch.load(log_path+'weight.pth'))
        # とりあえずvalidationで確認 テストデータあとで作る
        confusion.make_confusion_matrix(
            net, val_loader, CLASS_MAP, device, log_path)

    elif args.train_or_loadnet == "train":
        # CNN or CAE or VAE
        if args.model == "CNN":
            train_loss, train_acc, val_loss, val_acc = trainer.train(
                net, train_loader, val_loader, args.epoch, args.lr, device, log_path
            )
            # torch.save(net.state_dict(), log_path + "weight.pth")
            train_result.result(train_acc, train_loss,
                                val_acc, val_loss, log_path)
            # とりあえずvalidationで確認 テストデータあとで作る
            confusion.make_confusion_matrix(
                net, val_loader, CLASS_MAP, device, log_path)

        elif args.model == "CAE":
            train_loss, val_loss = trainer.train_cae(
                net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "cae_weight.pth")
            print("saved net weight!")
            train_result.result_cae(train_loss, val_loss, log_path)

            # val_loader_iter = iter(val_loader)
            # image, _ = next(val_loader_iter)
            # image = image.to(device)
            # net.to(device)
            # net.eval()
            # with torch.no_grad():
            #     output = net(image)
            # output = output.cpu()
            # print(output[0].size())
            # pil_img = Image.fromarray(np.flip(output[0].numpy().reshape(
            #     80, 80, 80).transpose(2, 0, 1)[50], 0) * 255)
            # pil_img = pil_img.convert("L")
            # pil_img.save(log_path+"img/cae_output_img.jpg")
            #plt.imshow(np.flip(output[0].numpy().reshape(80, 80, 80).transpose(2,0,1)[50],0), cmap="gray")

        elif args.model == "VAE":
            train_loss, val_loss = trainer.train_vae(
                net, train_loader, val_loader, args.epoch, args.lr, device, log_path)
            torch.save(net.state_dict(), log_path + "vae_weight.pth")
            print("saved net weight!")
            train_result.result_cae(train_loss, val_loss, log_path)

            # val_loader_iter = iter(val_loader)
            # image, _ = next(val_loader_iter)
            # image = image.to(device)
            # net.to(device)
            # net.eval()
            # with torch.no_grad():
            #     output, _, _ = net.forward(image)
            # output = output.cpu()
            # print(output[0].size())
            # pil_img = Image.fromarray(np.flip(output[0].numpy().reshape(
            #     80, 80, 80).transpose(1, 2, 0)[50], 0) * 255)
            # pil_img = pil_img.convert("L")
            # pil_img.save(log_path+"img/vae_output_img.jpg")


if __name__ == "__main__":
    main()
