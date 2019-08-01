import os
from os.path import join
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import DataLoaderHelper, AdiosDataLoader

from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import G, D, weights_init, dynamic_weights_init
from util import load_image, save_image, read_adios_bp, save_image_adios
from skimage.measure import compare_ssim as ssim
#from scipy.misc import imsave

import numpy

#### EXAMPLE RUN:  python3 train.py --dataset /path/to/folder_with_bp_files

parser = argparse.ArgumentParser(description='DeepRendering-implemention')
parser.add_argument('--dataset', required=True, help='output from unity')
# the cGAN is sensitive to images of different sizes due to the
# con/deconvolution operations. the current network is designed for 256x256
parser.add_argument('--image_width', type=int, default=256, help='width images in dataset')
parser.add_argument('--image_height', type=int, default=256, help='height images in dataset')
parser.add_argument('--train_batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for testing')
parser.add_argument('--n_epoch', type=int, default=400, help='number of iterations')
parser.add_argument('--n_channel_input', type=int, default=3, help='number of input channels')
parser.add_argument('--n_channel_output', type=int, default=3, help='number of output channels')
parser.add_argument('--n_generator_filters', type=int, default=64, help='number of initial generator filters')
parser.add_argument('--n_discriminator_filters', type=int, default=64, help='number of initial discriminator filters')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate')
# beta1 changed from 0.5 (low) for adam optimization update rate
parser.add_argument('--beta1', type=float, default=0.4, help='beta1')
parser.add_argument('--cuda', action='store_true', help='cuda')
parser.add_argument('--resume_G', help='resume G')
parser.add_argument('--resume_D', help='resume D')
parser.add_argument('--workers', type=int, default=4, help='number of threads for data loader')
parser.add_argument('--seed', type=int, default=123, help='random seed')
#increased from 170 to penalize generator for not being nearer
parser.add_argument('--lamda', type=int, default=260, help='L1 regularization factor')
opt = parser.parse_args()

cudnn.benchmark = True

torch.cuda.manual_seed(opt.seed)

print('=> Loading datasets')

root_dir = opt.dataset#"../PathTracer/build"
conditional_names = ["outputs", "direct", "depth", "normals", "albedo"]

# some higher resolution images
#root_dir = "../path_tracer/raytracingtherestofyourlife/dataset/"
train_dir = join(os.path.join(root_dir ,opt.dataset), "train")
test_dir = join(os.path.join(root_dir , opt.dataset), "val")

train_set = AdiosDataLoader(opt.dataset,split="train") #DataLoaderHelper(train_dir)
val_set = AdiosDataLoader(opt.dataset, split="val") #DataLoaderHelper(test_dir)
test_set = AdiosDataLoader(opt.dataset, split="test")

batch_size = opt.train_batch_size
n_epoch = opt.n_epoch

train_data = DataLoader(dataset=train_set, num_workers=opt.workers, batch_size=opt.train_batch_size, shuffle=True)
val_data = DataLoader(dataset=val_set, num_workers=opt.workers, batch_size=opt.test_batch_size, shuffle=False)
test_data = DataLoader(dataset=test_set, num_workers=opt.workers, batch_size=opt.test_batch_size, shuffle=False)


print('=> Building model')

netG = G(opt.n_channel_input*4, opt.n_channel_output, opt.n_generator_filters)
netG.apply(dynamic_weights_init)
netD = D(opt.n_channel_input*4, opt.n_channel_output, opt.n_discriminator_filters)
netD.apply(weights_init)

criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()


albedo = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
direct = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
normal = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
depth = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)

gt = torch.FloatTensor(opt.train_batch_size, opt.n_channel_output, 256, 256)

label = torch.FloatTensor(opt.train_batch_size)
fake_label = 0.1#numpy.random.uniform(0. , 0.1)
real_label = 0.9# numpy.random.uniform(0.9, 1.0)

netD = netD.cuda()
netG = netG.cuda()
criterion = criterion.cuda()
criterion_l1 = criterion_l1.cuda()


albedo = albedo.cuda()
direct = direct.cuda()
normal = normal.cuda()
depth = depth.cuda()
gt = gt.cuda()
label = label.cuda()

albedo = Variable(albedo)
direct = Variable(direct)
normal = Variable(normal)
depth = Variable(depth)
gt = Variable(gt)
label = Variable(label)


optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
lastEpoch = 0

if opt.resume_G:
    if os.path.isfile(opt.resume_G):
        print("=> loading generator checkpoint '{}'".format(opt.resume_G))
        checkpoint = torch.load(opt.resume_G)
        lastEpoch = checkpoint['epoch']
        n_epoch = n_epoch - lastEpoch
        netG.load_state_dict(checkpoint['state_dict_G'])
        optimizerG.load_state_dict(checkpoint['optimizer_G'])
        print("=> loaded generator checkpoint '{}' (epoch {})".format(opt.resume_G, checkpoint['epoch']))

    else:
        print("=> no checkpoint found")

if opt.resume_D:
    if os.path.isfile(opt.resume_D):
        print("=> loading discriminator checkpoint '{}'".format(opt.resume_D))
        checkpoint = torch.load(opt.resume_D)
        netD.load_state_dict(checkpoint['state_dict_D'])
        optimizerD.load_state_dict(checkpoint['optimizer_D'])
        print("=> loaded discriminator checkpoint '{}'".format(opt.resume_D))

# adjust loss, weights, bias based on Fresnel and stokes theorem
# for identifying refracted light

# do we want to focus on global illumination
# rendering books?
# augmentations to identify illumination attributes desired

### Notable changes:
# random small float for fake label, random near one positive label
# lower learning rate by factor 1/10
# increase L1 learning weight by fifty
# dynamic weight initialization

# possible changes:
# Noise to discriminator images to provide stronger gradient for generator
# different statistical measure for loss of discriminator / generator
#  to avoid mode collapse
#could do monte-carlo esque adaption where noise (gaussian blur) is added
# randomly to images given to the discriminator ( e.g. np.rand(0,1) < 0.2)
# randomly interject real images of simple shapes to capture shading

def train(epoch):
    for (i, images) in enumerate(train_data):
        netD.zero_grad()
        (albedo_cpu, direct_cpu, normal_cpu, depth_cpu, gt_cpu) = (images[0], images[1], images[2], images[3], images[4])
        # to avoid killing gradient
        fake_label = 0.1#numpy.random.uniform(0. , 0.1)
        real_label = 0.9#numpy.random.uniform(0.9, 1.0)
        with torch.no_grad():
            albedo.resize_(albedo_cpu.size()).copy_(albedo_cpu)
            direct.resize_(direct_cpu.size()).copy_(direct_cpu)
            normal.resize_(normal_cpu.size()).copy_(normal_cpu)
            depth.resize_(depth_cpu.size()).copy_(depth_cpu)
            gt.resize_(gt_cpu.size()).copy_(gt_cpu)
        output = netD(torch.cat((albedo, direct, normal, depth, gt), 1))
        with torch.no_grad():
            label.resize_(output.size()).fill_(real_label)
        err_d_real = criterion(output, label)
        err_d_real.backward()
        with torch.no_grad():
            d_x_y = output.mean()
        fake_B = netG(torch.cat((albedo, direct, normal, depth), 1))
        output = netD(torch.cat((albedo, direct, normal, depth, fake_B.detach()), 1))
        with torch.no_grad():
            label.resize_(output.size()).fill_(fake_label)
        err_d_fake = criterion(output, label)
        err_d_fake.backward()
        with torch.no_grad():
            d_x_gx = output.mean()
        err_d = (err_d_real + err_d_fake) * 0.5
        optimizerD.step()

        netG.zero_grad()
        output = netD(torch.cat((albedo, direct, normal, depth, fake_B), 1))
        with torch.no_grad():
            label.resize_(output.size()).fill_(real_label)
        err_g = criterion(output, label) + opt.lamda \
            * criterion_l1(fake_B, gt) 
        err_g.backward()
        with torch.no_grad():
            d_x_gx_2 = output.mean()
        optimizerG.step()
        print ('=> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
            epoch,
            i,
            len(train_data),
            err_d.item(),#data[0],
            err_g.item(),#data[0],
            d_x_y,
            d_x_gx,
            d_x_gx_2,
            ))

def save_checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset.split('/')[-1])):
        os.mkdir(os.path.join("checkpoint", opt.dataset.split('/')[-1]))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset.split('/')[-1], epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset.split('/')[-1], epoch)
    torch.save({'epoch':epoch+1, 'state_dict_G': netG.state_dict(), 'optimizer_G':optimizerG.state_dict()}, net_g_model_out_path)
    torch.save({'state_dict_D': netD.state_dict(), 'optimizer_D':optimizerD.state_dict()}, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset.split('/')[-1]))

    if not os.path.exists("validation"):
        os.mkdir("validation")
    if not os.path.exists(os.path.join("validation", opt.dataset.split('/')[-1])):
        os.mkdir(os.path.join("validation", opt.dataset.split('/')[-1]))

    for index, images in enumerate(val_data):
        (albedo_cpu, direct_cpu, normal_cpu, depth_cpu, gt_cpu) = (images[0], images[1], images[2], images[3], images[4])
        with torch.no_grad():
            albedo.resize_(albedo_cpu.size()).copy_(albedo_cpu)
            direct.resize_(direct_cpu.size()).copy_(direct_cpu)
            normal.resize_(normal_cpu.size()).copy_(normal_cpu)
            depth.resize_(depth_cpu.size()).copy_(depth_cpu)
        out = netG(torch.cat((albedo, direct, normal, depth), 1))
        out = out.cpu()
        out_img = out.data[0]
        save_image_adios(out_img,"validation/{}/{}_Fake.bp".format(opt.dataset.split('/')[-1], index),opt.image_width, opt.image_height, out_img.shape[0])
        save_image_adios(gt_cpu[0], "validation/{}/{}_Real.bp".format(opt.dataset.split('/')[-1], index),opt.image_width, opt.image_height,  out_img.shape[0])
        save_image_adios(direct_cpu[0],"validation/{}/{}_Direct.bp".format(opt.dataset.split('/')[-1], index),opt.image_width, opt.image_height,  out_img.shape[0])





for epoch in range(n_epoch):
    train(epoch+lastEpoch)
    if epoch % 5 == 0:
        save_checkpoint(epoch+lastEpoch)

