#from mpi4py import MPI
import os
from os.path import join
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from data import DataLoaderHelper, AdiosDataLoader

from torch.utils.data import DataLoader
import torch.utils.data.distributed

from torch.autograd import Variable
from model import G, D, weights_init, dynamic_weights_init
from util import load_image, save_image, read_adios_bp, save_image_adios
from skimage.measure import compare_ssim as ssim
#from scipy.misc import imsave

import numpy

import time

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
parser.add_argument('--nGPU', type=int, help='numbr of gpu to distribute between both neural net models')
parser.add_argument('--gpu_per_node', type=int, help='number GPU within each node')
parser.add_argument('--multiprocess_distributed', default=0, help='multi process parallel training distributed across nGPU between both neural network models. Requires nGPU and gpu_per_node arguments')
parser.add_argument('--world_size', type=int, default=-1, help='number of communications')
opt = parser.parse_args()

# profiling and logging
filename = 'run_log.txt'
proj_write_dir =  os.path.join(os.environ["PROJWORK"],"csc143","samlev")
if os.path.exists(filename):
    append_write = 'a' # append if already exists
else:
    append_write = 'w' # make a new file if not
print('Run log written to: ',os.path.join(proj_write_dir,filename))
log = open(os.path.join(proj_write_dir,filename),append_write)
log.write(" Beginning Training..............")
#log.write("WORLD SIZE: "+str(os.environ["WORLD_SIZE"])))
log.close()

#cuda auto-tuner to find best algorithm given hardware
#cudnn.benchmark = True

# sets seed of random number generator to make
# the results will be reproducible.
torch.cuda.manual_seed(opt.seed)
print('You have chosen to seed training. ',
      'seeding the random number generator allows reproducable results ',
      'This will turn on the CUDNN deterministic setting, ',
      'which can slow down your training considerably! ',
      'You may see unexpected behavior when restarting ',
      'from checkpoints.')

print('=> Loading datasets')

if opt.multiprocess_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    #if opt.world_size == -1:
    #    opt.world_size = int(os.environ["WORLD_SIZE"])
    opt.gpu_per_node = torch.cuda.device_count()
    opt.world_size = opt.gpu_per_node * opt.world_size
    #mp.spawn(build_model, nprocs=opt.gpu_per_node, args=(opt.gpu_per_node, opt))
#else:
#    build_model(opt.nGPU, opt.gpu_per_node)    


root_dir = opt.dataset#"../PathTracer/build"
conditional_names = ["outputs", "direct", "depth", "normals", "albedo"]

# some higher resolution images
proj_read_dir =  os.path.join(os.environ["PROJWORK"],"csc143","samlev","data")
train_set = AdiosDataLoader( os.path.join(proj_read_dir, opt.dataset),split="train") #DataLoaderHelper(train_dir)
val_set = AdiosDataLoader(  os.path.join(proj_read_dir, opt.dataset), split="val") #DataLoaderHelper(test_dir)
test_set = AdiosDataLoader(  os.path.join(proj_read_dir, opt.dataset), split="test")

batch_size = opt.train_batch_size
n_epoch = opt.n_epoch

# num_workers=opt.workers,
# Change if used in paper!
train_data = DataLoader(dataset=train_set, batch_size=opt.train_batch_size, shuffle=True)
val_data = DataLoader(dataset=val_set,  batch_size=opt.test_batch_size, shuffle=False)
test_data = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)


print('=> Building model')

netG = G(opt.n_channel_input*4, opt.n_channel_output, opt.n_generator_filters)
netG.apply(dynamic_weights_init)
netD = D(opt.n_channel_input*4, opt.n_channel_output, opt.n_discriminator_filters)
netD.apply(weights_init)

# specify gpus to use for data parallelism
def split_gpu(devices):
    split = len(devices)//2
    return devices[:split], devices[split:]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids_netG, device_ids_netD = [] , []
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    device_ids_netG, device_ids_netD = split_gpu(list(range( torch.cuda.device_count() )))
    node_gpus = list(range( torch.cuda.device_count()))
    #n = torch.cuda.device_count() // world_size
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #device_ids = list(range(rank * n, (rank + 1) * n))
    ##torch.nn.parallel.DistributedDataParallel(
                          
    netG = nn.DataParallel(netG, device_ids=node_gpus, dim=0)
    netD = nn.DataParallel(netD, device_ids=node_gpus, dim=1)
else:
    print("Using ",torch.cuda.device_count(), " GPU.")
    netG = nn.DataParallel(netG)#.to(deviceG)
    netD = nn.DataParallel(netD)#.to(deviceD)
    #netG = torch.nn.parallel.DistributedDataParallel(netG)

#netD.to(device)
#netG.to(device)

criterion = nn.BCELoss().to(device)
criterion_l1 = nn.L1Loss().to(device) # to use multiple gpu among multipl model

albedo = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
direct = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
normal = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)
depth = torch.FloatTensor(opt.train_batch_size, opt.n_channel_input, 256, 256)

gt = torch.FloatTensor(opt.train_batch_size, opt.n_channel_output, 256, 256)

label = torch.FloatTensor(opt.train_batch_size)
fake_label = 0.1#numpy.random.uniform(0. , 0.1)
real_label = 0.9# numpy.random.uniform(0.9, 1.0)


#for multigpu multimodel assign gpu to whole model
# with model already assigned to gpu devices.
# data parrallel later

criterion = criterion.to(device)#.cuda()   
criterion_l1 = criterion_l1.to(device)#.cuda()

albedo = albedo.to(device)#.cuda()
direct = direct.to(device)#.cuda()
normal = normal.to(device)#.cuda()
depth = depth.to(device)#.cuda()
gt = gt.to(device)#.cuda()
label = label.to(device)#.cuda()

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
    if not os.path.exists(os.path.join(proj_write_dir,"checkpoint")):
        os.mkdir(os.path.join(proj_write_dir,"checkpoint"))
    if not os.path.exists(os.path.join(proj_write_dir, "checkpoint", opt.dataset.split('/')[-1])):
        os.mkdir(os.path.join(proj_write_dir,"checkpoint", opt.dataset.split('/')[-1]))
    net_g_model_out_path = os.path.join(proj_write_dir,"checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset.split('/')[-1], epoch))
    net_d_model_out_path = os.path.join(proj_write_dir,"checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset.split('/')[-1], epoch))
    torch.save({'epoch':epoch+1, 'state_dict_G': netG.state_dict(), 'optimizer_G':optimizerG.state_dict()}, net_g_model_out_path)
    torch.save({'state_dict_D': netD.state_dict(), 'optimizer_D':optimizerD.state_dict()}, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + os.path.join(proj_write_dir,opt.dataset).split('/')[-1]))

    if not os.path.exists(os.path.join(proj_write_dir,"validation")):
        os.mkdir(os.path.join(proj_write_dir,"validation"))
    if not os.path.exists(os.path.join(proj_write_dir, "validation", opt.dataset.split('/')[-1])):
        os.mkdir(os.path.join(proj_write_dir,"validation", opt.dataset.split('/')[-1]))

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
        save_image_adios(out_img, os.path.join(proj_write_dir,"validation/{}/{}_Fake.bp".format(opt.dataset.split('/')[-1], index)), opt.image_width, opt.image_height, out_img.shape[0])
        save_image_adios(gt_cpu[0], os.path.join(proj_write_dir, "validation/{}/{}_Real.bp".format(opt.dataset.split('/')[-1], index)), opt.image_width, opt.image_height,  out_img.shape[0])
        save_image_adios(direct_cpu[0], os.path.join(proj_write_dir, "validation/{}/{}_Direct.bp".format(opt.dataset.split('/')[-1], index)) ,opt.image_width, opt.image_height,  out_img.shape[0])



#time profiling
time0 = time.time()

for epoch in range(n_epoch):
    train(epoch+lastEpoch)
    if epoch % 5 == 0:
        save_checkpoint(epoch+lastEpoch)
        # time profiling
        # profiling and logging                                                    
        filename = 'time_profiling.txt'
        filename = os.path.join(proj_write_dir, filename)
        if os.path.exists(filename):
            append_write = 'a' # append if already exists                       
        else:
            append_write = 'w' # make a new file if not                        
        tlog = open(filename,append_write)
        tlog.write(time0 - time.time())
        tlog.close()
