from os import listdir
from os.path import join

import torch.utils.data as data

from util import is_image, load_image, get_split, read_adios_bp, load_adios_image, format_tensor

class DataLoaderHelper(data.Dataset):
    def __init__(self, image_dir):
        super(DataLoaderHelper, self).__init__()
        self.albedo_path = join(image_dir, "albedo")
        self.depth_path = join(image_dir, "depth")
        self.direct_path = join(image_dir, "direct")
        self.normal_path = join(image_dir, "normal")
        self.gt_path = join(image_dir, "gt")
        self.image_filenames = [x for x in listdir(self.gt_path) if is_image(x)]
        for i, im_name in enumerate(self.image_filenames):
            base = im_name.split('-')
            im_name = base[1]+'-'+base[2]
            self.image_filenames[i] = im_name


    def __getitem__(self, index):
        albedo = load_image(join(self.albedo_path,'albedo-'+ self.image_filenames[index]))
        depth = load_image(join(self.depth_path, 'depth-'+self.image_filenames[index]))
        direct = load_image(join(self.direct_path,'direct-'+ self.image_filenames[index]))
        normal = load_image(join(self.normal_path, 'normals-'+self.image_filenames[index]))
        gt = load_image(join(self.gt_path, 'output-'+self.image_filenames[index]))
        return albedo, direct, normal, depth, gt

    def __len__(self):
        return len(self.image_filenames)

#### added split for training, validation and test set
#### using loaded Adios bp files. Each conditional training
#### set is assumed to have independent bp file
#### creating a pytorch dataset class
####
#### image filenames based on ground truth images available
#### read_adios_bp(.) takes buffer or ground truth image path
####                  to bp and returns tuple (image names, dict(name->image))
####                  conditional param: "direct", "depth", "normals"
####                                     "albedo", "trace" (or "output").
#### get_split(.) splits read adios image data into validation set (20%),
####             training set (60%), and test set (20%).
####
class AdiosDataLoader(data.Dataset):
    #split can be 'train', 'val', and 'test'
    def __init__(self, image_dir, split = 'train'):
        super(AdiosDataLoader, self).__init__()
        self.albedo_path = join(image_dir, "albedo.bp")
        self.depth_path = join(image_dir, "depth.bp")
        self.direct_path = join(image_dir, "direct.bp")
        self.normal_path = join(image_dir, "normals.bp")
        self.gt_path = join(image_dir, "outputs.bp")
        self.width = 256
        self.height = 256
        self.samplecount = 50
        # albedo
        self.image_data = read_adios_bp(self.albedo_path
                                             ,conditional="albedo"
                                             ,width=self.width
                                             ,height=self.height
                                             ,sample_count=self.samplecount )
        #collect adios var (image) names
        self.image_filenames = self.image_data[0]

        #albedo image name to image map
        self.name_to_image_albedo = self.image_data[1]

        # partition training, testing and validation set
        self.partitioned_image_filenames = get_split(self.image_filenames, split)
        #depth image name to image dict
        self.name_to_image_depth = read_adios_bp(self.depth_path
                                        ,conditional="depth"
                                        ,width=self.width
                                        ,height=self.height
                                        ,sample_count=self.samplecount )[1]
        # direct  image name to image dict
        self.name_to_image_direct = read_adios_bp(self.direct_path
                                        ,conditional="direct"
                                        ,width=self.width
                                        ,height=self.height
                                        ,sample_count=self.samplecount )[1]
        # path traced image image name to image dict
        self.name_to_image_trace = read_adios_bp(self.gt_path
                                        ,conditional="outputs"
                                        ,width=self.width
                                        ,height=self.height
                                        ,sample_count=self.samplecount )[1]
        # normal buffer image name to image dict
        self.name_to_image_normals = read_adios_bp(self.normal_path
                                        ,conditional="normals"
                                        ,width=self.width
                                        ,height=self.height
                                        ,sample_count=self.samplecount )[1]
        #for i, im_name in enumerate(self.image_filenames):
        #    base = im_name.split('-')
        #    im_name = base[1]+'-'+base[2]
        #    self.image_filenames[i] = im_name
        self.im_show_debug = 0
        self.im_show_debug += 1

    def __getitem__(self, index):
        name = self.partitioned_image_filenames[index]
        width = 128
        height = 128
        sample_count=100
        albedo = format_tensor(self.name_to_image_albedo[name])#format_tensor(load_adios_image(name,"albedo",  self.albedo_path, width=self.width, height=self.width, sample_count=self.samplecount))
        depth =  format_tensor(self.name_to_image_depth[name])#format_tensor(load_adios_image(name, "depth", self.depth_path, width=self.width, height=self.width, sample_count=self.samplecount))
        direct = format_tensor(self.name_to_image_direct[name])#format_tensor(load_adios_image(name, "direct", self.direct_path, width=self.width, height=self.width, sample_count=self.samplecount))
        normal = format_tensor(self.name_to_image_normals[name])#format_tensor(load_adios_image(name,"normals",  self.normal_path, width=self.width, height=self.width, sample_count=self.samplecount))
        gt =  format_tensor(self.name_to_image_trace[name])#format_tensor(load_adios_image(name, "trace", self.gt_path, width=self.width, height=self.width, sample_count=self.samplecount))
        return  albedo, direct, normal, depth, gt
        

    def __len__(self):
        return len(self.partitioned_image_filenames)
