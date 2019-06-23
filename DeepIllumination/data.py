from os import listdir
from os.path import join

import torch.utils.data as data

from util import is_image, load_image

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
        self.im_show_debug = 0
        self.im_show_debug += 1


    def __getitem__(self, index):
        self.im_show_debug += 1
        albedo = load_image(join(self.albedo_path,'albedo-'+ self.image_filenames[index]))
        depth = load_image(join(self.depth_path, 'depth-'+self.image_filenames[index]))
        direct = load_image(join(self.direct_path,'direct-'+ self.image_filenames[index]))
        normal = load_image(join(self.normal_path, 'normals-'+self.image_filenames[index]))
        gt = load_image(join(self.gt_path, 'output-'+self.image_filenames[index]))
        return albedo, direct, normal, depth, gt

    def __len__(self):
        return len(self.image_filenames)
