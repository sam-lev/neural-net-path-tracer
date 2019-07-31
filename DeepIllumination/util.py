import numpy as np
from scipy.misc import imread, imresize, imsave
import torch
import cv2
import adios2 as a2
# for debugging
from PIL import Image

def load_image(filepath):
    image = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
    print(image.shape)
    if image is None:
        print(filepath)
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
    #if im_show_debug< 5:
    #    im_temp = Image.fromarray(image, 'RGB')
    #    im_temp.show()
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    min = float(image.min())
    max = float(image.max())
    #print("min", min, " ", "max", max)
    image = torch.FloatTensor(image.size()).copy_(image)
    if max != min:
        image.mul_(1.0 / (max - min)) #add_(-min)
    #image = image.mul_(2.0).add_(-1.0)
    #if im_show_debug< 5:
    #    im_temp = Image.fromarray(image.numpy(), 'RGB')
    #    im_temp.show()
    return image

def format_tensor(image):
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
    #if im_show_debug< 5:
    #    im_temp = Image.fromarray(image, 'RGB')
    #    im_temp.show()
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    min = float(image.min())
    max = float(image.max())
    #print("min", min, " ", "max", max)
    image = torch.FloatTensor(image.size()).copy_(image)
    if max != min:
        image.mul_(1.0 / (max - min)) #add_(-min)
    #image = image.mul_(2.0).add_(-1.0)
    #if im_show_debug< 5:
    #    im_temp = Image.fromarray(image.numpy(), 'RGB')
    #    im_temp.show()
    return image

def save_image(image, filename):
    #image = image.add_(1.0).div_(2.0)
    image = image.numpy()
    #image *= 255.0
    image = image.clip(0, 255)
    image = np.transpose(image, (1, 2, 0))
    #image = image.astype(np.uint8)
    imsave(filename, image)
    print ("Image saved as {}".format(filename))

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg",".ppm"])

def translate(image,  samplecount=100, buffer_type = 0, shape = (256,256,3), mode="RGB", show=False):
    #image = image/int(self.samplecount)#/self.depthcount)#self.depthcount)
    if( buffer_type == "trace"):
        image = image/float(samplecount) #float
        image= 255.99*np.sqrt(image)
        image = image.astype(np.uint8) # must be int but division byfloat
    elif(buffer_type == "direct"):
        image[image < 0 ] = 0.
        image = image#/int(self.samplecount)#/self.depthcount)
        image= 255.99*np.sqrt(image)
        image = image.astype(np.uint8) ##must be int
    elif(buffer_type == "depth"): #no sqrt
        image[image < 0 ] = 0.
        image=255.99*image
        image = image.astype(np.uint8)
    elif(buffer_type == "albedo"):
        image[image < 0 ] = 0
        image = image#
        image= 255.99*image
        image = image.astype(np.uint8)
    else: #normal / unnamed
        #image = image/int(self.samplecount)#/self.depthcount)
        image[image < 0 ] = 0.
        image= 255.99*image
        image = image.astype(np.uint8)

    #image[image != image] = int(0)
    image = image.reshape(shape)
    if(buffer_type != "depth"):
        image = image[:,:,:3]
        
    if show:
        img = Image.fromarray(image, mode)
        img.show()
    return image

def read_adios_bp(filename=None, conditional = "direct", width=256, height=256, sample_count = 300):
    if not filename:
        print("No File Provided")
    if conditional not in ["direct", "depth", "normals", "albedo", "trace","outputs"]:
        print("Sample must be one of ",["direct", "depth", "normal", "albedo", "trace"], " but was given ", conditional)
    #if filename == "trace":
    #    filename="outputs"
    if conditional =="depth":   
        shape = [width, height]
        start = [0]
        count = [width*height]
        save_mode = "L"
    else:
        shape = [width, height,4]
        start = [0]
        count = [width*height*4]
        save_mode = "RGB" #remove alpha in translate

    print("LOADING: ", conditional, " ... ")
    test_view = ""
    image_samples = []
    image_names = []
    name_image = {}
    im_count = 0
    view = False
    with a2.open(filename, "r") as bundle: #mpi here when included
        for imgs in bundle:
            im = imgs.available_variables()
            for name, attributes in im.items():
                #print("name: ", name)
                image_names.append(name)

                """ For Testing: """
                #for key, value in attributes.items():
                #    print("\t" + key + ": " + value)
                #if im_count == 20:
                #    view = True
                #else:
                #    view = False
                im_count+=1
                IMAGE = imgs.read(name, start, count)
                #print("after")
                sample = translate(IMAGE,
                                   samplecount=sample_count,
                                   buffer_type=conditional,
                                  shape=shape,
                                   mode=save_mode
                                   ,show = view)
                image_samples.append(sample)
                name_image[name] = sample
                
    return (image_names, name_image)#image_samples)

def load_adios_image(image_name, conditional, filename=None, width=256, height=256, sample_count = 300):
    if not filename:
        print("No File Provided")
    if conditional not in ["direct", "depth", "normals", "albedo", "trace", "outputs"]:
        print("Sample must be one of ",["direct", "depth", "normals", "albedo", "trace"], " but was given ", conditional)
    #if filename == "trace":
    #    filename="outputs"
    if conditional =="depth":   
        shape = [width, height]
        start = [0]
        count = [width*height]
        save_mode = "L"
    else:
        shape = [width, height,4]
        start = [0]
        count = [width*height*4]
        save_mode = "RGB"

    #print("#########################  ", conditional, " ######")
    IMAGE =  None#np.zeros(count, dtype=np.float32)
    test_view = ""
    image_samples = []
    image_names = []
    with a2.open(filename, "r") as bundle: #mpi here when included
        for imgs in bundle:
            im = imgs.available_variables()
            for name, attributes in im.items():
                if conditional == "albedo":
                    print("name: ", name, "conditional: ", conditional)
                image_names.append(name)
                #for key, value in attributes.items():
                #    print("\t" + key + ": " + value)
                #if im_count == 20:
                #    test_view = name
                #    IMAGE = imgs.read(name, start, count)
                #im_count+=1
            try:
                IMAGE = imgs.read(image_name, start, count)
            except ValueError:
                print("____!!!___IMAGE READ FAILED___!!!___")
                print("conditional: ", conditional, " fname: ", filename)
                print("image: ", image_name)
            sample = translate(IMAGE,
                               samplecount=sample_count,
                               buffer_type=conditional,
                               shape=shape,
                               mode=save_mode, show=False)
                
    return sample
        
def format_dataLoader(traced=[], direct=[], depth=[], normals=[], albedo=[]):
    dataPack = []
    for tra, dir, dep, nor, alb in zip(traced, direct, depth, normals, albedo):
        dataPack.append((tra, dir, dep, nor, alb))
    return dataPack

#split can be 'train', 'val', and 'test'
#this is the function that splits a dataset into training, validation and testing set
#We are using a split of 60%-20%-20%, for train-val-test, respectively
#this function is used internally to the defined dataset classes
# In medical datasets possibly containing more than one example for the same subject/patient,
# this function should be applied to the list of patients/subjects, and not to the list of examples
# since in a real-world application you will not find the same subject/patient as your training data had,
# and therefore you should measure how well your model is doing in the same settings
def get_split(array_to_split, split):
    np.random.seed(0)
    np.random.shuffle(array_to_split)
    np.random.seed()
    if split == 'train':
        array_to_split = array_to_split[:int(len(array_to_split)*0.6)]
    elif split == 'val':
        array_to_split = array_to_split[int(len(array_to_split)*0.6):int(len(array_to_split)*0.8)]
    elif split == 'test':
        array_to_split = array_to_split[int(len(array_to_split)*0.8):]
    return array_to_split
