import trainingTracer_cuda as ttc
import numpy as np
from PIL import Image

class buffer_tests:
    def __init__(self, samplecount = 100, depthcount = 20, save_image = 0):
        class_purpose = "set of tests for path tracer to python neural net binding"
        self.rendered_buffers = []
        #self.get_buffers = get_buffers
        self.save_image = save_image
        self.path_trace, self.direct, self.normal, self.albedo = None, None, None, None
        self.depth = None
        self.buffers = [self.path_trace, self.direct, self.normal, self.albedo]
        self.samplecount = samplecount
        self.depthcount = depthcount
        self.theta = 0
        self.phi = 0
        
    def render_orientation(self, theta = 2.618, phi = 2.08*3.2,
                        samplecount = 100, depthcount = 20,
                           get_buffers = 1, save_image = None):
        if(theta):
            self.theta = theta
        if(phi):
            self.phi = phi
        if(samplecount):
            self.samplecount = samplecount
        if(depthcount):
            self.depthcount = depthcount
        if(save_image):
            self.save_image = save_image
            
        numTheta, numPhi = 30, 30
        rTheta =0# (2.0*np.pi)/float(numTheta)
        rPhi = 0#(np.pi/2.0)/float(numPhi)

        b_type = ["path","direct",  "normal", "albedo"]
        count = 0
        for t, b in zip(b_type, [self.path_trace, self.direct, self.normal, self.albedo] ):
            #if(t == "direct"):
            #    self.samplecount = 1000
            #    self.depthcount = 0
            print(t)
            buff =  ttc.renderFromOrientation(str(t),
                                           256,256,
                                           self.samplecount,self.depthcount,
                                              self.theta,
                                              self.phi,
                                           self.save_image)
            b = buff[count]
            count += 1
            self.rendered_buffers.append(b)
        # get depth buffer as well
        self.depth = ttc.renderDepthBuffer(256,256,
                                           self.samplecount,self.depthcount,
                                              self.theta,
                                              self.phi,
                                           self.save_image)
            
        rendered_buffers = self.rendered_buffers
        
        #convert to numpy ( copy = False?)
        self.path_trace = np.array(self.rendered_buffers[0])#.astype(np.uint8)
        self.direct = np.array(self.rendered_buffers[1])#.astype(np.uint8)
        self.normal = np.array(self.rendered_buffers[2])#.astype(np.uint8)
        self.albedo = np.array(self.rendered_buffers[3])#.astype(np.uint8)
        self.depth = np.array(self.depth)#.astype(np.uint8)
        
        print("shape of direct buffers " , self.direct.shape)
        print("shape of depth buffers " , self.depth.shape, " sum vals: ", np.sum(self.depth))
        print("shape of path traced image " , self.path_trace.shape)
        print("shape of normal buffers " , self.normal.shape)
        print("shape of albedo buffers " , self.albedo.shape)
        #if self.get_buffers == 0:
        #    self.direct.reshape((256,256,3))
        #else:
        #    self.direct.reshape((256,256))
        #img = Image.fromarray(self.direct)
        #img.show()
        
    def translate(self, image,  buffer_type = 0, mode="RGB"):
        #image = image/int(self.samplecount)#/self.depthcount)#self.depthcount)
        if( buffer_type == "path"):
            image = image/float(self.samplecount) #float
            image= 255.99*image
            image = image.astype(np.uint8) # must be int but division byfloat
        elif(buffer_type == "direct"):
            image[image < 0 ] = 0
            image = image#/int(self.samplecount)#/self.depthcount)
            image= 255.99*np.sqrt(image)
            image = image.astype(np.uint8) ##must be int
            #image.asty
        elif(buffer_type == "depth"): #no sqrt
            image[image < 0 ] = 0
            image= 555.99999*image
            image = image.astype(np.uint8)
        elif(buffer_type == "albedo"):
            image[image < 0 ] = 0
            image = image#
            image= 255.99*image
            image = image.astype(np.uint8)
        else: #normal / unnamed
            #image = image/int(self.samplecount)#/self.depthcount)
            image[image < 0 ] = 0
            image= 255.99*image
            image = image.astype(np.uint8)


        image[image == None] = int(0)
        #for p in image:w
        #    p = int(p)
        if(buffer_type == "depth"):
            image = image.reshape((256,256))#.astype(np.uint8)
        else:
            image = image.reshape((256,256,3))#.astype(np.uint8)
        img = Image.fromarray(image, mode)
        img.show()

    def test1(self):
        bt1 = buffer_tests(samplecount = 300, depthcount = 50)
        #bt1.render_orientation(save_image = 1)
        bt1.render_orientation(save_image = 0)
        bt1.translate(bt1.path_trace, buffer_type = "path")
        bt1.translate(bt1.direct, buffer_type = "direct")
        bt1.translate(bt1.normal, buffer_type = "normal")
        bt1.translate(bt1.albedo, buffer_type = "albedo")
        bt1.translate(bt1.depth,buffer_type = "depth", mode = "L")
        return bt1

    

    

bt1 = buffer_tests()
bt1.test1()
