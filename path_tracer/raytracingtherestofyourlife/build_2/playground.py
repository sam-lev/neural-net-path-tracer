import trainingTracer_cuda as ttc
import numpy as np
from PIL import Image

class buffer_tests:
    def __init__(self):
        class_purpose = "set of tests for path tracer to python neural net binding"
        self.rendered_buffers = []
        self.get_buffers = 0
        self.direct, self.depth, self.normal, self.albedo = None, None, None, None
    def render_orientation(self):
        phi = 2.08*np.pi #2.0*np.pi
        theta = 2.6180 #np.pi/2.0
        numTheta, numPhi = 30, 30
        rTheta =0# (2.0*np.pi)/float(numTheta)
        rPhi = 0#(np.pi/2.0)/float(numPhi)
        
        self.get_buffers = 1
        self.rendered_buffers = ttc.renderFromOrientation(256,256,
                                                          100,20,
                                                          theta+5.*(rTheta),
                                                          phi+5.*(rPhi),
                                                          self.get_buffers,
                                                          0)
        
        rendered_buffers = self.rendered_buffers 
        print("type returned buffers ", type(self.rendered_buffers))
        print("len of buffer ", len(self.rendered_buffers))
        direct = np.array(self.rendered_buffers[0]).astype(np.uint8)
        depth = np.array(self.rendered_buffers[1]).astype(np.uint8)
        normal = np.array(self.rendered_buffers[2]).astype(np.uint8)
        albedo = np.array(self.rendered_buffers[3]).astype(np.uint8)
        print("shape of direct buffers " , direct.shape)
        print("shape of depth buffers " , depth.shape)
        print("shape of normal buffers " , direct.shape)
        print("shape of albedo buffers " , direct.shape)
        if self.get_buffers == 0:
            direct.reshape((256,256,3))
        else:
            direct.reshape((256,256))
        img = Image.fromarray(direct)
        img.show()
