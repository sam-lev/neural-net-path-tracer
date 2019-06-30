import trainingTracer_cuda as ttc
import numpy as np
from PIL import Image

phi = 2.0*np.pi
theta = np.pi/2.0
numTheta, numPhi = 30, 30
rTheta = (2.0*np.pi)/float(numTheta)
rPhi = (np.pi/2.0)/float(numPhi)

get_buffers = 1
rendered_buffers = ttc.renderFromOrientation(256,256,100,20,
                                             theta+5.*(rTheta), phi+5.*(rPhi),
                                             get_buffers,1)
print("type returned buffers ", type(rendered_buffers))
print("len of buffer ", len(rendered_buffers))
direct = np.array(rendered_buffers[0]).astype(np.uint8)
print("shape of buffers " , direct.shape)
if get_buffers = 0:
    direct.reshape((256,256,3))
else:
    direct.reshape((256,256))
img = Image.fromarray(direct)
img.show()
