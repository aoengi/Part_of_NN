import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
#from ImageCrossUnion import imageCrossUnion

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.vgg_conv = nn.Sequential(*list(model.children())[:-1])
        self.linear = nn.Sequential(
            torch.nn.Linear(512*7*7, 10)
        )

    def forward(self, x):
        x = self.vgg_conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model_conv, target_layers):
        self.model_conv = model_conv
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        #print(x)
        #x = self.model_conv(x)
        #print(x[0].size())
        #x = self.model_conv2(x)
        #print(x[0].size())
        #x = self.model_conv3(x)
        #print(x[0].size())
        #x = self.model_conv4(x)
        #for name, module in self.model_conv1._modules.items():

            #pass
        for name, module in self.model_conv._modules.items():
            x = module(x)
            #print(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            #print(x.shape)
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.vgg_conv, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        #print(output.shape)
        output = output.view(output.size(0), -1)
        #print(output.shape)
        output = self.model.linear(output)
        return target_activations, output

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img) #转化为tensor
    preprocessed_img.unsqueeze_(0)  #
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask, num, path):
    #print(mask)
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    #cam = np.float32(img).copy()
    #print(heatmap.shape)
    #for i in range(224):
        #for j in range(224):
            #if(mask[i, j]<=0.00001):
                #cam[i,j,0]=0
                #cam[i, j, 1] = 0
                #cam[i, j, 2] = 0
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path + "/cam"+str(num)+".jpg", np.uint8(255 * cam))
    return heatmap

aa = []
zt = []
name = {0:"cane", 1:"cavallo", 2:"elefante", 3:"farfalla", 4:"gallina", 5:"gatto", 6:"mucca", 7:"pecora", 8:"ragno", 9:"scoiattolo"}

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):    #参数分别为网络模型、目标xx、是否cuda
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, num, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        #print(len(features[0][0][0]))
        #  1*1*256*6*6   1*1000
        #print(num)
        #print(len(features[0][0][0][0]), len(output[0]))
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        self.model.vgg_conv.zero_grad()
        self.model.linear.zero_grad()
        #one_hot.backward(retain_variables=True)
        one_hot.backward()
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]  # 512*6*6
        #print(len(target[0]))
        #print(len(target), len(target[0]))
        #print(grads_val.shape)
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        feature = np.zeros(target.shape[1:], dtype = np.float32)  # 6*6
        w_z = np.zeros(target.shape[1:], dtype=np.float32)
        w_f = np.zeros(target.shape[1:], dtype=np.float32)
        a = model.state_dict()["linear.0.weight"].cpu().numpy().copy()
        a = a.reshape(10, 512, 7, 7)
        for i, w in enumerate(weights):
            if(i==num):
                #print("w ", w)
                feature += target[i, :, :]
                w_z += a[index, i, :, :]
                w_f -= a[index, i, :, :]
                #print(a[index, i, :, :])
                # * w
                #
                cos = sum(sum(np.multiply(target[i, :, :], a[index, i, :, :])))
                xx = np.linalg.norm(target[i, :, :])
                yy = np.linalg.norm(a[index, i, :, :])
                if(xx!=0 and yy!=0):
                    cos = cos/(xx*yy)
                    zt.append(cos)
                else:
                    cos = 0
                #print(xx, yy, cos, cos/(xx*yy))
                aa.append(cos)

                #print(cam)
        #print(target.shape)
        #print(name[index])
        feature = np.maximum(feature, 0)
        feature = cv2.resize(feature, (224, 224))
        feature = feature - np.min(feature)
        feature = feature / np.max(feature)
        w_z = np.maximum(w_z, 0)
        w_z = cv2.resize(w_z, (224, 224))
        w_z = w_z - np.min(w_z)
        w_z = w_z / np.max(w_z)
        w_f = np.maximum(w_f, 0)
        w_f = cv2.resize(w_f, (224, 224))
        w_f = w_f - np.min(w_f)
        w_f = w_f / np.max(w_f)
        return feature, w_z, w_f

def myrange(i:float,j:float,k=1)->list:
    xlen=str((len(str(k-int(k)))-2)/10)+"f" #根据k步长，判断format位数公式小数点位数/10+"f"
    print("xlen=",xlen)
    lista=[]
    while i<j:
        lista.append(format(i, xlen))
        i+=k
    return lista

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        #print(one_hot, output)
        #print(output.cpu().detach().numpy()[0][index])
        #print(one_hot)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward()
        #print(input)
        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    image_path = "0abdu-rahman-2934-unsplash.jpg"

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    vgg = models.vgg16(pretrained=False)
    model = Net(model=vgg)
    #model = model().cuda()
    model.load_state_dict(torch.load('vgg16+.pkl', map_location=torch.device('cpu')))
    grad_cam = GradCam(model=model, target_layer_names = ["1"], use_cuda=False)

    img = cv2.imread(image_path, 1)
    #print(img)
    img = np.float32(cv2.resize(img, (224, 224))) / 255   #图像归一化
    #cv2.imshow('1', img)
    #cv2.waitKey(0)
    input = preprocess_image(img)
    #print(input.shape)
    #cv2.imshow('1', input)
    #print(img)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    union_1 = []
    union_2 = []
    path = ""
    for i in range(512):
        print(i)
        feature, w_z, w_f = grad_cam(input, i, target_index)
        show_cam_on_image(img, feature, i, "image1")
        show_cam_on_image(img, w_z, i, "image2")
        show_cam_on_image(img, w_f, i, "image3")
    #print(aa)
    #dic, fu = np.histogram(aa, bins=range(-0.5, 0.5, 0.05))
    #print(dic, fu)
    #print(range(0, 2, 1))
    #plt.hist(aa, bins=50)
    #plt.show()
    #print("期望", np.mean(zt),"方差" , np.var(zt))
    #print(np.sort(aa))
    #aa = np.argsort(aa)
    #print(aa)
