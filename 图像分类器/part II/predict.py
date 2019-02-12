import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
from PIL import Image
import json

parser = argparse.ArgumentParser(description='This is an Image Predictor Application')

parser.add_argument('image_path',help='Path to image.')
parser.add_argument('checkpoint_path', help='Path to checkpoint.')
parser.add_argument('--topk',type=int,help='Return top k most likely classes,Defaults to 1',default=1)
parser.add_argument('--category_names',help='Map of categories to real names,Defaults to cat_to_name.json',
                    default=cat_to_name.json)
parser.add_argument('--gpu', action='store_true', help='Use GPU to predict or not,Defaults to True',
                    default=True)

args = parser.parse_args()
image_path = args.image_path
checkpoint_path = args.checkpoint_path
topk = args.topk
category_names = args.category_names
use_gpu = args.gpu

print('- Selected Predict Options -')
print('Path to image:     {}'.format(data_dir))
print('Path to checkpoint:{}'.format(data_dir + save_dir))
print('Top K:             {}'.format(arch))
print('Category Names:    {}'.format(lr))
print('Use GPU:           {}'.format(use_gpu))
print('-' * 40)

#gpu
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#load names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
    else:
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier

    model.class_to_idx = class_to_idx
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #open image
    im = Image.open(image)
    width,height = im.size
    whr = width/height

    if width > height:
        height = 256
        width = int(height * whr)
    if width < height:
        width = 256
        height = int(width * whr)
    im = im.resize((width,height))

    left = int((width-224)/2)
    upper = int((height-224)/2)
    right = left + 224
    lower = upper + 224
    im = im.crop((left,upper,right,lower))

    im_array = np.array(im)/255
    im_mean = np.array([0.485, 0.456, 0.406])
    im_std = np.array([0.229, 0.224, 0.225])
    im_array = (im_array - im_mean)/im_std
    im_array = im_array.transpose(2,0,1)
    return torch.from_numpy(im_array)


def predict(image_path, model,device,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device)
    image = process_image(image_path)
    image = image.to(device)
    image = image.float().unsqueeze(0)
    output = model.forward(image)
    topk_ps = torch.exp(output).data.topk(topk)
    ps = topk_ps[0].cpu().numpy()[0]
    idx = topk_ps[1].cpu().numpy()[0]
    name_dict = {model.class_to_idx[i]: i for i in model.class_to_idx}
    classes = []
    for index in idx:
        classes.append(name_dict[index])

    return list(ps), classes

if __name__ == '__main__':
    model = load_checkpoint(checkpoint_path)
    ps,classes = predict(image_path,model,device,topk)
    for i in range(len(ps)):
        print('Probability of image name {} is {}.'.format(classes[i],ps[i]))

