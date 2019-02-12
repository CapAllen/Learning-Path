import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

parser = argparse.ArgumentParser(description='This is an Image Classifier Application')

parser.add_argument("data_directory", help="Your data directory")
parser.add_argument('--save_dir', help='Set directory to save checkpoints,Defaults to /checkpoint',
                    default='/checkpoint')
parser.add_argument('--arch', help='Choose architecture:DenseNet121(Default) or vgg13 ',
                    default='densenet121', choices=['densenet121', 'vgg13'])
parser.add_argument('--learning_rate', type=float, help='Set learning rate of model, Defaults to 0.002', default=0.002)
parser.add_argument('--hidden_units', type=int, help='Set hidden units of model', default=512)
parser.add_argument('--epochs', type=int, help='Set epochs, Defaults to 1', default=1)
parser.add_argument('--gpu', action='store_true', help='Use GPU to train your model or not,Defaults to True',
                    default=True)

args = parser.parse_args()
data_dir = args.data_directory
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
use_gpu = args.gpu

print('- Selected Model Options -')
print('Data Directory:            {}'.format(data_dir))
print('Save Checkpoints Directory:{}'.format(data_dir + save_dir))
print('Architecture:              {}'.format(arch))
print('Learning Rate:             {}'.format(lr))
print('Hidden Units:              {}'.format(hidden_units))
print('Epochs:                    {}'.format(epochs))
print('Use GPU:                   {}'.format(use_gpu))
print('-' * 40)

# load data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)

#build and train network
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# build model
def build_model(arch, hidden_units):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))]))
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))]))
    else:
        print('Sorry,Please try other architecture:densenet121 or vgg13.')

    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier
    return model


# train model
def train_model(model, device, learning_rate, epochs, train_loader, validation_loader):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 10

    for e in range(epochs):
        for images, labels in train_loader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(images)
            train_loss = criterion(log_ps, labels)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

            if steps % print_every == 0:
                vali_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        batch_loss = criterion(log_ps, labels)

                        vali_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {vali_loss / len(validation_loader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(validation_loader):.3f}")

                running_loss = 0
                model.train()


def test_model(model, device, test_loader):
    model.eval()

    model.to(device)
    criterion = nn.NLLLoss()
    test_loss, test_accuracy = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        test_loss += criterion(outputs, labels).item()
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    loss = round(test_loss / len(test_loader), 3)
    accuracy = round(test_accuracy / len(test_loader), 3)
    print('Test loss: {}'.format(loss))
    print('Test accuracy: {}'.format(accuracy))

    return loss, accuracy

#save checkpoints
def save_checkpoints(arch,model,hidden_units,learning_rate,epochs, save_dir):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    checkpoint = {'arch': arch,
              'hidden_units': hidden_units,
              'learning_rate': learning_rate,
              'epochs':epochs,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': train_dataset.class_to_idx}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint,save_dir+'/{}_checkpoint.pth'.format(arch))
    print('Checkpoints has been saved in {}'.format(save_dir+'/{}_checkpoint.pth'.format(arch)))


if __name__ == '__main__':
    model = build_model(arch,hidden_units)
    train_model(model,device,lr,epochs,train_loader,validation_loader)
    test_model(model,device,test_loader)
    save_checkpoints(arch,model,hidden_units,lr,epochs)
