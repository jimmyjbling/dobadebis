import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

data_dir = '/home/jwellni/Extra_credit/dogs/'
train_dir = os.path.join(data_dir, 'train/')
val_dir = os.path.join(data_dir, 'val/')

train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(180),
                                       transforms.CenterCrop((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valTest_transforms = transforms.Compose([transforms.Resize(size=256),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=valTest_transforms)

trainLoader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)

valLoader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size=128,
                                        shuffle=True,
                                        num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Gamma = models.resnet18(pretrained=True)

out_size = Gamma.fc.in_features
Gamma.fc = nn.Identity(out_size, out_size)
Gamma = Gamma.to(device)

g = nn.Linear(out_size, 120, bias=True).to(device)

criterion = nn.CrossEntropyLoss()

params = list(g.parameters()) + list(Gamma.parameters())
optimizer = optim.SGD(params, lr=0.0005, momentum=0.85)


def train(n_epochs, loaders, model, optimizer, criterion, Gamma):
    since = time.time()
    for epoch in range(1, n_epochs + 1):
        Gamma.eval()
        g.train()
        running_loss = 0.0
        running_corrects = 0
        total_examples = 0
        val_loss = 0.0

        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = g(Gamma(data))
            loss = criterion(output, target)
            _, preds = torch.max(output, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data)
            total_examples += data.size(0)

            if batch_idx % 100 == 99:
                batch_loss = running_loss / total_examples
                batch_acc = running_corrects.double() / total_examples
                running_loss = 0.0
                running_corrects = 0
                total_examples = 0

                print(f'Avg Loss: {batch_loss:.4f} Avg Acc: {batch_acc:.4f}')
        g.eval()
        for batch_idx, (data, target) in enumerate(loaders['val']):
            data, target = data.to(device), target.to(device)
            output = g(Gamma(data))
            loss = criterion(output, target)
            val_loss += ((1 / (batch_idx + 1)) * (loss.data - val_loss))
        print("Epoch {a}: validation loss = {b}".format(a=epoch, b=val_loss))
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return g


loaders = {'train': trainLoader, 'val': valLoader}
model = train(14, loaders, g, optimizer, criterion, Gamma)

try:
    torch.save(model.state_dict(), "my_model2.nn")
except Exception:
    pass

class_names = train_dataset.classes


def predict_breed(model, Gamma, class_names, img_path):
    img = Image.open(img_path)

    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    img_tensor = preprocess(img).unsqueeze_(0)

    img_tensor = img_tensor.cuda()

    model.eval()

    with torch.no_grad():
        output = model(Gamma(img_tensor))
        prediction = torch.argmax(output).item()

    model.train()

    breed = class_names[prediction]

    return breed


predicts = {}
for image_path in os.listdir('/home/jwellni/Extra_credit/dogs/test/test/'):
    image_name = image_path[:-4]
    image_path = '/home/jwellni/Extra_credit/dogs/test/test/' + str(image_path)
    breed = predict_breed(model, Gamma, class_names, image_path)
    predicts[image_name] = breed

print(predicts)

import csv

with open('results2.csv', 'w') as f:
    for key in predicts.keys():
        f.write("%s,%s\n" % (key, predicts[key]))
    f.close()



