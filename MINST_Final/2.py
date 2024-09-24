import torchvision
from torch import nn
from numpy import squeeze
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Flatten
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms, models, datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch
from torch import nn

data_transforms = {
    'train': transforms.Compose([transforms.Grayscale(),
                                 transforms.RandomRotation(45),
                                 transforms.Resize(28),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.485, 0.226)
                                 ]),
    'test': transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(28),
                                transforms.ToTensor(),
                                transforms.Normalize(0.485, 0.226)
                                 ]),
}


root_dir=''
train_dataset=torchvision.datasets.MNIST(root_dir,train=True,transform=data_transforms['train'],
                                      download=True)
test_dataset=torchvision.datasets.MNIST(root_dir,train=False,transform=data_transforms['test'],
                                      download=True)

test_size=len(test_dataset)

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=False)

class_names = train_dataset.classes


class mymodule(nn.Module):

    def __init__(self):
        super(mymodule,self).__init__()

        self.model=Sequential(
            Conv2d(1,16,3,stride=1,padding=1),
            ReLU(),
            MaxPool2d(2),
            # 14x14
            Conv2d(16, 32, 3, 1, 1),
            ReLU(),
            MaxPool2d(2),
            # 7x7
            Conv2d(32, 64, 3, 1, 1),
            ReLU(),

            Flatten(),
            Linear(64*7*7,128),
            ReLU(),
            Linear(128, 10)
        )

    def forward(self,input):
        output=self.model(input)
        return output

aozhi=mymodule()
aozhi.cuda()

loss_fun=nn.CrossEntropyLoss()
loss_fun.cuda()

learn_rate=0.001
optim=torch.optim.Adam(aozhi.parameters(),learn_rate)


train_step=0
epoch=20

for i in range(epoch):
    print('第{}次训练：'.format(i+1))
    total_loss=0
    for data in train_dataloader:
        img,label=data
        img=img.cuda()
        label=label.cuda()

        output=aozhi(img)
        loss=loss_fun(output,label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss+=loss
        train_step+=1
        if train_step % 100 == 0:
            print('训练次数：{},loss：{}'.format(train_step,loss))

    total_test_loss=0
    toatal_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            img, label = data
            img = img.cuda()
            label = label.cuda()

            output = aozhi(img)
            loss = loss_fun(output, label)

            total_test_loss+=loss
            accuracy=(output.argmax(1)==label).sum()
            toatal_accuracy+=accuracy

    print('整体测试集loss:{}'.format(total_test_loss))
    print('整体测试集准确率:{}'.format(toatal_accuracy / test_size))

torch.save(aozhi,'aoozhi_{}.modoel'.format(epoch))












































