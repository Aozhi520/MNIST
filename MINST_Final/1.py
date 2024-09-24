
from torch import nn
from numpy import squeeze
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Flatten
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms, models, datasets
from torch.utils.tensorboard import SummaryWriter
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


root_dir='Numbers'
batch_size=2

train_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train'), data_transforms['train'])
test_dataset = datasets.ImageFolder(os.path.join(root_dir, 'test'), data_transforms['test'])
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class_names = train_dataset.classes

test_data_size=len(test_dataset)


class mymodule(nn.Module):

    def __init__(self):
        super(mymodule,self).__init__()
        self.model=Sequential(

            Conv2d(1,16,3,stride=1,padding=1),
            ReLU(),
            MaxPool2d(2),
            #14x14
            Conv2d(16,32,3,stride=1,padding=1),
            ReLU(),
            MaxPool2d(2),
            #7x7
            Conv2d(32, 64, 3, stride=1, padding=1),
            ReLU(),

            Flatten(),
            Linear(64*7*7,128),
            ReLU(),
            Linear(128,2),
        )

    def forward(self,input):
        output=self.model(input)
        return output



aozhi=mymodule()
aozhi.cuda()

loss_fun=nn.CrossEntropyLoss()
loss_fun.cuda()

learn_rate=0.0001
optim=torch.optim.Adam(aozhi.parameters(),learn_rate)
# optim=torch.optim.SGD(aozhi.parameters(),learn_rate)
# optim=torch.optim.Adagrad(aozhi.parameters(),learn_rate)

writer=SummaryWriter('Adam')

train_step=0
test_step=0
total_loss=0
epoch=40
for i in range(epoch):

    print('第{}次训练：'.format(i+1))

    for data in train_dataloader:
        img,label=data
        img=img.cuda()
        label=label.cuda()

        output=aozhi(img)
        loss=loss_fun(output,label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_step += 1
        if train_step%10==0:
            print('训练次数：{},loss:{}'.format(train_step,loss))



    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, label = data
            img = img.cuda()
            label = label.cuda()
            output = aozhi(img)
            loss = loss_fun(output, label)
            total_test_loss += loss

            accuracy = (output.argmax(1) == label).sum()
            total_accuracy += accuracy

            test_step+=1
            if test_step % 100 == 0:
                writer.add_scalar('accuracy',total_accuracy / test_data_size,test_step)

    print('整体测试集的LOSS：{}'.format(total_test_loss))

    print('整体测试集的准确率：{}'.format(total_accuracy / test_data_size))


torch.save(aozhi, 'aozhi_{}.model'.format(epoch))

writer.close()


#预测
class Mydata(Dataset):

    def __init__(self,root_dir,transform=None):
         self.img_path=[os.path.join(root_dir,f) for f in os.listdir(root_dir)]
         self.transform=transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img=Image.open(self.img_path[idx]).convert('L')
        img_tensor=self.transform(img)
        return img_tensor

transform_tensor=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(0.485, 0.226)
])


root_dir= 'Numbers/pre'
dataset=Mydata(root_dir,transform_tensor)

dataloader=DataLoader(dataset,batch_size=1,shuffle=False)

model=torch.load('aozhi_40.model')

predictions = []
with torch.no_grad():
    for data in dataloader:
        img=data
        img=img.cuda()
        output=model(img)
        predict=output.argmax(1)

        preds=predict.cpu().numpy()
        predictions.extend(preds)
    predicted_class_names = [class_names[pred] for pred in predictions]
    print(predicted_class_names)











