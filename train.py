from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import time
import copy
import wandb
import os

from dataset import ImageDataset
from config import data_transforms, batch_size
from utils import parse_args
from model import model

cudnn.benchmark = True
wandb.login()

def train_val_dataset(dataset, val_split=0.15):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def main():
    args = parse_args()
    
    datadir = args.dataset
    # devided_data = {}
    devided_data = {x: datasets.ImageFolder(os.path.join(datadir,x), data_transforms)
                      for x in ['train', 'test']}
    # devided_data['train'] = datasets.ImageFolder(os.path.join(datadir,'train'), data_transforms)
    pills_class_to_idx = devided_data['train'].class_to_idx
    test_classes = devided_data['test'].classes
    # devided_data['test'] = datasets.ImageFolder(os.path.join(datadir,'train'), data_transforms, target_transform=pills_class_to_idx)
    # devided_data = ImageDataset(args)
    # devided_data = train_val_dataset(image_datasets)
    # print(len(devided_data['train']))
    # print(len(devided_data['test']))
    # print(devided_data['train'].classes)
  
    dataloaders = {x: torch.utils.data.DataLoader(devided_data[x], batch_size=batch_size,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'test']
                }

    dataset_sizes = {x:len(devided_data[x])
                    for x in ['train', 'test']}
    # print(dataset_sizes)
    # return



    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        wandb.init(
            project='Pill_classification',
            name=args.model + ('_pretrained_full_100epochs' if args.pretrained else '_not_pretrained_full_100epochs'),
            config={
                'epoch': num_epochs,
                'batch_size': batch_size,
                'architecture': args.model
            }
        )
               
        device = torch.device('cuda:0' if args.use_gpu else 'cpu')
        start_time = time.time()
        
        # best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    if phase == 'test':
                        map_labels = []
                        for label in labels:
                            map_label = pills_class_to_idx[test_classes[label]]
                            map_labels.append(map_label)
                            
                        labels = torch.Tensor(map_labels)
                        labels = labels.long()
                        # print(labels)
                        # labels.to(torch.int64)                     
                            
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # print(outputs)
                        _, preds = torch.max(outputs, 1)
                        # labels.to(torch.long)
                        # print(outputs, labels)            
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # print(labels.data)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                if phase == 'train':
                    train_metric = {
                        'train/train_loss': epoch_loss,
                        'train/train_accuracy': epoch_acc
                    }
                else:
                    val_metrics = {
                        'val/val_loss': epoch_loss,
                        'val/val_accuracy': epoch_acc
                    }
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            # name = './epoch{}acc{:.4f}.pth'.format(epoch, epoch_acc)
            # torch.save(model_ft, name)
            wandb.log({**train_metric, **val_metrics})
            print()
            
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        ))
        print('Best val Acc: {:4f}'.format(best_acc))
        
        wandb.summary['Best accuracy'] = best_acc
        wandb.finish()
        
        model.load_state_dict(best_model_wts)
        return model,best_acc

    model_ft = model(args)

    criterion = nn.CrossEntropyLoss()
    # # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    num_epochs = args.num_epochs
    model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs)
    
    if args.save_weights:
        # from datetime import datetime
        # now = datetime.now().strftime('_%Y%m%d%H%M%S')
        name = './weights/{}_{:.4f}.pth'.format(args.model, best_acc)
        torch.save(model_ft, name)
    # # train_model('', '', '', '')

if __name__ == '__main__':
    main()