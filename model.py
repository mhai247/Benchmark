from torchvision import models, datasets
import torch
import torch.nn as nn
import os

def model(args):
    if args.weights is not None:
        model_ft = torch.load(args.weights)
        return model_ft
    
    assert args.model is not None, 'Require model'
    # print(args.pretrained)
    dict_model = {
        'r18': models.resnet18(pretrained=args.pretrained),
        'r34': models.resnet34(pretrained=args.pretrained),
        'r50': models.resnet50(pretrained=args.pretrained),
        'r101': models.resnet101(pretrained=args.pretrained),
        'r152': models.resnet152(pretrained=args.pretrained)        
    }
    model_ft = dict_model[args.model]
    
    if args.trainable == False:
        for param in model_ft.parameters():
            param.requires_grad = False
    
    image_dataset = datasets.ImageFolder(os.path.join(args.dataset, 'train'))
    num_classes = len(image_dataset.classes)
    # print(num_classes)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    model_ft = model_ft.to(device)
    
    return model_ft
    
    
    