import json
from PIL import Image
import argparse
import torch
import glob
import os
import csv
import time

from config import classes_file, data_transforms

def parse_args():
    def str2bool(v):
        return v.lower().rstrip() in ('true', '1', 't')
    args = argparse.ArgumentParser()
    args.add_argument('-w', '--weight', type=str, required=True,
                      help='Path to weight file')
    args.add_argument('-i', '--image', type=str,
                      help='Path to image file')
    args.add_argument('-f', '--folder', type=str,
                      help='Path to image folder')
    args.add_argument('-g', '--use_gpu', type=str2bool, default=True,
                      help='Use gpu or not')
    return args.parse_args()    

def main():
    fs = open(classes_file, 'r')
    pill_name = json.load(fs)
    args = parse_args()
    # print(args.use_gpu)
    
    assert bool(args.image) ^ bool(args.folder), 'Must be image or directory path as argument'
    
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    
    model = torch.load(args.weight,
                       map_location=device)
    
    model.eval()
    model.to(device)
    
    def predict(pil_img):
        transformed = data_transforms(pil_img)
        tensor = torch.unsqueeze(transformed, 0)
        # print(tensor.shape)
        tensor = tensor.to(device)
        # print(tensor)
        output = model(tensor)
        # print(output.shape)
        _, pred = torch.max(output, 1)
        return pill_name[pred[0]]

    start = time.time()
    
    if args.image is not None:
        img = Image.open(args.image).convert('RGB')
        name = predict(img)
        print(name)
        
    if args.folder is not None:
        folder_name = args.folder.split('/')[-1 if args.folder[-1] != '/'
                                             else -2]
        f = open(folder_name + '.csv', 'w')
        writer = csv.writer(f)
        
        files = glob.glob(os.path.join(args.folder, '*.jpg'))
        
        for file in files:
            img = Image.open(file).convert('RGB')
            name = predict(img)
            img_name = file.split('/')[-1]
            writer.writerow((img_name, name))
            
        print('Result saved in {}'.format(folder_name + '.csv'))
        
    time_elapsed = int(time.time() - start)
    print('Time: {}m {}s'.format(time_elapsed//60, time_elapsed%60))
    f.close()
    fs.close()

if __name__ == '__main__':
    main()