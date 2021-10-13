## Code is loosely based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import os
import logging
from pathlib import Path
import datetime
import torch
import numpy as np
from tqdm import tqdm
import model.pointtransformer_cls as pt_cls
from helper.ModelNetDataLoader import ModelNetDataLoader
from helper.optimizer import RangerVA
import helper.provider as provider

def train():

    def log_string(str):
        logger.info(str)
        print(str)

    ## Hyperparameters
    config = {'num_points' : 1024,
            'batch_size': 11,
            'use_normals': True,
            'optimizer': 'RangerVA',
            'lr': 0.001,
            'decay_rate': 1e-06,
            'epochs': 500,
            'num_classes': 40,
            'dropout': 0.4,
            'M': 4,
            'K': 64,
            'd_m': 512,
    }

    ## Create LogDir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)

    with open(str(experiment_dir) + "/config.txt", "w") as f:
        f.write(str(config))
        f.close()

    ## logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    file_handler = logging.FileHandler(f"{experiment_dir}/logs.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('Hyperparameters:')
    log_string(config)

    ## Create Dataloader
    # data_path = 'data/modelnet40_normal_resampled/'
    # train_ds = ModelNetDataLoader(root=data_path, npoint=config['num_points'], split='train', normal_channel=config['use_normals'])
    # test_ds = ModelNetDataLoader(root=data_path, npoint=config['num_points'], split='test', normal_channel=config['use_normals'])

    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    # test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    ## Create Point Transformer model
    model = pt_cls.Point_Transformer(config).cuda()
    # model = pt_cls.SortNet(128,6, top_k=64).cuda()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    from helper.summary import summary
    #summary(model, input_data=[(1, 128, 1024),(6, 1024)])
    summary(model, input_data=(6, 1024))
    # from pytictoc import TicToc

    # t = TicToc()

    # t.tic()
    # for i in range(100):
        
    #     a = torch.zeros(2, 1, 128, 1024).cuda()
    #     b = torch.zeros(2, 6, 1024).cuda()
    #     out = model(a, b)
    # t.toc()


    exit()
    #
    criterion = pt_cls.Loss().cuda()

    ## Create optimizer
    optimizer = None
    if config['optimizer'] == 'RangerVA':
        optimizer = RangerVA(model.parameters(), 
                            lr=config['lr'], 
                            weight_decay=config['decay_rate'])
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=config['decay_rate']
    )
    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []
            
    ## Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    for epoch in range(config['epochs']):
        log_string(f"Epoch {epoch}/{config['epochs']}")

        scheduler.step()

        for data in tqdm(train_dl, total=len(train_dl), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            model = model.train()
            pred = model(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string(f"Train Instance Accuracy: {train_instance_acc}")

        with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), test_dl, config)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            log_string(f"Test Instance Accuracy: {instance_acc}, Class Accuracy: {class_acc}")
            log_string(f"Best Instance Accuracy: {best_instance_acc}, Class Accuracy: {best_class_acc}")

            if (instance_acc >= best_instance_acc):
                log_string('Save model...')
                savepath = str(experiment_dir) + '/best_model.pth'
                log_string(f"Saving at {savepath}")
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            global_epoch += 1
    
    


def test(model, loader, config):
    mean_correct = []
    class_acc = np.zeros((config['num_classes'],3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


if __name__ == '__main__':
    train()
