from __future__ import print_function
import argparse
import sys
import time
import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from Data import TrainDatasetDataLoader
from models import create_model
from data_manager import *
from config import Config
from util.visualizer import Visualizer

config = Config().parser()                              #get training config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
np.random.seed(0)

dataset = config.dataset

if dataset == 'sysu':
    data_path = './Dataset/SYSU-MM01/'
    log_path = config.log_path + 'sysu_log/'
    test_mode = [1, 2] # thermal to visible

elif dataset =='regdb':
    data_path = './Dataset/RegDB/'
    log_path = config.log_path + 'regdb_log/'
    test_mode = [2, 1] # visible to thermal

checkpoint_path = config.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)




if __name__ == '__main__':
    config = Config.parse()
    data_loader = TrainDatasetDataLoader(config=config)
    dataset_size = len(data_loader)
    print('The number of training images  = %d'%dataset_size)

    model = create_model(config=config)
    model.setup(config)
    visualizer = Visualizer(config)
    total_iters = 0

    for epoch in range(config.epoch_count,config.niter + config.niter_decay+1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        for i ,data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % config.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % config.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % config.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % config.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if config.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % config.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if config.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % config.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, config.niter + config.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()







































