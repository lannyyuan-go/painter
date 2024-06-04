import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import torch.nn.functional as F

def DiffAugment(x, policy='color,translation,cutout'):
    if policy:
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
    return x

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)
    return x

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + x_mean
    return x

def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, [x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, [x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device))
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.shape[2] + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.shape[3] + 1)
    x = F.pad(x, [1, 1, 1, 1])
    x = x.permute(0, 2, 3, 1)
    x = x[grid_batch, grid_x, grid_y]
    return x.permute(0, 3, 1, 2)

def rand_cutout(x, ratio=0.5):
    cutout_size = (torch.rand(x.size(0), device=x.device) * ratio).view(-1, 1, 1, 1)
    cutout_size = (cutout_size * torch.tensor(x.shape[2:]).to(x.device)).long()
    offset_x = torch.randint(0, x.shape[2] + (1 - cutout_size[:, 0]), [x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.shape[3] + (1 - cutout_size[:, 1]), [x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device))
    grid_x = torch.clamp(grid_x - offset_x, 0, x.shape[2] - cutout_size[:, 0])
    grid_y = torch.clamp(grid_y - offset_y, 0, x.shape[3] - cutout_size[:, 1])
    mask = torch.ones(x.shape, device=x.device)
    mask[grid_batch, :, grid_x, grid_y] = 0
    x = x * mask
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            torch.cuda.synchronize()
            optimize_start_time = time.time()
            data['A'] = DiffAugment(data['A'], policy='color,translation,cutout')
            data['B'] = DiffAugment(data['B'], policy='color,translation,cutout')
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize()
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
#             model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
