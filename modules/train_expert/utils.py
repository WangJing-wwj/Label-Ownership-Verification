import torch

from modules.base_utils.util import generate_full_path


def checkpoint_callback(model, opt, epoch, iteration, expert_id,save_iter, output_dir):
    '''Saves model and optimizer state dicts at fixed intervals.'''
    if iteration % save_iter == 0 and iteration != 0:
        checkpoint_path = f'{output_dir}model_id_{str(expert_id)}_ep_{str(epoch)}_iter_{str(iteration)}.pth'
        opt_path = f'{output_dir}model_id_{str(expert_id)}_ep_{str(epoch)}_iter_{str(iteration)}_opt.pth'
        torch.save(model.state_dict(), generate_full_path(checkpoint_path))
        torch.save(opt.state_dict(), generate_full_path(opt_path))

def checkpoint_clean_model_callback(model, epoch, expert_id, acc,output_dir):
        if epoch in  list(range(50, 301, 10)):
            checkpoint_path = f'{output_dir}clean_model_id_{str(expert_id)}_ep_{str(epoch)}_acc_{acc}.pth'
            torch.save(model.state_dict(), generate_full_path(checkpoint_path))