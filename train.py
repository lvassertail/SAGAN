from data_loader import DataLoader
import torch
import torch.optim as optim
import os
from models.generator import *
from models.discriminator import *
from evaluator import Inception
from parameter import get_parameters
from trainer import Trainer
from utils import CheckpointData


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loader
    data_loader = DataLoader(config.dataset, config.image_path, config.im_size, config.batch_size, config.im_center_corp, shuf=True)
    dl, n_classes = data_loader.load()

    samples_dir = make_folder(config.sample_path, config.version)
    checkpoint_dir = make_folder(config.model_save_path, config.version)

    gen, dis, gen_optimizer, dis_optimizer, checkpoint_data = \
        create_model(checkpoint_dir, n_classes, device)

    evaluator = Inception(5000, 100, 1, device)

    labeled = True if n_classes > 0 else False
    trainer = Trainer(dl, gen, dis, gen_optimizer, dis_optimizer, config.num_epochs,
                      evaluator, checkpoint_dir, samples_dir, config.model_save_epoch,
                      config.calc_score_step, config.sample_save_step, config.version,
                      device, checkpoint_data, labeled)

    trainer.train()


def create_model(checkpoint_dir, n_classes, device):

    model_type = config.model

    gen_lr = dis_lr = config.baseline_lr
    if model_type == 'baseline':

        if config.im_size == 32:
            gen = BaselineGenerator32(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
            dis = SNProjectionDiscriminator32(ch=config.d_ch, n_classes=n_classes).to(device)
        else:
            gen = BaselineGenerator(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
            dis = SNProjectionDiscriminator(ch=config.d_ch, n_classes=n_classes).to(device)

    elif model_type == 'sn_on_g_d':

        if config.im_size == 32:
            gen = SNGenerator32(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
            dis = SNProjectionDiscriminator32(ch=config.d_ch, n_classes=n_classes).to(device)
        else:
            gen = SNGenerator(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
            dis = SNProjectionDiscriminator(ch=config.d_ch, n_classes=n_classes).to(device)
    else:
        gen_lr = config.ttur_gen_lr
        dis_lr = config.ttur_dis_lr

        if model_type == 'sn_on_g_d_ttur':

            if config.im_size == 32:
                gen = SNGenerator32(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
                dis = SNProjectionDiscriminator32(ch=config.d_ch, n_classes=n_classes).to(device)
            else:
                gen = SNGenerator(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
                dis = SNProjectionDiscriminator(ch=config.d_ch, n_classes=n_classes).to(device)

        elif model_type == 'sagan':
            if config.im_size == 32:
                gen = SaganGenerator32(feat_k=config.feat_k, ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
                dis = SaganDiscriminator32(feat_k=config.feat_k, ch=config.d_ch, n_classes=n_classes).to(device)
            else:
                gen = SaganGenerator(feat_k=config.feat_k, ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
                dis = SaganDiscriminator(feat_k=config.feat_k, imsize=config.im_size, ch=config.d_ch, n_classes=n_classes).to(device)

    # optimizers
    gen_optimizer = create_optimizer(gen.parameters(), gen_lr)
    dis_optimizer = create_optimizer(dis.parameters(), dis_lr)

    # Load Checkpoint
    checkpoint_data = CheckpointData()

    checkpoint_file_final = os.path.join(checkpoint_dir, config.final_checkpoint_name)
    if (config.load_checkpoint and os.path.isfile(checkpoint_file_final)):
        print('*** Loading final checkpoint file %s' % config.final_checkpoint_name)

        checkpoint = torch.load(checkpoint_file_final, map_location=device)

        gen.load_state_dict(checkpoint['gen_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        dis.load_state_dict(checkpoint['dis_state_dict'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])

        checkpoint_data.start_from_epoch = checkpoint['epoch']
        checkpoint_data.start_from_step = checkpoint['step']
        checkpoint_data.prev_scores_steps = checkpoint['saved_scores_steps']
        checkpoint_data.prev_scores = checkpoint['scores']

    return gen, dis, gen_optimizer, dis_optimizer, checkpoint_data


def make_folder(path, version):
    dir = os.path.join(path, version)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def create_optimizer(model_params, opt_lr):
    optimizer_type = config.optimizer_type
    opt_params=dict(
        lr = opt_lr,
        weight_decay = config.weight_decay,
        betas = (config.beta1, config.beta2)
    )
    return optim.__dict__[optimizer_type](model_params, **opt_params)


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    train()