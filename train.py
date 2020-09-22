from data_loader import DataLoader
import torch
import torch.optim as optim
import os
from models.generator import BaselineGenerator
from models.generator import SNGenerator
from models.discriminator import SNProjectionDiscriminator
from evaluator import Inception
from parameter import get_parameters
from trainer import Trainer


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loader
    data_loader = DataLoader(config.dataset, config.image_path, config.im_size, config.batch_size, shuf=True)
    dl = data_loader.load()
    n_classes = len(dl.dataset.classes)

    samples_dir = make_folder(config.sample_path, config.version)
    checkpoint_dir = make_folder(config.model_save_path, config.version)

    gen, dis, gen_optimizer, dis_optimizer = create_model(checkpoint_dir, n_classes, device)

    evaluator = Inception(5000, 100, 1, device)

    trainer = Trainer(dl, gen, dis, gen_optimizer, dis_optimizer, config.num_epochs,
                      evaluator, checkpoint_dir, samples_dir, config.model_save_epoch,
                      config.model_calc_score, config.sample_save_epoch, device)

    trainer.train()


def create_model(checkpoint_dir, n_classes, device):

    # Checkpoint
    checkpoint_file_final_G = os.path.join(checkpoint_dir, 'gan_final_G.pt')
    checkpoint_file_final_D = os.path.join(checkpoint_dir, 'gan_final_D.pt')

    # Model
    if os.path.isfile(checkpoint_file_final_G) and os.path.isfile(checkpoint_file_final_D):
        print('*** Loading final checkpoint file instead of training')
        gen = torch.load(checkpoint_file_final_G, map_location=device)
        dis = torch.load(checkpoint_file_final_D, map_location=device)
    else:
        model_type = config.model

        if model_type == 'baseline':
            gen = BaselineGenerator(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
            dis = SNProjectionDiscriminator(ch=config.d_ch, n_classes=n_classes).to(device)
        elif model_type == 'sn_on_g_d':
            gen = SNGenerator(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
            dis = SNProjectionDiscriminator(ch=config.d_ch, n_classes=n_classes).to(device)
        else:
            gen_lr = config.ttur_gen_lr
            dis_lr = config.ttur_dis_lr

            if model_type == 'sn_on_g_d_ttur':
                gen = SNGenerator(ch=config.g_ch, dim_z=config.z_dim, n_classes=n_classes).to(device)
                dis = SNProjectionDiscriminator(ch=config.d_ch, n_classes=n_classes).to(device)

    # optimizers
    gen_optimizer = create_optimizer(gen.parameters(), gen_lr)
    dis_optimizer = create_optimizer(dis.parameters(), dis_lr)

    return gen, dis, gen_optimizer, dis_optimizer


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