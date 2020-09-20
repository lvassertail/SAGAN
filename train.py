from data_loader import Data_Loader
from models.sagan import *
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
from torch.optim import Optimizer
from torchvision.utils import save_image
from models.generator import ResNetGenerator
from models.discriminator import SNResNetProjectionDiscriminator
from evaluator import Inception


def train(hp):

    # Show hypers
    print(hp)

    num_epochs = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = hp['batch_size']
    z_dim = hp['z_dim']
    im_size = hp['im_size']

    # Data loader
    data_loader = Data_Loader('cifar', './data', im_size, batch_size, shuf=True)
    dl, n_classes = data_loader.loader()

    samples_dir = make_folder('samples', hp['version'])

    # Checkpoint
    checkpoint_dir = make_folder('checkpoints', hp['version'])
    checkpoint_file_final_G = os.path.join(checkpoint_dir, 'gan_final_G.pt')
    checkpoint_file_final_D = os.path.join(checkpoint_dir, 'gan_final_D.pt')
    #if os.path.isfile(f'{checkpoint_file}.pt'):
    #    os.remove(f'{checkpoint_file}.pt')

    # Model
    if os.path.isfile(checkpoint_file_final_G) and os.path.isfile(checkpoint_file_final_D):
        print('*** Loading final checkpoint file instead of training')
        gen = torch.load(checkpoint_file_final_G, map_location=device)
        dsc = torch.load(checkpoint_file_final_D, map_location=device)
    else:
        #gen = Generator(im_size, z_dim).to(device)
        #dsc = Discriminator(im_size).to(device)
        gen = ResNetGenerator(ch=im_size, dim_z=z_dim, n_classes=n_classes, bottom_width=4).to(device)
        dsc = SNResNetProjectionDiscriminator(ch=im_size, n_classes=n_classes).to(device)

    # optimizers
    gen_optimizer = create_optimizer(gen.parameters(), hp['generator_optimizer'])
    dsc_optimizer = create_optimizer(dsc.parameters(), hp['discriminator_optimizer'])

    # optimizers - sagan
    #gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen.parameters()), self.g_lr, [self.beta1, self.beta2])
    #dsc_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dsc.parameters()), self.d_lr, [self.beta1, self.beta2])

    #ins = InceptionScore('inception_v3_google-1a9a5a14.pth')
    evaluator = Inception(5000, 100, 1, device)
    for epoch_idx in range(num_epochs):
        # We'll accumulate batch losses and show an average once per epoch.
        dsc_losses = []
        gen_losses = []
        print(f'--- EPOCH {epoch_idx+1}/{num_epochs} ---')

        with tqdm.tqdm(total=len(dl.batch_sampler), file=sys.stdout) as pbar:
            for batch_idx, (x_real, y_real) in enumerate(dl):
                x_real = x_real.to(device)
                y_real = y_real.to(device)
                dsc_loss, gen_loss = train_batch(
                    dsc, gen,
                    dsc_optimizer, gen_optimizer,
                    x_real, y_real)
                dsc_losses.append(dsc_loss)
                gen_losses.append(gen_loss)
                pbar.update()

        dsc_avg_loss, gen_avg_loss = np.mean(dsc_losses), np.mean(gen_losses)
        print(f'Discriminator loss: {dsc_avg_loss}')
        print(f'Generator loss:     {gen_avg_loss}')

        if (epoch_idx+1) % hp["model_save_epoch"]==0:
            torch.save(gen.state_dict(),
                       os.path.join(checkpoint_dir, 'gan{}_G.pt'.format(epoch_idx + 1)))
            torch.save(dsc.state_dict(),
                       os.path.join(checkpoint_dir, 'gan{}_D.pt'.format(epoch_idx + 1)))
        if (epoch_idx+1) % hp["model_calc_score"]==0:
            gen.eval()
            # Sample images
            #fake_images,_,_= gen(fixed_z)
            #fake_images = gen(fixed_z)
            fake_images_x, fake_images_y = gen.sample(batch_size, with_grad=False)
            #fake_images_x = gen.sample(batch_size, with_grad=False)
            save_image(denorm(fake_images_x.data),
                        os.path.join(samples_dir, '{}_fake.png'.format(epoch_idx + 1)))
            #inception_score = ins.calculate(fake_images_x, resize=True)
            #print(f'the generated fake images were saved. the Inception Score is: {inception_score}')

            score, _ = evaluator.eval_gen(gen)
            print("[%d] evaluated inception score: %.4f" % (epoch_idx + 1, score))
            gen.train()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def make_folder(path, version):
    dir = os.path.join(path, version)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def train_batch_gan(dsc_model: Discriminator, gen_model: Generator,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: torch.utils.data.DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    sample = gen_model.sample(x_data.shape[0], with_grad=True)

    real = dsc_model(x_data)
    fake = dsc_model(sample)

    dsc_loss = discriminator_loss_fn(real[0], fake[0])

    dsc_loss.backward(retain_graph=True)
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    gen_loss = generator_loss_fn(dsc_model(sample)[0])

    # train the weights using the optimizer
    gen_loss.backward(retain_graph=True)
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()

def train_batch_resnet_old(dsc_model, gen_model,
                dsc_optimizer, gen_optimizer,
                x_real, y_real):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # Discriminator update
    dsc_optimizer.zero_grad()
    batch_size = x_real.shape[0]
    x_fake, y_fake = gen_model.sample(batch_size, with_grad=True)

    dsc_real = dsc_model(x_real, y_real)
    dsc_fake = dsc_model(x_fake, y_fake)

    dsc_loss = discriminator_loss_fn(dsc_real, dsc_fake)

    dsc_loss.backward(retain_graph=True)
    dsc_optimizer.step()

    # Generator update
    gen_optimizer.zero_grad()

    dsc_fake = dsc_model(x_fake, y_fake)
    gen_loss = generator_loss_fn(dsc_fake)

    # train the weights using the optimizer
    gen_loss.backward(retain_graph=True)
    gen_optimizer.step()

    return dsc_loss.item(), gen_loss.item()

def train_batch(dis, gen,
                opt_dis, opt_gen,
                x_real, y_real):
    batchsize = x_real.size(0)
    device = torch.device('cuda')

    z_fake = torch.randn(batchsize, gen.dim_z, dtype=torch.float, device=device)
    y_fake = torch.randint(0, gen.n_classes, (batchsize,), device=device, dtype=torch.long)
    x_fake = gen(batchsize, y=y_fake, z=z_fake)
    dis_fake = dis(x_fake, y=y_fake)
    loss_gen = generator_loss_fn(dis_fake)
    opt_gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    y_fake = torch.randint(0, gen.n_classes, (batchsize,), device=device, dtype=torch.long)
    with torch.no_grad():
        x_fake = gen(batchsize, y=y_fake).detach()

    dis_real = dis(x_real, y=y_real)
    dis_fake = dis(x_fake, y=y_fake)
    loss_dis = discriminator_loss_fn(dis_real, dis_fake)
    opt_dis.zero_grad()
    loss_dis.backward()
    opt_dis.step()

    return loss_dis.item(), loss_gen.item()

# Optimizer
def create_optimizer(model_params, opt_params):
    opt_params = opt_params.copy()
    optimizer_type = opt_params['type']
    opt_params.pop('type')
    return optim.__dict__[optimizer_type](model_params, **opt_params)


# Loss
#def dsc_loss_fn(y_data, y_generated):
#    return discriminator_loss_fn(y_data, y_generated, hp['data_label'], hp['label_noise'])

def discriminator_loss_fn(dsc_real, dsc_fake):
    d_loss_real = F.relu(1.0 - dsc_real).mean()
    d_loss_fake = F.relu(1.0 + dsc_fake).mean()

    return d_loss_real + d_loss_fake

def generator_loss_fn(g_out_fake):
    g_loss_fake = -g_out_fake.mean()

    return g_loss_fake

def gen_loss_fn(y_generated):
    return generator_loss_fn(y_generated, hp['data_label'])

# TODO: Change to parameters
def gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['version'] = 'resnet_cifar64'
    hypers["batch_size"] = 64
    hypers["im_size"] = 64
    hypers["model_save_epoch"] = 10
    hypers["model_calc_score"] = 1
    hypers["z_dim"] = 128
    hypers["data_label"] = 1
    hypers["label_noise"] = 0.3

    #gan
    #hypers["discriminator_optimizer"]["type"] = hypers["generator_optimizer"]["type"] = 'Adam'
    #hypers["discriminator_optimizer"]["lr"] = hypers["generator_optimizer"]["lr"] = 0.0003
    #hypers["discriminator_optimizer"]["weight_decay"] = hypers["generator_optimizer"]["weight_decay"] = 0.02
    #hypers["discriminator_optimizer"]["betas"] = hypers["generator_optimizer"]["betas"] = (0.5, 0.999)

    #sagan
    hypers["discriminator_optimizer"]["type"] = hypers["generator_optimizer"]["type"] = 'Adam'
    hypers["discriminator_optimizer"]["lr"] = 0.0002  # TTUR - a slower update rule is used for the generator
    hypers["generator_optimizer"]["lr"] = 0.0002  # TTUR - a faster update rule is used for the discriminator
    hypers["discriminator_optimizer"]["weight_decay"] = hypers["generator_optimizer"]["weight_decay"] = 0
    hypers["discriminator_optimizer"]["betas"] = hypers["generator_optimizer"]["betas"] = (0.0, 0.9)

    # ========================
    return hypers

if __name__ == '__main__':
    #config = get_parameters()
    hp = gan_hyperparams()
    train(hp)