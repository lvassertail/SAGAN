import os
import sys
import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import *

class Trainer():
    def __init__(self, data_loader, gen, dis, gen_optimizer, dis_optimizer, num_epochs,
                 evaluator, checkpoint_dir, samples_dir, model_save_epoch,
                 model_calc_score, sample_save_epoch, version, device, checkpoint_data):

        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size

        self.gen = gen
        self.dis = dis
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer

        self.num_epochs = num_epochs
        self.evaluator = evaluator

        self.model_save_epoch = model_save_epoch
        self.model_calc_score = model_calc_score
        self.sample_save_epoch = sample_save_epoch
        self.checkpoint_dir = checkpoint_dir
        self.samples_dir = samples_dir

        self.device = device
        self.version = version

        self.fixed_noise, self.fixed_y = self.sample_noises()
        self.n_sample_row = max(int(math.sqrt(self.batch_size)), 1)

        self.start_from_epoch=checkpoint_data.start_from_epoch
        self.start_from_step=checkpoint_data.start_from_step
        self.prev_scores_steps=checkpoint_data.prev_scores_steps
        self.prev_scores=checkpoint_data.prev_scores

    def train(self):

        self.gen.train()
        self.dis.train()

        step_idx = self.start_from_step

        saved_scores_steps = []
        saved_scores_steps.extend(self.prev_scores_steps)
        scores = []
        scores.extend(self.prev_scores)

        for epoch_idx in range(self.start_from_epoch, self.num_epochs + self.start_from_epoch):
            # We'll accumulate batch losses and show an average once per epoch.
            dsc_losses = []
            gen_losses = []
            print(f'--- EPOCH {epoch_idx+1}/{self.num_epochs + self.start_from_epoch} ---')

            with tqdm.tqdm(total=len(self.data_loader.batch_sampler), file=sys.stdout) as pbar:
                for batch_idx, (x_real, y_real) in enumerate(self.data_loader):
                    step_idx += 1
                    x_real = x_real.to(self.device)
                    y_real = y_real.to(self.device)
                    dis_loss, gen_loss = self.train_batch(x_real, y_real)
                    dsc_losses.append(dis_loss)
                    gen_losses.append(gen_loss)
                    pbar.update()

            dsc_avg_loss, gen_avg_loss = np.mean(dsc_losses), np.mean(gen_losses)
            print(f'Discriminator loss: {dsc_avg_loss}')
            print(f'Generator loss:     {gen_avg_loss}')

            self.gen.eval()

            if (epoch_idx - self.start_from_epoch + 1) % self.model_calc_score == 0:

                score, _ = self.evaluator.eval_gen(self.gen)
                print("[%d] evaluated inception score: %.4f" % (epoch_idx + 1, score))

                scores.append(score)
                saved_scores_steps.append(step_idx)

            if (epoch_idx - self.start_from_epoch + 1) % self.model_save_epoch == 0:
                torch.save({'epoch': epoch_idx + 1,
                            'step': step_idx,
                            'saved_scores_steps': saved_scores_steps,
                            'scores': scores,
                            'gen_state_dict': self.gen.state_dict(),
                            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
                            'dis_state_dict': self.dis.state_dict(),
                            'dis_optimizer_state_dict': self.dis_optimizer.state_dict()},
                        os.path.join(self.checkpoint_dir,
                                    '{}_{}.pt'.format(self.version, epoch_idx + 1)))

                save_scores_plot(self.checkpoint_dir, self.version, saved_scores_steps , scores, epoch_idx + 1)
                print("[%d] Checkpoint and plot were saved" % (epoch_idx + 1))

            if (epoch_idx - self.start_from_epoch + 1) % self.sample_save_epoch == 0:

                fake = self.gen_samples()
                save_image(fake, os.path.join(self.samples_dir, '{}_fake.png'.format(epoch_idx + 1)),
                            nrow=self.n_sample_row, padding=2)

                print("[%d] Sample image was saved" % (epoch_idx + 1))

            self.gen.train()


    def gen_samples(self):
        with torch.no_grad():
            fake = self.gen(self.fixed_noise, self.fixed_y).detach().cpu() * .5 + .5
        return fake

    def sample_noises(self):
        fake_noise = torch.randn(self.batch_size, self.gen.dim_z, dtype=torch.float, device=self.device)
        fake_y = torch.randint(low=0, high=self.gen.n_classes, size=(self.batch_size,), dtype=torch.long,
                               device=self.device)

        return fake_noise, fake_y

    def train_batch(self, x_real, y_real):

        self.gen_optimizer.zero_grad()

        z_fake, y_fake = self.sample_noises()
        x_fake = self.gen(z_fake, y_fake)
        dis_fake = self.dis(x_fake, y=y_fake)
        loss_gen = self.generator_loss_fn(dis_fake)

        loss_gen.backward()
        self.gen_optimizer.step()

        z_fake, y_fake = self.sample_noises()
        self.dis.zero_grad()

        with torch.no_grad():
            x_fake = self.gen(z_fake, y_fake).detach()

        dis_real = self.dis(x_real, y=y_real)
        dis_fake = self.dis(x_fake, y=y_fake)
        loss_dis = self.discriminator_loss_fn(dis_real, dis_fake)

        loss_dis.backward()
        self.dis_optimizer.step()

        return loss_dis.item(), loss_gen.item()

    @staticmethod
    def discriminator_loss_fn(dis_real, dis_fake):
        d_loss_real = F.relu(1.0 - dis_real).mean()
        d_loss_fake = F.relu(1.0 + dis_fake).mean()

        return d_loss_real + d_loss_fake

    @staticmethod
    def generator_loss_fn(g_out_fake):
        g_loss_fake = -g_out_fake.mean()

        return g_loss_fake