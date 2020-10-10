import os
import sys
import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class Trainer():

    def __init__(self, data_loader, gen, dis, gen_optimizer, dis_optimizer, num_epochs,
                 evaluator, checkpoint_dir, samples_dir, model_save_epoch,
                 calc_score_step, sample_save_step, version, device, checkpoint_data, labeled):

        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size

        self.gen = gen
        self.dis = dis
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer

        self.num_epochs = num_epochs
        self.evaluator = evaluator

        self.model_save_epoch = model_save_epoch
        self.calc_score_step = calc_score_step
        self.sample_save_step = sample_save_step
        self.checkpoint_dir = checkpoint_dir
        self.samples_dir = samples_dir

        self.device = device
        self.version = version

        self.labeled = labeled
        self.fixed_noise, self.fixed_y = self.sample_noises()
        self.n_sample_row = max(int(math.sqrt(self.batch_size)), 1)

        self.start_from_epoch=checkpoint_data.start_from_epoch
        self.start_from_step=checkpoint_data.start_from_step
        self.prev_scores_steps=checkpoint_data.prev_scores_steps
        self.prev_scores=checkpoint_data.prev_scores

        self.log_path = os.path.join(self.checkpoint_dir, "log")

    def train(self):

        self.gen.train()
        self.dis.train()

        step_idx = self.start_from_step

        saved_scores_steps = []
        saved_scores_steps.extend(self.prev_scores_steps)
        saved_scores = []
        saved_scores.extend(self.prev_scores)

        epoch_logs = []

        for epoch_idx in range(self.start_from_epoch, self.num_epochs + self.start_from_epoch):
            # We'll accumulate batch losses and show an average once per epoch.
            dsc_losses = []
            gen_losses = []
            self.print_and_save(f'--- EPOCH {epoch_idx+1}/{self.num_epochs + self.start_from_epoch} ---', epoch_logs)

            with tqdm.tqdm(total=len(self.data_loader.batch_sampler), file=sys.stdout) as pbar:
                for batch_idx, (x_real, y_real) in enumerate(self.data_loader):

                    x_real = x_real.to(self.device)

                    if self.labeled:
                        y_real = y_real.to(self.device)
                    else:
                        y_real = None

                    dis_loss, gen_loss = self.train_batch(x_real, y_real)

                    dsc_losses.append(dis_loss)
                    gen_losses.append(gen_loss)

                    self.evaluate_model(epoch_idx, saved_scores_steps, saved_scores, step_idx, epoch_logs)

                    step_idx += 1
                    pbar.update()

            dsc_avg_loss, gen_avg_loss = np.mean(dsc_losses), np.mean(gen_losses)
            self.print_and_save(f'Discriminator loss: {dsc_avg_loss}', epoch_logs)
            self.print_and_save(f'Generator loss:     {gen_avg_loss}', epoch_logs)

            # save checkpoint
            if (epoch_idx + 1 - self.start_from_epoch) % self.model_save_epoch == 0:
                self.save_model(epoch_idx, epoch_logs, saved_scores, saved_scores_steps, step_idx)

            self.write_to_logs(epoch_logs)
            epoch_logs.clear()

        # last save
        self.calc_score(epoch_idx, epoch_logs, saved_scores, saved_scores_steps, step_idx)
        self.save_model(epoch_idx, epoch_logs, saved_scores, saved_scores_steps, step_idx)

    def evaluate_model(self, epoch_idx, saved_scores_steps, saved_scores, step_idx, epoch_logs):

        self.gen.eval()

        current_step = step_idx + 1 - self.start_from_step

        if current_step % self.calc_score_step == 0:
            self.calc_score(epoch_idx, epoch_logs, saved_scores, saved_scores_steps, step_idx)

        if current_step % self.sample_save_step == 0:
            # save sample
            fake = self.gen_samples()
            save_image(fake, os.path.join(self.samples_dir, '{}_fake.png'.format(step_idx + 1)),
                       nrow=self.n_sample_row, padding=2)
            self.print_and_save("[%d] Sample image was saved" % (step_idx + 1), epoch_logs)

        self.gen.train()

    def save_model(self, epoch_idx, epoch_logs, saved_scores, saved_scores_steps, step_idx):

        # save model
        torch.save({'epoch': epoch_idx + 1,
                    'step': step_idx,
                    'saved_scores_steps': saved_scores_steps,
                    'scores': saved_scores,
                    'gen_state_dict': self.gen.state_dict(),
                    'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
                    'dis_state_dict': self.dis.state_dict(),
                    'dis_optimizer_state_dict': self.dis_optimizer.state_dict()},
                   os.path.join(self.checkpoint_dir,
                                '{}_{}.pt'.format(self.version, epoch_idx + 1)))

        # save plot
        self.save_scores_plot(saved_scores_steps, saved_scores, epoch_idx + 1)
        self.print_and_save("[%d] Checkpoint and plot were saved" % (epoch_idx + 1), epoch_logs)

    def calc_score(self, epoch_idx, epoch_logs, saved_scores, saved_scores_steps, step_idx):
        score, _ = self.evaluator.eval_gen(self.gen)
        self.print_and_save("[%d] evaluated inception score: %.4f" % (epoch_idx + 1, score), epoch_logs)
        saved_scores.append(score)
        saved_scores_steps.append(step_idx)

    def save_scores_plot(self, steps, scores, epoch):
        plt.plot(steps, scores)
        plt.xlabel('Iteration')
        plt.ylabel('Inception score')
        plt.title('IS ' + self.version)
        #plt.show()
        plot_name = 'IS_{}_plot_{}.png'.format(self.version, epoch)
        #plot_name = 'IS_' + model_version + '_plot.png'
        plot_dir = os.path.join(self.checkpoint_dir, plot_name)
        plt.savefig(plot_dir)


    @staticmethod
    def print_and_save(log, epoch_logs):
        print(log)
        epoch_logs.append(log)

    def write_to_logs(self, messages):
        with open(self.log_path, 'a') as log_file:
            for i in range(len(messages)):
                log_file.write(messages[i] + '\n')

    def gen_samples(self):
        with torch.no_grad():
            fake = self.gen(self.fixed_noise, self.fixed_y).detach().cpu() * .5 + .5
        return fake

    def sample_noises(self):
        fake_noise = torch.randn(self.batch_size, self.gen.dim_z, dtype=torch.float, device=self.device)

        fake_y = None
        if self.labeled:
            fake_y = torch.randint(low=0, high=self.gen.n_classes, size=(self.batch_size,), dtype=torch.long,
                                   device=self.device)
        return fake_noise, fake_y

    def train_batch(self, x_real, y_real):

        ######################
        # Generator update
        ######################

        self.gen_optimizer.zero_grad()

        z_fake, y_fake = self.sample_noises()
        x_fake = self.gen(z_fake, y_fake)
        dis_fake = self.dis(x_fake, y=y_fake)
        loss_gen = self.generator_loss_fn(dis_fake)

        loss_gen.backward()
        self.gen_optimizer.step()

        ######################
        # Discriminator update
        ######################

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

class CheckpointData():
    def __init__(self, start_from_epoch=0, start_from_step=0, prev_scores_steps=[], prev_scores=[]):
        self.start_from_epoch = start_from_epoch
        self.start_from_step = start_from_step
        self.prev_scores_steps = prev_scores_steps
        self.prev_scores = prev_scores