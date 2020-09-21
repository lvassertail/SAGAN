#from torchvision.models import inception_v3
from torchvision.models.inception import inception_v3
import torch
import math

class Inception(object):

    def __init__(self, n_images=50000, batch_size=100, splits=10, device=torch.device("cpu")):
        self.n_images = n_images
        self.batch_size = batch_size
        self.splits = splits
        self.n_batches = int(math.ceil(float(n_images)/float(batch_size)))
        self.device = device

        self.inception_model = inception_v3(pretrained=False, transform_input=False).type(torch.cuda.FloatTensor)
        self.inception_model.load_state_dict(torch.load('inception_v3_google-1a9a5a14.pth'))
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2).to(self.device)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2).to(self.device)
        self.inception_model.eval().to(device)

    def generate_images(self, gen):
        with torch.no_grad():
            batch_noise, batch_y = self.sample_noises(self.batch_size, gen.dim_z, gen.n_classes, self.device)
            # batch_images = (gen(batch_noise, batch_y).detach() * .5 + .5)
            # batch_images = (batch_images - self.mean) / self.std
            batch_images = gen(z=batch_noise, y=batch_y).detach()
        return batch_images

    def sample_noises(self, n_samples, noise_dim, n_categories=None, device=torch.device("cpu")):
        noise = torch.randn(n_samples, noise_dim, device=device)
        if n_categories:
            y_fake = torch.randint(low=0, high=n_categories, size=(n_samples,), dtype=torch.long,
                                   device=device)
        else:
            y_fake = None
        return noise, y_fake

    def inception_softmax(self, batch_images):
        with torch.no_grad():
            if batch_images.shape[-1] != 299 or batch_images.shape[-2] != 299:
                batch_images = torch.nn.functional.interpolate(batch_images, size=(299, 299), mode='bilinear',
                                                               align_corners=False)

            y = self.inception_model(batch_images)
            y = torch.nn.functional.softmax(y, dim=1)
        return y

    def kl_scores(self, ys):
        scores = []
        with torch.no_grad():
            for j in range(self.splits):
                part = ys[(j*self.n_images//self.splits): ((j+1)*self.n_images // self.splits), :]
                kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
                kl = torch.mean(torch.sum(kl, 1))
                kl = torch.exp(kl)
                scores.append(kl.unsqueeze(0))
            scores = torch.cat(scores, 0)
            m_scores = torch.mean(scores).detach().cpu().numpy()
            m_std = torch.std(scores).detach().cpu().numpy()
        return m_scores, m_std

    def eval_gen(self, gen):
        ys = []
        for i in range(self.n_batches):
            batch_images = self.generate_images(gen)
            y = self.inception_softmax(batch_images)
            ys.append(y)
        ys = torch.cat(ys, 0)
        m_scores, m_std = self.kl_scores(ys)
        return m_scores, m_std

    def eval_dataset(self, dataset):
        ys = []
        for i in range(self.n_batches):
            batch_images = dataset.get_next()
            if isinstance(batch_images, list):
                batch_images = batch_images[0]
            batch_images = batch_images.to(self.device)
            y = self.inception_softmax(batch_images)
            ys.append(y)
        ys = torch.cat(ys, 0)
        m_scores, m_std = self.kl_scores(ys)
        return m_scores, m_std

