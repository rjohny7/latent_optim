from collections import OrderedDict

import torchvision
import numpy as np
import math
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from bicubic import BicubicDownSample

from dcgan import Generator

def get_transformation(image_size):
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

class LatentOptimizer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['image_size'][0] != config['image_size'][1]:
            raise Exception('Non-square images are not supported yet.')

        self.reconstruction = 'invert'
        #self.project = config["project"]
        self.steps = 1000

        self.layer_in = None
        self.best = None
        self.skip = None
        self.lr = 0.1
        self.lr_record = []
        self.current_step = 0

        # prepare images
        device = 'cuda'
        resized_imgs = []
        original_imgs = []

        transform_lpips = get_transformation(256)
        transform = get_transformation(256)

        for imgfile in '/content/dcgan/alex.png':
            resized_imgs.append(transform_lpips(Image.open(imgfile).convert("RGB")))
            original_imgs.append(transform(Image.open(imgfile).convert("RGB")))

        self.resized_imgs = torch.stack(resized_imgs, 0).to(device)
        self.original_imgs = torch.stack(original_imgs, 0).to(device)

        self.downsampler_1024_image = BicubicDownSample(4)

        # Load models and pre-trained weights
        gen = Generator(1024, 512, 8)
        gen.load_state_dict(torch.load(config["ckpt"])["g_ema"], strict=False)
        gen.eval()
        self.gen = gen.to(device)
        self.gen.start_layer = 0
        self.gen.end_layer = 4


        self.mpl = MappingProxy(torch.load('gaussian_fit.pt'))
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg",
                                            use_gpu=device.startswith("cuda"))


        self.cls = imagenet_models.resnet50()
        state_dict = torch.load('imagenet_l2_3_0.pt')['model']
        new_dict = OrderedDict()

        for key in state_dict.keys():
            if 'module.model' in key:
                new_dict[key[13:]] = state_dict[key]

        self.cls.load_state_dict(new_dict)
        self.cls.to(config['device'])

        bs = self.original_imgs.shape[0]

        # initialization
        if self.gen.start_layer == 0:
            noises_single = self.gen.make_noise(bs)
            self.noises = []
            for noise in noises_single:
                self.noises.append(noise.normal_())
            self.latent_z = torch.randn(
                        (bs, 18, 512),
                        dtype=torch.float,
                        requires_grad=True, device='cuda')
            self.gen_outs = [None]
        else:
            # restore noises
            self.noises = torch.load(config['saved_noises'][0])
            self.latent_z = torch.load(config['saved_noises'][1]).to(config['device'])
            self.gen_outs = torch.load(config['saved_noises'][2])
            self.latent_z.requires_grad = True

    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp


    def invert_(self, start_layer, noise_list, steps, verbose=False, project=False):
        # noise_list containts the indices of nodes that we will be optimizing over
        for i in range(len(self.noises)):
            if i in noise_list:
                self.noises[i].requires_grad = True
            else:
                self.noises[i].requires_grad = False

        with torch.no_grad():
            if start_layer == 0:
                var_list = [self.latent_z] + self.noises
            else:
                intermediate_out = torch.ones(self.gen_outs[-1].shape, device=self.gen_outs[-1].device) * self.gen_outs[-1]
                intermediate_out.requires_grad = True
                var_list = [self.latent_z] + self.noises + [self.gen_outs[-1]]

            # set network that we will be optimizing over
            self.gen.start_layer = start_layer
            self.gen.end_layer = 4

        optimizer = optim.Adam(var_list, lr=self.lr)
        #ps = SphericalOptimizer([self.latent_z] + self.noises)
        pbar = tqdm(range(steps))
        self.current_step += steps

        if self.reconstruction == 'inpaint':
            mask = torch.ones(self.config['image_size'], device=self.config['device'])
            _, _, x, y = torch.where(self.original_imgs == -1)
            mask[x, y] = 0

        mse_min = np.inf

        lr_func = lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10)
        for i in pbar:
            t = i / steps
            lr = self.get_lr(t, self.lr)
            optimizer.param_groups[0]["lr"] = lr
            self.lr_record.append(lr)
            latent_w = self.mpl(self.latent_z)
            img_gen, _ = self.gen([latent_w], 
                                  input_is_latent=True,
                                  noise=self.noises, 
                                  layer_in=self.gen_outs[-1],)
            batch, channel, height, width = img_gen.shape
            factor = height // 256
            # calculate loss
            if self.reconstruction == 'inpaint':
                # downsample generared images
                downsampled = self.downsampler_1024_image(img_gen)
                # mask
                masked = downsampled * mask
                # compute loss
                diff = torch.abs(masked - self.original_imgs) - self.config['dead_zone_linear_alpha']
                dead_zone_linear_loss = torch.max(torch.zeros(diff.shape, device=diff.device), diff).sum()
                mse_loss = F.mse_loss(masked, self.original_imgs)
                if self.config['lpips_method'] == 'mask':
                    p_loss = self.percept(self.downsampler_image_256(masked),
                                          self.downsampler_image_256(self.original_imgs)).sum()
                elif self.config['lpips_method'] == 'fill':
                    filled = mask * self.original_imgs + (1 - mask) * downsampled
                    p_loss = self.percept(self.downsampler_1024_256(img_gen), self.downsampler_image_256(filled)).sum()
                else:
                    raise NotImplementdError('LPIPS policy not implemented')
            elif self.reconstruction == 'invert':
                diff = torch.abs(self.downsampler_1024_image(img_gen) - self.original_imgs) - self.config['dead_zone_linear_alpha']
                dead_zone_linear_loss = torch.max(torch.zeros(diff.shape, device=diff.device), diff).sum()
                mse_loss = F.mse_loss(self.downsampler_1024_image(img_gen), self.original_imgs)
                p_loss = self.percept(self.downsampler_1024_256(img_gen), self.resized_imgs).sum()
            elif self.reconstruction == 'denoise':
                diff = torch.abs(self.downsampler_1024_image(img_gen) - self.original_imgs) - self.config['dead_zone_linear_alpha']
                dead_zone_linear_loss = torch.max(torch.zeros(diff.shape, device=diff.device), diff).sum()
                mse_loss = F.mse_loss(self.downsampler_1024_image(img_gen), self.original_imgs)
                p_loss = self.percept(self.downsampler_1024_256(img_gen), self.resized_imgs).sum()

            loss = p_loss + mse_loss
            # if self.config['cls']:
            #     downsampled = self.downsampler_1024_128(img_gen)
            #     cls_out = self.cls(downsampled)
            #     cls_loss = F.cross_entropy(cls_out, self.config['target'] * torch.ones(cls_out.shape[0], device=img_gen.device, dtype=torch.int64))
            #     loss += self.config['cls'] * cls_loss
            #     cls_prob = F.softmax(cls_out, dim=-1)[0, self.config['target']].item()
            # else:
            #     cls_prob = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if self.project:
            #     ps.step()

            if mse_loss < mse_min:
                mse_min = mse_loss
                self.best = img_gen.detach().cpu()
            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {0:.4f};"
                    f" mse: {mse_loss.item():.4f};"
                    #f" cls_prob: {cls_prob:.4f} lr: {lr:.4f}"
                )
            )
            if False and i % 50 == 0:
                torchvision.utils.save_image(
                    img_gen,
                    f'gif_{start_layer}_{i}.png',
                    nrow=int(img_gen.shape[0] ** 0.5),
                    normalize=True)
        # TODO: check what happens when we are in the last layer
        with torch.no_grad():
            latent_w = self.mpl(self.latent_z)
            self.gen.end_layer = self.gen.start_layer
            intermediate_out, _  = self.gen([latent_w],
                                             input_is_latent=True,
                                             noise=self.noises,
                                             layer_in=self.gen_outs[-1],
                                             skip=self.skip)
            self.gen_outs.append(intermediate_out)
            self.gen.end_layer = self.config['end_layer']

    def invert(self):
        for i, steps in enumerate(self.steps.split(',')):
            begin_from = i + self.config['start_layer']
            if begin_from > self.config['end_layer']:
                raise Exception('Attempting to go after end layer...')
            self.invert_(begin_from, range(5 + 2 * begin_from), int(steps))
        return (self.latent_z, self.noises, self.gen_outs), self.best