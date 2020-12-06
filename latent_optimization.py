import cv2
import torch
from dcgan import Generator
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
import torch.nn.functional as F
import lpips

class LatentOptim:
    def __init__(self, generator, z_size, lr, device, target_image, logProbWeight=0.0, shiftLossWeight=1.0, loss_type="default"):
        self.logProbWeight = logProbWeight
        self.loss_type = loss_type
        self.generator = generator
        self.z_size = z_size
        self.z_pdf = torch.distributions.Normal(0, 1)

        # Latent Vector to optimize over
        self.z_vec = torch.randn(1, z_size, 1, 1, device=device).clone().detach()
        self.z_vec.requires_grad = True

        # Affine transformation variables to optimize over
        self.dy = torch.tensor(0.0).clone().detach()  # Shift image in y direction. (+) -> down, (-) -> up, range [-1, 1]
        self.dy.requires_grad = True
        self.dx = torch.tensor(0.0).clone().detach()  # Shift image in x direction. (+) -> right, (-) -> left, range [-1, 1]
        self.dx.requires_grad = True
        self.scale = torch.tensor(0.0).clone().detach()  # Zoom in or out. (+) -> zoom in, (-) -> zoom out, range [-1, 1]
        self.scale.requires_grad = True
        self.rot = torch.tensor(0.0).clone().detach()  # Rotate image. (+) -> Clockwise, (-) -> Counterclockwise
        self.rot.requires_grad = True
        self.shiftLossWeight = shiftLossWeight

        # Define optimizer and params to optimize over
        self.params = [self.z_vec]

        # Add affine parameters if affine loss is specified
        if "affine" in self.loss_type:
            self.params.extend([self.dx, self.dy])

            if "scale" in self.loss_type:
                self.params.extend([self.scale])

            if "rot" in self.loss_type:
                self.params.extend([self.rot])

        self.optim = torch.optim.Adam(self.params, lr=lr, betas=(0.9, 0.99))
        #self.optim = torch.optim.SGD(self.params, lr=lr, momentum=0.9)
        #self.scheduler = CosineAnnealingWarmRestarts(self.optim, 100, 1)
        #self.scheduler = CyclicLR(self.optim, base_lr=1e-2, max_lr=0.1, step_size_up=500)
        self.target_image = target_image

        # Gradient Convolution Kernels
        self.kernel_x = torch.FloatTensor([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        self.kernel_y = torch.FloatTensor([[-1, -1, -1],
                                       [0, 0, 0],
                                       [1, 1, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        self.lapacian_operator = torch.FloatTensor([[0, 1, 0],
                                                [1, -4, 1],
                                                [0, 1, 0]]).unsqueeze(0).unsqueeze(0).to(device)

        if 'lpips' in self.loss_type:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
            #self.lpips_loss = lpips_pytorch.LPIPS(net_type='vgg')

    def get_grads(self, image):
        # Convert to grayscale if not already
        if image.shape[1] > 1:
            image = torch.mean(image, dim=1, keepdim=True)

        grad_x = F.conv2d(image, self.kernel_x)
        grad_y = F.conv2d(image, self.kernel_y)
        grad_mag = (grad_x ** 2 + grad_y ** 2) ** 0.5
        laplacian = F.conv2d(image, self.lapacian_operator)
        return grad_mag, laplacian

    def step(self):
        self.optim.zero_grad()

        # Compute Loss
        image = self.generator(self.z_vec)

        # Step optimizer

        if self.loss_type == 'mse' or self.loss_type == 'default':
            logProb = self.z_pdf.log_prob(self.z_vec).mean()  # From https://github.com/ToniCreswell/InvertingGAN/blob/master/scripts/invert.py
            loss = torch.mean((self.target_image - image)**2) - self.logProbWeight * logProb
        elif self.loss_type == 'edge+mse':
            loss = torch.mean((self.get_grads(self.target_image)[0] - self.get_grads(image)[0]) ** 2)
        elif self.loss_type == 'edge+lpips':
            loss = self.lpips_loss(self.get_grads(self.target_image)[0], self.get_grads(image)[0])
        elif self.loss_type == 'lpips':
            loss = self.lpips_loss(self.target_image, image)
        elif self.loss_type == 'lpips+mse':
            loss = 0.8 * self.lpips_loss(self.target_image, image) + 0.2 * torch.mean((self.target_image - image)**2)
        elif self.loss_type == 'edge+lpips&lpips':
            edge_lpips = self.lpips_loss(self.get_grads(self.target_image)[0], self.get_grads(image)[0])
            lpips_loss = self.lpips_loss(self.target_image, image)
            loss = edge_lpips + lpips_loss
        elif self.loss_type == 'affine_debug':
            loss = torch.pow(self.dx, 2) + torch.pow(self.dy, 2)  # Must use torch.pow(x, 2) instead of x**2 for autograd (idk why but x**2 doesn't work as well)
        elif self.loss_type == 'affine(trans)_lpips':
            theta = torch.zeros(1, 2, 3)
            theta[:, 0, 0] = 1
            theta[:, 0, 1] = 0
            theta[:, 0, 2] = -self.dx * 2
            theta[:, 1, 0] = 0
            theta[:, 1, 1] = 1
            theta[:, 1, 2] = -self.dy * 2
            theta = theta.to(device)

            grid = F.affine_grid(theta, image.shape, align_corners=False)
            affined_target = F.grid_sample((-self.target_image+1)/2, grid, align_corners=False) * -2 + 1
            lpips_loss = self.lpips_loss(affined_target, image)
            shift_loss = torch.pow(self.dx, 2) + torch.pow(self.dy, 2)  # Must use torch.pow(x, 2) instead of x**2 for autograd (idk why but x**2 doesn't work as well)
            loss = lpips_loss + self.shiftLossWeight * shift_loss
        elif self.loss_type == 'affine(trans/scale)_lpips':
            theta = torch.zeros(1, 2, 3)
            theta[:, 0, 0] = 1 + (-self.scale * 2)
            theta[:, 0, 1] = 0
            theta[:, 0, 2] = -self.dx * 2
            theta[:, 1, 0] = 0
            theta[:, 1, 1] = 1 + (-self.scale * 2)
            theta[:, 1, 2] = -self.dy * 2
            theta = theta.to(device)

            grid = F.affine_grid(theta, image.shape, align_corners=False)
            affined_target = F.grid_sample((-self.target_image + 1) / 2, grid, align_corners=False) * -2 + 1
            lpips_loss = self.lpips_loss(affined_target, image)

            # Must use torch.pow(x, 2) instead of x**2 for autograd (idk why but x**2 doesn't work as well)
            shift_loss = torch.pow(self.dx, 2) + torch.pow(self.dy, 2) + torch.pow(self.scale, 2)

            loss = lpips_loss + self.shiftLossWeight * shift_loss
        elif self.loss_type == 'affine(trans/scale/rot)_lpips':
            theta = torch.zeros(1, 2, 3)
            theta[:, 0, 0] = 1 + (-self.scale * 2)
            theta[:, 0, 1] = self.rot * 2
            theta[:, 0, 2] = -self.dx * 2
            theta[:, 1, 0] = -self.rot * 2
            theta[:, 1, 1] = 1 + (-self.scale * 2)
            theta[:, 1, 2] = -self.dy * 2
            theta = theta.to(device)

            grid = F.affine_grid(theta, image.shape, align_corners=False)
            affined_target = F.grid_sample((-self.target_image + 1) / 2, grid, align_corners=False) * -2 + 1
            lpips_loss = self.lpips_loss(affined_target, image)

            # Must use torch.pow(x, 2) instead of x**2 for autograd (idk why but x**2 doesn't work as well)
            shift_loss = torch.pow(self.dx, 2) + torch.pow(self.dy, 2) + torch.pow(self.scale, 2) + torch.pow(self.rot, 2)

            loss = lpips_loss + self.shiftLossWeight * shift_loss
        else:
            assert False, "Invalid loss type"

        print(f"Loss: {loss}, dx: {self.dx}, dy: {self.dy}, scale: {self.scale}")

        loss.backward()
        self.optim.step()
        #self.scheduler.step()

        return loss.item(), self.z_vec.detach()

def biggest_rectangle(r):
    #return w*h
    return r[2]*r[3]

def cropFace(image_path, crop_size=(256, 256), resize_dims=(64, 64)):
    """
    Load an image from file, detect face, crop to square bounding box, and return torch tensor in (N, C, H, W) order
    :param image_path: path to image on disk
    :param crop_size: dimensions of returned image
    :return: torch tensor
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # detector options
    anime_face_cascade = cv2.CascadeClassifier('./lbpcascade_animeface.xml')
    human_face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faces = anime_face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(90, 90))
    faces2 = human_face_cascade.detectMultiScale(gray)

    faces = [x for x in faces]
    faces2 = [x for x in faces2]
    faces.extend(faces2)

    # if any faces are detected, we only extract the biggest detected region
    if len(faces) == 0:
        assert False, "No face detected in image"
    elif len(faces) > 1:
        sorted(faces, key=biggest_rectangle, reverse=True)

    x, y, w, h = faces[0]
    cropped_image = image[y:y + h, x:x + w, :]
    resized_image = cv2.resize(cropped_image, crop_size)

    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    resized_image = Image.fromarray(resized_image).resize(resize_dims)

    image_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Rescale to [0, 1]
    image_tensor = (image_tensor - 0.5) / 0.5  # Rescale [0, 1] to [-1, 1]
    return image_tensor

def img2tensor(image_path, resize_dims=(64, 64)):
    """
    Read image from disk, return pytorch tensor (N, C, H, W)
    :param image_path: path to image on disk
    :return: numpy array
    """
    target_image = Image.open(image_path)
    target_image = target_image.resize(resize_dims)
    target_image = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Rescale to [0, 1]
    target_image = (target_image - 0.5) / 0.5  # Rescale [0, 1] to [-1, 1]
    return target_image

def tensor2numpy_image(image_tensor):
    """
    Convert a tensor (N, C, H, W) to a numpy array (H, W, C) [for plotting with plt.imshow()]
    :param image_tensor: input image in tensor form
    :return: numpy array
    """
    return np.array(image_tensor.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5)

if __name__ == '__main__':
    # Get GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Set up Generator
    generator = Generator().to(device)
    save = torch.load('iter48800.save')
    generator.load_state_dict(save['gen_params'])
    generator.eval()


    target_image = img2tensor('billie_eilish_shifted.jpg')  # Load target image (precropped 64x64)
    #target_image = cropFace('dimakis-alex.jpg')
    print(target_image.shape)
    plt.imshow(tensor2numpy_image(target_image))
    plt.show()

    latent_optim = LatentOptim(generator=generator, z_size=100, lr=0.1,  loss_type='affine(trans/scale/rot)_lpips', device=device, target_image=target_image.to(device), logProbWeight=0, shiftLossWeight=1e-2)

    # Early Stopping Config
    early_stopping = False
    min_loss = 1000000
    min_loss_iter = -1
    best_zvec = None

    print(latent_optim.dx)
    print(latent_optim.dy)
    # Run optimization
    for i in range(1, 10000):
        print(f"{i}: ", end="")
        loss, z_vec = latent_optim.step()
        if loss < min_loss:
            min_loss = loss
            min_loss_iter = i
            best_zvec = z_vec

        elif i - min_loss_iter >= 1000 and early_stopping:
            break

    image = generator(latent_optim.z_vec).cpu().detach()
    plt.imshow(tensor2numpy_image(image))
    plt.title(f"Last z_vec, loss{loss}")
    plt.show()

    image = generator(best_zvec).cpu().detach()
    plt.imshow(tensor2numpy_image(image))
    plt.title(f"Best Z_vec, loss={min_loss}, iter={min_loss_iter}")
    plt.show()
    print(torch.min(image), torch.max(image))
