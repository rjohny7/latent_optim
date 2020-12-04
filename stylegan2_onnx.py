import torch
from torchvision import utils
from tqdm import tqdm
from stylegan2.model import Generator


def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

stylegan_gen = Generator(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2).to(device)

checkpoint = torch.load('./twdne3.pt')

stylegan_gen.load_state_dict(checkpoint["g_ema"])


mean_latent = None

#generate(args, stylegan_gen, device, mean_latent)