Trained with:

For everything except lpips_alternateAdam and lpips_mse_*
Adam optimizer, lr=0.1
Steps: 10000

DCGAN iter48800.save

lpips_alternateAdam*
---------------------------
lpips_alternateAdam
Adam optimizer, lr=0.1, betas=(0.9, 0.99)

lpips_alternateAdam2
Adam optimizer, lr=0.1, betas=(0.9, 0.98)

lpips_alternateAdam3
Adam optimizer, lr=0.1, betas=(0.9, 0.985)

lpips_alternateAdam4
Adam optimizer, lr=0.1, betas=(0.9, 0.995)

lpips_mse_*
---------------------------
lpips_mse_1
Adam optimizer, lr=0.1, betas=(0.9, 0.99)
Weights (lpips, mse): 1, 1

lpips_mse_2
Adam optimizer, lr=0.1, betas=(0.9, 0.99)
Weights (lpips, mse): 0.8, 0.2

lpips_mse_2
Adam optimizer, lr=0.1, betas=(0.9, 0.99)
Weights (lpips, mse): 0.2, 0.8

try beta 0.999 with above weights