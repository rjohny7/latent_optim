from stylegan2_pytorch.stylegan2_pytorch import Trainer
import os

trainer = Trainer()
trainer.set_data_src(os.path.join(os.getcwd(), "data/ds1_fullsize"))
trainer.train()