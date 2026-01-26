from dataset.conditional_dataset import CONDITIONAL_DATASET
from nn_models.flows import ConditionalBijectionNet
from trainer.train import ConditionalTrainer
import torch
from SE3_DS.utils.utils import *

from pathlib import Path
# 准备数据
device = 'cuda'  if torch.cuda.is_available() else 'cpu'

data_path='D:\\PhD\Research\\URControl\\SE3_DS\\dataset\\Conditional_data'

_THIS_DIR = Path(__file__).resolve().parent

data_path = _THIS_DIR / f"dataset/Conditional_data/"
save_path = _THIS_DIR / f"nn_models/models/Conditional_models/best_model.pth"

data=CONDITIONAL_DATASET(data_path,isImg=False)
diffieo=ConditionalBijectionNet(num_dims=6, num_condition=1, num_blocks=10, num_hidden=128)
trainer=ConditionalTrainer(model=diffieo, dataset=data, batch_size=128, num_epochs=1000)
# trainer.train(save_path)

# test
trainer.test(save_path)