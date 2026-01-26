from dataset.dataset import SE3_DATASET
from nn_models.flows import BijectionNet
from trainer.train import Trainer
import torch
from SE3_DS.utils.utils import *
from pathlib import Path

device = 'cuda'  if torch.cuda.is_available() else 'cpu'

_THIS_DIR = Path(__file__).resolve().parent
data_name='000'
data_path = _THIS_DIR / f"dataset/PegInHole_data/teach_data_{data_name}.txt"
save_path = _THIS_DIR / f"nn_models/models/PegInHole_models/best_model_{data_name}.pth"

data=SE3_DATASET(data_path, k=1085) # k is the lehgth of the trajectory in teach data
diffieo=BijectionNet(num_dims=6, num_blocks=6, num_hidden=64)
trainer=Trainer(model=diffieo, dataset=data.dataset, batch_size=64, num_epochs=1000)
trainer.train(save_path)

# test
# trainer.test(data.goal_H, save_path)