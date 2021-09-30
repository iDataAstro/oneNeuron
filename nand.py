from utils.model import Perceptron
from utils.all_utils import *

NAND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [1,1,1,0],
}

X, y, _ = prepare_data(df_name=NAND)

LR = 0.3
EPOCHS = 10

model_NAND = Perceptron(learning_rate=LR, epochs=EPOCHS)
model_NAND.fit(X, y)
_ = model_NAND.total_loss()

filename = "nand_model"
root_dir = "/Users/jatin/oneNeuron/oneNeuron"
save_model(model_NAND, root_dir, filename)