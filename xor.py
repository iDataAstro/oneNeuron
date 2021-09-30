from utils.model import Perceptron
from utils.all_utils import *

XOR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,0],
}

X, y, _ = prepare_data(df_name=XOR)

LR = 0.3
EPOCHS = 10

model_XOR = Perceptron(learning_rate=LR, epochs=EPOCHS)
model_XOR.fit(X, y)
_ = model_XOR.total_loss()

filename = "xor_model"
root_dir = "/Users/jatin/oneNeuron/oneNeuron"
save_model(model_XOR, root_dir, filename)