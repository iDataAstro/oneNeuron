from utils.model import Perceptron
from utils.all_utils import *

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1]
}

X, y, _ = prepare_data(df_name=AND)

LR = 1e-4
EPOCHS = 10

model_AND = Perceptron(learning_rate=LR, epochs=EPOCHS)
model_AND.fit(X, y)
_ = model_AND.total_loss()

filename = "and_model"
save_model(model_AND, filename)
save_plot(AND, filename, model_AND)