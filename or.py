from utils.model import Perceptron
from utils.all_utils import *

OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

X, y, _ = prepare_data(df_name=OR)

LR = 0.3
EPOCHS = 10

model_OR = Perceptron(learning_rate=LR, epochs=EPOCHS)
model_OR.fit(X, y)
_ = model_OR.total_loss()

filename = "or_model"
save_model(model_OR, filename)
save_plot(OR, filename, model_OR)