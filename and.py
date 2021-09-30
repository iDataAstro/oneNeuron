from utils.model import Perceptron
from utils.all_utils import *

def main(df_name, learning_rate, epochs, filename):
    X, y, _ = prepare_data(df_name)

    model_AND = Perceptron(learning_rate=learning_rate, epochs=epochs)
    model_AND.fit(X, y)
    _ = model_AND.total_loss()

    save_model(model_AND, filename)
    save_plot(df_name, filename, model_AND)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1]
    }

    LR = 1e-4
    EPOCHS = 10
    
    main(df_name=AND, learning_rate=LR, epochs=EPOCHS, filename="and_model")