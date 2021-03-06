from utils.model import Perceptron
from utils.all_utils import *
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "or.py.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format=logging_str, filemode="a")

def main(df_name, learning_rate, epochs, filename):
    X, y, _ = prepare_data(df_name)

    model_OR = Perceptron(learning_rate=learning_rate, epochs=epochs)
    model_OR.fit(X, y)
    _ = model_OR.total_loss()

    save_model(model_OR, filename)
    save_plot(df_name, filename, model_OR)

if __name__ == '__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }

    LR = 1e-4
    EPOCHS = 10
    
    try:
        logging.info("\n>>>>>>>>>>> START OF TRAINING <<<<<<<<<<")
        main(df_name=OR, learning_rate=LR, epochs=EPOCHS, filename="or_model")
        logging.info("<<<<<<<<<<< TRAINING DONE SUCCESSFULLY >>>>>>>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise(e)