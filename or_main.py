from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os 


logging_str = "[%(asctime)s: %(levelname)s: %(module)s ] %(message)s"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"Runnig_logs_OR.log"),level=logging.INFO, format=logging_str,filemode="a")


def main (data,eta,epochs,filename,plotname):

    df= pd.DataFrame(data)
    logging.info(f"This is the actual DataFrame{df}")
    ## Implementing Prepare_data method from all_utils.py
    x,y = prepare_data(df) 

    

    model_or = Perceptron(eta=eta, epochs=epochs)
    model_or.fit(x,y)

    _ = model_or.total_loss()

    save_model(model_or,filename=filename)
    save_plot(df,plotname,model_or)



if __name__ == "__main__": ## << entry point

        ##    Implementing Truth Table [AND DATASET] 
    OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1]
    }

    ETA = 0.3 ## 0 AND 1
    EPOCHS = 10
    try:
        logging.info(">>>>> starting training >>>>>")
        main(data=OR,eta=ETA,epochs=EPOCHS,filename="or_main.model",plotname="or_main.png")
        logging.info("<<<<< training done successfully<<<<<\n")
    except Exception as e: 
        logging.exception(e)
        raise e