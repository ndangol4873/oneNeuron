from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np


def main (data,eta,epochs,filename,plotname):

    df= pd.DataFrame(data)
    print(df)
    ## Implementing Prepare_data method from all_utils.py
    x,y = prepare_data(df) 

    

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(x,y)

    _ = model.total_loss()

    save_model(model,filename=filename)
    save_plot(df,plotname,model)



if __name__ == "__main__": ## << entry point

        ##    Implementing Truth Table [AND DATASET] 
    AND = {
            "x1": [0,0,1,1],
            "x2": [0,1,0,1],
            "y": [0,0,0,1]
    }

    ETA = 0.3 ## 0 AND 1
    EPOCHS = 10


    main(data=AND,eta=ETA,epochs=EPOCHS,filename="and_main.model",plotname="and_main.png")