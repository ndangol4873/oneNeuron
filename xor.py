from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np


## Implementing XOR Truth [XOR DATA SET]

XOR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,0]
}

df = pd.DataFrame(XOR)
df

x,y = prepare_data(df) ## Implementing prepare_data method from all_utils

ETA = 0.3 ## 0 AND 1
EPOCHS = 10

model_XOR = Perceptron(eta = ETA, epochs= EPOCHS)
model_XOR.fit(x,y)

_ = model_XOR.total_loss()


save_plot(df,"or.png", model_XOR) ## Implementing save_plot methods from all_utils
save_model(model_XOR,"or.model") ## Implementing save_model methods from all_utils

