from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

## Implementing Truth [OR Data SET]

OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1]
}

df= pd.DataFrame(OR)
print(df)


x,y = prepare_data(df)

ETA = 0.3 ## 0 AND 1
EPOCHS = 10

model_OR = Perceptron(eta = ETA, epochs= EPOCHS)
model_OR.fit(x,y)

_ = model_OR.total_loss()

save_plot(df,"or.png", model_OR) ## Implementing save_plot methods from all_utils
save_model(model_OR,"or.model") ## Implementing save_model methods from all_utils