import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

x = np.load('image.npz')['arr_0']
x = 255.0 - x
y = pd.read_csv('labels.csv')['labels']
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nClasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.75, random_state = 1)

xtrainscale = xtrain/255

LR = LogisticRegression(solver='saga', multi_class='multinomial')
LR.fit(xtrainscale, ytrain)

def predictor(img):
    im_pil = Image.open(img)
    img_bw = im_pil.convert("L")
    img_bw_rs = img_bw.resize((22, 30), Image.ANTIALIAS)
    px_filter = 20
    minxpix = np.percentile(img_bw_rs, px_filter)
    img_bw_rs_inv_sc = np.clip(img_bw_rs-minxpix, 0, 255)
    maxpix = np.max(img_bw_rs)
    img_bw_rs_inv_sc = np.asarray(img_bw_rs_inv_sc)/maxpix

    sample = np.array(img_bw_rs_inv_sc).reshape(1,660)
    prediction = LR.predict(sample)
    return prediction[0]