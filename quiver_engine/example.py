from scipy.misc import imsave
from quiver_engine import server

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = load_model('C:/Users/jenni/Downloads/artifacts_model.h5')
model.summary()

# temp_folder not working - creates a \tmp file where the project is instead (delete folder if error) 
server.launch(model, classes=['Accepted','Crack', 'Chip'], temp_folder='C:/Data/temp', input_folder='C:/Data/catdog')
