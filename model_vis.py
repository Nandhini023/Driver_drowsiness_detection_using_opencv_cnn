import visualkeras
from keras.models import load_model
model = load_model('model.h5')
visualkeras.layered_view(model).show()
visualkeras.layered_view(model, to_file = 'output.png')

