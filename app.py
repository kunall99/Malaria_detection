
import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding',False)       #for ignoring the warnings
@st.cache(allow_output_mutation=True)
def load_model():
	model=tf.keras.models.load_model('model_vgg19.h5')
	return model
model=load_model()
st.write("""
	    # Malaria Classification
	     """
	    )
file = st.file_uploader("Please upload the cell image",type=["jpg","png"])
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):

	size = (240,240)
	image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
	img = np.asarray(image)
	img_reshape = img[np.newaxis,...]
	prediction = model.predict(img_reshape)
	return prediction
if file is None:
	st.text("Please upload an image file")
else:
	image = Image.open(file)
	st.image(image,use_column_width=True)
	predictions = import_and_predict(image,model)
	class_names=['Parasite','Uninfected']
	string="This image most likely is:"+class_names[np.argmax(predictions)]
	st.success(string)