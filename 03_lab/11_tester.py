from keras import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

imdb_dir = os.path.join(os.getcwd(), 'aclImdb')
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
	dir_name = os.path.join(test_dir, label_type)
	for fname in sorted(os.listdir(dir_name)):
		if fname[-4:] == '.txt':
			f = open(os.path.join(dir_name, fname), encoding='UTF-8')
			texts.append(f.read())
			f.close()
			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)

maxlen = 100  
max_words = 10000 
tokenizer = Tokenizer(num_words=max_words) 

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)




model = models.load_model('11_model.h5')
model.load_weights('11_pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)















