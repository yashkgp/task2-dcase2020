import os
import gc
import pickle
import numpy as np
import keras.models
from keras import regularizers
from keras.layers import Input, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn import metrics 
from scipy import spatial

onehot_encoder = OneHotEncoder(sparse = False)

def get_model(inputDim):

	inputLayer = Input(shape=(inputDim,))
	encoded = Dense(units = 784, activation = 'relu')(inputLayer)
	encoded = Dense(units = 128, activation = 'relu')(encoded)
	encoded = Dense(units = 64, activation = 'relu')(encoded)
	decoded = Dense(units = 128, activation ='relu')(encoded)
	decoded = Dense(units = 784, activation = 'relu')(decoded)
	decoded = Dense(units = inputDim, activation = 'sigmoid')(decoded)

	return Model(inputs=inputLayer, outputs=decoded)


def train_model(classes, X_train):
	print("Classes = {}".format(classes))
	print("# Classes = {}".format(len(classes)))
	try:
		print('Loading Model')
		model = load_model('saved/autoencoder_vggish.h5')
	except:
		print('Loading Model Failed, Training')
		model = get_model(int(X_train[0].shape[0]))
		model.compile(loss='mean_squared_error',
	              optimizer='adam',
	              metrics=['mae'])
		print(model.summary())
		model.fit(X_train, X_train, epochs = 100, batch_size = 64)
		model.save('saved/autoencoder_vggish.h5')
	return model


def make_train_data(classes):
	train_data = {'X':[], 'y':[]}
	for clas in classes:
		with open(os.path.join('data', clas, 'vggish_embeddings_train'), 'rb') as file:
			D = pickle.load(file)
		for file in D:
			if clas == 'ToyCar':
				train_data['X'].append((np.array(D[file][:-1]).flatten()))
			else:	
				train_data['X'].append((np.array(D[file]).flatten()))
			train_data['y'].append([classes.index(clas)])
	# X_train, X_test, y_train, y_test = train_test_split(train_data['X'], train_data['y'], test_size=0.25, random_state=42)
	X_train, X_test, y_train, y_test = train_data['X'], [], train_data['y'], []
	del train_data
	gc.collect()
	y_train = onehot_encoder.fit_transform(y_train)
	# y_test = onehot_encoder.fit_transform(y_test)
	return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def make_test_data(classes):

	result = {}
	for clas in classes:

		with open(os.path.join('data', clas, 'vggish_embeddings_test'), 'rb') as file:
			test_dict = pickle.load(file)

		result[clas] = {}
		# Get ids
		IDS = []
		for file in test_dict:
			temp = file.split('_')
			ID = temp[2]
			IDS.append(ID)
		IDS = list(set(IDS))
		for ID in IDS:
			result[clas][ID] = {'X':[], 'y':[], 'anomaly':[]}

		test_data = {'X':[], 'y':[], 'anomaly':[]}
		for file in test_dict:
			temp = file.split('_')
			ID = temp[2]
			if clas == 'ToyCar':
				result[clas][ID]['X'].append((np.array(test_dict[file][:-1]).flatten()))
			else:
				result[clas][ID]['X'].append((np.array(test_dict[file]).flatten()))
			result[clas][ID]['y'].append([classes.index(clas)])
			if file.split('_')[0] == 'anomaly':
				result[clas][ID]['anomaly'].append(1)
			else:
				result[clas][ID]['anomaly'].append(0)
	return result

def calculate_centroid(clas, intermediate_layer_model):
	with open(os.path.join('data', clas, 'vggish_embeddings_train'), 'rb') as file:
		train_dict = pickle.load(file)
	centroid = []
	for file in train_dict:
		if clas == 'ToyCar':
			emb = intermediate_layer_model.predict(normalize(np.array([train_dict[file][:-1].flatten()])))
		else:
			emb = intermediate_layer_model.predict(normalize(np.array([train_dict[file].flatten()])))
		centroid.append(emb)
	return np.mean(centroid, axis = 0)

def test_using_similarity(model, test_data):

	layer_name = 'dense_5'
	intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)

	print('\n\nTesting Using Similarity')

	for clas in test_data:
		centroid = calculate_centroid(clas, intermediate_layer_model)
		print('\n', clas, ':')
		print("M_ID\tAUC\tpAUC")
		avg_auc = []
		avg_pauc = []
		for ids in test_data[clas]:
			test_feats = np.array(test_data[clas][ids]['X'])
			y_pred = intermediate_layer_model.predict(test_feats)
			pred_anom = [spatial.distance.cosine(centroid, i) for i in y_pred]
			real_anom = test_data[clas][ids]['anomaly']
			
			auc = metrics.roc_auc_score(real_anom, pred_anom)
			p_auc = metrics.roc_auc_score(real_anom, pred_anom, max_fpr = 0.1)
			
			print("%d\t%.2f\t%.2f"%(int(ids), auc*100, p_auc*100))
			avg_auc.append(auc)
			avg_pauc.append(p_auc)
		avg_auc = np.mean(avg_auc)
		avg_pauc = np.mean(avg_pauc)
		print("AVG\t%.2f\t%.2f"%(avg_auc*100, avg_pauc*100))

def test_using_reconstruction_loss(model, test_data):

	print('\n\nTesting Reconstruction Loss')

	for clas in test_data:
		print('\n', clas, ':')
		print("M_ID\tAUC\tpAUC")
		avg_auc = []
		avg_pauc = []
		for ids in test_data[clas]:
			test_feats = np.array(test_data[clas][ids]['X'])
			y_pred = model.predict(test_feats)
			# mae = [np.sum(np.abs(test_feats - y_pred), axis = 1)]
			mae = [mean_absolute_error(test_feats[i], y_pred[i]) for i in range(len(test_feats))]
			# pred_anom = [spatial.distance.cosine(centroid, i) for i in y_pred]
			real_anom = test_data[clas][ids]['anomaly']
			
			auc = metrics.roc_auc_score(real_anom, mae)
			p_auc = metrics.roc_auc_score(real_anom, mae, max_fpr = 0.1)
			
			print("%d\t%.2f\t%.2f"%(int(ids), auc*100, p_auc*100))
			avg_auc.append(auc)
			avg_pauc.append(p_auc)
		avg_auc = np.mean(avg_auc)
		avg_pauc = np.mean(avg_pauc)
		print("AVG\t%.2f\t%.2f"%(avg_auc*100, avg_pauc*100))


if __name__ == '__main__':
	classes = ['slider', 'valve', 'pump', 'fan', 'ToyCar', 'ToyConveyor']
	X_train, X_test, y_train, y_test = make_train_data(classes)
	model = train_model(classes, X_train)
	test_data = make_test_data(classes)
	test_using_reconstruction_loss(model, test_data)
	test_using_similarity(model, test_data)

