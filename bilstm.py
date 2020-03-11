import os
import gc
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import regularizers
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics 
from scipy import spatial

onehot_encoder = OneHotEncoder(sparse = False)


def bi_lstm(nclasses):
	model = Sequential()
	model.add(BatchNormalization(input_shape=(10, 128)))
	model.add(Dropout(0.5))
	model.add((LSTM(64, activation='relu',
	        kernel_regularizer=regularizers.l2(0.01),
	        activity_regularizer=regularizers.l2(0.01),
	        return_sequences=True)))

	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add((LSTM(64, activation='relu',
	        kernel_regularizer=regularizers.l2(0.01),
	        activity_regularizer=regularizers.l2(0.01))))

	model.add(Dense(nclasses, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	print(model.summary())
	return model


def train_model(classes, X_train, X_test, y_train, y_test):
	print("Classes = {}".format(classes))
	print("# Classes = {}".format(len(classes)))
	model = bi_lstm(len(classes))
	model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size = 32)
	return model


def make_train_data(classes):
	train_data = {'X':[], 'y':[]}
	for clas in classes:
		with open(os.path.join('data', clas, 'vggish_embeddings_train'), 'rb') as file:
			D = pickle.load(file)
		for file in D:
			train_data['X'].append(np.array(D[file]))
			train_data['y'].append([classes.index(clas)])
	X_train, X_test, y_train, y_test = train_test_split(train_data['X'], train_data['y'], test_size=0.25, random_state=42)
	del train_data
	gc.collect()
	y_train = onehot_encoder.fit_transform(y_train)
	y_test = onehot_encoder.fit_transform(y_test)
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
			result[clas][ID]['X'].append(np.array(test_dict[file]))
			result[clas][ID]['y'].append([classes.index(clas)])
			if file.split('_')[0] == 'anomaly':
				result[clas][ID]['anomaly'].append(1)
			else:
				result[clas][ID]['anomaly'].append(0)
	return result


def test_using_labels(model, test_data):
	print('Testing Using Labels')
	for clas in test_data:
		print('\n', clas, ':')
		print("M_ID\tAUC\tpAUC")
		avg_auc = []
		avg_pauc = []
		for ids in test_data[clas]:
			test_feats = np.array(test_data[clas][ids]['X'])
			y_pred = model.predict(test_feats)
			y_pred_labels = np.argmax(y_pred, axis = 1)
			y_true = np.array(test_data[clas][ids]['y'])
			pred_anom = [y_pred[i][y_pred_labels[i]] if y_pred_labels[i] == y_true[i] else (1 - y_pred[i][y_pred_labels[i]]) for i in range(len(y_true))]
			real_anom = test_data[clas][ids]['anomaly']
			
			auc = metrics.roc_auc_score(real_anom, pred_anom)
			p_auc = metrics.roc_auc_score(real_anom, pred_anom, max_fpr = 0.1)
			
			print("%d\t%.2f\t%.2f"%(int(ids), auc*100, p_auc*100))
			avg_auc.append(auc)
			avg_pauc.append(p_auc)
		avg_auc = np.mean(avg_auc)
		avg_pauc = np.mean(avg_pauc)
		print("AVG\t%.2f\t%.2f"%(avg_auc*100, avg_pauc*100))

def calculate_centroid(clas, intermediate_layer_model):
	with open(os.path.join('data', clas, 'vggish_embeddings_train'), 'rb') as file:
		train_dict = pickle.load(file)
	centroid = []
	for file in train_dict:
		emb = intermediate_layer_model.predict(np.array([train_dict[file]]))
		centroid.append(emb)
	return np.mean(centroid, axis = 0)

def test_using_similarity(model, test_data):

	layer_name = 'lstm_2'
	intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)

	print('\n\nTesting Using Similarity')

	for clas in test_data:
		print('\n', clas, ':')
		print("M_ID\tAUC\tpAUC")
		avg_auc = []
		avg_pauc = []
		centroid = calculate_centroid(clas, intermediate_layer_model)
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

if __name__ == '__main__':
	classes = ['slider', 'valve']
	X_train, X_test, y_train, y_test = make_train_data(classes)
	model = train_model(classes, X_train, X_test, y_train, y_test)
	test_data = make_test_data(classes)
	test_using_labels(model, test_data)
	test_using_similarity(model, test_data)


