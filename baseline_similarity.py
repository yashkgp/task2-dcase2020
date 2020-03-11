import os
import sys
import pickle
import numpy as np
from sklearn import metrics 
from scipy import spatial

def read(path):
	with open(path, 'rb') as file:
		d = pickle.load(file)
	return d

def calculate_centroid(train_dict):
	centroid = []
	for file in train_dict:
		emb = train_dict[file]
		centroid.append(np.mean(emb, axis = 0))
	return np.mean(centroid, axis = 0)

def calculate_similarity(centroid, test_dict):
	
	normal_dist = []
	anomaly_dist = []
	
	result = {}
	# Get ids
	IDS = []
	for file in test_dict:
		temp = file.split('_')
		ID = temp[2]
		IDS.append(ID)
	IDS = list(set(IDS))
	for ID in IDS:
		result[ID] = {}

	for file in test_dict:
		
		temp = file.split('_')
		ID = temp[2]
		emb = test_dict[file]
		distance = spatial.distance.cosine(centroid, np.mean(emb, axis = 0))

		if temp[0] == 'normal':
			normal_dist.append(distance)
			try:
				result[ID]['y_true'].append(0)
			except:
				result[ID]['y_true'] = []
				result[ID]['y_true'].append(0)
			try:
				result[ID]['y_pred'].append(distance)
			except:
				result[ID]['y_pred'] = []
				result[ID]['y_pred'].append(distance)

		if temp[0] == 'anomaly':
			anomaly_dist.append(distance)
			try:
				result[ID]['y_true'].append(1)
			except:
				result[ID]['y_true'] = []
				result[ID]['y_true'].append(1)
			try:
				result[ID]['y_pred'].append(distance)
			except:
				result[ID]['y_pred'] = []
				result[ID]['y_pred'].append(distance)

	return normal_dist, anomaly_dist, result

def analyze(dist):
	print("MEAN = {}".format(np.mean(dist)))
	print("STD DEV = {}".format(np.std(dist)))
	print("MEDIAN = {}".format(np.quantile(dist, .50))) 
	print("Q1 quantile = {}".format(np.quantile(dist, .25))) 
	print("Q3 quantile = {}".format(np.quantile(dist, .75)))

def eval(result):
	print("M_ID\tAUC\tpAUC")
	avg_auc = []
	avg_pauc = []
	for ids in result:
		y_pred = result[ids]['y_pred']
		y_true = result[ids]['y_true']
		auc = metrics.roc_auc_score(y_true, y_pred)
		p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr = 0.1)
		print("%d\t%.2f\t%.2f"%(int(ids), auc*100, p_auc*100))
		avg_auc.append(auc)
		avg_pauc.append(p_auc)
	avg_auc = np.mean(avg_auc)
	avg_pauc = np.mean(avg_pauc)
	print("AVG\t%.2f\t%.2f"%(avg_auc*100, avg_pauc*100))

def main():
	# path1 = machine type
	path1 = sys.argv[1]
	train_dict = read(os.path.join('data', path1, 'vggish_embeddings_train'))
	test_dict = read(os.path.join('data', path1, 'vggish_embeddings_test'))

	centroid = calculate_centroid(train_dict)
	normal, anomaly, result = calculate_similarity(centroid, test_dict)

	print("\nSimilarity Score Stats:")
	print('\nNormal Dist:')
	analyze(normal)
	print('\nAnomaly Dist:')
	analyze(anomaly)

	print("\nResults:\n")
	eval(result)

if __name__ == '__main__':
	main()