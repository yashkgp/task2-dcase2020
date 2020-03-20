import os
import gc
import tqdm
import pickle
import librosa
import numpy as np 

frames = 41
window_size = 512 * (frames - 1)

def windows(data, window_size):
	start = 0
	while start < len(data):
		yield start, start + window_size
		start += int(window_size / 2)

def make_train_data(clas):
    try:
        with open(os.path.join('data', clas, 'spec_train.pkl'), 'rb') as file:
            train_data = pickle.load(file)
        return train_data
    except:
        train_data = {'X':[]}
        for clip in tqdm.tqdm(os.listdir(os.path.join('data', clas, 'train'))):
            librosa_load, sr = librosa.load(os.path.join('data', clas, 'train', clip))
            for (start, end) in windows(librosa_load, window_size):
                if len(librosa_load[start:end]) == window_size:
                    sig = librosa_load[start:end]
                    mfcc = np.array(librosa.feature.melspectrogram(sig, sr = 22050, n_fft = 1024, hop_length = 512, n_mels = 60))
                    logmfcc = librosa.amplitude_to_db(mfcc)
                    delta = np.array(librosa.feature.delta(logmfcc))
                    arr = np.stack([logmfcc, delta], axis = 2)
                    if np.mean(logmfcc) > -70.0:
                        train_data['X'].append(arr)
        train_data['X'] = np.array(train_data['X'])
        print('Saving Spectrograms')
        with open(os.path.join('data', clas, 'spec_train.pkl'), 'wb') as file:
            pickle.dump(train_data, file)
        return train_data

def rearrange_test(test_data):
    new_test = {}
    mids = list(set([int(test_data['id'][i].split('_')[2]) for i in range(len(test_data['id']))]))
    for mid in mids:
        new_test[mid] = {}
    for i in tqdm.tqdm(range(len(test_data['id']))):
        machine_id = int(test_data['id'][i].split('_')[2])
        clip_id = test_data['id'][i].split('_')[-1][:-4] + '_' + test_data['id'][i].split('_')[0]
        try:
            new_test[machine_id][clip_id].append(test_data['X'][i])
        except:
            new_test[machine_id][clip_id] = []
            new_test[machine_id][clip_id].append(test_data['X'][i])
    return new_test

def make_test_data(clas):
    try:
        with open(os.path.join('data', clas, 'spec_test.pkl'), 'rb') as file:
            test_data = pickle.load(file)
        return test_data
    except:
        test_data = {'X':[], 'id':[]}
        for clip in tqdm.tqdm(os.listdir(os.path.join('data', clas, 'test'))):
            librosa_load, sr = librosa.load(os.path.join('data', clas, 'test', clip))
            for (start, end) in windows(librosa_load, window_size):
                if len(librosa_load[start:end]) == window_size:
                    sig = librosa_load[start:end]
                    mfcc = np.array(librosa.feature.melspectrogram(sig, sr = 22050, n_fft = 1024, hop_length = 512, n_mels = 60))
                    logmfcc = librosa.amplitude_to_db(mfcc)
                    delta = np.array(librosa.feature.delta(logmfcc))
                    arr = np.stack([logmfcc, delta], axis = 2)
                    if np.mean(logmfcc) > -70.0:
                        test_data['X'].append(arr)
                        test_data['id'].append(clip)
        test_data['X'] = np.array(test_data['X'])
        test_data = rearrange_test(test_data)
        print('Saving Spectrograms')
        with open(os.path.join('data', clas, 'spec_test.pkl'), 'wb') as file:
            pickle.dump(test_data, file)
        return test_data

def test_using_cum_mae(model, test_data, clas):
    print('\n\nTesting Reconstruction MAE Loss')
    print('\n', clas, ':')
    print("M_ID\tAUC\tpAUC")
    avg_auc = []
    avg_pauc = []
    for mid in test_data:
        for clip_id in test_data[mid]:
            isanomaly = clip_id.split('_')[1]
            test_feats = test_data[mid][clip_id]
            y_pred = [model.predict(i) for i in test_feats]
            ####DEFINE MAE FOR IMAGE#############
            # mae = [mean_absolute_error(test_feats[i], y_pred[i]) for i in range(len(test_feats))]
            real_anom = 1 if isanomaly == 'anomaly' else 0
            auc = metrics.roc_auc_score(real_anom, mae)
            p_auc = metrics.roc_auc_score(real_anom, mae, max_fpr = 0.1)
            print("%d\t%.2f\t%.2f"%(int(ids), auc*100, p_auc*100))
            avg_auc.append(auc)
            avg_pauc.append(p_auc)
    avg_auc = np.mean(avg_auc)
    avg_pauc = np.mean(avg_pauc)
    print("AVG\t%.2f\t%.2f"%(avg_auc*100, avg_pauc*100))

def train_model(clas, train_data):
    # Define and train model
    return None

if __name__ == '__main__':
    
    # Train data format - 
    # {'X':[Spectrograms]}
    # spectrograms shape = (60, 41, 2)

    # Test data format - 
    # {machine_id:{clip_id:[]}}

    classes  = ['pump']
    for clas in classes:
        test_data = make_test_data(clas)
        train_data = make_train_data(clas)
        model = train_model(clas, train_data)
        test_using_cum_mae(model, test_data, clas)