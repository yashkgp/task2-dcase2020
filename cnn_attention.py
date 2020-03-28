import os
import gc
import tqdm
import pickle
import librosa
import numpy as np 
import torch
from torch.autograd import Variable
from torch import nn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score


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

def test_using_cum_mse(model, test_data, clas):
    print('\n\nTesting Reconstruction MSE Loss')
    criterion = nn.MSELoss()
    print('\n', clas, ':')
    print("M_ID\tAUC\tpAUC")
    avg_auc = []
    avg_pauc = []
    evaluate = {}
    for mid in test_data:
        evaluate[mid] = {'pred':[], 'real':[]}
        for clip_id in test_data[mid]:
            isanomaly = clip_id.split('_')[1]
            test_feats = test_data[mid][clip_id]
            if torch.cuda.is_available() == True:
                y_pred = [model(Variable(torch.tensor([i.T])).cuda()) for i in test_feats]
                mae = [(criterion(y_pred[i][0], torch.tensor(test_feats[i].T).cuda()).data).cpu() for i in range(len(test_feats))]
            else:
                y_pred = [model(Variable(torch.tensor([i.T]))) for i in test_feats]
                mae = [criterion(y_pred[i][0], torch.tensor(test_feats[i].T)).data for i in range(len(test_feats))]
            evaluate[mid]['pred'].append(np.sum(mae))
            real_anom = 1 if isanomaly == 'anomaly' else 0
            evaluate[mid]['real'].append(real_anom)
        auc = roc_auc_score(evaluate[mid]['real'], evaluate[mid]['pred'])
        p_auc = roc_auc_score(evaluate[mid]['real'], evaluate[mid]['pred'], max_fpr = 0.1)
        print("%d\t%.2f\t%.2f"%(int(ids), auc*100, p_auc*100))
        avg_auc.append(auc)
        avg_pauc.append(p_auc)
    avg_auc = np.mean(avg_auc)
    avg_pauc = np.mean(avg_pauc)
    print("AVG\t%.2f\t%.2f"%(avg_auc*100, avg_pauc*100))

    # for mid in test_data:
    #     evaluate[mid] = {'pred':[], 'real':[]}
    #     for clip_id in test_data[mid]:
    #         isanomaly = clip_id.split('_')[1]
    #         real_anom = 1 if isanomaly == 'anomaly' else 0
    #         evaluate[mid]['real'].append(real_anom)




class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 12, 5, stride = 1, padding = 3),  
            nn.ReLU(True),
            nn.MaxPool2d((4,2), stride = (4,2)),  
            nn.Conv2d(12, 24, 5, stride = 1, padding = 3),  
            nn.ReLU(True),
            nn.MaxPool2d((4,2), stride = (4,2)),
            nn.Conv2d(24, 24, 5, stride = 1, padding = 3),  
            nn.ReLU(True),
            nn.MaxPool2d((4,2), stride = (4,2))
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(24, 24, 5, stride = 1, padding = 2),  
            nn.ReLU(True),
            nn.Upsample(size = (3, 16), mode = "nearest"),
            nn.Conv2d(24, 24, 5, stride = 1, padding = 1),  
            nn.ReLU(True),
            nn.Upsample((10, 31), mode = "nearest"),
            nn.Conv2d(24, 12, 5, stride = 1, padding = 0),
            nn.ReLU(True),
            nn.Upsample((41, 60), mode = "nearest"),
            nn.Conv2d(12, 2, 5, stride = 1, padding = 2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# def autoencoder_model():

#     input_img = Input(shape=(60, 41, 2))

#     x = Conv2D(24, (5, 5), activation='relu', strides = (1, 1), padding='same')(input_img)
#     x = MaxPooling2D((4, 2), strides = (4, 2), padding='same')(x)
#     x = Conv2D(48, (5, 5), strides = (1, 1),activation='relu', padding='same')(x)
#     x = MaxPooling2D((4, 2), strides = (4, 2), padding='same')(x)
#     x = Conv2D(48, (5, 5), strides = (1, 1), activation='relu', padding='same')(x)
#     encoded = MaxPooling2D((4, 2), strides = (4, 2), padding='same')(x)

#     x = Conv2D(48, (5, 5), strides = (1, 1), activation='relu', padding='same')(encoded)
#     x = UpSampling2D((1, 1))(x)
#     x = Conv2D(48, (5, 5), strides = (1, 1),activation='relu', padding='same')(x)
#     x = UpSampling2D((1, 1))(x)
#     x = Conv2D(24, (5, 5), activation='relu', strides = (1, 1), padding='same')(input_img)
#     x = UpSampling2D((1, 1))(x)
#     decoded = Conv2D(2, (5, 5), activation='sigmoid', padding='same')(x)

#     autoencoder = Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss='mae')

#     return autoencoder



def train_model(train_data, clas):

    num_epochs = 2
    batch_size = 128
    learning_rate = 1e-3
    
    if torch.cuda.is_available() == True:
        model = autoencoder().cuda()
    else:
        model = autoencoder()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

    try:
        if torch.cuda.is_available() == True:
            model.load_state_dict(torch.load('saved/'+clas+'_conv_autoencoder.pth', map_location = 'cuda:0'))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load('saved/'+clas+'_conv_autoencoder.pth'))
        model.eval()
        return model
    except:
        for epoch in range(num_epochs):
            num_batch = 0
            bs = 0
            batch = []
            for data in tqdm.tqdm(train_data['X']):
                img = data.T
                if bs < batch_size:
                    batch.append(img)
                    bs += 1
                else:
                    batch = np.array(batch)
                    if torch.cuda.is_available() == True:
                        batch = Variable(torch.tensor(batch)).cuda()
                    else:
                        batch = Variable(torch.tensor(batch))
                    # ===================forward=====================
                    output = model(batch)
                    loss = criterion(output, batch)
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch = []
                    bs = 0
                    num_batch += 1
            # For Final batch 
            if len(train_data['X']) % batch_size != 0:
                batch = np.array(batch[:bs])
                if torch.cuda.is_available() == True:
                    batch = Variable(torch.tensor(batch)).cuda()
                else:
                    batch = Variable(torch.tensor(batch))
                # ===================forward=====================
                output = model(batch)
                loss = criterion(output, batch)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))

        torch.save(model.state_dict(), 'saved/'+clas+'_conv_autoencoder.pth')
        return model

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
        model = train_model(train_data, clas)
        test_using_cum_mse(model, test_data, clas)