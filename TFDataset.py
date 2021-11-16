import os
import numpy as np
from math import ceil
from tensorflow.keras.utils import Sequence

import cv2

class TFDataset(Sequence):

    def __init__(self, dir_rec_lodopab, part, batch_size,
        shuffle=False, return_with_channel=True):

        self.dir_rec_lodopab = dir_rec_lodopab
        self.part = part
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_with_channel = return_with_channel

        self.len_parts = {"train":35820, "validation":3522, "test":3553}

        self.indexes = [x for x in range(self.len_parts[self.part])]
        self.on_epoch_end()


    # Retorna dois np.array: [x/obs], [y/gt]
    def __data_generation(self, list_idxs):
        vet_gt = []
        vet_obs = []
        for idx in list_idxs:
            nome_gt = f"ground_truth_{self.part}_{idx}.npy"
            nome_obs = f"observation_{self.part}_{idx}.npy"
            dir_gt = os.path.join(self.dir_rec_lodopab, nome_gt)
            dir_obs = os.path.join(self.dir_rec_lodopab, nome_obs)
            vet_gt.append(np.asarray(np.load(dir_gt, 'r').copy()))
            vet_obs.append(np.asarray(np.load(dir_obs, 'r').copy()))
        
        vet_obs, vet_gt = np.asarray(vet_obs), np.asarray(vet_gt)
        if self.return_with_channel:
            vet_obs = np.expand_dims(vet_obs, axis=-1) 
            vet_gt = np.expand_dims(vet_gt, axis=-1) 

        return vet_obs, vet_gt


    # Seleciona a lista de ids e requisita amostras ao __data_generation
    def __getitem__(self, idx):
        # Prepara o range de ids sem dar overflow
        range_idx = range(idx*self.batch_size, (idx+1)*self.batch_size)
        if (idx+1)*self.batch_size >= self.len_parts[self.part]:
            range_idx = range(idx*self.batch_size, self.len_parts[self.part])
        list_idxs = [self.indexes[x] for x in range_idx]
        
        # Requisita amostras. list_x/y.shape: (batch_size, linha, coluna, 1)
        list_x, list_y = self.__data_generation(list_idxs)
        return list_x, list_y


    def __len__(self):
        return ceil(self.len_parts[self.part] / self.batch_size)


    # Aleatoriza os indices a cada fim de época se self.shuffle == True
    def on_epoch_end(self):
        if self.shuffle == True: np.random.shuffle(self.indexes)




if __name__ == '__main__':

    batch_size = 10
    part = "test"

    dataset = TFDataset(
        "/home/baltz/dados/Dados_2/tcc-database/unziped",
        "/home/baltz/dados/Dados_2/tcc-database/reco_dataset",
        part, batch_size=batch_size, shuffle=False, return_with_channel=True)
    
    # x, y = dataset.__getitem__(dataset.len_parts[part]-1)
    x, y = dataset.__getitem__(ceil(dataset.len_parts[part]/batch_size)-1)
    print(x.shape)
    print(y.shape)


    for i in range(len(x)):
        y_cv2 = np.squeeze(y[i]*255, axis=-1)

        x_cv2 = np.squeeze(x[i]*255, axis=-1)

        vis = np.concatenate((y_cv2, x_cv2), axis=1)

        cv2.imwrite("aux/concat_"+str(i)+".png", vis)
