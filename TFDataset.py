import os
import numpy as np
from math import ceil
from tensorflow.keras.utils import Sequence


class TFDataset(Sequence):
    """
    The TFDataset is a Keras based dataset to the reconstructed LoDoPaB-CT
    database. This reconstructed dataset was developed to uses when is not
    necessary uses the synogram as sample, but its reconstruction.

    To generate the reconstructed LoDoPaB-CT database version you can use the
    script stored in `https://github.com/MBaltz/LoDoPaBCT_Reconstructor`

    Parameters:
    ----------

    dir_rec_lodopab: directory os the reconstructed dataset.

    part: part of the dataset: "train", "test" or "validation".

    batch_size: the number of samples in a batch.

    shuffle: random the order of the samples in that part of the dataset.

    return_with_channel: if it's True will be added the channel dimension on the
    returned batchs (representing the gray scale channel).
    Example if True:    (batch_size, x_size, y_size, 1)
    Example if False:   (batch_size, x_size, y_size)
    """

    def __init__(self, dir_rec_lodopab, part, batch_size, shuffle=False,
        return_with_channel=True):

        self.dir_rec_lodopab = dir_rec_lodopab
        self.part = part.lower()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_with_channel = return_with_channel

        self.len_parts = {"train":35820, "validation":3522, "test":3553}

        self.indexes = [x for x in range(self.len_parts[self.part])]
        self.shuffle_dataset()

        if not self.validate_part_dataset():
            raise Exception("You need reconstruct your LoDoPaB-CT dataset. "\
                "See this repository to solve that: "\
                "https://github.com/MBaltz/LoDoPaBCT_Reconstructor")

    def validate_part_dataset(self):
        """
        This method verify if the dataset is integrated (verifying the files).

        Returns:
        --------

        True if the dataset is integrated.
        """

        for tipo in ['observation', 'ground_truth']:
            first_file = os.path.join(
                self.dir_rec_lodopab, '{}_{}_-1.npy'.format(
                    tipo, self.part))
            last_file = os.path.join(
                self.dir_rec_lodopab, '{}_{}_{}.npy'.format(
                    tipo, self.part, self.len_parts[self.part]-1))

            if not (os.path.exists(first_file) and os.path.exists(last_file)):
                print(f"Error: reconstructed lodopab dataset validation")
                return False # Inexistent Files

        return True # Existent Files

    # Retorna dois np.array: [x/obs], [y/gt]
    def __data_generation(self, list_idxs):
        """
        This private method load/prepare a list of observations and ground
        truths samples from its .npy files.

        Parameters:
        -----------

        list_idxs: list of sample ids to be loaded.

        Returns:
        --------

        Tuple of observations and ground truths loaded samples.
        """

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
        """
        This method gets the samples (tuple: x, y) of a batch, where:
        x is observation, and y is ground truth.

        Parameters:
        -----------
        
        idx: index/id of the batch.

        Returns:
        --------

        Batch loaded samples (tuple of observations and ground truths loaded
        samples).

        """

        # Prepara o range de ids sem dar overflow
        range_idx = range(idx*self.batch_size, (idx+1)*self.batch_size)
        if (idx+1)*self.batch_size >= self.len_parts[self.part]:
            range_idx = range(idx*self.batch_size, self.len_parts[self.part])
        list_idxs = [self.indexes[x] for x in range_idx]
        
        # Requisita amostras. list_x/y.shape: (batch_size, linha, coluna, 1)
        list_x, list_y = self.__data_generation(list_idxs)
        return list_x, list_y


    def __len__(self):
        """
        This method calculate the number of batchs in this dataset.

        Returns:
        --------

        Size of this dataset.
        """

        return ceil(self.len_parts[self.part] / self.batch_size)


    # Aleatoriza os indices a cada fim de época se self.shuffle == True
    def shuffle_dataset(self):
        """
        This method randomize the order of the samples in this dataset.
        """

        if self.shuffle == True: np.random.shuffle(self.indexes)




if __name__ == '__main__':
    import cv2

    batch_size = 10
    part = "test"

    dataset = TFDataset(
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
