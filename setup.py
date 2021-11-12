<<<<<<< HEAD
#! /bin/python3

"""
This script generate a reconstructed LoDoPaB-CT dataset.
The new dataset is saved in ".npy" files (sample by sample).

Author: Mateus Baltazar (github.com/MBaltz/)
"""

import os
import numpy as np
from math import ceil
from glob import glob
from cv2 import resize, INTER_LINEAR
from h5py import File as h5py_File
from dival.reference_reconstructors import get_reference_reconstructor


class LoDoPaB_Reconstructor():
    """
    The LoDoPaB reconstructed dataset was created to eliminate the need to
    reconstruct the synograms every time if you don't use it.

    So, this new dataset has the pairs (groundtruth, reconstructed observation).
    
    Parameters:
    -----------

    dir_in_LoDoPaB: directory of the LoDoPaB-CT trivial (and unziped)
        dataset. This directory contains the downloaded dataset.

    dir_out_LoDoPaB: directory of the new dataset. In this directory this
        script will extract the dataset in it's new format.

    impl: ["skimage", "astra_cpu", "astra_cuda"].
        Default is "astra_cuda"

    rec_type: method to reconstruct the dataset, example:
        ["fbp", "fbpunet", "iradomap", etc]
        Default is "fbp".

    dsize: tuple of integers which represent the shape of the samples of the
        new dataset. Example: dsize = (256, 256).
        If none, the size of sample will be de defalt size (362, 362).
        Default is None.
        
    """


    def __init__(self, dir_in_LoDoPaB, dir_out_LoDoPaB, impl="astra_cuda",
        rec_type="fbp", dsize=None):

        self.dir_in_LoDoPaB = dir_in_LoDoPaB
        self.dir_out_LoDoPaB = dir_out_LoDoPaB
        self.impl = impl
        self.rec_type = rec_type
        self.dsize = dsize # Tuple (linhes, columns)
        
        self.parts = ['train', 'validation', 'test']
        self.len_parts = {"train":35820, "validation":3522, "test":3553}
        
        self.reconstrutor = get_reference_reconstructor(
            self.rec_type, 'lodopab', impl=self.impl)


    def verify_trivial_dataset(self):
        """
        Verify os the trivial (downloaded and unziped) dataset is intact in the
        self.dir_in_LoDoPaB directory.

        Returns:
        --------

        True if the trivial dataset is intact. Else, False.
        """
        for part in self.parts:
            for tipo in ['observation', 'ground_truth']:
                first_file = os.path.join(
                    self.dir_in_LoDoPaB, '{}_{}_000.hdf5'.format(tipo, part))
                last_file = os.path.join(
                    self.dir_in_LoDoPaB, '{}_{}_{:03d}.hdf5'.format(
                        tipo, part, ceil(self.len_parts[part]/128)-1))

                if not (os.path.exists(first_file) and os.path.exists(last_file)):
                    print(f"Error: database in \"{self.dir_in_LoDoPaB}\" is not integrate.")
                    if not (os.path.exists(first_file)):
                        print(f"File \"{first_file}\" does not exists!")
                    else: print(f"File \"{last_file}\" does not exists!")
                    return False # Arquivos inexistentes
        return True


    def verify_processed_dataset(self):
        """
        Verify os the processed dataset is intact in the self.dir_out_LoDoPaB
        directory. The shape of samples if verify too, because a (256, 256) data
        set is different of a (362, 362) dataset.

        Returns:
        --------

        True if the processed dataset is intact. Else, False.
        """
        for part in self.parts:
            for tipo in ['observation', 'ground_truth']:
                first_file = os.path.join(
                    self.dir_out_LoDoPaB, '{}_{}_0.npy'.format(tipo, part))
                last_file = os.path.join(
                    self.dir_out_LoDoPaB, '{}_{}_{}.npy'.format(
                        tipo, part, self.len_parts[part]-1))

                if not (os.path.exists(first_file) and os.path.exists(last_file)):
                    print(f"Error: database in \"{self.dir_out_LoDoPaB}\" is not integrate.")
                    if not (os.path.exists(first_file)):
                        print(f"File \"{first_file}\" does not exists!")
                    else: print(f"File \"{last_file}\" does not exists!")
                    return False # Arquivos inexistentes

        if self.dsize != None:
            amostra = np.load(os.path.join(
                self.dir_out_LoDoPaB, 'observation_test_0.npy'), 'r').copy()
            if (amostra.shape[0], amostra.shape[1]) != self.dsize:
                print("Error: the dimension of the found sample is different:")
                print(f"\t({amostra.shape[0]}, {amostra.shape[1]}) != {self.dsize}")
                return False

        return True


    def __clear_out_directory(self):
        """
        Remove all files in self.dir_out_LoDoPaB directory.
        """
        answer = input(f"Clear \"{self.dir_out_LoDoPaB}\" directory? [y/n] ")
        if answer.lower() in ["y", "yes", "sim", "s"]:
            for f in glob(os.path.join(self.dir_out_LoDoPaB, "*")):
                print(f"> Removendo \"{f}\" ", end="\r"); os.remove(f)
            print(f"\n{self.dir_out_LoDoPaB} is clear.\n")
        else: print("Nothing to do."); exit()


    def reconstruct(self):
        """
        Generate the new dataset, reconstructing the synograms and saving the
        pairs (groundtruth, reconstructed synogram) individually for each sample
        in ".npy" files in self.dir_out_LoDoPaB directory.

        For the reconstruction, will be used the impl, reconstructed method end
        shape/size passed in the constructor parameters .

        Example of generated files:
        - ground_truth_train_0.npy
        - ground_truth_train_2004.npy
        - observation_train_93.npy
        - observation_train_3.npy
        """
        if not os.path.exists(self.dir_in_LoDoPaB):
            print(f"Directory {self.dir_in_LoDoPaB} does not exists!"); exit()
        if not os.path.exists(self.dir_out_LoDoPaB):
            print(f"Directory {self.dir_out_LoDoPaB} does not exists!"); exit()

        if not self.verify_trivial_dataset(): exit()

        if self.verify_processed_dataset():
            print(f"Dataset already created in {self.dir_out_LoDoPaB}"); exit()
        elif len(glob(os.path.join(self.dir_out_LoDoPaB, "*"))) > 0:
            self.__clear_out_directory()

        answer = input("You have more than 45GiB free in disk? [y,n] ")
        if not answer.lower() in ["y", "yes", "sim", "s"]:
            print("Nothing to do."); exit()

        print("[Extracting hdf5 files, Reconstructing, Resizing and Normalizing]")
        print(f"Saving Files:\n> Carregando...", end="\r")

        for p in self.parts:

            # Coleta os diretórios de todos os arquivos hdf5 de uma das
            # parts e um dos tipos por vez
            arqs_hdf5_obs = glob(os.path.join(self.dir_in_LoDoPaB, 'observation_'+p+'*'))
            arqs_hdf5_gt = glob(os.path.join(self.dir_in_LoDoPaB, 'ground_truth_'+p+'*'))
            # Ordena as listas pelo número do arquivo hdf5
            arqs_hdf5_obs.sort(key = lambda x: x.split("_")[-1])
            arqs_hdf5_gt.sort(key = lambda x: x.split("_")[-1])

            qnt_arquivos = len(arqs_hdf5_obs)
            for i_arq in range(qnt_arquivos):
                # Carrega conteúdo dos arquivos na memória
                i_arq_hdf5_obs = h5py_File(arqs_hdf5_obs[i_arq], 'r')
                i_arq_hdf5_gt = h5py_File(arqs_hdf5_gt[i_arq], 'r')
                id_arq_hdf5 = int(arqs_hdf5_obs[i_arq].split('_')[-1].split('.')[0])

                # Reconstrói (se for sinograma) e Salva as amostras (gt e rec_obs)
                qnt_amostras = len(i_arq_hdf5_obs['data'])
                for i_amostra in range(qnt_amostras):
                    conteudo_obs = i_arq_hdf5_obs['data'][i_amostra]
                    conteudo_gt = i_arq_hdf5_gt['data'][i_amostra]

                    # Reconstrução do sinograma
                    conteudo_obs = self.reconstrutor.reconstruct(conteudo_obs)
                    conteudo_obs = np.array(conteudo_obs)

                    # Redimensiona a imagem final caso tenha sido indicado
                    if self.dsize != None:
                        if self.dsize == (362, 362): self.dsize = None # Just Verify
                        else:
                            conteudo_obs = resize(
                                np.asarray(conteudo_obs), dsize=self.dsize,
                                interpolation=INTER_LINEAR)
                            conteudo_gt = resize(
                                np.asarray(conteudo_gt), dsize=self.dsize,
                                interpolation=INTER_LINEAR)

                    # Normalização entre [0, 1], pois o valor da
                    # reconstrução está entre números positivos e negativos
                    conteudo_obs -= np.min(conteudo_obs)
                    conteudo_obs /= np.max(conteudo_obs)
                    # Multiplica a reconstrução normalizada pelo maior número
                    # do groundtruth, ou seja: o valor branco do groundtruth
                    # será o valor branco da reconstrução (aproxima contraste)
                    conteudo_obs *= np.max(conteudo_gt)


                    # 128 porque dentro de um hdf5 tem no máximo 128 amostras,
                    # com exceção do último aquivo hdf5, que pode ter menos.
                    nome_arq_salvar_obs = 'observation_'+p+'_'+str(i_amostra+(id_arq_hdf5*128))+'.npy'
                    nome_arq_salvar_gt = 'ground_truth_'+p+'_'+str(i_amostra+(id_arq_hdf5*128))+'.npy'
                    dir_arq_salvar_obs = os.path.join(self.dir_out_LoDoPaB, nome_arq_salvar_obs)
                    dir_arq_salvar_gt = os.path.join(self.dir_out_LoDoPaB, nome_arq_salvar_gt)

                    # Indicativo da progressão da extração
                    print(f"> \"{nome_arq_salvar_obs}\", \"{nome_arq_salvar_gt}\"", end=" ")
                    print(f"\t[{str(i_amostra+(id_arq_hdf5*128))}/{self.len_parts[p]}]", end="\r")
                    
                    np.save(dir_arq_salvar_obs, conteudo_obs)
                    np.save(dir_arq_salvar_gt, conteudo_gt)

        print("Success! All dataset are reconstructed.")


# rec_db = LoDoPaB_Processor(
#     dir_in_LoDoPaB="/home/baltz/dados/Dados_2/tcc-database/unziped/", dir_out_LoDoPaB="/tmp/a",
#     impl="astra_cpu", rec_type="fbp", dsize=(256, 256))
# rec_db.reconstruct()
||||||| 997384f
=======
#! /bin/python3

"""
This script generate a reconstructed LoDoPaB-CT dataset.
The new dataset is saved in ".npy" files (sample by sample).

Author: Mateus Baltazar (github.com/MBaltz/)
"""

import os
import numpy as np
from math import ceil
from glob import glob
from cv2 import resize, INTER_LINEAR
from h5py import File as h5py_File
from dival.reference_reconstructors import get_reference_reconstructor


class LoDoPaB_Reconstructo():
    """
    The LoDoPaB reconstructed dataset was created to eliminate the need to
    reconstruct the synograms every time if you don't use it.

    So, this new dataset has the pairs (groundtruth, reconstructed observation).
    
    Parameters:
    -----------

    dir_in_LoDoPaB: directory of the LoDoPaB-CT trivial (and unziped)
        dataset. This directory contains the downloaded dataset.

    dir_out_LoDoPaB: directory of the new dataset. In this directory this
        script will extract the dataset in it's new format.

    impl: ["skimage", "astra_cpu", "astra_cuda"].
        Default is "astra_cuda"

    rec_type: method to reconstruct the dataset, example:
        ["fbp", "fbpunet", "iradomap", etc]
        Default is "fbp".

    dsize: tuple of integers which represent the shape of the samples of the
        new dataset. Example: dsize = (256, 256).
        If none, the size of sample will be de defalt size (362, 362).
        Default is None.
        
    """


    def __init__(self, dir_in_LoDoPaB, dir_out_LoDoPaB, impl="astra_cuda",
        rec_type="fbp", dsize=None):

        self.dir_in_LoDoPaB = dir_in_LoDoPaB
        self.dir_out_LoDoPaB = dir_out_LoDoPaB
        self.impl = impl
        self.rec_type = rec_type
        self.dsize = dsize # Tuple (linhes, columns)
        
        self.parts = ['train', 'validation', 'test']
        self.len_parts = {"train":35820, "validation":3522, "test":3553}
        
        self.reconstrutor = get_reference_reconstructor(
            self.rec_type, 'lodopab', impl=self.impl)


    def verify_trivial_dataset(self):
        """
        Verify os the trivial (downloaded and unziped) dataset is intact in the
        self.dir_in_LoDoPaB directory.

        Returns:
        --------

        True if the trivial dataset is intact. Else, False.
        """
        for part in self.parts:
            for tipo in ['observation', 'ground_truth']:
                first_file = os.path.join(
                    self.dir_in_LoDoPaB, '{}_{}_000.hdf5'.format(tipo, part))
                last_file = os.path.join(
                    self.dir_in_LoDoPaB, '{}_{}_{:03d}.hdf5'.format(
                        tipo, part, ceil(self.len_parts[part]/128)-1))

                if not (os.path.exists(first_file) and os.path.exists(last_file)):
                    print(f"Error: database in \"{self.dir_in_LoDoPaB}\" is not integrate.")
                    if not (os.path.exists(first_file)):
                        print(f"File \"{first_file}\" does not exists!")
                    else: print(f"File \"{last_file}\" does not exists!")
                    return False # Arquivos inexistentes
        return True


    def verify_processed_dataset(self):
        """
        Verify os the processed dataset is intact in the self.dir_out_LoDoPaB
        directory. The shape of samples if verify too, because a (256, 256) data
        set is different of a (362, 362) dataset.

        Returns:
        --------

        True if the processed dataset is intact. Else, False.
        """
        for part in self.parts:
            for tipo in ['observation', 'ground_truth']:
                first_file = os.path.join(
                    self.dir_out_LoDoPaB, '{}_{}_0.npy'.format(tipo, part))
                last_file = os.path.join(
                    self.dir_out_LoDoPaB, '{}_{}_{}.npy'.format(
                        tipo, part, self.len_parts[part]-1))

                if not (os.path.exists(first_file) and os.path.exists(last_file)):
                    print(f"Error: database in \"{self.dir_out_LoDoPaB}\" is not integrate.")
                    if not (os.path.exists(first_file)):
                        print(f"File \"{first_file}\" does not exists!")
                    else: print(f"File \"{last_file}\" does not exists!")
                    return False # Arquivos inexistentes

        if self.dsize != None:
            amostra = np.load(os.path.join(
                self.dir_out_LoDoPaB, 'observation_test_0.npy'), 'r').copy()
            if (amostra.shape[0], amostra.shape[1]) != self.dsize:
                print("Error: the dimension of the found sample is different:")
                print(f"\t({amostra.shape[0]}, {amostra.shape[1]}) != {self.dsize}")
                return False

        return True


    def __clear_out_directory(self):
        """
        Remove all files in self.dir_out_LoDoPaB directory.
        """
        answer = input(f"Clear \"{self.dir_out_LoDoPaB}\" directory? [y/n] ")
        if answer.lower() in ["y", "yes", "sim", "s"]:
            for f in glob(os.path.join(self.dir_out_LoDoPaB, "*")):
                print(f"> Removendo \"{f}\" ", end="\r"); os.remove(f)
            print(f"\n{self.dir_out_LoDoPaB} is clear.\n")
        else: print("Nothing to do."); exit()


    def reconstruct(self):
        """
        Generate the new dataset, reconstructing the synograms and saving the
        pairs (groundtruth, reconstructed synogram) individually for each sample
        in ".npy" files in self.dir_out_LoDoPaB directory.

        For the reconstruction, will be used the impl, reconstructed method end
        shape/size passed in the constructor parameters .

        Example of generated files:
        - ground_truth_train_0.npy
        - ground_truth_train_2004.npy
        - observation_train_93.npy
        - observation_train_3.npy
        """
        if not os.path.exists(self.dir_in_LoDoPaB):
            print(f"Directory {self.dir_in_LoDoPaB} does not exists!"); exit()
        if not os.path.exists(self.dir_out_LoDoPaB):
            print(f"Directory {self.dir_out_LoDoPaB} does not exists!"); exit()

        if not self.verify_trivial_dataset(): exit()

        if self.verify_processed_dataset():
            print(f"Dataset already created in {self.dir_out_LoDoPaB}"); exit()
        elif len(glob(os.path.join(self.dir_out_LoDoPaB, "*"))) > 0:
            self.__clear_out_directory()

        answer = input("You have more than 45GiB free in disk? [y,n] ")
        if not answer.lower() in ["y", "yes", "sim", "s"]:
            print("Nothing to do."); exit()

        print("[Extracting hdf5 files, Reconstructing, Resizing and Normalizing]")
        print(f"Saving Files:\n> Carregando...", end="\r")

        for p in self.parts:

            # Coleta os diretórios de todos os arquivos hdf5 de uma das
            # parts e um dos tipos por vez
            arqs_hdf5_obs = glob(os.path.join(self.dir_in_LoDoPaB, 'observation_'+p+'*'))
            arqs_hdf5_gt = glob(os.path.join(self.dir_in_LoDoPaB, 'ground_truth_'+p+'*'))
            # Ordena as listas pelo número do arquivo hdf5
            arqs_hdf5_obs.sort(key = lambda x: x.split("_")[-1])
            arqs_hdf5_gt.sort(key = lambda x: x.split("_")[-1])

            qnt_arquivos = len(arqs_hdf5_obs)
            for i_arq in range(qnt_arquivos):
                # Carrega conteúdo dos arquivos na memória
                i_arq_hdf5_obs = h5py_File(arqs_hdf5_obs[i_arq], 'r')
                i_arq_hdf5_gt = h5py_File(arqs_hdf5_gt[i_arq], 'r')
                id_arq_hdf5 = int(arqs_hdf5_obs[i_arq].split('_')[-1].split('.')[0])

                # Reconstrói (se for sinograma) e Salva as amostras (gt e rec_obs)
                qnt_amostras = len(i_arq_hdf5_obs['data'])
                for i_amostra in range(qnt_amostras):
                    conteudo_obs = i_arq_hdf5_obs['data'][i_amostra]
                    conteudo_gt = i_arq_hdf5_gt['data'][i_amostra]

                    # Reconstrução do sinograma
                    conteudo_obs = self.reconstrutor.reconstruct(conteudo_obs)
                    conteudo_obs = np.array(conteudo_obs)

                    # Redimensiona a imagem final caso tenha sido indicado
                    if self.dsize != None:
                        if self.dsize == (362, 362): self.dsize = None # Just Verify
                        else:
                            conteudo_obs = resize(
                                np.asarray(conteudo_obs), dsize=self.dsize,
                                interpolation=INTER_LINEAR)
                            conteudo_gt = resize(
                                np.asarray(conteudo_gt), dsize=self.dsize,
                                interpolation=INTER_LINEAR)

                    # Normalização entre [0, 1], pois o valor da
                    # reconstrução está entre números positivos e negativos
                    conteudo_obs -= np.min(conteudo_obs)
                    conteudo_obs /= np.max(conteudo_obs)
                    # Multiplica a reconstrução normalizada pelo maior número
                    # do groundtruth, ou seja: o valor branco do groundtruth
                    # será o valor branco da reconstrução (aproxima contraste)
                    conteudo_obs *= np.max(conteudo_gt)


                    # 128 porque dentro de um hdf5 tem no máximo 128 amostras,
                    # com exceção do último aquivo hdf5, que pode ter menos.
                    nome_arq_salvar_obs = 'observation_'+p+'_'+str(i_amostra+(id_arq_hdf5*128))+'.npy'
                    nome_arq_salvar_gt = 'ground_truth_'+p+'_'+str(i_amostra+(id_arq_hdf5*128))+'.npy'
                    dir_arq_salvar_obs = os.path.join(self.dir_out_LoDoPaB, nome_arq_salvar_obs)
                    dir_arq_salvar_gt = os.path.join(self.dir_out_LoDoPaB, nome_arq_salvar_gt)

                    # Indicativo da progressão da extração
                    print(f"> \"{nome_arq_salvar_obs}\", \"{nome_arq_salvar_gt}\"", end=" ")
                    print(f"\t[{str(i_amostra+(id_arq_hdf5*128))}/{self.len_parts[p]}]", end="\r")
                    
                    np.save(dir_arq_salvar_obs, conteudo_obs)
                    np.save(dir_arq_salvar_gt, conteudo_gt)

        print("Success! All dataset are reconstructed.")


# rec_db = LoDoPaB_Processor(
#     dir_in_LoDoPaB="/home/baltz/dados/Dados_2/tcc-database/unziped/", dir_out_LoDoPaB="/tmp/a",
#     impl="astra_cpu", rec_type="fbp", dsize=(256, 256))
# rec_db.reconstruct()
>>>>>>> f317cc1fc23f76bc12902ff370c98c441ade52c5
