from dival.reference_reconstructors import get_reference_reconstructor
from dival import get_standard_dataset
import numpy as np

rec = get_reference_reconstructor('fbp', 'lodopab', impl='astra_cpu')
dataset = get_standard_dataset('lodopab', impl='astra_cpu')


# A img de indice 3550 é interessante de observar (gt mais escuro)
for i in range(3553):
    x, y = dataset.get_data_pairs_per_index('test', i)[0]
    x, y = np.asarray(x), np.asarray(y)

    x_reco = rec.reconstruct(x)
    x_reco = np.asarray(x_reco)



    x_reco_mod = np.exp(x_reco)
    print(f"min: {np.min(x_reco_mod)}, max: {np.max(x_reco_mod)}")

    x_reco_mod = x_reco_mod - np.min(x_reco_mod)
    x_reco_mod = x_reco_mod / np.max(x_reco_mod)
    x_reco_mod = x_reco_mod * np.max(y)

    # >>>>>>> Essa multiplicação não é boa!! TODO: MELHORAR ISSO!


    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(1,3))
    
    fig.add_subplot(1, 3, 1)
    # plt.figure('reco')
    plt.imshow(x_reco[:,:], cmap='gray')
    
    fig.add_subplot(1, 3, 2)
    # plt.figure('reco_mod')
    plt.imshow(x_reco_mod[:,:], cmap='gray')

    fig.add_subplot(1, 3, 3)
    # plt.figure('gt')
    plt.imshow(y[:,:], cmap='gray')

    plt.show()