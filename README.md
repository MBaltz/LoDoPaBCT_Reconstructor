# LoDoPaB-CT Reconstructor
This script generate a reconstructed LoDoPaB-CT dataset.

The synograms are reconstructed to increase performance, if you will not use it's.

The new dataset is composed by ".npy" files. Each file represents a sample.

---

Example of use:

```python
rec_db = LoDoPaB_Processor(
    dir_in_LoDoPaB="./trivial_db/unziped/", dir_out_LoDoPaB="./rec_db/",
    impl="astra_cuda", rec_type="fbp", dsize=(256, 256))
rec_db.reconstruct()
```

---

Example of generated files:
- ground_truth_train_0.npy
- ground_truth_train_2004.npy
- observation_train_93.npy
- observation_train_3.npy

---

_Thank you [DIVal](https://github.com/jleuschn/dival)!_
