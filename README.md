# Application of SwinV2-T on the Pneumonia Dataset

## Parameters
- Model Configuration:
  - Embedding Dimension: 96
  - Depths: [2, 2, 6, 2]
  - Number of Heads: [2, 4, 8, 16]
  - Window Size: 8
  - Absolute Position Encoding: False
  - Patch Normalization: True
  - Number of Classes: 2
  - Drop Path Rate: 0.5
 
- Optimizer and Loss Function:
  - Learning Rate: 1e-06
  - Betas: (0.9, 0.999)
  - Weight Decay: 0.01
  - Loss Function: CrossEntropyLoss()

- Training Parameters:
  - Batch Size: 256
  - Image Size: 256
  - Device: cuda
  - Number of Epochs: 30

## Training
##### Epoch: 1 | train_loss: 0.658464 | train_acc: 0.471540 | val_loss: 1.213098 | val_acc: 0.319754 | lr: 0.00000100
##### Epoch: 2 | train_loss: 0.511008 | train_acc: 0.566158 | val_loss: 0.842121 | val_acc: 0.471168 | lr: 0.00000100
##### Epoch: 3 | train_loss: 0.434949 | train_acc: 0.647321 | val_loss: 0.748352 | val_acc: 0.597098 | lr: 0.00000100
##### Epoch: 4 | train_loss: 0.373820 | train_acc: 0.706411 | val_loss: 0.640981 | val_acc: 0.670573 | lr: 0.00000100
##### Epoch: 5 | train_loss: 0.332641 | train_acc: 0.762897 | val_loss: 0.558566 | val_acc: 0.736607 | lr: 0.00000100
##### Epoch: 6 | train_loss: 0.310131 | train_acc: 0.787760 | val_loss: 0.526036 | val_acc: 0.736979 | lr: 0.00000100
##### Epoch: 7 | train_loss: 0.279436 | train_acc: 0.815724 | val_loss: 0.485411 | val_acc: 0.760975 | lr: 0.00000100
##### Epoch: 8 | train_loss: 0.263184 | train_acc: 0.834635 | val_loss: 0.486995 | val_acc: 0.759673 | lr: 0.00000100
##### Epoch: 9 | train_loss: 0.252353 | train_acc: 0.841580 | val_loss: 0.452744 | val_acc: 0.787760 | lr: 0.00000100
##### Epoch: 10 | train_loss: 0.243718 | train_acc: 0.854725 | val_loss: 0.435229 | val_acc: 0.822359 | lr: 0.00000100
##### Epoch: 11 | train_loss: 0.224149 | train_acc: 0.871900 | val_loss: 0.405211 | val_acc: 0.829613 | lr: 0.00000100
##### Epoch: 12 | train_loss: 0.206025 | train_acc: 0.878162 | val_loss: 0.437871 | val_acc: 0.810082 | lr: 0.00000100
##### Epoch: 13 | train_loss: 0.196171 | train_acc: 0.887401 | val_loss: 0.405618 | val_acc: 0.825707 | lr: 0.00000100
##### Epoch: 14 | train_loss: 0.190114 | train_acc: 0.895089 | val_loss: 0.432887 | val_acc: 0.818266 | lr: 0.00000100
##### Epoch: 15 | train_loss: 0.176067 | train_acc: 0.902158 | val_loss: 0.413758 | val_acc: 0.823475 | lr: 0.00000100
##### Epoch: 16 | train_loss: 0.184269 | train_acc: 0.888393 | val_loss: 0.402322 | val_acc: 0.830729 | lr: 0.00000100
##### Epoch: 17 | train_loss: 0.171889 | train_acc: 0.904886 | val_loss: 0.431907 | val_acc: 0.824219 | lr: 0.00000100
##### Epoch: 18 | train_loss: 0.178078 | train_acc: 0.898748 | val_loss: 0.431624 | val_acc: 0.828497 | lr: 0.00000100
##### Epoch: 19 | train_loss: 0.164385 | train_acc: 0.909040 | val_loss: 0.441365 | val_acc: 0.829799 | lr: 0.00000100
##### Epoch: 20 | train_loss: 0.158487 | train_acc: 0.912760 | val_loss: 0.427060 | val_acc: 0.830729 | lr: 0.00000100
##### Epoch: 21 | train_loss: 0.153747 | train_acc: 0.916915 | val_loss: 0.456757 | val_acc: 0.811756 | lr: 0.00000100
##### Epoch: 22 | train_loss: 0.159078 | train_acc: 0.913008 | val_loss: 0.455332 | val_acc: 0.814174 | lr: 0.00000100

Epoch 00022: reducing learning rate of group 0 to 1.0000e-07.

##### Epoch: 23 | train_loss: 0.155827 | train_acc: 0.921875 | val_loss: 0.454591 | val_acc: 0.828125 | lr: 0.00000010
##### Epoch: 24 | train_loss: 0.165102 | train_acc: 0.911334 | val_loss: 0.458545 | val_acc: 0.820685 | lr: 0.00000010
##### Epoch: 25 | train_loss: 0.150349 | train_acc: 0.915365 | val_loss: 0.451405 | val_acc: 0.835751 | lr: 0.00000010
##### Epoch: 26 | train_loss: 0.155043 | train_acc: 0.917597 | val_loss: 0.449801 | val_acc: 0.820312 | lr: 0.00000010
##### Epoch: 27 | train_loss: 0.151180 | train_acc: 0.913938 | val_loss: 0.472542 | val_acc: 0.820685 | lr: 0.00000010
##### Epoch: 28 | train_loss: 0.150512 | train_acc: 0.915489 | val_loss: 0.443284 | val_acc: 0.827195 | lr: 0.00000010

Epoch 00028: reducing learning rate of group 0 to 1.0000e-08.

##### Epoch: 29 | train_loss: 0.154057 | train_acc: 0.912946 | val_loss: 0.454844 | val_acc: 0.827567 | lr: 0.00000001
##### Epoch: 30 | train_loss: 0.148519 | train_acc: 0.924231 | val_loss: 0.453439 | val_acc: 0.831845 | lr: 0.00000001
