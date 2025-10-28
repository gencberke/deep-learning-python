# torch'un data set importları
import torch

from hafta_5.toy_example import my_net

# data setten tensor set etme

# data loader (ds, batch_size = 2, shuffle=True) batchsize önemli parametre

# neural network'ü aynı şekilde yaratıyoruz bir şey değişmiyor

# bu sefer tek farkl loss_historys sonrası olan for döngüsünde 2. bir for döngüsü olması gerek neden?
"""
for data in dl:
    x, y = data
"""

val_x = [[10], [20], [30]]
val_x = torch.Tensor(val_x).float()
my_net(val_x)

# torch summary'yi import et kesinlikle çok güzel özet çıkarıyor