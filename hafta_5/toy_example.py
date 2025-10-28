import torch

x = [[1,2],[4,5],[7,8]]
y = [[3],[4],[5],[7]]

x = torch.tensor(x).float()
y = torch.tensor(y).float()

device = torch.device("cpu")

X = x.to(device)
Y = y.to(device)

print(X)
print(Y)
print("============================")

# pytorchta bir tensor cpu bir tensor gpu da ise bunlarla beraber işlem yapamayız. device'ı tanımlayıp tensorları
# device'a taşıyacağız.

class MyNeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = torch.nn.Linear(2, 8) ## hidden layer'a input
        self.hidden_layer_activation = torch.nn.ReLU() # activation layer'ım (fonksiyon ReLU seçtim)
        self.hidden_to_output_layer = torch.nn.Linear(8, 1) # outputum

    # burada layerlarımızdan ilerleyişimiz gösteriliyor zaten basitçe anlaşılabiliyor
    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x

print(torch.nn.Linear(2, 7)) # in features: girdilerim out features: nöronlarım
print("============================")
print(MyNeuralNet())
my_net = MyNeuralNet().to(device)
print("")

my_net.input_to_hidden_layer.bias.data.fill_(0)
print(my_net.input_to_hidden_layer.bias.data)

for param in my_net.parameters():
    print(param)

print("=============================")

loss = torch.nn.MSELoss()
_Y = my_net(X) # iterasyonu yaptık forward progression
loss_value = loss(_Y, _Y) # burada loss(_Y, Y) yaptı hoca ama ben yapamadım?
print(loss_value)

print("=============================")

from torch.optim import SGD # sgd -> bir örnek al gradiant ve weight (yani tek örnek üzerinden öğrenme, çok büyük dataset
# lerde bu yüzden kullanılmıyor genelde )
opt = SGD(my_net.parameters(), lr=0.001) # lr = learning rate

my_net.parameters()

print("============================")
# this is not a part of building a model this is just a illustration of how we perform
opt.zero_grad()
loss_value = loss_func(my_net(X), Y) # error??
loss_value.backward()
opt.step()

# epok? hyperparameter? hyperparameter tuning önemli dedi hoca. belli bir trashold değerden sonrası
# başarılı olarak mı değerlendiriliyor?
print("============================")

loss_history = []
epoch = 50

for _ in range(epoch):
    opt.zero_grad()
    loss_value = loss(my_net(X), Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value.item())