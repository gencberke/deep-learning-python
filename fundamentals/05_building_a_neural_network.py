"""
1) Neural Network nedir?
2) Model, Layer ve parameter kavramları
3) Toy dataset oluşturmak
4) Forward ve Backward akışı
5) Loss function
6) Optimizer
7) Training loop
8) Modelin öğrendiğini gözlemlemek
"""

import torch
from torch import nn

print("===================================== 1) Neural Network Nedir?")

"""
1) Yapay sinir ağları (neural networks), birbirine bağlanan basit layer'lardan oluşur.
   Her katman (layer) girişte aldığı tensor'u weight ve bias değerleriyle işler
   ve aktivasyon fonksiyonu ile non-linear (doğrusal olmayan) bir dönüşüm uygular. 

        y = activation(Wx + b)

   PyTorch'ta bu yapıyı kurmak için: 
        1 → Katmanlar (nn.Linear, nn.Conv2d, vs.)
        2 → Aktivasyon fonksiyonları (nn.ReLU, nn.Sigmoid, vs.)
        3 → Model tanımı (nn.Module)
        4 → Kayıp fonksiyonu (Loss Function)
        5 → Optimizer (ör: SGD, Adam)
   Bu bileşenlerin hepsi bir araya geldiğinde öğrenebilen bir model oluşur.

   Bu notebook’ta bu kavramların hepsini en basit haliyle uygulayacağız.
   Küçük bir “toy dataset” (ör. y = 2x + 1 gibi) üzerinde model eğitimi yapacağız.
"""

print("===================================== 2) Model, Layer ve Parametre Kavramları:")

"""
2) Bir model PyTorch'ta nn.Module sınıfından türetilir. 
   Her modelin:
        - Katmanları (layers)
        - Parametreleri (weights & biases)
        - forward() metodu vardır (verinin modelden geçişi burada tanımlanır)
"""


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(in_features=1, out_features=4)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


model = SimpleNN()
print(model)

"""
    nn.Linear → y = xW^T + b işlemini yapar
    nn.ReLU → negatif değerleri sıfırlar, pozitifleri korur
    forward() → verinin modelden nasıl geçtiğini (flow) tanımlar
"""

print("===================================== 3) Toy Dataset Oluşturmak:")

"""
3) Küçük bir veri seti (toy dataset) oluşturacağız.
   Gerçek hayattaki veriler kusursuz değildir, bu yüzden veriye biraz rastgelelik (noise) ekleyeceğiz.
"""
torch.manual_seed(42)
x = torch.linspace(0, 5, steps=100).unsqueeze(1)
# .unsqueeze(1): tensora yeni bir boyut ekler → nn.Linear (batch, feature) beklediği için gerekli.
# Koymasaydık linspace bize (100,) boyutunda bir tensor üretecekti.
# nn.Linear, bunu 100 sample ve 1 feature olarak parametrize eder.

noise = torch.randn_like(x) * 0.2
# Gerçek verilerde hata ve gürültü (noise) olur.
# Burada Gaussian dağılımlı küçük rastgele gürültü ekliyoruz.
# Eğer hiç noise eklemezsek, model lineer fonksiyonu çok kolay ezberler (0 training effort).

y = 2 * x + 1 + noise
# Bu satır bizim hedef (target) değerlerimizi oluşturuyor: y = 2x + 1 + noise

print("===================================== 4) Forward ve Backward Akışı:")

"""
4) Modelin çıktısını hesaplayalım:
"""
y_prediction = model(x)
print(y_prediction[:5])

"""
    PyTorch tüm parametreleri autograd sistemiyle takip eder.
"""
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)

"""
    Bu parametreler (weight, bias) eğitim süresince güncellenecek.
    forward() → input’tan output’a
    backward() → loss’tan parametrelere (gradient hesaplama)
"""

print("===================================== 4.1) Weight ve Bias Kavramı:")

"""
    Bir yapay sinir ağı (neural network) katmanında iki temel öğrenilebilir parametre vardır:
        1) weight (ağırlık)
        2) bias (sapma değeri)

    Matematiksel olarak bir lineer katman (nn.Linear) şu işlemi yapar:
        y = W * x + b

    Buradaki:
        - W (weight): girişteki her bir feature'ın çıktı üzerindeki etkisini belirler.
                      Yani modelin 'eğimi'dir (hangi yön ve büyüklükte etkili olduğunu belirler).
        - b (bias): fonksiyonu yukarı/aşağı kaydıran sabit değerdir.
                    Eğer bias olmasa, model sadece orijinden (0,0) geçen doğrular öğrenebilirdi.
                    Bias fonksiyonu esnekleştirir.

    Eğitim sürecinde model, weight ve bias değerlerini loss fonksiyonunu minimize edecek şekilde
    günceller. Bu sayede model veriye daha uygun hale gelir.

    Her nn.Linear layer'ı otomatik olarak şu parametreleri taşır:
        - weight -> shape = (out_features, in_features)
        - bias   -> shape = (out_features,)
"""
y_prediction = model(x)
print(y_prediction[:5])

for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)

"""
    Çıktı yorumlama:
    -------------------------------------
    ▪ layer1.weight -> (4, 1) → 1 input, 4 nöron
    ▪ layer1.bias   -> (4,)   → her nöronun 1 bias'ı var
    ▪ layer2.weight -> (1, 4) → 4 input alıp 1 çıktı üretir
    ▪ layer2.bias   -> (1,)   → tek çıktı için tek bias
    ▪ tensor(..., grad_fn=<SliceBackward0>) → bu değerler autograd graph'ına bağlıdır, türev alınabilir.

    Eğitim boyunca:
        - weight ve bias rastgele başlatılır (random initialization).
        - loss.backward() çağrıldığında bu parametrelerin .grad değerleri hesaplanır.
        - optimizer.step() bu parametreleri gradient doğrultusunda günceller.
        - Böylece model yavaş yavaş doğru fonksiyonu öğrenir.
"""

print("===================================== 5) Loss Function:")

"""
5) Loss, tahmin edilen değerle gerçek değer arasındaki farkı ölçer.
   Modelin ana amacı bu farkı minimize etmektir.
"""
loss_fn = nn.MSELoss()  # Mean Squared Error

y_prediction = model(x)
loss = loss_fn(y_prediction, y)
print("Initial Loss:", loss)

# Loss function ileride daha detaylı ele alınacak.
# Başlangıçta loss’un yüksek olması anormal değildir.
# Önemli olan epoch'lar ilerledikçe loss’un düzenli olarak azalmasıdır.

print("===================================== 6) Optimizer:")

"""
6) Optimizer gradient'leri kullanarak parametreleri günceller.
   Yani modelin 'öğrenme mekanizması'dır.
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# SGD (Stochastic Gradient Descent) → klasik gradient descent algoritması.
# lr (learning rate): adım boyu, ne kadar büyükse o kadar hızlı ama riskli öğrenme.

"""
optimizer.step():
    -> backward() ile hesaplanan gradient'leri kullanarak parametreleri günceller.
    -> Yani weight ve bias değerleri loss'u azaltacak yönde değiştirilir.

optimizer.zero_grad():
    -> Gradient belleğini temizler.
    -> backward() çağrısı her seferinde mevcut gradientleri eklediği için
       önceki iterasyondan kalan gradientler sıfırlanmazsa birikir (accumulate).
    -> Bu yüzden her epoch başında gradientleri sıfırlarız.
"""

print("===================================== 7) Training Loop:")

"""
7) Modeli eğitmek için her epoch'ta aynı işlemler tekrarlanır.
"""
epochs = 100
for epoch in range(epochs):
    # 1) forward
    y_prediction = model(x)

    # 2) loss hesapla
    loss = loss_fn(y_prediction, y)

    # 3) backward
    loss.backward()

    # 4) parametreleri güncelle
    optimizer.step()

    # 5) gradientleri sıfırla
    optimizer.zero_grad()

    if (epoch + 1) % 40 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

"""
loss.backward() → gradient hesaplar
optimizer.step() → parametreleri günceller
optimizer.zero_grad() → eski gradientleri sıfırlar
"""

print("===================================== 7.1) Epoch Kavramı:")

"""
    "Epoch" bir eğitim döngüsünün tamamını tanımlar.

    ▪ Tanım:
        Bir epoch, modelin eğitim verisinin tamamını bir kez görüp
        forward() → backward() → optimizer.step() işlemlerini tamamladığı tam bir turdur.

    ▪ Neden birden fazla epoch?
        Model tek bir turda genellikle yeterince öğrenemez çünkü weight ve bias değerleri
        her adımda küçük miktarlarda güncellenir. 
        Bu nedenle veriyi defalarca görmesi gerekir.
        Her epoch'ta model biraz daha iyi hale gelir, loss genellikle giderek azalır.

    ▪ Epoch - Iteration - Batch farkı:
        Dataset → eğitimde kullanılan tüm örnekler (ör. 1000 sample)
        Batch → bir seferde işlenen örnek grubu (ör. 100 sample)
        Iteration → her batch'in işlenmesi (ör. 1000/100 = 10 iteration)
        Epoch → tüm dataset'in baştan sona 1 kez işlenmesi (10 iteration = 1 epoch)

    ▪ Örnek:
        1000 örnek, batch=100 → 1 epoch = 10 iteration
        200 epoch → 200 × 10 = 2000 iteration

    ▪ Epoch sayısı nasıl seçilir?
        - Az epoch → model yeterince öğrenemez (underfitting)
        - Çok epoch → model aşırı öğrenir (overfitting)
        - Uygun epoch sayısı → loss’un düşüp sabitlendiği nokta

        Basit linear toy model → genelde 100–300 epoch yeterlidir.
        Daha karmaşık modellerde (CNN, RNN) epoch sayısı artar.

    ▪ Loss - Epoch ilişkisi (örnek):
            Epoch 1   | Loss: 31.2060
            Epoch 40  | Loss: 4.1224
            Epoch 80  | Loss: 1.0820
            Epoch 120 | Loss: 0.3821
            Epoch 160 | Loss: 0.1563
            Epoch 200 | Loss: 0.0897

        Loss azaldıkça model veriye daha çok yaklaşır.
        Bu azalan ama stabilize olan loss eğrisi modelin doğru öğrendiğini gösterir.

    ▪ Özetle:
        - Epoch → modelin tüm veriyi baştan sona bir kez işlemesi
        - Iteration → bir batch'in işlenmesi
        - Batch → bir seferde kullanılan örnek sayısı
        - Çok az epoch → underfitting
        - Çok fazla epoch → overfitting
        - Epoch arttıkça loss azalır ama bir noktadan sonra sabitlenir (convergence)

    💡 Kısa tanım:
        Epoch, modelin tüm eğitim verisini bir kez işleyip ağırlıklarını güncellediği döngüdür.
        Eğitim genellikle onlarca veya yüzlerce epoch sürer, her seferinde loss biraz daha azalır.
"""

print("===================================== 8) Modelin Öğrendiğini Gözlemlemek:")

with torch.no_grad():
    y_eval = model(x)

"""
Loss azalıyor → model doğru yönde öğreniyor.
Model artık yaklaşık olarak y ≈ 2x + 1 fonksiyonunu öğrenmiş durumda.
"""

print("===================================== *) Hot-Takes:")

"""
- nn.Module -> PyTorch'taki tüm modellerin temelidir.
- forward() -> input’tan output’a veri akışı.
- backward() -> autograd sistemiyle gradient hesaplama.
- optimizer.step() -> parametreleri günceller.
- zero_grad() -> gradient belleğini temizler.
- loss.backward() + optimizer.step() döngüsü → “learning”.
- Bu yapı tüm training loop’ların temel iskeletidir.
"""

print("===================================== 📘 Summary Table: Neural Network on Toy Dataset")

"""
| Konsept | Açıklama | Örnek Kod |
|----------|-----------|-----------|
| nn.Module | Model yapısı tanımlamak için PyTorch sınıfı | class Model(nn.Module): ... |
| nn.Linear | Fully connected layer (W*x + b) | nn.Linear(1,4) |
| forward() | Verinin modeldeki akışını belirler | def forward(self, x): ... |
| MSELoss | Tahmin ve gerçek farkını ölçer | nn.MSELoss() |
| optimizer | Gradient’e göre ağırlıkları günceller | torch.optim.SGD(...) |
| backward() | Gradient hesaplamasını başlatır | loss.backward() |
| zero_grad() | Gradient belleğini temizler | optimizer.zero_grad() |
| Training Loop | Modelin öğrenme süreci (forward → loss → backward → update) | for epoch in range(...): ... |
| model.parameters() | Öğrenilebilir tensor’lar (weights & biases) | model.parameters() |
"""
