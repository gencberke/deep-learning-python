"""
1) Tensor nedir? (ve neden PyTorch’un kalbidir)
2) Tensor özellikleri (shape, dtype, device)
3) Tensor oluşturma yolları
4) Tensor’lar arasında işlemler (ekleme, çarpma, reshape)
5) Numpy ile fark ve dönüşüm (array ↔ tensor)
"""
from random import randint
import torch
import numpy as np

# reproducibility - her çalıştırmada aynı random sonuçlar
torch.manual_seed(42)

print("===================================== 1) Tensor nedir?")
"""
1) Tensor nedir?

    Tensor PyTorch'taki core data typelardan biridir. aslında cpu'da veya gpu'da yaşayabilen çok boyutlu bir array'dir
(multi-dimensional array) ve autograd'i destekler deep learning için. 

    A scalar -> tek sayı (örnek: 5)
    A vector -> tek boyutlu liste (örnek: [1, 2, 3])
    A matrix -> iki boyutlu tablo (örnek: [[1,2],[3,4]])
    A tensor -> bunların genelleştirilmiş halidir; 0, 1, 2, 3, … n boyutlu olabilir.
"""

# burada python listesini torch tensor objesine dönüştürdük. yani artık üzerinde matematiksel
# işlemler yapılmasını destekliyor.
x = torch.tensor([[1, 2], [3, 4]])
print(x)

print("===================================== 2) Tensor özellikleri:")

"""
2) Tensor özellikleri:
"""
print(x.shape)  # tensorun boyutu (rows, columns)
print(x.dtype)  # veri tipi (örnek: int64 veya float32)
print(x.device)  # CPU mu GPU mu?
print(x.ndim)  # kaç boyutlu? (2D tensor)

print("===================================== 3) Tensor oluşturma yolları:")

"""
3) Tensor oluşturma yolları:
"""

# tüm elemanları "sıfır"'dan oluşan 3 satır 4 sütunluk bir tensor (multi-dimensional array)
zeros = torch.zeros((3, 4))

# tüm elemanları "bir"'den oluşan 3 satır 4 sütunluk bir tensor (multi-dimensional array)
ones = torch.ones((3, 4))

# 0, 1 arası rastgele değerler üretir. her çalıştırıldığında farklı sonuç çıkarır
random_tensor = torch.rand((3, 4))
print(random_tensor)
# Rastgele normalize değerler üretmek

# -∞ ile +∞ arası değerler üretir, ortalama 0 civarında dağılır (Gaussian distribution)
# dikkat! yukarıdakinden sadece bir harf farklı: rand -> randn
random_tensor = torch.randn((3, 4))
print(random_tensor)
# Model initialization (örneğin weight’ler) gibi yerlerde kullanılır

# verilen değerler arası rastgele int içerikli tensor üretir:
my_randint = torch.randint(low=0, high=5, size=(3, 4))
print(my_randint)

# step kadar arttırıp end'e kadar gider (end hariçtir)
a = torch.arange(start=0, end=30, step=3)

# start ve end arasında eşit aralıklı 10 değer üretir (end dahildir)
b = torch.linspace(start=0, end=30, steps=10)

print(a)
print(b)

"""
    Tensor oluştururken dtype ve device belirlenebilir. dtype tensorun data type'ını temsil ederken device ise cpu'da mı
gpuda mı oluşturulacağını temsil eder. bu önemlidir çünkü arada hem performans farkı çok yüksektir hem iki tensor arası
işlem yapılacaksa aynı birimlerde depolanması gerekmektedir. biz şimdilik cpu kullanıcaz. 
"""

x = torch.ones((3, 4), dtype=torch.float32, device='cpu')
print(x)

"""
    "cuda" yazdığında GPU’da oluşturulur ama önce torch.cuda.is_available() kontrolü yapılmalı.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand((3, 3), device=device)
print(x.device)

print("===================================== 4) Tensor Operations:")

"""
4) Tensor Operations: 
"""

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cpu')
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device='cpu')

# “element-wise” = her bir hücre kendi karşılığıyla işlem görür.
print(a + b)  # element-wise addition
print(a - b)  # element-wise subtraction
print(a * b)  # element-wise multiplication
print(a / b)  # element-wise division

# matrix multiplication (2x2 @ 2x2 -> 2x2)
# not: eğer bu işlem vector'lar üzerinde yapılırsa buna dot product denir.
matmul_result = torch.matmul(a, b)
print(matmul_result)

# reshaping tensors: tensorun verisini değiştirmeden boyutlarını yeniden düzenler
# örneğin içinde 12 değer olan bir matrisi 3x4 e reshapeleyebiliriz.
x = torch.arange(0, 12)
reshaped = x.reshape(3, 4)
print(reshaped)

print("===================================== *) Hot-Takes:")

"""
    1) numpy ve pytorch birbiriyle iletişimde olabilir.
"""

torch_tensor = torch.tensor([[1, 2], [3, 4]])
numpy_array = torch_tensor.numpy()  # torch tensor to -> numpy array
print(type(numpy_array), numpy_array)

numpy_array_second = np.array([[5, 6], [7, 8]])
torch_tensor_second = torch.from_numpy(numpy_array_second).float()  # numpy array to -> torch tensor
                                                                    # torch.from_numpy aynı belleği paylaşır yani numpy array değişirse
                                                                    # torch_tensor da değişir ama ayrı nesneleri göstersin istiyorsak
                                                                    # torch.tensor() ile yeni bir tensor yaratıp ona numpy array'i kopyalarız
print(type(torch_tensor_second), torch_tensor_second)

"""
    2) bahsettiğimiz gibi iki tensor öğesi arasında işlem yapılacaksa bu ikisi aynı
        birimde initialize edilmiş olması gerekir. bir tensor'u taşımak için <tensor ismi>.to kullanılabilir
"""
my_tensor = torch.tensor([[1, 2], [3, 4]])
my_tensor = my_tensor.to("cpu")  # my_tensor'u taşımamız gerektiğinde "<tensor ismi> = " yaparız yoksa kopyasını
# taşır tensorun kendisini değil

"""
    3) tensor initialization shortcuts: 
"""
eye = torch.eye(3)  # birebir (identity) matris
full = torch.full((3, 3), 5)  # tüm elemanları verilen fill_value olan matris
randn = torch.randn(3, 3)  # Gaussian dağılımından rastgele sayılar (ortalama 0, varyans 1)

"""
    4) requires_grad=true nedir?
        bu ilerideki notlarda daha detaylı incelenecek olsada backward progression yaparken işimize yarayacak.
        geçmişi hafızada tutmaya yarıyacak. true'ya çekmediğimiz sürece geçmişi akılda tutmayacak. 
"""
x = torch.tensor([[2.0, 3.0]], requires_grad=True)
y = (x ** 2).sum()  # scalar output için .sum() çünkü backward yapmak için tek dimension şimdilik
y.backward()
print(x.grad)  # gradient = 2*x yani tensor([[4., 6.]])

print("===================================== 📘 Summary Table: Tensor Fundamentals")

"""
| Konsept | Açıklama | Örnek Kod |
|----------|-----------|-----------|
| tensor | PyTorch'un temel veri yapısı (multi-dimensional array). | x = torch.tensor([[1,2],[3,4]]) |
| shape / dtype / device | Boyut, veri tipi ve işlem birimi (CPU/GPU). | x.shape, x.dtype, x.device |
| tensor initialization | Farklı tensor oluşturma yolları. | torch.zeros(), torch.ones(), torch.rand() |
| random generation | Rastgele tensor üretir (uniform veya normal dist.). | torch.rand(), torch.randn() |
| arange / linspace | Sayı aralıklarından tensor oluşturur. | torch.arange(0,10,2), torch.linspace(0,1,5) |
| device placement | Tensor’un CPU veya GPU’da oluşturulması. | x.to("cuda") |
| numpy interoperability | NumPy array ↔ Torch tensor dönüşümü. | t.numpy(), torch.from_numpy(a) |
| element-wise operations | Tüm elemanlarda aynı anda işlem yapar. | a + b, a * b |
| requires_grad | Türevi takip edip etmeyeceğini belirler. | torch.tensor([1.0], requires_grad=True) |
"""
