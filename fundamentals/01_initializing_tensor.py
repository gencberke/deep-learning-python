"""
1) Tensor nedir? (ve neden PyTorch’un kalbidir)
2) Tensor özellikleri (shape, dtype, device)
3) Tensor oluşturma yolları
4) Tensor’lar arasında işlemler (ekleme, çarpma, reshape)
5) Numpy ile fark ve dönüşüm (array ↔ tensor)
"""
from random import randint

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

import torch

# burada python listesini torch tensor objesine dönüştürdük. yani artık üzerinde matematiksel
# işlemler yapılmasını destekliyor.
x = torch.tensor([[1,2], [3,4]])
print(x)

print("===================================== 2) Tensor özellikleri:")

"""
2) Tensor özellikleri:
"""
print(x.shape)      # tensorun boyutu (rows, columns)
print(x.dtype)      # veri tipi (örnek: int64 veya float32)
print(x.device)     # CPU mu GPU mu?
print(x.ndim)       # kaç boyutlu? (2D tensor)

print("===================================== 3) Tensor oluşturma yolları:")

"""
3) Tensor oluşturma yolları:
"""

# tüm elemanları "sıfır"'dan oluşan 3 satır 4 sütunluk bir tensor (multi-dimensional array)
zeros = torch.zeros((3,4))

# tüm elemanları "bir"'den oluşan 3 satır 4 sütunluk bir tensor (multi-dimensional array)
ones = torch.ones((3,4))

# 0, 1 arası rastgele değerler üretir. her çalıştırıldığında farklı sonuç çıkarır
random_tensor = torch.rand((3,4))
print(random_tensor)
# Rastgele normalize değerler üretmek

# - sonsuz + sonsuz arası değerler üretir ortalama sıfır civarı olmak üzre dikkat! yukarıdakinden 1 harf farklı
random_tensor = torch.randn((3,4))
print(random_tensor)
# Model initialization (weight’ler) gibi yerlerde kullanılır

# verilen değerler arası rastgele int içerikli tensor üretir:
my_randint = torch.randint(low=0, high=5, size=(3,4))
print(my_randint)

# step kadar arttırıp end e ulaşır
a = torch.arange(start=0, end=30, step=3)

# end'i step'e bölüp adımları döndürür
b = torch.linspace(start=0, end=30, steps=10)

print(a)
print(b)

"""
    Tensor oluştururken dtype ve device belirlenebilir. dtype tensorun data type'ını temsil ederken device ise cpu'da mı
gpuda mı oluşturulacağını temsil eder. bu önemlidir çünkü arada hem performans farkı çok yüksektir hem iki tensor arası
işlem yapılacaksa aynı birimlerde depolanması gerekmektedir. biz şimdilik cpu kullanıcaz. 
"""

x = torch.ones((3,4), dtype=torch.float32, device='cpu')
print(x)

"""
    "cuda" yazdığında GPU’da oluşturulur ama önce torch.cuda.is_available() kontrolü yapılmalı.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand((3, 3), device=device)
print(x.device)

print("===================================== 4) Tensor Operations:")

"""
4) Tensor Operations: 
"""

a = torch.tensor([[1,2], [3,4]], dtype=torch.float32, device='cpu')
b = torch.tensor([[5,6], [7, 8]], dtype=torch.float32, device='cpu')

# “element-wise” = her bir hücre kendi karşılığıyla işlem görür.
print(a + b)      # element-wise addition
print(a - b)      # element-wise subtraction
print(a * b)      # element-wise multiplication
print(a / b)      # element-wise division

# matrix multiplication (Dot Product)
# bize 2x2 bir matrix döndürür.
matmul_result = torch.matmul(a, b)
print(matmul_result)

# reshaping tensors: tensorun verisini değiştirmeden boyutlarını yeniden düzenler
# örneğin içinde 12 değer olan bir matrisi 3x4 e reshapeleyebiliriz.
x = torch.arange(0, 12)
reshaped = x.reshape(3, 4)
print(reshaped)

