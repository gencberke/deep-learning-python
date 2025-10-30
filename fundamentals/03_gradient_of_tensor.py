"""
1) What is Autograd?
2) Computation Graph
3) Backward Propagation
4) Gradient Accumulation
"""
import torch
import numpy as np
from torch.onnx.symbolic_opset9 import tensor

print("===================================== 1) What is Autograd?")
"""
1) autograd -> pytorch'un otomatik difransiyel engine'i yani pytorch bir tensor üzerindeki tüm işlemleri bir graph olarak
    trackler ve bu grafiği kullanarak gradyen yani türev hesaplayabilir. bu işlemde deep learningte backpropagation'un
    temelidir.
"""
# requires_grad=True demek, bu tensor üzerindeki işlemleri autograd sisteminde takip et.
x = torch.tensor([3.0, 4.0], requires_grad=True)
print(x)
y = (x ** 2).sum() # y = 3'ün karesi + 4'ün karesi

# şimdi burada y'ye göre x'in türevi hesaplanır.
y.backward()
print(x.grad)

print("===================================== 2) Computation Graph:")

"""
2) Autograd sistemi her bir işlemi bir node olarak düşünür. sonucu ise bir edge olarak. "computation graph" dinamik olarak
    oluşturulur yani her pass forward işlemi sırasında tekrar tekrar çizilir. bu sisteme define-by-run denilir
    
        x -> (x^2) -> sum -> y
        
    bu zincirde x girdi, (x^2) operation sonucu oluşan intermediate tensor, y final output. pytorch her adımı takip eder.
    ve backward çağrıldığında chain rule'a türev uygular.  
"""

print("===================================== 2.1) Chain Rule (Zincir Kuralı): ")

"""
2.1) Chain Rule (Zincir Kuralı):

    Matematikte chain rule, bileşik fonksiyonların türevini alırken kullanılır.
    Eğer y = f(g(x)) ise:
        dy/dx = (dy/dg) * (dg/dx)

    PyTorch'un autograd sistemi de aynen bu mantıkla çalışır.
    Backward propagation sırasında son çıktının (loss) gradient'i en baştaki input tensor'lara kadar
    zincirleme olarak (reverse mode differentiation) hesaplanır.

    Örnek:
        x = 2
        y = x^2
        z = 3y + 5

        z = 3(x^2) + 5
        dz/dx = 3 * 2x = 6x

    PyTorch bu işlemleri manuel yazmadan otomatik olarak yapar:
"""
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = 3 * y + 5
z.backward()
print("x.grad:", x.grad)  # 6x = 12.0

print("===================================== 2.2) non scalar outputs ve gradient arguments: ")

"""
2.2) .backward() metodu yalnızca skaler sonuçlar için çalışır. 
    Eğer sonuç bir vektörse (örneğin [y1, y2, y3]), hangi yön boyunca türev alınacağını belirtmen gerekir.
"""
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2 # y = [1.0, 4.0, 9.0]
v = torch.tensor([0.1, 0.01, 0.001]) # custom weights (directional vektor)

y.backward(v)
print(x.grad)

"""
    Burada PyTorch şu hesabı yapar:

    ∂(v·y)/∂x = v * 2x

    Jacobian-vector product (JVP) olarak bilinir. Tam türev matrisi yerine, belirli bir vektör yönünde türev alır 
    bu deep learning’de gradient optimizasyonlarını hızlandırmak için kullanılır.
"""

print("===================================== 2.3) detach() ve no_grad() :")

"""
2.3) autograd bazen istemediğimiz tensorlarıda izler. bu durumda iki yöntem vardır:
"""
# .detach() -> tensor'u graph'tan ayırır ama veriyi korur
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y.detach()  # artık z'nin requires_grad=False
print(z.requires_grad)  # False

# torch.no_grad() -> context manager olarak çalışır ve içerideki hiçbir işlem graph'a eklenmez.
x = torch.tensor([3.0], requires_grad=True)
with torch.no_grad():
    y = x ** 2
print(y.requires_grad)  # False

"""
        .detach() → tensor seviyesinde koparma
        no_grad() → kod bloğu seviyesinde geçici olarak autograd’ı kapatma

    Model evaluation (örneğin test setinde) sırasında performans için no_grad kullanmak best practice’tir. (çok mantıklı)
"""

print("===================================== 3) Gradient Accumulation: ")

"""
3) .backward() her çağrıldığında gradient'ler var olan .grad üzerine eklenir (accumulate) 
    Bu bazen eğitim döngülerinde “gradient accumulation” olarak bilerek kullanılır, ama genellikle temizlemek gerekir. 
"""
x = torch.tensor([2.0], requires_grad=True)
y1 = x ** 2
y2 = x * 3
y1.backward()

print(x.grad) # 4.0

y2.backward()
print(x.grad) # 4.0 + 3.0 = 7.0

x.grad.zero_() # gradyantları sıfırlamak için
print(x.grad)

"""
        .zero_() bir in-place operation’dır, gradient belleğini temizler.

    Optimizer’larda (optimizer.zero_grad()) da bu mantık kullanılır.
"""

print("===================================== 3.1) requires_grad_() ve leaf tensors :")

"""
3.1) Her tensor oluşturulduğunda varsayılan olarak requires_grad=False gelir.
    Bunu sonradan aktif etmek istersen .requires_grad_() fonksiyonunu kullanabilirsin. 
"""
a = torch.ones(3)
print(a.requires_grad)  # False
a.requires_grad_()      # inplace olarak değiştirir
print(a.requires_grad)  # True

"""
Önemli Not: Bir tensor autograd graph’ında doğrudan kullanıcı tarafından oluşturulmuş ve bir operation sonucu değilse, 
            ona leaf tensor denir. Gradient’ler sadece leaf tensor’lar için saklanır (.grad attribute’u yalnızca onlarda
            dolu olur).
"""

print("===================================== *) Hot-takes ")

"""
1) bir gradyan hesaplamasını temporarily durdurmak için
"""
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
with torch.no_grad():
    z = y * 3
print(z.requires_grad)  # False

# veya
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y.detach() * 3  # sadece y'yi kopardık

"""
2) Gradients for multi-step functions:
    Autograd chain rule sayesinde çok adımlı fonksiyonlarda da otomatik türev alır:
"""
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1
y.backward()
print(x.grad)  # 2x + 2 = 8

print("===================================== 📘 Summary Table: Autograd Basics")

"""
| Konsept | Açıklama | Örnek Kod |
|----------|-----------|-----------|
| requires_grad | PyTorch'un gradient hesaplamasını aktif eder. | x = torch.tensor([2.0], requires_grad=True) |
| backward() | Scalar bir output'un türevini hesaplar. | y.backward() |
| grad | Hesaplanan türevi döndürür. | print(x.grad) |
| computation graph | Tüm işlemlerin zincir olarak izlendiği dinamik yapı. | x → x² → sum → y |
| chain rule | Bileşik fonksiyonlarda türevi zincirleme uygular. | dy/dx = (dy/dg)*(dg/dx) |
| Jacobian-vector product (JVP) | Non-scalar sonuçlarda yön vektörüne göre türev alır. | y.backward(v) |
| detach() | Tensor'u graph'tan ayırır (gradient takibi durur). | z = y.detach() |
| no_grad() | Geçici olarak gradient takibini kapatır. | with torch.no_grad(): ... |
| gradient accumulation | backward() çağrıları gradyanları toplar (eklenir). | x.grad.zero_() ile temizlenir |
| leaf tensor | Doğrudan kullanıcı tarafından oluşturulmuş tensor. | a = torch.ones(3, requires_grad=True) |
"""
