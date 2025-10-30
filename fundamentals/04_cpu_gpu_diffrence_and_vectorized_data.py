"""
1) NumPy vs PyTorch - Speed Comparison
2) Vectorized Computation
3) GPU (CUDA) Execution Theory
"""
import numpy as np
import torch
import time

print("===================================== 1) NumPy vs PyTorch - CPU Speed Comparison:")

"""
    NumPy ve PyTorch her ikisi de CPU'da çalışabilir. 
    Ancak PyTorch C++ backend (ATen) kullandığı için bazı matematiksel işlemleri daha optimize biçimde yürütür.
    Burada CPU üzerinde basit bir hız karşılaştırması yapacağız.
"""

# boyut (ne kadar büyük, o kadar fark)
size = 10_000_000

# NumPy versiyonu
a_np = np.random.rand(size)
b_np = np.random.rand(size)

start = time.time()
c_np = a_np * b_np  # element-wise çarpma
end = time.time()
print(f"NumPy time: {end - start:.6f} seconds")

# PyTorch versiyonu
a_torch = torch.rand(size)
b_torch = torch.rand(size)

start = time.time()
c_torch = a_torch * b_torch  # element-wise çarpma
end = time.time()
print(f"PyTorch (CPU) time: {end - start:.6f} seconds")

"""
    Her iki işlem de CPU'da olsa bile PyTorch genellikle optimize edilmiş low-level C++ implementasyonu sayesinde
    küçük farklarla da olsa daha hızlı olabilir.
    Ancak bu fark genellikle CPU'da çok belirgin değildir, asıl fark GPU'da ortaya çıkar.
"""

print("===================================== 2) Vectorized Computation (Vektörleştirilmiş Hesaplama):")

"""
    Python'da klasik for-loop ile hesap yapmak yerine NumPy veya PyTorch'un vektörleştirilmiş fonksiyonlarını
    kullanmak 100 kata kadar performans farkı yaratabilir. (loop overhead ortadan kalkar) (ileride ele alınacak)

    Aşağıda klasik Python döngüsüyle yapılan bir işlemi, PyTorch ve NumPy vektörleştirilmiş haliyle karşılaştıralım:
"""

# saf Python versiyonu (çok yavaş)
x_list = [i for i in range(10_000_00)]
start = time.time()
y_list = [i**2 for i in x_list]
print(f"Pure Python time: {time.time() - start:.6f} seconds")

# NumPy vectorized
x_np = np.arange(10_000_00)
start = time.time()
y_np = x_np ** 2
print(f"NumPy vectorized time: {time.time() - start:.6f} seconds")

# PyTorch vectorized
x_torch = torch.arange(10_000_00)
start = time.time()
y_torch = x_torch ** 2
print(f"PyTorch vectorized time: {time.time() - start:.6f} seconds")

"""
    Gözlem: for-loop → milyonlarca iteration = Python interpreter overhead
            vectorized → C düzeyinde işlem → çok daha hızlı
"""

print("===================================== 3) GPU (CUDA) Execution Theory:")

"""
    GPU hızlandırması (CUDA) teorik olarak şu şekilde çalışır:

    1. Tensor'ları GPU belleğine taşırsın:
        x = torch.rand(10000, device='cuda')

    2. GPU üzerinde işlem yaparsın (örneğin matrix multiplication):
        y = torch.mm(x, x)

    3. GPU asenkron çalıştığı için zaman ölçümü yaparken torch.cuda.synchronize() eklenir.
       (CPU zamanlamasıyla GPU'nun bitiş zamanlaması farklı olabilir.)

    4. Sonuçlar CPU'ya geri taşınabilir:
        y_cpu = y.to('cpu')

    GPU işlemleri çoklu çekirdek (thousands of cores) ve paralel floating-point execution sayesinde
    CPU'ya kıyasla 10-100x hız artışı sağlayabilir.

    Ancak bu yalnızca NVIDIA GPU + CUDA driver ortamında geçerlidir.
    Intel Iris Pro CUDA desteklemediği için işlemler CPU’da çalışacaktır.
"""

if torch.cuda.is_available():
    print("✅ CUDA GPU detected:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not available — running on CPU (Intel Iris Pro detected)")

print("===================================== *) Hot-takes:")

"""
    - CPU'da NumPy ve PyTorch hızları genellikle benzerdir, bazen PyTorch az farkla önde olabilir.
    - GPU (CUDA) varsa, PyTorch büyük matrix işlemlerinde dramatik hız farkı yaratır.
    - torch.cuda.synchronize() zaman ölçümünde önemlidir (GPU işlemleri asenkron çalışır).
    - Vectorized operations, Python loop’larından kat kat hızlıdır.
    - Intel Iris Pro gibi entegre GPU'lar CUDA desteklemez → PyTorch işlemleri CPU üzerinde yürür.
"""

print("===================================== 📘 Summary Table: NumPy vs PyTorch Computation Speed")

"""
| Konsept | Açıklama | Örnek Kod |
|----------|-----------|-----------|
| Computation Speed | PyTorch C++ backend (ATen) kullandığı için NumPy'ye kıyasla bazı işlemlerde CPU’da daha optimize çalışabilir. | a_torch * b_torch |
| Vectorized Computation | Döngü (loop) yerine toplu işlemler (C düzeyinde) kullanarak 10-100x hız kazanımı sağlar. | y = x ** 2 |
| Pure Python vs Vectorized | for-loop yerine NumPy/PyTorch kullanmak büyük fark yaratır. | [i**2 for i in x] → x**2 |
| GPU (CUDA) Execution | CUDA destekli GPU’larda işlemler paralel yürütülür; CPU’ya göre 10–100x daha hızlı olabilir. | x = torch.rand(1000,1000, device='cuda') |
| torch.cuda.is_available() | Sistemde CUDA destekli GPU olup olmadığını kontrol eder. | torch.cuda.is_available() |
| torch.cuda.synchronize() | GPU asenkron çalıştığı için zaman ölçümünde CPU ve GPU senkronizasyonu sağlar. | torch.cuda.synchronize() |
| device placement | Tensor’un CPU veya GPU’da oluşturulacağı yeri belirler. | torch.device("cuda" if torch.cuda.is_available() else "cpu") |
| Intel Iris Pro durumu | CUDA desteklemediğinden tüm işlemler CPU üzerinde yürür (GPU hızlanması olmaz). | n/a |
| Performance Insight | Küçük işlemlerde fark azdır; büyük matrislerde PyTorch fark yaratır. | matrix multiplication test |
"""
