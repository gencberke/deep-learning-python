"""
1) Aritmetik İşlemler
2) In-place operations
3) Matrix Multiplication (Matris çarpımı)
4) Tensor Indexing and Slicing
5) Reshaping and Views
6) Combining and Splitting Tensors
"""
import torch

print("===================================== 1) Aritmetik İşlemler: ")

"""
    PyTorch tensor'ları üzerinde NumPy array'leri gibi matematiksel işlemler yapılabilir (element-wise operations).
"""

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print(x + y)  # element-wise addition
print(x * y)  # element-wise multiplication
print(x / y)  # element-wise division (float döner)
print(x - y)  # element-wise subtraction

"""
    Bu işlemler function çağrılarıyla da yapılabilir; ikisi eşdeğerdir.
    Function versiyonunda output parametresi verilebilir (out=...).
"""

torch.add(x, y)
torch.sub(x, y)
torch.mul(x, y)
torch.div(x, y)

print("===================================== 2) In-place operations: ")

"""
    In-place operation: işlem sonucu yeni bir tensor'a yazılmak yerine mevcut tensor'un üstüne yazılır (mutates in place).
    PyTorch'ta in-place operasyonların sonuna '_' konur.
    Dikkat: in-place işlemler autograd (automatic differentiation) sırasında gradient hesaplarını bozabilir.
"""

print(x)
x.add_(y)     # x = x + y (in-place)
print(x)

print("===================================== 3) Matrix Multiplication: ")

"""
    PyTorch'ta '*' operatörü element-wise çarpar. Gerçek matris çarpımı (matrix multiplication) için
    '@' operatörü veya torch.matmul() kullanılır.
"""
a = torch.tensor([[1, 2], [4, 5]])
b = torch.tensor([[6, 7], [8, 9]])

print(a @ b)
print(a.matmul(b))
# eğer a ve b 1D vektörse -> dot product (torch.dot kullanılabilir)
# eğer a ve b 2D matrisse -> satır x sütun çarpımı
# eğer tensor'lar 3D+ ise -> batch matrix multiplication (ör: (batch_size x n x m) @ (batch_size x m x p))

print("===================================== 4) Tensor Indexing and Slicing: ")

"""
    NumPy'daki gibi PyTorch'ta da tensor'lardan belli elemanlar ve dilimler alınabilir.
"""
my_tensor = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
print(my_tensor.size())  # torch.Size([...]); .shape ile aynıdır

print(my_tensor[0, 0])  # ilk satır, ilk sütun
print(my_tensor[:, 1])  # tüm satırlar, 2. sütun
print(my_tensor[2, :])  # 3. satır, tüm sütunlar
# indexing -> tek bir eleman seçmek
# slicing  -> bir aralığı/dilimi seçmek
# ':' tüm elemanlar demektir

print("===================================== 5) Reshaping and Views: ")

"""
    Bir Tensor’un şeklini (shape) değiştirmek için:
        1) reshape()
        2) view()
    Not (reshape vs view):
      - reshape(): mümkünse view döndürür; mümkün değilse kopya (copy) döndürebilir. Esnek davranır.
      - view(): her zaman contiguous (bitişik) memory ister; değilse .contiguous() çağırman gerekir.
"""

x = torch.arange(9)  # tensor([0,1,2,3,4,5,6,7,8])
x_reshaped = x.reshape(3, 3)
# reshape, memory layout uygunsa sadece "görünümü" değiştirir (no copy). Uygun değilse kopya oluşturabilir.

# view örneği:
x = torch.arange(10)
# bazı operasyonlardan sonra tensor contiguous olmayabilir; o durumda:
y = x.contiguous().view(2, 5)  # view contiguous ister
print(y.shape)

print("===================================== 6) Combining and Splitting Tensors: ")

"""
    Tensor'ları birleştirmek (concatenate) ve bölmek (split) için:
"""
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

print(torch.cat((x, y), dim=0))  # dim=0 -> satır yönünde birleştir
print(torch.cat((x, y), dim=1))  # dim=1 -> sütun yönünde birleştir

t = torch.arange(10)
print(torch.chunk(t, 3))  # chunk(n_chunks): t'yi ~eşit 3 parçaya böler (son parça daha kısa olabilir)
print(torch.split(t, 3))  # split(split_size): her parçanın boyutu 3 olacak şekilde böler (son parça daha kısa olabilir)
# Ek not:
#   - torch.chunk(input, chunks)   -> parça sayısını verirsin.
#   - torch.split(input, split_size_or_sections) -> parça boyutunu verirsin ya da [boyutlar] listesi verebilirsin.

print("===================================== *) Hot Takes: ")

"""
    1) Interoperability with NumPy:
       torch_tensor.numpy() -> NumPy array ile aynı memory'yi paylaşır (shared memory).
       torch.from_numpy(ndarray) -> yine shared memory; ndarray değişirse tensor da etkilenir.
       Tam bağımsız kopya istiyorsan torch.tensor(ndarray) kullan.
"""

"""
    2) Broadcasting:
       Farklı boyutlu tensor'lar, broadcasting kurallarıyla otomatik genişletilerek (virtually expanded) element-wise işlem
       yapılabilir. Bir eksen boyutu 1 ise diğerine "yayılır" (expand) kurala uygun olduğu sürece.
"""
x = torch.ones(2, 1)
y = torch.ones(1, 5)
print((x + y).shape)  # -> torch.Size([2, 5])
print(x + y)
