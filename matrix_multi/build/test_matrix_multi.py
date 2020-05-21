import time
import numpy as np
import matrix_multi_library

A = np.random.randn(20, 20).astype(np.float32).reshape(-1)
B = np.random.randn(20, 20).astype(np.float32).reshape(-1)
C = np.random.randn(20, 20).astype(np.float32).reshape(-1)
C[:] = 0

t1 = time.time()
matrix_multi_library.matrix_multi(A, B, C, 20, 20, 20)
t2 = time.time()
print("in the gpu cost: {}ms".format((t2-t1)*1000))
# print(C.reshape(10, 10))

t1 = time.time()
temp = np.dot(A.reshape(20, 20), B.reshape(20, 20))
t2 = time.time()
print("in the cpu cost: {}ms".format((t2-t1)*1000))
# print(C.reshape(10, 10))

t1 = time.time()
matrix_multi_library.matrix_multi(A, B, C, 20, 20, 20)
t2 = time.time()
print("in the gpu cost: {}ms".format((t2-t1)*1000))

t1 = time.time()
matrix_multi_library.matrix_multi(A, B, C, 20, 20, 20)
t2 = time.time()
print("in the gpu cost: {}ms".format((t2-t1)*1000))

t1 = time.time()
matrix_multi_library.matrix_multi(A, B, C, 20, 20, 20)
t2 = time.time()
print("in the gpu cost: {}ms".format((t2-t1)*1000))

t1 = time.time()
matrix_multi_library.matrix_multi(A, B, C, 20, 20, 20)
t2 = time.time()
print("in the gpu cost: {}ms".format((t2-t1)*1000))

t1 = time.time()
matrix_multi_library.matrix_multi(A, B, C, 20, 20, 20)
t2 = time.time()
print("in the gpu cost: {}ms".format((t2-t1)*1000))
