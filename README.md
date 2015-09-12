dct_simd
========

The code is 4x4, 8x8, and 16x16 Fast DCT implimentaion with C++ and SIMD intrinsics, and has an OpenCV demo.
The DCT code itself does not require OpenCV lib.

The code is based on

* for 4x4, 8x8
Christoph Loeffler ,Adriaan Ligtenberg Moschytz “Practical fast 1-D DCT algorithm with 11 Multiplications,” Proc. IEEE ICASSP,vol 2,pp.988-991,Feb 1989.

* for 16x16 
Plonka, Gerlind, and Manfred Tasche. "Fast and numerically stable algorithms for discrete cosine transforms." Linear algebra and its applications 394 (2005) : 309 - 345.
