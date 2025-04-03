# CFD-HW3

CFD HW3为主要程序，包括Lax-Friedrichs格式、一阶迎风格式、二阶迎风格式的构造，每一种格式都利用了显式方法维护周期性边界条件。

CFD HW3 L2 Error 验证了上述三种格式的精度：在给定CFL的情况下，通过改变网格数量（N），考察了每一种格式的L2误差，结果显示在不同网格数下的L2误差曲线的斜率与dx的参考线斜率几乎一致（1），验证了三种格式的精度阶数为1.
