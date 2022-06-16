# Ultra-Lite-Convolutional-Neural-Network-for-Automatic-Modulation-Classification

In this paper, we designed a ultra lite CNN for AMC, and its simulation is based on RML2016A

We are also modifying the structure to improve performance.

![Uploading image.png…]()

#**20220616**

改成衰减学习率（0.8/10），SCNN性能好点儿了，但是其他方法基本没啥用；

MCLDNN用了4倍数据增强后training acc一直是0.09左右，可能是随机数设置问题；

计算模型复杂度用MACC的话，只能得到一个理论值，需要在设备上实际测试速度

更新一版结果，ULCNN平均acc可以达到62%
