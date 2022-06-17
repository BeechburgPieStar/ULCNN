# Ultra-Lite-Convolutional-Neural-Network-for-Automatic-Modulation-Classification

In this paper, we designed a ultra lite CNN for AMC, and its simulation is based on RML2016A

We are also modifying the structure to improve performance.

![image](https://user-images.githubusercontent.com/107237593/174008584-2e72e305-9474-4a3d-af03-2f94e5844f5c.png)

#**20220616**

改成衰减学习率（0.8/10），SCNN性能好点儿了，但是其他方法基本没啥用；

MCLDNN用了4倍数据增强后training acc一直是0.09左右，可能是随机数设置问题；

计算模型复杂度用MACC的话，只能得到一个理论值，需要在设备上实际测试速度

更新一版结果，ULCNN平均acc可以达到62%

明天继续修改模型结构

#**20220617**

继续简化模型结果，目前参数量只有1万左右，计算量大概在0.27M，平均准确率可以达到62.24%，非常接近MCLDNN的性能（62.25%）！

下午考虑用跨层次特征融合继续提高性能！

继续调试MCLDNN的结果
