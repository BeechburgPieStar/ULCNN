# Ultra-Lite-Convolutional-Neural-Network-for-Automatic-Modulation-Classification

In this paper, we designed a ultra lite CNN for AMC, and its simulation is based on RML2016A

We are also modifying the structure to improve performance.

![image](https://user-images.githubusercontent.com/107237593/173319130-64a6e458-c5a7-4070-a322-c1962ef41abd.png)

![image](https://user-images.githubusercontent.com/107237593/173318852-8b09785b-788d-4ed2-a4a0-8d132536994f.png)

#**20220616**
改成衰减学习率（0.8/10），SCNN性能好点儿了，但是其他方法基本没啥用.
MCLDNN用了4倍数据增强后training acc一直是0.09左右，可能是随机数设置问题
