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

好蠢那，MCLDNN最后一层用的linear没用softmax，原来做知识蒸馏没改过来。。。

更新一版结果，目前是62.25%，目标62.5%左右/参数量在1.2万左右/计算量低于0.3M

![image](https://user-images.githubusercontent.com/107237593/174310001-8c099ca3-8e22-4ea6-b358-eaf3b6b94722.png)

#**20220618**

元素级的操作太占用时间了，虽然FLOPs/MACCs没增加多少，但是MAC明显增大

跨层次融合最多能到62.27%，继续简化结构

性能更新完成，目前ULCNN可以达到62.42%，仅有9751个参数/0.196M的MACCs

![image](https://user-images.githubusercontent.com/107237593/174430367-9926fb1a-95e1-4c3a-bd97-e9b3fc418abc.png)

开始消融实验，去除冗余结构。下周一测试模型在树莓派/1080Ti上的运行时间

#**20220618**

消融实验 

![image](https://user-images.githubusercontent.com/107237593/174464421-d64c3d48-61e4-4c40-8653-d96a1fe5629d.png)

不同卷积核

![image](https://user-images.githubusercontent.com/107237593/174464466-df66f621-7504-4804-8df2-2d804f87489b.png)

不同神经元数量

![image](https://user-images.githubusercontent.com/107237593/174464476-70af005f-392c-4c4f-9463-cc34ca1d7853.png)














