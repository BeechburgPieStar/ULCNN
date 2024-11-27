# Ultra-Lite-Convolutional-Neural-Network-for-Automatic-Modulation-Classification

In this paper, we designed a ultra lite CNN (ULCNN) (9,751 trainable parameters and 0.2M MACCs) for AMC, and its simulation is based on RML2016.10a

# Paper

http://arxiv.org/abs/2208.04659

L. Guo, Y. Wang, Y. Liu, Y. Lin, H. Zhao and G. Gui, "Ultralight Convolutional Neural Network for Automatic Modulation Classification in Internet of Unmanned Aerial Vehicles," in IEEE Internet of Things Journal, vol. 11, no. 11, pp. 20831-20839, June 2024.

# Requirements

keras=2.1.4 tensorflow=1.14

# Codes
MCLDNN [1]

SCNN [2]

MCNet [3]

PET-CGDNN [4]

ULCNN is the proposed structure.

The model weights are given in "model/"

# Dataset
RML2016.10a

Train/val/test samples: 77000/33000/110000

https://pan.baidu.com/s/1T36jgWlZ3oWmFWYpQLyiZg, passwd：f7qy
or run dataset2016.py

# Structure

![image](https://user-images.githubusercontent.com/107237593/183788614-404b5743-21bb-4e9a-915c-04725fb36162.png)

# Classification performances

![image](https://user-images.githubusercontent.com/107237593/181512551-f537bf6f-c9ff-4a74-a5a8-b03ae80ee49d.png)

# Ablation studies
![image](https://user-images.githubusercontent.com/107237593/181512604-917152a7-b753-4150-98a1-3858dd093d2e.png)

# Loss and accuracy curves

![image](https://user-images.githubusercontent.com/107237593/174467709-0c04b16a-c260-4355-a942-cb1a0a6bf775.png)

# Complexity analysis

![image](https://user-images.githubusercontent.com/107237593/176838730-5fcc0c0b-3fe8-46b1-a3df-1e3d1bd4413a.png)

# Reference
[1] J. Xu, C. Luo, G. Parr and Y. Luo, "A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition," in IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1629-1632, Oct. 2020, doi: 10.1109/LWC.2020.2999453.

[2] X. Fu et al., "Lightweight Automatic Modulation Classification Based on Decentralized Learning," in IEEE Transactions on Cognitive Communications and Networking, vol. 8, no. 1, pp. 57-70, March 2022, doi: 10.1109/TCCN.2021.3089178.

[3] T. Huynh-The, C. Hua, Q. Pham and D. Kim, "MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification," in IEEE Communications Letters, vol. 24, no. 4, pp. 811-815, April 2020, doi: 10.1109/LCOMM.2020.2968030.

[4] F. Zhang, C. Luo, J. Xu and Y. Luo, "An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation," in IEEE Communications Letters, vol. 25, no. 10, pp. 3287-3290, Oct. 2021, doi: 10.1109/LCOMM.2021.3102656.


# Acknowledgement
Note that our code is partly based on [leena201818](https://github.com/leena201818/radioml), [wzjialang](https://github.com/wzjialang/MCLDNN), [ThienHuynhThe](https://github.com/ThienHuynhThe/MCNet) and [Richardzhangxx](https://github.com/Richardzhangxx/PET-CGDNN).

Thanks for your great works!

# License / 许可证

本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
















