# code-for-my-graduate-paper-python-version
Python Code for Paper [Iterative-Saliency-via-Dynamic-Image-Region-Partitioning](https://github.com/168WenFangjun/Iterative-Saliency-via-Dynamic-Image-Region-Partitioning/blob/master/Iterative%20Saliency%20via%20Dynamic%20Image%20Region%20Partitioning.pdf)


# Iterative-Saliency-via-Dynamic-Image-Region-Partitioning
![image](https://github.com/168WenFangjun/Iterative-Saliency-via-Dynamic-Image-Region-Partitioning/blob/master/code-for-my-graduate-paper/test/3_95_95850.jpg)
![image](https://github.com/168WenFangjun/Iterative-Saliency-via-Dynamic-Image-Region-Partitioning/blob/master/code-for-my-graduate-paper/saliencymap/3_95_95850.png)


# Introduction 

Vison Old, Vision New. 

After reading this paper , you will get new insights of computer vision. 

For more detail information, please contact with me via gmail [168fangjunwen@gmail.com].

# ISDIP Python Implementation

Python版本的ISDIP（Iterative-Saliency-via-Dynamic-Image-Region-Partitioning）算法实现。


## 使用方法

```bash
python ./main.py
```

## 文件结构

- `saliencymap/` - 输出显著图
- `superpixels/` - 输出超像素
- `test/` - 测试图片

## 算法特性

- 超像素分割（SLIC）
- 四方向扫描
- 多颜色空间特征（RGB+LAB+XYZ）
- 高效的Python实现
