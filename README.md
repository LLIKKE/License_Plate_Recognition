

# 传统视觉方法实现车牌检测与识别

计算机视觉课程的期中实验考核，着重于传统计算机视觉方法实现车牌检测与识别。

Mid-term examination for computer vision course focusing on traditional computer vision methods to achieve licence plate detection and recognition.


<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/">
    <img src="fig/workflow.svg" alt="Logo" width="500" height="400">
  </a>

  <h3 align="center">传统视觉方法实现车牌检测与识别</h3>
  <p align="center">

</p>

### 项目结构说明

```
License_Plate_Recognition
├── LPR_chi.py
├── LPR_en.py
├── README.md
├── Real Tme LPR system.py
├── Video_LPR.py
├── README.md
├─fig
│      workflow.svg
├─image_chi
│      sample.jpg
└─image_en
        06 CGN 419.jpg
```

### 上手指南
1. 根据导入头文件安装相应包
2. 访问[https://digi.bib.uni-mannheim.de/tesseract/](https://digi.bib.uni-mannheim.de/tesseract/) 安装相应tesseract，选择其程序路径
```python
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
```
3. "LPR_chi.py"实现蓝底白字车牌识别,输入相应图片路径即可
```python
# todo: 国内车牌检测，国内车牌蓝底白字，特征提取不明显，因此做了额外的增强
img = cv2.imread("image_chi/sample2.png")
Plate_Recognition(img)
```
4. "LPR_chi.py"实现白底黑字车牌识别,输入相应图片路径即可
```python
img = cv2.imread("image_en/06 CGN 419.jpg")
Plate_Recognition(img)
```

5. "Video_LPR.py"实现对视频数据的车牌识别，在上方输入视频路径
```python
cap = cv2.VideoCapture('car1.mp4')
```
6. "Real Tme LPR system.py"实现使用设备摄像头实时获得车辆图像进行车牌识别

### Demo效果
传统方法实现车牌定位受图像分辨率、色偏、车距影响较大，因此很多时候需要自己调整一下
###### **白底黑字车牌效果**
白底黑字车牌效果，不知道是不是OCR的原因，白底黑字效果要好很多。
<div style="text-align:center;">
    <img src="image_en/26_EJ_668.jpeg" alt="26 20EJ 20668.jpeg" width="200" height="200" />
</div>
<div style="text-align:center;">
    <img src="fig/26EJ668.png" alt="Workflow" width="500" height="140" />
</div>


###### **蓝底白字车牌效果**
<div style="text-align:center;">
    <img src="fig/L66666.png" alt="image_chi/sample4.jpg" width="220" height="200" />
</div>
<div style="text-align:center;">
    <img src="fig/LB66666.png" alt="Workflow" width="500" height="140" />
</div>


###### **视频车牌检测*
[video_result.mp4
](https://github.com/user-attachments/assets/5f9a2fb4-7d5e-49d6-8316-79e1b0ec6a7a)
### 作者

LLIKKE







