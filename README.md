# CAM

Implementation of Learning Deep Features for Discriminative Localization([arxiv](https://arxiv.org/pdf/1512.04150.pdf))

I used custom model with Global Average Pooling and Convolution(No FC).
It was trained by using STL10 about 5000 images with 10 classes. To visualize much bigger image, I upsampled the data to 128x128(orginally 96x96). The model will be saved in model folder.


## Results
![test1](https://user-images.githubusercontent.com/25279765/36484699-7928832c-175d-11e8-9c8c-ac166404ce64.jpg) ![cam1](https://user-images.githubusercontent.com/25279765/36484700-7958af98-175d-11e8-80ce-7d8a6239308c.jpg)

![test2](https://user-images.githubusercontent.com/25279765/36484702-7b559ef0-175d-11e8-9359-4727cd4cadd9.jpg) ![cam2](https://user-images.githubusercontent.com/25279765/36484704-7b88e27e-175d-11e8-8032-95654cb1e051.jpg)

![test3](https://user-images.githubusercontent.com/25279765/36484707-7cda1332-175d-11e8-82a0-711c86a6a454.jpg) ![cam3](https://user-images.githubusercontent.com/25279765/36484708-7d05851c-175d-11e8-8141-ff4e23958c44.jpg)

## Usage
To train a model

```bash
python train.py
```

if you want to use custom dataset, change code in utils.py or train.py

To create CAM(it needs saved model)
```bash
python create_cam.py
```

it will generate CAM only randomly chosen one in the test dataset of STL10. if you want to create more than one, change codes in create_cam.py

## Defendency

Need opencv-python(cv2) and pytorch
```bash
pip install opencv-python
```

## Reference

Codes of create_cam.py is influenced by https://github.com/metalbubble/CAM
