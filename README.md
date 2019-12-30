# CAM

Implementation of Learning Deep Features for Discriminative Localization([arxiv](https://arxiv.org/pdf/1512.04150.pdf))

![test1](https://user-images.githubusercontent.com/25279765/36484699-7928832c-175d-11e8-9c8c-ac166404ce64.jpg) ![cam1](https://user-images.githubusercontent.com/25279765/36484700-7958af98-175d-11e8-80ce-7d8a6239308c.jpg)

![test2](https://user-images.githubusercontent.com/25279765/36484702-7b559ef0-175d-11e8-9359-4727cd4cadd9.jpg) ![cam2](https://user-images.githubusercontent.com/25279765/36484704-7b88e27e-175d-11e8-8032-95654cb1e051.jpg)

![test3](https://user-images.githubusercontent.com/25279765/36484707-7cda1332-175d-11e8-82a0-711c86a6a454.jpg) ![cam3](https://user-images.githubusercontent.com/25279765/36484708-7d05851c-175d-11e8-8141-ff4e23958c44.jpg)


## Dependency

Need opencv-python(cv2) and pytorch
```bash
pip install opencv-python
```

## Training

To train a model

```bash
python train.py --dataset CIFAR --dataset_path ./data --model_path ./model --model_name model.pth --img_size 128 --batch_size 32 --epoch 30 --log_step 10 --lr 0.001
```

###### Arguments 

- `--dataset` : Specify which dataset you use. 
Three types are supported: (STL, CIFAR, OWN)
If you want to train model with your own dataset, use `OWN`

- `--dataset_path` : Specify the path to your dataset.
If you use STL10 or CIFAR10, it will download the dataset at the path.

- `--model_path` : Specify the path where the model to be saved

- `--model_name` : Specify the name of .pth file

- `--img_size` : The size of images to train

- `--batch_size` : The number of images in each batch

- `--epoch` : The number of epochs to train

- `--lr` : Learning rate

- `--log_step` : The number of iterations to print loss

- `-s`, `--save_model_in_epoch` : Basically the model will be saved after an epoch finished. If `-s` is true, the model will be saved after each log_step too.

## Create CAM

it needs saved model

```bash
python create_cam.py --dataset CIFAR --dataset_path ./data --model_path ./model --model_name model.pth --result_path ./result --img_size 128 --num_result 1
```

###### Arguments
- `--dataset` : Specify which dataset you use, (STL, CIFAR, OWN)
If you want to test model with your own dataset, use `OWN`

- `--dataset_path` : Specify the path to your dataset.
If you use STL10 or CIFAR10, it will download the dataset at the path.

- `--model_path` : Specify the path where the model is saved

- `--model_name` : Specify the name of .pth file

- `--result_path` : Specify the path where the CAM and the original image to be saved

- `--img_size` : The size of images to save

- `--num_result` : The number of result to create. It will be randomly chosen from the test dataset.

## Reference

Codes of create_cam.py is influenced by https://github.com/metalbubble/CAM
