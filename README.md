# Image Super Resolution Using UNET ComputerVision with MLFlow tracking and ML lifecycle management
Understanding and Implementing Image Super Resolution model in Computer vision using UNET architecture using Kaggle dataset.

The project builds a model which will learn to improve the resolution of the low-resolution images. Model will be trained on the pair of (low_res, high_res) images and later low resolution image will be sent to predict high resolution version of that image.

************************************************

# Image dataset

![](https://github.com/joshir199/Image-Super-Resolution-Using-UNET-ComputerVision-with-MLFlow/blob/main/images/1_low.png)                       ![](https://github.com/joshir199/Image-Super-Resolution-Using-UNET-ComputerVision-with-MLFlow/blob/main/images/1_high.png)

---------------- low res --------------||-------------------high res-------



# Training model

To train the model with mlflow tracking context, run following command:
```bash
python train.py
```
