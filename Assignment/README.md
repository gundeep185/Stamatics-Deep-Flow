# Dog Breed Identification using CNN & Transfer Learning

##  Objective
To identify the breed of a dog from an image using Convolutional Neural Networks and Transfer Learning.

##  Model Architecture
- Base Model: `MobileNetV2` (pretrained on ImageNet)
- Added Layers:
  - Global Average Pooling
  - Dense layer with ReLU
  - Dropout for regularization
  - Final softmax layer with 120 outputs (number of dog breeds)

##  Transfer Learning Strategy
- Used pretrained `MobileNetV2` as a fixed feature extractor
- Fine-tuned top layers after freezing initial layers
- Applied data augmentation with `ImageDataGenerator`

##  Performance Metrics
- Training Accuracy: ~1.4%
- Validation Accuracy: ~1.3%
(Note: Model trained for 5 epochs for demonstration. Accuracy is low due to time/resource constraints.)

##  Saved Model
- File: `light_cnn_model.h5`  
- Size: ~94MB (not uploaded here due to GitHub file size limits)

##  How to Load Saved Model

```python
from tensorflow.keras.models import load_model

model = load_model('light_cnn_model.h5')
