# CIFAR-10 CNN Classifier

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.  
The model achieves up to **88% accuracy** with full training.  
**Early stopping** is used to ensure at least **80% accuracy** while preventing overfitting.

## What is CIFAR-10?

CIFAR-10 is a dataset of **60,000 32x32 color images** categorized into **10 classes**:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## How to Run the Code

1. Open a terminal and navigate to the project root folder.
2. To train the model, run:
   ```sh
   python main.py --train
   ```
3. To test the trained model, run:
   ```sh
   python main.py --test
   ```

### Notes:

- The trained model is saved automatically.
- Make sure all dependencies are installed before running the script.

```sh
pip install -r requirements.txt
```
