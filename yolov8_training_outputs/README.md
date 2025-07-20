# YOLOv8 Training Artifacts – Head Detection Model

This directory contains key output artifacts from training a custom YOLOv8 model for head detection using the JHU-CROWD dataset.

##  Included Files

- **`args.yaml`**  
  Configuration file that contains the hyperparameters and settings used during training (e.g., number of epochs, image size, batch size, optimizer).

- **`confusion_matrix.png`**  
  Visual representation of the confusion matrix showing the model's performance in distinguishing between classes (heads vs. background).

- **`results.csv`**  
  Tabular log of training and validation metrics (e.g., precision, recall, mAP) recorded over each epoch. Useful for plotting custom curves.

- **`results.png`**  
  Automatically generated visualization of key training metrics like precision, recall, mAP, loss values over epochs.

##  Training Details

- **Model**: YOLOv8 (medium size – `yolov8m`)
- **Dataset**: JHU-CROWD (head detection subset)
- **Framework**: Ultralytics YOLOv8
- **Objective**: Real-time head detection for crowd density estimation and stampede risk analysis

##  Notes

- These artifacts provide evidence of successful model training.
- The actual trained model weights file `best.pt` is not included here due to file size restrictions, but can be made available via request or external link.

