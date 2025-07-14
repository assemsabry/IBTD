# Intelligent Brain Tumor Detector (IBTD)

## Overview

The Intelligent Brain Tumor Detector (IBTD) is a deep learning-based classifier built to accurately detect and classify brain tumor types from MRI images. The model is trained using a fine-tuned ResNet50 architecture and leverages modern training techniques like data augmentation and MixUp regularization.

## Model Evaluation Report

The model was evaluated on a balanced test set containing four brain tumor categories: **GLIOMA**, **MENINGIOMA**, **NOTUMOR**, and **PITUITARY**.

### Classification Performance:

| Class      | Precision | Recall | F1-score | Support |
| ---------- | --------- | ------ | -------- | ------- |
| GLIOMA     | 99.65%    | 95.00% | 97.27%   | 300     |
| MENINGIOMA | 93.87%    | 95.10% | 94.48%   | 306     |
| NOTUMOR    | 99.26%    | 99.51% | 99.38%   | 405     |
| PITUITARY  | 96.44%    | 99.33% | 97.87%   | 300     |

**Overall Accuracy:** 97.41%

### Confusion Matrix:

```
               Predicted
            G     M     N     P
Actual G   285   15     0     0
       M     1  291     3    11
       N     0    2   403     0
       P     0    2     0   298
```

(G: GLIOMA, M: MENINGIOMA, N: NOTUMOR, P: PITUITARY)

## Real-World Interpretation

If the model is used in a hospital setting to classify MRI scans for 100 patients:

* **On average, it will correctly diagnose about 97 to 98 patients.**
* The likelihood of misclassifying a tumor as another type is relatively low, especially for **NOTUMOR**, where the model achieves over 99% recall and precision.

### Example Use Case

A radiologist uploads an MRI image of a patient suspected of having a brain tumor:

* The IBTD model processes the image.
* It returns: `Class: PITUITARY` with a high confidence score.
* The radiologist uses this prediction to prioritize diagnosis and further testing.

## Model Details

* **Architecture:** ResNet50 (fine-tuned)
* **Training Set Size:** 4856 images
* **Validation Set Size:** 856 images
* **Epochs Trained:** 30
* **Data Augmentation:** Rotation, flipping, affine transforms, color jitter
* **Regularization:** MixUp

## Developer

**Assem Sabry**
AI Engineer and Creator of IBTD
