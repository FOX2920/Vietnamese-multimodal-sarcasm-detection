# Multimodal Sarcasm Detection on Vietnamese Social Media Datasets

This repository contains the implementation of a multimodal deep learning approach for detecting sarcasm in Vietnamese social media content, using both text and image inputs.

## Overview

Sarcasm is a complex linguistic phenomenon often used to express opinions contrary to what is literally stated, typically for humor or criticism. With the rise of multimodal content on social media platforms like Facebook and Instagram, detecting sarcasm has become increasingly complex due to the combination of text, images, and emojis.

This project presents a study on multimodal sarcasm detection using data from the DSC UIT Challenge - Table B, with a deep learning model that combines linguistic and visual features.

## Problem Definition

The task is to detect and classify sarcasm in multimodal posts consisting of:
- An image (I)
- A text description/caption (C) related to the image

The output is one of four labels:
- **non-sarcasm**: The content does not contain sarcasm
- **multi-sarcasm**: Sarcasm is conveyed through the combination of caption and image
- **image-sarcasm**: Sarcasm is primarily conveyed through the image
- **text-sarcasm**: Sarcasm is primarily conveyed through the caption

## Model Architecture

Our model uses a multimodal fusion approach that combines features from both text and image processing components:

### Image Processing
- **Vision Transformer (ViT)**: Extracts global semantic features from images
- **ResNet152**: Extracts detailed regional features from images
- These features are combined and reduced to a 512-dimensional vector

### Text Processing
- **ViSoBERT**: Vietnamese Social Media BERT, optimized for Vietnamese social media content
- **XLM-RoBERTa**: Multilingual model for capturing cross-lingual features
- A BiLSTM layer processes the combined text features

### Multimodal Feature Fusion
- Features from image and text components are concatenated
- The combined features pass through fully-connected layers for classification
- Outputs the predicted label using argmax on the logits

![Model Architecture](multimodal.drawio (1).png)

## Dataset

The ViMMSD (Vietnamese Multimodal Sarcasm Detection) dataset used in this study includes:
- **Training set**: 10,805 samples
- **Public test set**: 1,413 samples
- **Private test set**: 1,504 samples

The label distribution in the training set shows a significant imbalance:
- non-sarcasm: 6,062 samples (56.1%)
- multi-sarcasm: 4,224 samples (39.1%)
- image-sarcasm: 442 samples (4.1%)
- text-sarcasm: 77 samples (0.7%)

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning rate | 5 × 10⁻⁵ |
| Weight decay | 0.01 |
| Epochs | 8 |
| Early stopping | Patience = 3 |

We used the AdamW optimizer and Focal Loss to address the data imbalance issue.

## Results

The model combining ViSoBERT and XLM-RoBERTa for text processing, along with ViT and ResNet152 for image processing, achieved the best performance:

| Model | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| ViSoBERT | 41.7 | 44.3 | 41.6 |
| mBERT | 35.1 | 38.1 | 35.2 |
| PhoBERT | 37.2 | 39.3 | 37.3 |
| XLM-R | 37.5 | 40.2 | 37.6 |
| ViSoBERT + XLM-R | 42.0 | 44.8 | 42.4 |

## Key Findings

1. **ViSoBERT's Effectiveness**: ViSoBERT performed best among single-language models, proving its suitability for Vietnamese social media data that contains informal elements like teencode and emojis.

2. **XLM-RoBERTa's Multilingual Capabilities**: XLM-RoBERTa outperformed mBERT in the multilingual model category due to its robust semantic representation capabilities.

3. **Complementary Models**: Combining ViSoBERT and XLM-RoBERTa improved performance by leveraging ViSoBERT's Vietnamese language specialization and XLM-RoBERTa's cross-lingual context capabilities.

4. **Multimodal Superiority**: The multimodal approach consistently outperformed single-modality models, confirming the importance of considering both text and image information for sarcasm detection.

## Limitations and Future Work

- **Data Imbalance**: The significant imbalance between labels remains a challenge, affecting performance on underrepresented labels.
- **Cultural Context**: Some multilingual contexts or culture-specific elements are not well-handled by the model.
- **Resource Requirements**: Combined models require substantial computational resources and longer training times.

Future work will focus on:
- Collecting more diverse data, including multilingual content and cultural elements
- Improving methods for processing informal elements like emojis and teencode
- Enhancing model fusion techniques for better efficiency and generalization
- Optimizing the training process to reduce time and resource costs

## Dependencies
To run this project, install the following dependencies:
```
pip install torch torchvision transformers datasets scikit-learn matplotlib seaborn pandas
```

## Training
Run the training script using:
```
python train.py --epochs 20 --batch_size 32 --lr 5e-5
```

## Evaluation
Evaluate the trained model on the test set:
```
python evaluate.py --model checkpoint.pth
```

## Contributions
Feel free to contribute by submitting issues and pull requests.

## License
This project is licensed under the MIT License.

