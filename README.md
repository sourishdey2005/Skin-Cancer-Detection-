# Skin-Cancer-Detection-
Skin Cancer Detection 
# 🔬 Advanced Skin Lesion Classification with CPU-Optimized CNN (`CPU_Light_Efficient_CNN`)

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/cafc038d-c7da-4d0b-9458-019d4a62f52e" />


## 🌟 Project Overview

This project presents `CPU_Light_Efficient_CNN`, a custom-built Convolutional Neural Network (CNN) specifically designed for the classification of dermoscopic images into seven distinct skin lesion categories. Developed with a focus on computational efficiency, this model incorporates modern architectural patterns like **Depthwise Separable Convolutions** (inspired by EfficientNet and MobileNet) to achieve respectable accuracy while being suitable for CPU-bound environments or scenarios requiring lightweight models.

The primary objective is to develop a robust and accessible solution for automated skin cancer preliminary screening using the HAM10000 dataset, which encompasses a diverse range of lesion types. The project provides a complete pipeline from efficient data preparation and real-time augmentation to model definition, training with advanced callbacks, and detailed performance evaluation.

---

## ✨ Key Features

* **CPU-Optimized Architecture (`CPU_Light_Efficient_CNN`):** Implements Depthwise Separable Convolutions to significantly reduce model parameters and computational cost, making it efficient for CPU inference and resource-constrained environments.
* **Comprehensive Data Pipeline:** Utilizes `tf.data.Dataset` for optimized, high-throughput data loading, preprocessing, and real-time augmentation (flips, rotations, brightness/contrast adjustments).
* **Robust Training Strategy:**
    * **Class Weighting:** Dynamically balances loss contributions from under-represented classes using `compute_class_weight` to effectively address dataset imbalance.
    * **Advanced Callbacks:** Employs `ReduceLROnPlateau` for adaptive learning rate scheduling, `EarlyStopping` to prevent overfitting, and `ModelCheckpoint` for saving the best performing model based on validation accuracy.
* **Reproducibility:** Seeded random states (`tf.random.set_seed`, `np.random.seed`) ensure consistent experimental results.
* **Modular Design:** Clearly separated components for data handling, model definition, and training logic for easy understanding, modification, and extension.
* **Detailed Evaluation:** Provides a full suite of metrics including Accuracy, F1-Score (weighted), Precision, Recall, Classification Report, and Confusion Matrix.
* **Retraining Capability:** Seamlessly loads and continues training from a previously saved model checkpoint, allowing for iterative improvement, fine-tuning, or extended training.

---

## 🧠 Model Architecture: `CPU_Light_Efficient_CNN`

The `CPU_Light_Efficient_CNN` is a custom-designed, lightweight, yet capable sequential CNN. Its design is heavily influenced by EfficientNet and MobileNet principles, primarily through the extensive use of **Depthwise Separable Convolutions**. This technique decomposes a standard convolution into a depthwise convolution (filtering inputs channels independently) and a pointwise convolution (combining the outputs), drastically reducing computational load and parameter count.

### Architecture Breakdown:

* **Input Layer:** `(128, 128, 3)` - Accepts normalized RGB images.
* **Initial Convolution:** A standard `Conv2D` layer with 32 filters, followed by `BatchNormalization` and ReLU activation, serving as the initial feature extractor.
* **Efficient Feature Blocks (3 Blocks):**
    * Each block begins with a `DepthwiseConv2D` (3x3 kernel, ReLU activation, `padding='same'`) to filter input channels independently.
    * This is immediately followed by a `Conv2D` (1x1 kernel, ReLU activation) to efficiently combine the outputs from the depthwise step (the "pointwise" convolution).
    * `BatchNormalization` is applied after each `Conv2D` for stable training.
    * `MaxPooling2D` (`2x2`) is used for spatial down-sampling, reducing the feature map dimensions.
    * `Dropout` layers (with rates 0.25, 0.35, 0.45) are strategically placed after each block to prevent overfitting by randomly deactivating neurons during training.
    * Filter counts progressively increase across blocks: $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ (for the 1x1 convolutions).
* **Global Average Pooling (`GlobalAveragePooling2D`):** This layer efficiently flattens the final 2D feature maps into a single 1D feature vector by averaging values, significantly reducing parameters compared to traditional `Flatten` layers.
* **Classifier Head:**
    * A `Dense` (fully connected) layer with 256 units (ReLU activation) for high-level feature combination.
    * Additional `BatchNormalization` and `Dropout` (0.5) for robust regularization before the final output.
    * **Output Layer:** A `Dense` layer with `CONFIG['num_classes']` (7) units and `softmax` activation, producing the final probability distribution over the seven skin lesion classes.

### Architecture


<img width="3999" height="1178" alt="image" src="https://github.com/user-attachments/assets/c1070e4a-c648-4583-a647-d4878d6cf060" />


## 📊 Dataset

This project utilizes the **HAM10000 ("Human Against Machine with 10000 training images")** dataset, a comprehensive collection of dermoscopic images of common pigmented skin lesions. It comprises 10,015 dermatoscopic images which are meticulously categorized into seven distinct classes of skin conditions:

1.  **`akiec`**: Actinic Keratoses and Intraepithelial Carcinoma / Bowen's disease
2.  **`bcc`**: Basal Cell Carcinoma
3.  **`bkl`**: Benign Keratosis-like lesions (solar lentigines / seborrheic keratoses / lichen planus-like keratoses)
4.  **`df`**: Dermatofibroma
5.  **`mel`**: Melanoma
6.  **`nv`**: Melanocytic nevi
7.  **`vasc`**: Vascular lesions (angiomas / angiokeratomas)

The dataset typically consists of image files (e.g., `HAM10000_images_part_1`, `HAM10000_images_part_2`) and a corresponding metadata CSV file (`HAM10000_metadata.csv`) that provides diagnostic information and patient demographics for each image.

---

## 🛠️ Setup and Installation

To get this project running on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project is configured for `tensorflow-cpu` for broader compatibility. If you have a GPU and the necessary CUDA/cuDNN setup, you can install the `tensorflow` package instead of `tensorflow-cpu` for accelerated performance.

    ```bash
    pip install -r requirements.txt
    # Alternatively, install manually:
    pip install tensorflow-cpu==2.13.0 scikit-learn matplotlib seaborn opencv-python-headless Pillow -q
    ```

4.  **Download the Dataset:**

    * The HAM10000 dataset is typically available on platforms like Kaggle. You will need to download the `HAM10000_metadata.csv` and the image folders (e.g., `HAM10000_images_part_1`, `HAM10000_images_part_2`, and potentially a `Skin Cancer/Skin Cancer` folder if it's part of your specific download).
    * Place these downloaded directories in a structured path similar to Kaggle's `/kaggle/input/skin-cancer-dataset/` or, more practically, create a `data/` directory within your project root and place them there.
    * **Important:** You might need to modify the `base_paths` list within the `find_image_paths` function in `main.py` to correctly point to the absolute or relative paths of your image directories on your local system.

    **Example Recommended Local Structure:**
    ```
    your_project_root/
    ├── main.py
    ├── models/
    ├── visualizations/
    ├── data/
    │   ├── HAM10000_metadata.csv
    │   ├── HAM10000_images_part_1/
    │   ├── HAM10000_images_part_2/
    │   └── Skin Cancer/
    │       └── Skin Cancer/
    └── requirements.txt
    ```
    If you follow this structure, you'd update `base_paths` to include paths like `'./data/HAM10000_images_part_1'`, etc.

---

## 🚀 Usage

The `main.py` script serves as a unified entry point, intelligently handling both initial training and continuation (retraining) of the model.

### Initial Training (First Run)

Simply execute the script. It will automatically detect the absence of a pre-trained model checkpoint (`custom_cnn_v2_cpu_light.h5`) and initiate training from scratch.

```bash
python main.py
```

*Expected Console Output Snippet:*

```
...
No existing model found at /kaggle/working/models/custom_cnn_v2_cpu_light.h5. Training from scratch...
...
Phase: Training CPU_Light_Efficient_CNN from epoch 1
Epoch 1/150
...
```

### Retraining (Continuing Training for Better Accuracy)

If a model checkpoint (`custom_cnn_v2_cpu_light.h5`) already exists in the `models/` directory, running the script will automatically load these weights and resume training. This is the intended workflow for fine-tuning the model or training for additional epochs to achieve higher accuracy.

```bash
python main.py
```

*Expected Console Output Snippet:*

```
...
Found existing model: /kaggle/working/models/custom_cnn_v2_cpu_light.h5. Loading for retraining...
Model loaded successfully. Continuing training from initial epoch 0.
...
Phase: Training CPU_Light_Efficient_CNN from epoch 1
Epoch 1/150
...
```

*(Note: The `initial_epoch` displayed refers to the start of the current `model.fit()` call. The model itself starts with the loaded weights from the best previous epoch.)*

### Key Configuration Parameters

The `CONFIG` dictionary within `main.py` allows for easy adjustment of critical training parameters:

* `img_size`: Tuple `(height, width)` for input image dimensions (e.g., `(128, 128)`).
* `batch_size`: Number of samples processed before the model's weights are updated.
* `epochs`: Total number of full passes over the training dataset. Set higher for extended retraining.
* `learning_rate`: Initial learning rate for the Adam optimizer. Often reduced during retraining for finer adjustments.
* `model_path`: Specifies the exact path where the model is saved and loaded from.

-----

## 📈 Results and Performance

After training or retraining, the script will output the final test set metrics to the console and generate detailed plots for training history.

**Achieved Performance (Latest Run):**

* **Model:** `CPU_Light_Efficient_CNN`
* **Final Test Accuracy:** `0.6693`
* **Final Test F1-Score (Weighted):** `0.5367`

**🎯 Target Accuracy:** `90%+`

The current performance indicates a solid baseline, especially for a CPU-optimized custom model. The journey towards the target 90%+ accuracy will involve continued retraining, potential hyperparameter tuning, and exploring further architectural refinements or advanced data augmentation strategies.

### Visualizations:

The script automatically saves the following insightful plots into the `visualizations/` directory after each training run:

#### 1\. Training & Validation Loss History

This plot illustrates the model's learning progression by showing how the loss values (error) change over epochs for both the training and validation datasets. A converging gap typically indicates a well-regularized model.

![Training Loss History](visualizations/CPU_Light_Efficient_CNN_training_history.png)

*(This image will be generated in your `visualizations/` directory after you run the script and will depict your actual training and validation loss history.)*

#### 2\. Training & Validation Accuracy History

This plot tracks the accuracy metric over epochs, allowing visual inspection of how well the model generalizes to unseen data during training (validation accuracy). The goal is to maximize validation accuracy without significant overfitting.



*(This image will be generated in your `visualizations/` directory after you run the script and will depict your actual training and validation accuracy history.)*

#### 3\. Classification Report (Console Output)

A detailed classification report provides per-class performance metrics (Precision, Recall, F1-Score) along with overall averages. This is crucial for understanding the model's strengths and weaknesses across different skin lesion types, especially given the class imbalance.



*(Note: The provided `main.py` currently only generates history plots. To produce the Classification Report and Confusion Matrix visuals as image files, you would need to re-integrate or extend the `plot_results_single` function from previous iterations and ensure it's called after final test evaluation.)*

---

## 🔄 Reproducibility

For consistent and verifiable experimental results, all relevant random seeds (TensorFlow, NumPy) are explicitly set at the beginning of the script:

```python
tf.random.set_seed(42)
np.random.seed(42)
```

This measure ensures that running the script multiple times with the same configuration and dataset should yield highly consistent outcomes.

---

## 🚀 Future Enhancements

The current model provides a solid foundation. Here are several avenues for future improvement and expansion:

* **Advanced Hyperparameter Optimization:** Implement automated search strategies (e.g., Keras Tuner, Optuna, Weights & Biases) to explore the optimal combinations of learning rates, batch sizes, dropout schedules, L2 regularization, and architectural variations.
* **Ensemble Modeling:** Combine predictions from several models (e.g., different initializations, architectures, or folds of data) to enhance overall robustness and potentially boost accuracy beyond single-model limits.
* **Interpretability with XAI:** Integrate Explainable AI (XAI) techniques such as Grad-CAM, LIME, or SHAP to visualize which specific regions of the dermoscopic images most influence the model's predictions, thereby increasing clinical trust and understanding.
* **Model Quantization & Pruning:** Explore techniques for model size reduction and acceleration (e.g., TensorFlow Lite, Post-training Quantization, Pruning) for efficient deployment on edge devices or mobile platforms.
* **Sophisticated Augmentation:** Experiment with advanced data augmentation techniques like CutMix, Mixup, or AutoAugment to further increase data diversity and model generalization.
* **Higher Resolution Inputs:** Investigate the impact of training with larger image resolutions if computational resources permit, potentially capturing finer diagnostic details.
* **Multi-Modal Data Fusion:** Explore integrating patient metadata (age, sex, localization) as auxiliary inputs to the model to potentially improve diagnostic accuracy by providing clinical context.
* **Active Learning:** Implement an active learning loop to strategically select the most informative unlabelled samples for expert annotation, optimizing the data labeling process.

---

## 🤝 Contributing

Contributions are highly welcome and essential for improving this project\! If you have suggestions for enhancements, new features, bug fixes, or documentation improvements, please feel free to:

1.  **Open an Issue:** Describe the bug, feature request, or suggestion.
2.  **Submit a Pull Request:**
    * Fork the repository.
    * Create a new branch: `git checkout -b feature/your-feature-name` or `bugfix/your-bugfix-name`.
    * Make your changes, ensuring code quality and adherence to existing style.
    * Commit your changes with clear and descriptive messages.
    * Push to your fork: `git push origin feature/your-feature-name`.
    * Open a Pull Request to the `main` branch of this repository, providing a detailed explanation of your changes.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for full details.

---

## 📞 Contact

For any questions, feedback, or collaborations, please feel free to reach out:

* **Your Name:** Sourish Dey 
* **GitHub:** [sourishdey2005](https://github.com/sourishdey2005)
* **LinkedIn (Optional):** [https://www.linkedin.com/in/sourish-dey-20b170206/]

---
