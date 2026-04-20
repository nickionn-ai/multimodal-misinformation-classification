# Multimodal Misinformation Classification

This project presents a multimodal machine learning system that combines textual and visual information to classify content more effectively than single-modality approaches.

By leveraging **DistilBERT** for text embeddings and **CLIP** for image embeddings, the model integrates both modalities to improve classification performance and robustness.

---

## 🚀 Project Overview

Traditional classification models rely on a single data source, such as text or images. However, real-world data often contains multiple modalities that provide complementary information.

In this project, we explore whether combining:
- **Text (semantic information)**
- **Images (visual context)**

can improve classification accuracy compared to individual models.

---

## 🧠 Methodology

The pipeline consists of the following steps:

1. **Data Preprocessing**
   - Clean text and image URLs
   - Remove invalid samples
   - Construct a valid multimodal dataset

2. **Feature Extraction**
   - Text embeddings using **DistilBERT**
   - Image embeddings using **CLIP**

3. **Multimodal Fusion**
   - Early fusion via feature concatenation

4. **Model Training**
   - Logistic Regression classifier

5. **Baselines**
   - Text-only model
   - Image-only model

6. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC Curve & AUC
   - Precision-Recall Curve
   - Calibration Analysis

7. **Advanced Analysis**
   - Error Analysis
   - Hard / Uncertain examples
   - Feature importance
   - Robustness checks
   - Ablation-style insights

---

## 📊 Results

| Model        | Accuracy |
|-------------|---------|
| Text-only    | ~0.66 |
| Image-only   | ~0.71 |
| Multimodal   | ~0.76 |

The multimodal model achieves the best overall performance, demonstrating that combining textual and visual features provides a more informative representation.

- Improved **recall** for the positive class  
- More balanced **precision–recall trade-off**  
- Better overall **robustness**

---

## 📈 Key Insights

- Multimodal learning significantly improves performance over single-modality models  
- Visual features provide strong signals but are enhanced when combined with text  
- Simple fusion strategies already yield strong results  
- Some ambiguity remains in challenging samples  

---

## ⚠️ Limitations

- Uses simple fusion (concatenation)  
- Limited dataset size  
- Not fully calibrated probabilities  
- Struggles with ambiguous or conflicting samples  

---

## 🔮 Future Work

- Explore advanced multimodal architectures (e.g., transformers with cross-attention)  
- Improve calibration techniques  
- Increase dataset size and quality  
- Experiment with more sophisticated fusion strategies  

---

## 🛠️ Tech Stack

- Python  
- NumPy / Pandas  
- Scikit-learn  
- PyTorch  
- HuggingFace Transformers  
- Matplotlib / Seaborn  

---

## ▶️ How to Run

1. Clone the repository:
git clone https://github.com/your-username/multimodal-misinformation-classification.git
cd multimodal-misinformation-classification

2. Install dependencies:
pip install -r requirements.txt

3. Run the notebook:
jupyter notebook multimodal_misinformation_classification.ipynb

---

## 📌 Project Highlights

- End-to-end multimodal ML pipeline  
- Real-world data handling (missing images, noisy data)  
- Comprehensive evaluation beyond accuracy  
- Strong emphasis on model interpretability and analysis  

---

## 📎 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- HuggingFace Transformers  
- OpenAI CLIP  
- Scikit-learn  
