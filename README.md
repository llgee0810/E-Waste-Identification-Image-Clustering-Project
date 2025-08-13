# E-Waste Identification — Image Clustering Project

## Portfolio Summary
Developed an unsupervised machine learning pipeline to classify damaged returned products as e-waste or non-e-waste using 2,400 unstructured product images. Applied VGG16 for deep learning-based feature extraction, PCA for dimensionality reduction, and K-Means clustering to group products by type and detect visible damage patterns. Delivered actionable recommendations to improve operational efficiency, reduce manual inspection costs, and enhance sustainability management for GlobalStore.

---

## Business Problem
**How can GlobalStore identify e-waste in returned products without labelled training data to reduce processing costs and improve sustainability performance?**  
Returned products include both functional and damaged items, making manual inspection time-consuming and costly. With no labelled dataset available, this project applies unsupervised learning to group similar products and flag potential damage, enabling prioritisation of inspection, improved recycling processes, and creation of labelled datasets for future supervised learning.

---

## Dataset & Data Model

**Source data:**  
- 2,400 unlabelled images of returned products in various categories (electronics, furniture, appliances).  
- Images stored in `/product_images` folder with mixed resolutions and file formats.  

**Data preprocessing and feature engineering:**  
- Images resized to 150×150 pixels for model input consistency.  
- Features extracted using pre-trained VGG16 convolutional neural network (without top classification layers).  
- Dimensionality reduction applied via Principal Component Analysis (PCA) to remove noise and retain key patterns.  

**Data outputs:**  
- `clustering_results.csv`: First-round clustering output with `k=8` (Davies-Bouldin Index).  
- `cluster_{id}_reclustered.csv`: Second-round clustering outputs with `k=2` for damage-focused separation.
<img width="423" height="235" alt="image" src="https://github.com/user-attachments/assets/e8704d45-d748-4f3c-aae8-11fe4e518ea6" />

---

## Approach

**1. Feature Extraction**  
- Used VGG16 CNN to transform images into high-dimensional numerical feature maps.  
- Extracted features represent textures, edges, and object shapes relevant to identifying damage.  

**2. Dimensionality Reduction**  
- Applied PCA to reduce dimensionality while retaining detail necessary to detect cracks or missing parts.  
- Initial round: `n_components=100` to maintain fine-grained details.  

**3. First-Round Clustering**  
- Determined optimal number of clusters using the Davies-Bouldin Index.  
- Set `k=8` to group by product type before attempting damage detection.  
<img width="475" height="265" alt="image" src="https://github.com/user-attachments/assets/0df29e56-b23c-4ebc-97c6-f4d2971d9aba" />

**4. Second-Round Clustering**  
- Applied sub-clustering to selected clusters (e.g., Cluster 2) with `k=2` to distinguish likely e-waste from non-e-waste within each product type.  
<img width="488" height="293" alt="image" src="https://github.com/user-attachments/assets/4dd45abc-7520-41f8-988c-4fd3b99dc6b1" />

---

## Key Findings
- Cluster 2 contained the largest and most varied set of products, including items with visible damage.  
- Furniture clusters (tables, chairs) could be confidently classified as non-e-waste.  
- Sub-clustering allowed separation of newer and older product versions, with older versions more often damaged.  
- The model can identify some visible damage but struggles with internal defects or subtle wear.  
<img width="405" height="210" alt="image" src="https://github.com/user-attachments/assets/a1606861-d295-4acb-879c-1616f176a288" />
<img width="407" height="211" alt="image" src="https://github.com/user-attachments/assets/3d023212-edba-46c6-9a1b-c4202ab86c9e" />

---

## Actionable Insights
1. Manually label a small subset of damaged and non-damaged products to enable semi-supervised learning for improved accuracy.  
2. Request customers to indicate whether a returned product is damaged to provide metadata for model training.  
3. Apply version-based filtering to prioritise inspection of older products.  

---

## Technologies Used
- **Python**: NumPy, Pandas, Matplotlib, OpenCV  
- **Deep Learning**: TensorFlow / Keras (VGG16)  
- **Machine Learning**: Scikit-learn (PCA, K-Means, Davies-Bouldin Index)  
- **Data Visualization**: Matplotlib, Seaborn  

---


