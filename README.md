# nuclei-cGAN: Conditional GAN for Nuclei Image Generation  
🚀 **Generating synthetic nuclei images with controllable morphological features using Conditional GANs (cGANs)**  

## 📌 Project Overview  
This project implements a **Conditional Generative Adversarial Network (cGAN)** to generate realistic **nuclei images** conditioned on clinical metadata. The model allows control over key **morphological features** such as **size, shape, and texture**, making it useful for **digital pathology augmentation, synthetic dataset generation, and AI-assisted histopathology research**.  

### 🔬 **Why This Matters**
- **Synthetic Data Augmentation**: Overcome dataset limitations in histopathology.  
- **Controlled Image Generation**: Create **specific nuclei types** using conditioning labels.  
- **Enhancing AI Models**: Train pathology AI systems with **diverse, balanced datasets**.  

---

## 🏗️ **How It Works**
### 1️⃣ **Conditional GAN Architecture**
- **Generator (G):** Creates synthetic nuclei images based on conditioning labels (e.g., "large round nuclei with smooth texture").  
- **Discriminator (D):** Evaluates image authenticity and ensures metadata alignment.  

### 2️⃣ **Conditioning Input (Clinical Metadata)**
- **Morphological Features**: Nucleus **size, shape, texture, staining intensity**  
- **Pathology Metadata**: Cancer **subtype, patient age, stain type**  

### 3️⃣ **Training Data**
- Uses **public histopathology datasets** like **MoNuSeg, PanNuke, TNBC**.  
- Extracts **nucleus masks and feature distributions** for conditioning.  

---

## 🛠️ **Tech Stack**
- **Deep Learning:** PyTorch, TensorFlow/Keras  
- **GAN Framework:** Conditional GAN with label conditioning  
- **Image Processing:** OpenCV, scikit-image  
- **Data Handling:** Pandas, NumPy  

---

## 📂 **Project Structure**
```bash
📂 nuclei-cGAN  
 ├── 📂 data/  # Nuclei images & metadata  
 ├── 📂 models/  # Generator & Discriminator architectures  
 ├── 📂 training/  # Training scripts & loss functions  
 ├── 📂 evaluation/  # Image quality assessment (FID, SSIM)  
 ├── 📂 utils/  # Preprocessing, augmentation functions  
 ├── README.md  # Project Overview  
 ├── requirements.txt  # Dependencies  
 ├── train.py  # Training pipeline  
 ├── generate.py  # Inference script to generate synthetic nuclei  
