# 🚀 Automated Question Generation (AQG)  

Automated Question Generation (AQG) is an **AI-driven system** that automatically generates **multiple-choice, fill-in-the-blank, and short-answer questions** from text. It is useful for **educational assessments, quizzes, and NLP applications**, significantly improving efficiency in content creation.  

---

## 📖 Project Overview  
This project focuses on **automated question generation** from **PDF documents, scripts, or lessons**. Using **Transformer models (T5, BART, and GPT-3)**, it processes text and generates meaningful **questions and answers** for learning and assessment purposes.  

---

## ⚙️ Features  
✅ Supports **PDFs, scripts, or text-based lessons** as input.  
✅ Uses **T5, BART, and OpenAI GPT models** for question generation.  
✅ Fine-tunes **T5 on the MS MARCO dataset** for better accuracy.  
✅ Supports **multiple NLP-based question generation techniques**.  
✅ Saves and loads trained models for **efficient reusability**.  

---

## 🛠 Models Used  

| Model | Purpose |  
|--------|---------|  
| **T5 (Google's Transformer)** | Fine-tuned for **question generation** on the `MS MARCO` dataset. |  
| **BART (Facebook's Transformer)** | Generates questions based on **context using beam search**. |  
| **GPT-3 (OpenAI text-davinci-003)** | Generates **creative and diverse** questions using OpenAI’s API. |  

---

## 📝 Training Process  

### **1️⃣ Dataset Used**  
- The model is trained using the **MS MARCO (Microsoft Machine Reading Comprehension)** dataset.  
- This dataset contains **passage-question-answer pairs**, making it ideal for **question generation** tasks.  

### **2️⃣ Data Preprocessing**  
- Extracts text from **PDFs, scripts, or raw documents**.  
- Tokenizes text using **NLTK / SpaCy**.  
- Converts text into **input-output pairs** (`context → question`).  
- Formats the dataset to match **T5/BART’s training structure**.  

### **3️⃣ Model Fine-Tuning**  

#### **🔹 T5 Fine-Tuning**  
✅ Trained on **passage-based question generation tasks**.  
✅ Uses **encoder-decoder Transformer architecture**.  
✅ Fine-tuned using **Hugging Face Transformers**.  
✅ **Loss function**: Cross-Entropy Loss.  

#### **🔹 BART Fine-Tuning**  
✅ Uses **beam search decoding** for better fluency.  
✅ Optimized to generate **diverse and grammatically correct questions**.  

#### **🔹 GPT-3 Integration**  
✅ Generates **creative & complex** questions.  
✅ Uses **OpenAI’s API** for inference (without explicit fine-tuning).  

### **4️⃣ Training Hyperparameters**  

| Parameter | Value |  
|-----------|------|  
| **Batch Size** | `4` |  
| **Epochs** | `3` |  
| **Learning Rate** | `2e-5` |  
| **Optimizer** | AdamW |  
| **Loss Function** | Cross-Entropy |  
| **Framework** | PyTorch + Hugging Face Transformers |  

### **5️⃣ Model Evaluation**  
- **Evaluation Metrics**: BLEU, ROUGE, METEOR scores.  
- Compares **generated vs. ground-truth questions**.  
- **Visualizes** training loss & model performance.  

### **6️⃣ Saving & Loading Models**  

#### **🔹 Saving the Model**  
After training, you can save the model using:  python save.py

📊 Expected Output
✔ Model-generated questions from input text.
✔ Fine-tuning logs and training loss graphs.
✔ Saved model available for future use (./fine_tuned_t5_model/).

🔥 Future Improvements
🔹 Improve accuracy using hybrid Transformer-based ensembles.
🔹 Add support for multiple datasets (SQuAD, HotpotQA).
🔹 Develop a Streamlit Web App for an interactive user interface.
