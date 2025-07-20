# MediRead

**MediRead: An AI-Powered System for Understanding Medical PDFs and Answering Questions**

---

## Introduction

Medical documents often contain dense information and specialized terminology, making them difficult to navigate. **MediRead** is a smart solution that uses biomedical language models to analyze and comprehend uploaded PDFs, delivering **accurate, document-grounded answers** without hallucination. It's designed for students, professionals, and researchers seeking an efficient way to query medical literature.

---

## Problem Statement

Understanding medical PDFs is challenging due to:

- Their **unstructured** nature.
- The use of **domain-specific terminologies and abbreviations**.
- The need for **manual effort** to extract information.

**MediRead** addresses these issues by:

- Understanding biomedical language.
- Accepting and processing **multiple PDFs** at once.
- Providing **fact-based answers** strictly grounded in the uploaded content.

---

## Models Used

- **BioBERT**:  
  Used for embedding text chunks from uploaded PDFs into a vector database.

- **SmolLM3-3B** (by Hugging Face):  
  A general-purpose LLM used for generating answers based on the relevant context retrieved from the vector store.

---

## System Overview

MediRead is a **Streamlit-based Retrieval-Augmented Generation (RAG)** system consisting of:

- PDF ingestion using `fitz` (PyMuPDF)
- Chunking with `RecursiveCharacterTextSplitter` from LangChain
- Biomedical embedding with **BioBERT**
- Vector indexing and search using **FAISS**
- Natural language answering via **SmolLM3-3B**

---

## Methodology

### **Pipeline Workflow**

```text
PDFs
 → Text Extraction
   → Chunking
     → BioBERT Embeddings
       → FAISS Indexing

Question
 → BioBERT Embedding
   → Top-k Search (FAISS)
     → Context
       → SmolLM3-3B
         → Answer
```

---

## Implementation Details

- **UI**: Streamlit
- **PDF Parsing**: PyMuPDF (`fitz`)
- **Text Chunking**: LangChain’s RecursiveCharacterTextSplitter
- **Embeddings**: `pritamdeka/BioBERT`
- **Vector Store**: FAISS
- **LLM**: `HuggingFaceTB/SmolLM3-3B` (via InferenceClient)

---

## Challenges & Limitations

- API latency affects LLM response speed.
- BioBERT performs better on **GPU** than CPU.
- **OCR support is missing**, so scanned PDFs aren't processed.
- The system **does not hallucinate** – it answers only from the context it retrieves.

---

## Conclusion

MediRead significantly improves accessibility to complex medical literature. By integrating domain-specific embeddings and retrieval-aware generative models, it allows users to **ask questions naturally** while ensuring the answers remain **trustworthy and grounded** in the source documents.

---

## Future Work

- Add support for **OCR** (scanned document processing).
- Enable **chart and figure interpretation**.
- Fine-tune models on **medical research datasets** for better accuracy.
- Add **multi-language support**.
- Develop **offline/mobile** versions of the app.

---

<a href="https://drive.google.com/file/d/1GJUgor4c5EndLDVG_lv4FOMLsKPoggFc/view?usp=sharing">**Watch Demo Video**</a><br>
**Contact**: fareshatahet491@outlook.com
