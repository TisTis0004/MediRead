import os
import re
import numpy as np
import fitz
import torch
import faiss
import tempfile
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------- Page Setup -----------------
st.set_page_config(page_title="MediRead", layout="centered")
st.title("🩺 MediRead")
st.markdown(
    "**MediRead** helps you understand your medical PDFs by answering questions using AI. "
    "Upload one or more PDFs, ask a question, and get answers directly from your documents!"
)


# ----------------- PDF Handling -----------------
def extract_text_from_pdf(pdf_folder_path, progress_callback):
    pages = []
    files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith(".pdf")]
    for file_idx, file in enumerate(files):
        full_path = os.path.join(pdf_folder_path, file)
        try:
            doc = fitz.open(full_path)
            for page_num, page in enumerate(doc, start=1):
                pages.append({"file": file, "page": page_num, "text": page.get_text()})
        except Exception as e:
            st.error(f"[ERROR] Failed to open {file}: {e}")
        progress_callback((file_idx + 1) / len(files))
    return pages


def chunking(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    file_chunks = []
    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])
        for i, chunk in enumerate(chunks):
            file_chunks.append(
                {
                    "chunk": chunk,
                    "metadata": {
                        "file": page_data["file"],
                        "page": page_data["page"],
                        "chunk_id": i,
                    },
                }
            )
    return file_chunks


# ----------------- Embedding -----------------
@st.cache_resource(show_spinner=False)
def load_bio_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    model = AutoModel.from_pretrained(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    return tokenizer, model


def embed_chunks(_chunks, _tokenizer, _model, progress_callback):
    _model.eval()
    embeddings = []
    total = len(_chunks)
    with torch.no_grad():
        for i, chunk in enumerate(_chunks):
            inputs = _tokenizer(
                chunk["chunk"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            output = _model(**inputs)
            last_hidden = output.last_hidden_state
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden.size())
                .float()
            )
            pooled = (last_hidden * mask).sum(1) / mask.sum(1)
            embeddings.append(
                {
                    "embedding": pooled.squeeze().cpu().numpy(),
                    "metadata": chunk["metadata"],
                }
            )
            if i % 5 == 0 or i == total - 1:
                progress_callback((i + 1) / total)
    return embeddings


def build_faiss_index(embedded_chunks):
    vectors = np.array([c["embedding"] for c in embedded_chunks]).astype("float32")
    metadata = [c["metadata"] for c in embedded_chunks]
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, metadata


# ----------------- Inference Setup -----------------
HF_TOKEN = os.getenv("HF_MediRead_final")
if HF_TOKEN is None:
    st.error(
        "Please set the environment variable `HF_MediRead_final` to access the inference model."
    )
    st.stop()

client = InferenceClient(model="HuggingFaceTB/SmolLM3-3B", token=HF_TOKEN)


# ----------------- RAG QA -----------------
def rag_ask_question(query, index, chunks, tokenizer, embed_model, top_k=4):
    embed_model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        output = embed_model(**inputs)
        last_hidden = output.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)
        query_vector = pooled.squeeze().cpu().numpy().astype("float32").reshape(1, -1)

    D, I = index.search(query_vector, top_k)
    retrieved_info = [
        {"text": chunks[i]["chunk"][:800], "metadata": chunks[i]["metadata"]}
        for i in I[0]
    ]
    context = "\n\n".join([info["text"] for info in retrieved_info])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly knowledgeable medical research assistant. "
                "Your goal is to help users understand information found *strictly* within the provided context. "
                "You must not guess or fabricate information. "
                "If the answer cannot be derived from the context, say: 'The information is not available in the provided documents.'\n\n"
                "Do not elaborate, infer, or generalize beyond the context. If the information is even partially missing, clearly say it is unavailable.\n\n"
                "When answering:\n"
                "- Be precise, factual, and use formal language.\n"
                "- Structure your response in clear paragraphs.\n"
                "- If appropriate, cite the source file and page (e.g., 'Source: filename.pdf, page 4').\n"
                "- Avoid speculation, opinions, or general knowledge.\n"
                "- Never mention anything not present in the context."
            ),
        },
        {
            "role": "user",
            "content": f"### Context:\n{context}\n\n### Question:\n{query}",
        },
    ]

    try:
        completion = client.chat.completions.create(messages=messages)
        raw_answer = completion.choices[0].message.content.strip()
        cleaned_answer = re.sub(
            r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL
        ).strip()
        answer = cleaned_answer if cleaned_answer else "⚠️ No answer generated."
    except Exception as e:
        answer = f"⚠️ LLM call failed: {e}"

    sources = "\n".join(
        [
            f"- {info['metadata']['file']} (page {info['metadata']['page']})"
            for info in retrieved_info
        ]
    )
    return answer, sources, retrieved_info


# ----------------- UI Flow -----------------
uploaded_files = st.file_uploader(
    "📤 Upload your medical PDFs", type=["pdf"], accept_multiple_files=True
)
st.caption(
    "📌 Only files **under 20MB** will be accepted. Files over this size will be ignored automatically."
)

MAX_MB = 20
if uploaded_files:
    too_large = [f.name for f in uploaded_files if f.size > MAX_MB * 1024 * 1024]
    if too_large:
        st.warning(
            f"⚠️ These files are too large (> {MAX_MB}MB) and were skipped:\n- "
            + "\n- ".join(too_large)
        )
        uploaded_files = [f for f in uploaded_files if f.size <= MAX_MB * 1024 * 1024]

if uploaded_files:
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False

    if not st.session_state.index_ready:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            progress_bar = st.progress(0, text="🔍 Extracting PDF text...")
            raw_pages = extract_text_from_pdf(
                tmp_dir, lambda x: progress_bar.progress(x, "📄 Extracting...")
            )

            progress_bar.progress(0, "🧩 Splitting into chunks...")
            chunks = chunking(raw_pages)

            progress_bar.progress(0, "📥 Loading BioBERT...")
            tokenizer, model = load_bio_embedding_model()

            progress_bar.progress(0, "🧠 Embedding chunks...")
            embeddings = embed_chunks(
                chunks,
                tokenizer,
                model,
                lambda x: progress_bar.progress(x, "📦 Embedding..."),
            )

            progress_bar.progress(0.95, "📊 Building index...")
            index, metadata = build_faiss_index(embeddings)
            progress_bar.progress(1.0, "✅ Ready!")

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.index_ready = True
            st.success("✅ Files processed! Ask your medical question below.")

    st.markdown("### 🎛️ Retrieval Behavior")
    retrieval_mode = st.radio(
        "How do you want MediRead to find relevant content?",
        options=["🤖 Normal (Flexible)", "🔒 Strict (Precise)"],
        index=0,
        key="retrieval_mode_radio",
    )
    top_k_value = 4 if retrieval_mode == "🤖 Normal (Flexible)" else 2

    with st.form("question_form", clear_on_submit=False):
        user_question = st.text_input("❓ Enter your question:")
        ask_button = st.form_submit_button("Ask")

    if ask_button and user_question.strip():
        st.session_state.answer = ""
        st.session_state.sources = ""
        st.session_state.retrieved_chunks = []

        with st.spinner("🧠 Thinking..."):
            answer, source_info, retrieved_chunks = rag_ask_question(
                user_question,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.tokenizer,
                st.session_state.model,
                top_k=top_k_value,
            )
            st.session_state.answer = answer
            st.session_state.sources = source_info
            st.session_state.retrieved_chunks = retrieved_chunks

    if st.session_state.get("answer"):
        st.markdown("**🧠 Answer:**")
        st.markdown(st.session_state.answer)

        copy_code = f"""
            <textarea id="copy-target" style="position: absolute; left: -9999px;">{st.session_state.answer}</textarea>
            <button onclick="navigator.clipboard.writeText(document.getElementById('copy-target').value);"
                style="
                    margin-top: 10px;
                    padding: 0.4em 1em;
                    font-size: 0.9em;
                    border-radius: 6px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    cursor: pointer;
                ">
                📋 Copy Answer
            </button>
        """
        st.components.v1.html(copy_code, height=50)

        info_not_found = (
            st.session_state.answer.strip()
            .lower()
            .startswith("the information is not available")
        )

        if not info_not_found:
            st.markdown("---")
            st.markdown(f"**📚 Sources:**\n{st.session_state.sources}")

            st.markdown("### Retrieved Chunks")
            for i, chunk in enumerate(st.session_state.retrieved_chunks, start=1):
                meta = chunk.get("metadata", {})
                file_name = meta.get("file", "Unknown")
                page_number = meta.get("page", "N/A")
                chunk_text = chunk.get("text", "No content")
                with st.expander(f"📄 Chunk {i} — {file_name} (Page {page_number})"):
                    st.markdown(f"```text\n{chunk_text}\n```")
else:
    st.info("Upload one or more medical PDFs to begin.")
