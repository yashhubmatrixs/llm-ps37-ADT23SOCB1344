import json
import re
import uuid

import chromadb
import pandas as pd
import requests
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"
TOP_K = 4  # chunks retrieved for each RAG question

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def call_llm(prompt, json_mode=False):
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


def read_pages(pdf_file):
    # returns [(page_number, text), ...]
    reader = PdfReader(pdf_file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


# ---------- Part 1: topic tagging + counting ----------

def tag_questions(page_text, subject):
    prompt = f"""You are analysing a university question paper for the subject: {subject}.
From the text below, list every exam question and give each ONE short topic name
(2-4 words, e.g. "Normalization", "Binary Trees", "Deadlock").
Use the same topic name for the same concept every time.
Reply ONLY with JSON in this format:
{{"questions": [{{"question": "...", "topic": "..."}}]}}
If there are no questions in the text, reply {{"questions": []}}.

TEXT:
{page_text}
"""
    raw = call_llm(prompt, json_mode=True)
    try:
        return json.loads(raw).get("questions", [])
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()).get("questions", [])
            except json.JSONDecodeError:
                pass
    return []


def clean_topic(t):
    return re.sub(r"\s+", " ", str(t)).strip().title()


# ---------- Part 2: RAG (chunks -> embeddings in ChromaDB -> retrieve -> LLM) ----------

def build_index(pages_by_paper):
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"papers_{uuid.uuid4().hex[:8]}")  # default embedder = MiniLM
    ids, docs, metas = [], [], []
    for paper, pages in pages_by_paper.items():
        for page_no, text in pages:
            for k, chunk in enumerate(splitter.split_text(text)):
                ids.append(f"{paper}-{page_no}-{k}")
                docs.append(chunk)
                metas.append({"paper": paper, "page": page_no})
    if docs:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def rag_answer(col, question):
    res = col.query(query_texts=[question], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(f"[{m['paper']}, page {m['page']}]\n{d}" for d, m in zip(docs, metas))
    prompt = f"""Answer the student's question using ONLY the question paper context below.
If the answer is not in the context, say "Not found in the uploaded papers."
Mention the paper name (year) when you refer to a question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return call_llm(prompt), metas


# ---------- UI ----------

st.set_page_config(page_title="Question Paper Analyzer", layout="wide")
st.title("Previous Year Question Paper Analyzer")
st.write("Upload past papers, see which topics repeat, and ask questions about the papers (RAG).")

subject = st.text_input("Subject name", placeholder="e.g. Database Management Systems")
files = st.file_uploader("Upload question paper PDFs", type="pdf", accept_multiple_files=True)
top_n = st.slider("Show top N topics", 5, 20, 10)

if st.button("Analyze") and files:
    rows, pages_by_paper = [], {}
    progress = st.progress(0.0, text="Starting...")
    for i, f in enumerate(files):
        pages = read_pages(f)
        if not any(t.strip() for _, t in pages):
            st.warning(f"{f.name}: no text found (scanned PDF?). Skipping.")
            continue
        pages_by_paper[f.name] = pages
        for j, (page_no, text) in enumerate(pages):
            if not text.strip():
                continue
            progress.progress((i + j / len(pages)) / len(files), text=f"Tagging {f.name} - page {page_no}")
            try:
                for q in tag_questions(text, subject or "General"):
                    rows.append(
                        {
                            "paper": f.name,
                            "page": page_no,
                            "question": q.get("question", ""),
                            "topic": clean_topic(q.get("topic", "Unknown")),
                        }
                    )
            except requests.RequestException as e:
                st.error(f"Could not reach Ollama. Is it running? ({e})")
                st.stop()
    progress.text("Building search index...")
    st.session_state["df"] = pd.DataFrame(rows).drop_duplicates(subset=["paper", "question"]) if rows else pd.DataFrame(rows)
    st.session_state["col"] = build_index(pages_by_paper)
    progress.empty()

df = st.session_state.get("df")
col = st.session_state.get("col")

if df is not None:
    tab1, tab2 = st.tabs(["Frequent topics", "Ask the papers (RAG)"])

    with tab1:
        if df.empty:
            st.info("No questions found. Try clearer PDFs or a different subject name.")
        else:
            counts = df["topic"].value_counts().head(top_n)
            st.subheader("Most frequently asked topics")
            st.bar_chart(counts)
            st.dataframe(counts.rename("times asked").reset_index(), use_container_width=True)

            pick = st.selectbox("See questions for a topic", counts.index)
            st.dataframe(df[df["topic"] == pick][["paper", "page", "question"]], use_container_width=True)
            st.download_button("Download tagged questions (CSV)", df.to_csv(index=False), "tagged_questions.csv")

    with tab2:
        st.caption("Chunks are embedded in ChromaDB, the top matches are retrieved and Llama 3 answers only from them.")
        q = st.text_input("Ask something", placeholder="Which deadlock questions were asked and in which year?")
        if st.button("Ask") and q and col is not None:
            try:
                ans, metas = rag_answer(col, q)
                st.write(ans)
                st.caption("Sources: " + ", ".join(sorted({f"{m['paper']} (p.{m['page']})" for m in metas})))
            except requests.RequestException as e:
                st.error(f"Could not reach Ollama. Is it running? ({e})")
