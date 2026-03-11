import json
import os
import re
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

DATA_DIR = "/data"
FILES_DIR = os.path.join(DATA_DIR, "files")
DB_DIR = os.path.join(DATA_DIR, "db")
PAGES_JSON = os.path.join(DATA_DIR, "pages.json")


class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ATLAS-DAEDALUS RAG API running"}


# ===============================
# TEXT HELPERS
# ===============================

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\.\-\+\(\)/ ]", " ", text)
    return text.split()


def load_pages_corpus() -> List[Dict[str, Any]]:
    if not os.path.exists(PAGES_JSON):
        return []

    with open(PAGES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pages_corpus(corpus: List[Dict[str, Any]]) -> None:
    with open(PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)


# ===============================
# METADATA HELPERS
# ===============================

def extract_section_hint(text: str) -> str:
    patterns = [
        r"\b\d+\.\d+\.\d+\.\d+\b",
        r"\b\d+\.\d+\.\d+\b",
        r"\b\d+\.\d+\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return ""


def extract_table_hint(text: str) -> str:
    match = re.search(r"\bTable\s+\d+[A-Za-z\-]*\b", text, flags=re.IGNORECASE)
    return match.group(0) if match else ""


def extract_annex_hint(text: str) -> str:
    match = re.search(r"\bAnnex\s+[A-Z]\b", text, flags=re.IGNORECASE)
    return match.group(0) if match else ""


def extract_figure_hint(text: str) -> str:
    match = re.search(r"\bFigure\s+\d+[A-Za-z\-]*\b", text, flags=re.IGNORECASE)
    return match.group(0) if match else ""


# ===============================
# FORMULA HELPERS
# ===============================

def clean_latex_escapes(text: str) -> str:
    replacements = {
        "\\\\[": "\\[",
        "\\\\]": "\\]",
        "\\\\(": "\\(",
        "\\\\)": "\\)",
        "\\\\frac": "\\frac",
        "\\\\times": "\\times",
        "\\\\cdot": "\\cdot",
        "\\\\mu": "\\mu",
        "\\\\alpha": "\\alpha",
        "\\\\beta": "\\beta",
        "\\\\gamma": "\\gamma",
        "\\\\Delta": "\\Delta",
        "\\\\delta": "\\delta",
        "\\\\min": "\\min",
        "\\\\max": "\\max",
        "\\\\left": "\\left",
        "\\\\right": "\\right",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_variable_names(text: str) -> str:
    replacements = {
        "Nmc": "N_mc",
        "Nmr": "N_mr",
        "Avl": "A_vl",
        "Av1": "A_v1",
        "Avi": "A_vi",
        "Nc": "N_c",
        "Nr": "N_r",
    }

    for old, new in replacements.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    return text


def replace_frac_once(text: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    return pattern.sub(r"(\1 / \2)", text)


def latex_to_pretty_text(latex: str) -> str:
    text = clean_latex_escapes(latex)

    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")

    text = normalize_variable_names(text)

    for _ in range(10):
        new_text = replace_frac_once(text)
        if new_text == text:
            break
        text = new_text

    replacements = {
        "\\times": "×",
        "\\cdot": "·",
        "\\mu": "μ",
        "\\alpha": "α",
        "\\beta": "β",
        "\\gamma": "γ",
        "\\Delta": "Δ",
        "\\delta": "δ",
        "\\leq": "≤",
        "\\geq": "≥",
        "\\neq": "≠",
        "\\approx": "≈",
        "\\min": "min",
        "\\max": "max",
        "\\left": "",
        "\\right": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_json_block(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


# ===============================
# QUESTION TYPE
# ===============================

def classify_question_type(query: str) -> str:
    q = query.lower()

    table_keywords = [
        "table",
        "row",
        "category",
        "casting factor",
        "severity level",
        "drain pipe size",
        "dimensions",
        "tolerances",
        "value from the relevant table",
        "quote the table",
    ]

    formula_keywords = [
        "formula",
        "equation",
        "calculate",
        "calculation",
        "correction factor",
        "multiplier",
        "write the equation",
        "allowable nozzle forces and moments",
    ]

    procedure_keywords = [
        "procedure",
        "describe the procedure",
        "verification",
        "verify",
        "worksheet",
        "steps",
        "how do you verify",
    ]

    if any(k in q for k in table_keywords):
        return "table"

    if any(k in q for k in formula_keywords):
        return "formula"

    if any(k in q for k in procedure_keywords):
        return "procedure"

    return "fact"


# ===============================
# RETRIEVAL HELPERS
# ===============================

def doc_key(hit: Dict[str, Any]) -> Tuple[str, Any, str]:
    return (
        hit.get("source", ""),
        hit.get("page", ""),
        hit.get("content", "")[:500]
    )


def unique_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []

    for hit in hits:
        key = doc_key(hit)
        if key not in seen:
            seen.add(key)
            result.append(hit)

    return result


def build_context(hits: List[Dict[str, Any]], max_docs: int = 10) -> str:
    selected = hits[:max_docs]
    parts = []

    for i, hit in enumerate(selected, start=1):
        section_hint = hit.get("section_hint", "")
        table_hint = hit.get("table_hint", "")
        annex_hint = hit.get("annex_hint", "")
        figure_hint = hit.get("figure_hint", "")

        meta_parts = [
            f"page={hit['page']}",
            f"source={hit['source']}",
            f"chunk_id={hit.get('chunk_id', 'unknown')}"
        ]

        if section_hint:
            meta_parts.append(f"section_hint={section_hint}")
        if table_hint:
            meta_parts.append(f"table_hint={table_hint}")
        if annex_hint:
            meta_parts.append(f"annex_hint={annex_hint}")
        if figure_hint:
            meta_parts.append(f"figure_hint={figure_hint}")

        parts.append(
            f"[Document {i} | {' | '.join(meta_parts)}]\n{hit['content']}"
        )

    return "\n\n".join(parts)


def bm25_search(query: str, corpus: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    if not corpus:
        return []

    tokenized_corpus = [tokenize_for_bm25(item["content"]) for item in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [corpus[idx] for idx in ranked_indices]


def vector_search(db: Chroma, query: str, k: int = 10, fetch_k: int = 40) -> List[Dict[str, Any]]:
    docs = db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    results = []
    for doc in docs:
        results.append(
            {
                "page": doc.metadata.get("page", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "section_hint": doc.metadata.get("section_hint", ""),
                "table_hint": doc.metadata.get("table_hint", ""),
                "annex_hint": doc.metadata.get("annex_hint", ""),
                "figure_hint": doc.metadata.get("figure_hint", ""),
                "content": doc.page_content
            }
        )

    return results


def rrf_fuse(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    score_map: Dict[Tuple[str, Any, str], float] = {}
    obj_map: Dict[Tuple[str, Any, str], Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            key = doc_key(hit)
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank)
            obj_map[key] = hit

    ranked_keys = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
    return [obj_map[key] for key in ranked_keys]


# ===============================
# QUERY EXPANSION
# ===============================

def heuristic_expansions(query: str) -> List[str]:
    q = query.lower()
    expansions = [query]

    if "external thrust" in q and "gear coupling" in q:
        expansions.extend([
            "API 617 external thrust force gear coupling formula",
            "API 617 gear coupling external force equation",
            "API 617 annex external forces and moments gear couplings",
            "external thrust force allowable gear coupling API 617",
            "formula external thrust gear couplings rated power speed shaft diameter",
        ])

    if "shaft vibration" in q and "mechanical running test" in q:
        expansions.extend([
            "API 617 mechanical running test shaft vibration limit",
            "API 617 mechanical test vibration limit A_vl",
            "API 617 A_vl equation",
            "API 617 equation 8 A_vl",
            "API 617 maximum allowable shaft vibration equation",
            "API 617 25.4 um 1.0 mil shaft vibration",
            "API 617 N_mc A_vl",
        ])

    if "table" in q or "category" in q or "casting factor" in q or "severity level" in q:
        expansions.extend([
            query,
            "API 617 relevant table exact row value",
            "API 617 table value condition",
            "API 617 exact table lookup",
        ])

    if "formula" in q or "equation" in q or "write the equation" in q:
        expansions.extend([
            query,
            "API 617 equation formula exact expression",
            "API 617 explicit formula",
            "API 617 correction factor formula",
            "API 617 annex formula",
        ])

    return list(dict.fromkeys(expansions))


def llm_expand_query(query: str, llm: ChatOpenAI) -> List[str]:
    prompt = f"""
Rewrite the engineering question into 5 alternative search queries
that could appear in a technical standard.

Rules:
- Keep the meaning unchanged.
- Prefer standard wording.
- Include likely keyword-heavy phrasing.
- Return only the queries, one per line.
- Do not number them.

Question:
{query}
"""
    try:
        response = llm.invoke(prompt)
        alternatives = [
            line.strip()
            for line in response.content.split("\n")
            if line.strip()
        ]
        return alternatives[:5]
    except Exception:
        return []


def build_query_set(query: str, llm: ChatOpenAI) -> List[str]:
    queries = []
    queries.extend(heuristic_expansions(query))
    queries.extend(llm_expand_query(query, llm))
    return list(dict.fromkeys([q.strip() for q in queries if q.strip()]))


# ===============================
# EXTRACTION
# ===============================

def extraction_prompt_for_mode(question: str, context: str, mode: str) -> str:
    if mode == "table":
        return f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context.
Do not invent values, categories, units, conditions, sections, tables, annexes, figures, or formulas.

The user is asking a TABLE-STYLE question.
Your job is to extract the exact value or category from the context if it is explicitly present.

Return ONLY valid JSON with this schema:

{{
  "answer": "short engineering answer in plain English",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "short description of the matched row or matched condition",
  "formula_plain": "",
  "formula_latex": "",
  "variables": [],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": true
}}

Rules:
- If the exact value/category is present, extract it directly.
- If the context only says 'see Table X' but does not include the actual row value, return found=false.
- Do NOT answer with just 'specified in Table X' unless the actual value is explicitly visible in the context.
- If not found, return exactly:

{{
  "answer": "Not found in the provided standard excerpt.",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "",
  "formula_plain": "",
  "formula_latex": "",
  "variables": [],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": false
}}

Question:
{question}

Context:
{context}
"""

    if mode == "formula":
        return f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context.
Do not invent formulas, values, units, limits, section numbers, table numbers, annexes, or correction factors.

The user is specifically asking for an equation or formula.

Return ONLY valid JSON with this schema:

{{
  "answer": "short engineering answer in plain English",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "",
  "formula_plain": "formula in readable plain text",
  "formula_latex": "formula in proper LaTeX if available",
  "variables": [
    {{
      "symbol": "",
      "meaning": "",
      "units": ""
    }}
  ],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": true
}}

Rules:
- If the context contains only a prose requirement but no explicit formula, return found=false.
- If a correction factor is explicitly present, include it in answer or formula_plain.
- Do NOT reconstruct a formula unless it is explicitly supported by context.
- If not found, return exactly:

{{
  "answer": "Not found in the provided standard excerpt.",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "",
  "formula_plain": "",
  "formula_latex": "",
  "variables": [],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": false
}}

Question:
{question}

Context:
{context}
"""

    if mode == "procedure":
        return f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context.
Do not invent steps, worksheets, section numbers, annexes, or requirements.

The user is asking for a procedure.

Return ONLY valid JSON with this schema:

{{
  "answer": "clear procedural summary in plain English",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "",
  "formula_plain": "",
  "formula_latex": "",
  "variables": [],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": true
}}

Rules:
- Summarize only steps explicitly supported by context.
- If worksheets or multipliers are explicitly mentioned, include them in answer.
- If not found, return exactly:

{{
  "answer": "Not found in the provided standard excerpt.",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "",
  "formula_plain": "",
  "formula_latex": "",
  "variables": [],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": false
}}

Question:
{question}

Context:
{context}
"""

    return f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context.
Do not invent values, units, conditions, section numbers, table numbers, annexes, or figures.

Return ONLY valid JSON with this schema:

{{
  "answer": "short engineering answer in plain English",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "",
  "formula_plain": "",
  "formula_latex": "",
  "variables": [],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": true
}}

Rules:
- Answer only what is explicitly supported by context.
- Do NOT substitute a nearby but different requirement.
- If not found, return exactly:

{{
  "answer": "Not found in the provided standard excerpt.",
  "value": "",
  "units": "",
  "condition": "",
  "table_row": "",
  "formula_plain": "",
  "formula_latex": "",
  "variables": [],
  "reference": {{
    "section": "",
    "table": "",
    "annex": "",
    "figure": "",
    "page": ""
  }},
  "found": false
}}

Question:
{question}

Context:
{context}
"""


def answer_from_context(question: str, hits: List[Dict[str, Any]], llm: ChatOpenAI) -> Dict[str, Any]:
    context = build_context(hits, max_docs=10)
    mode = classify_question_type(question)
    prompt = extraction_prompt_for_mode(question, context, mode)

    try:
        response = llm.invoke(prompt)
        parsed = extract_json_block(response.content)

        if not parsed:
            return {
                "answer": "Not found in the provided standard excerpt.",
                "value": "",
                "units": "",
                "condition": "",
                "table_row": "",
                "formula_plain": "",
                "formula_latex": "",
                "variables": [],
                "reference": {"section": "", "table": "", "annex": "", "figure": "", "page": ""},
                "found": False,
                "mode": mode
            }

        parsed["mode"] = mode
        return parsed

    except Exception:
        return {
            "answer": "Not found in the provided standard excerpt.",
            "value": "",
            "units": "",
            "condition": "",
            "table_row": "",
            "formula_plain": "",
            "formula_latex": "",
            "variables": [],
            "reference": {"section": "", "table": "", "annex": "", "figure": "", "page": ""},
            "found": False,
            "mode": mode
        }


def second_pass_queries(question: str, llm: ChatOpenAI) -> List[str]:
    mode = classify_question_type(question)

    prompt = f"""
Generate up to 6 very short keyword-heavy retrieval queries
for finding the exact answer in a technical standard.

Question mode: {mode}

Rules:
- Focus on exact sections, exact tables, exact annexes, exact row conditions, formulas, and variables.
- Return only the queries, one per line.
- Do not explain anything.

Question:
{question}
"""
    try:
        response = llm.invoke(prompt)
        queries = [line.strip() for line in response.content.split("\n") if line.strip()]
        return queries[:6]
    except Exception:
        return []


# ===============================
# ASK
# ===============================

@app.post("/ask")
def ask(question: Question):
    if not os.path.exists(DB_DIR):
        return {"error": "Vector database not initialized yet. Upload a PDF first."}

    corpus = load_pages_corpus()
    if not corpus:
        return {"error": "Page corpus not initialized yet. Upload a PDF first."}

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    mode = classify_question_type(question.query)
    query_set = build_query_set(question.query, llm)

    vector_result_lists: List[List[Dict[str, Any]]] = []
    bm25_result_lists: List[List[Dict[str, Any]]] = []

    for q in query_set:
        try:
            vector_result_lists.append(vector_search(db, q, k=10, fetch_k=40))
        except Exception:
            pass

        try:
            bm25_result_lists.append(bm25_search(q, corpus, k=10))
        except Exception:
            pass

    fused_hits = rrf_fuse(vector_result_lists + bm25_result_lists)
    fused_hits = unique_hits(fused_hits)

    if not fused_hits:
        return {
            "answer": "No documents retrieved from database.",
            "value": "",
            "units": "",
            "condition": "",
            "table_row": "",
            "formula_plain": "",
            "formula_latex": "",
            "formula_pretty_from_latex": "",
            "variables": [],
            "reference": {"section": "", "table": "", "annex": "", "figure": "", "page": ""},
            "mode": mode,
            "sources": [],
            "retrieved_chunks_preview": [],
            "queries_used": query_set
        }

    answer_obj = answer_from_context(question.query, fused_hits, llm)

    if not answer_obj.get("found", False):
        extra_queries = second_pass_queries(question.query, llm)
        extra_lists: List[List[Dict[str, Any]]] = []

        for q in extra_queries:
            try:
                extra_lists.append(vector_search(db, q, k=10, fetch_k=40))
            except Exception:
                pass

            try:
                extra_lists.append(bm25_search(q, corpus, k=10))
            except Exception:
                pass

        if extra_lists:
            second_pass_hits = rrf_fuse([fused_hits] + extra_lists)
            second_pass_hits = unique_hits(second_pass_hits)
            second_answer_obj = answer_from_context(question.query, second_pass_hits, llm)

            if second_answer_obj.get("found", False):
                fused_hits = second_pass_hits
                answer_obj = second_answer_obj
                query_set = list(dict.fromkeys(query_set + extra_queries))

    formula_latex = clean_latex_escapes(answer_obj.get("formula_latex", ""))
    formula_pretty = latex_to_pretty_text(formula_latex) if formula_latex else ""

    return {
        "answer": answer_obj.get("answer", "Not found in the provided standard excerpt."),
        "value": answer_obj.get("value", ""),
        "units": answer_obj.get("units", ""),
        "condition": answer_obj.get("condition", ""),
        "table_row": answer_obj.get("table_row", ""),
        "formula_plain": answer_obj.get("formula_plain", ""),
        "formula_latex": formula_latex,
        "formula_pretty_from_latex": formula_pretty,
        "variables": answer_obj.get("variables", []),
        "reference": answer_obj.get(
            "reference",
            {"section": "", "table": "", "annex": "", "figure": "", "page": ""}
        ),
        "mode": answer_obj.get("mode", mode),
        "sources": [
            {
                "page": hit["page"],
                "source": hit["source"],
                "chunk_id": hit.get("chunk_id", ""),
                "section_hint": hit.get("section_hint", ""),
                "table_hint": hit.get("table_hint", ""),
                "annex_hint": hit.get("annex_hint", ""),
                "figure_hint": hit.get("figure_hint", ""),
            }
            for hit in fused_hits[:10]
        ],
        "retrieved_chunks_preview": [
            hit["content"][:900] for hit in fused_hits[:10]
        ],
        "queries_used": query_set
    }


# ===============================
# UPLOAD
# ===============================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    file_path = os.path.join(FILES_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    if os.path.getsize(file_path) == 0:
        return {"error": "Uploaded file is empty"}

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        return {"error": "PDF parsing failed or document is empty"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=[
            "\nAnnex ",
            "\nANNEX ",
            "\nTable ",
            "\nTABLE ",
            "\nFigure ",
            "\nFIGURE ",
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    chunk_documents: List[Document] = []
    corpus_items: List[Dict[str, Any]] = []
    chunk_counter = 0

    for doc in documents:
        cleaned_page_text = normalize_text(doc.page_content)
        if not cleaned_page_text:
            continue

        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", file_path)

        page_doc = Document(
            page_content=cleaned_page_text,
            metadata={
                "source": source,
                "page": page
            }
        )

        splits = splitter.split_documents([page_doc])

        for split_doc in splits:
            chunk_counter += 1
            chunk_text = normalize_text(split_doc.page_content)

            if not chunk_text:
                continue

            section_hint = extract_section_hint(chunk_text)
            table_hint = extract_table_hint(chunk_text)
            annex_hint = extract_annex_hint(chunk_text)
            figure_hint = extract_figure_hint(chunk_text)

            metadata = {
                "source": source,
                "page": page,
                "chunk_id": chunk_counter,
                "section_hint": section_hint,
                "table_hint": table_hint,
                "annex_hint": annex_hint,
                "figure_hint": figure_hint,
            }

            split_doc.page_content = chunk_text
            split_doc.metadata = metadata
            chunk_documents.append(split_doc)

            corpus_items.append(
                {
                    "page": page,
                    "source": source,
                    "chunk_id": chunk_counter,
                    "section_hint": section_hint,
                    "table_hint": table_hint,
                    "annex_hint": annex_hint,
                    "figure_hint": figure_hint,
                    "content": chunk_text
                }
            )

    if not chunk_documents:
        return {"error": "No valid text extracted from PDF"}

    save_pages_corpus(corpus_items)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    Chroma.from_documents(
        chunk_documents,
        embeddings,
        persist_directory=DB_DIR
    )

    return {
        "status": "PDF uploaded and indexed successfully",
        "chunks_indexed": len(chunk_documents),
        "file": file.filename
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)