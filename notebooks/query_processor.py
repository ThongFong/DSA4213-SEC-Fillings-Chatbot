
# Lightweight export of your QueryProcessor so other notebooks can import it
import re, unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# ---- toggles (you can flip them later from base_RAG if needed)
USE_SBERT   = True
USE_FLAN_T5 = True

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[&$%.\-]+")
def simple_tokenize(text: str) -> List[str]:
    toks = TOKEN_RE.findall(text)
    cleaned = []
    for t in toks:
        if t.lower() == "'s": continue
        if t.endswith("'s"): t = t[:-2]
        cleaned.append(t)
    return cleaned

DOMAIN_SYNONYMS = {
    "risk": ["risk factor","risk factors","uncertainty","exposure","threat"],
    "cyber": ["cybersecurity","information security","infosec","data breach","security incident"],
    "performance": ["revenue","growth","margin","profit","loss","guidance","results"],
    "strategy": ["roadmap","plan","initiative","expansion","capex","restructuring","acquisition"],
    "md&a": ["management discussion","md&a","results of operations"],
}

def keyword_expand(tokens: List[str]) -> List[str]:
    ex = []
    for t in tokens:
        t0 = t.strip(".-").lower()
        ex.extend(DOMAIN_SYNONYMS.get(t0, []))
    seen, out = set(), []
    for w in ex:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

def build_keywords(tokens: List[str], expansions: List[str]) -> List[str]:
    kept = []
    for t in tokens + expansions:
        t = t.lower()
        if not re.search(r"[a-z0-9]", t): 
            continue
        if t not in kept:
            kept.append(t)
    return kept

COMPANY_TICKERS = {"tesla":"TSLA","apple":"AAPL","microsoft":"MSFT","nvidia":"NVDA"}

# --- entities (quarter/year/company/ticker) ---
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

def extract_entities(raw_text: str) -> dict:
    out = {}
    low = raw_text.lower()
    q = re.findall(r"\b(q[1-4])\s*([12][0-9]{3})\b", low)
    if q: out["quarter"] = [f"{p.upper()} {y}" for p, y in q]
    years = re.findall(r"\b(20[0-4][0-9]|19[0-9]{2})\b", raw_text)
    if years: out["year"] = sorted(set(years))
    companies = set()
    if _nlp is not None:
        doc = _nlp(raw_text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.add(ent.text.strip())
    low_raw = raw_text.lower()
    for name in COMPANY_TICKERS:
        if name in low_raw: companies.add(name.title())
    if companies: out["company"] = sorted(companies)
    tickers = set(COMPANY_TICKERS.get(c.lower(),"") for c in companies if COMPANY_TICKERS.get(c.lower()))
    tickers.update(re.findall(r"\$([A-Z]{1,5})\b", raw_text))
    tickers.update(re.findall(r"\(([A-Z]{1,5})\)", raw_text))
    tickers.update(re.findall(r"\b(?:NASDAQ|NYSE)\s*:\s*([A-Z]{1,5})\b", raw_text))
    tickers = {t for t in tickers if t}
    if tickers: out["ticker"] = sorted(tickers)
    return out

# --- SBERT embedding (matches base_RAG Config) ---
try:
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") if USE_SBERT else None
except Exception:
    _sbert = None

def sbert_embed(text: str) -> Optional[List[float]]:
    if _sbert is None: return None
    v = _sbert.encode([text], normalize_embeddings=True)[0]
    return v.tolist()

# --- Flan-T5 paraphrasing (optional) ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _flan_device = "cuda" if USE_FLAN_T5 and torch.cuda.is_available() else "cpu"
    _flan_tok = AutoTokenizer.from_pretrained("google/flan-t5-small") if USE_FLAN_T5 else None
    _flan_mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(_flan_device).eval() if USE_FLAN_T5 else None
except Exception:
    _flan_tok = _flan_mdl = None
    _flan_device = "cpu"

def t5_paraphrases_safe(q: str, num_return: int = 5, max_new_tokens: int = 48) -> List[str]:
    if not (USE_FLAN_T5 and _flan_tok is not None and _flan_mdl is not None): return []
    import torch, re
    prompt = ("Rewrite the query into multiple short paraphrases without adding facts or numbers. "
              "Keep meaning; avoid speculation or meta text.\nQuery: " + q)
    x = _flan_tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    x = {k: v.to(_flan_device) for k, v in x.items()}
    with torch.no_grad():
        out = _flan_mdl.generate(
            **x, do_sample=True, top_k=50, top_p=0.92, temperature=0.9,
            num_return_sequences=num_return, max_new_tokens=max_new_tokens,
            repetition_penalty=1.1, no_repeat_ngram_size=3
        )
    paras = _flan_tok.batch_decode(out, skip_special_tokens=True)
    base = re.sub(r"\W+"," ", q).strip().lower()
    seen, kept = set(), []
    for p in paras:
        p2 = normalize(p)
        p2_cmp = re.sub(r"\W+"," ", p2).strip().lower()
        if p2_cmp == base: continue
        if p2 and p2 not in seen:
            seen.add(p2); kept.append(p2)
    return kept[:num_return]

# --- intent (hybrid)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
INTENT_LABELS = ["risk","performance","strategy"]
X_train = [
    "What new risk factors were disclosed?",
    "Cybersecurity breach details for Tesla",
    "Explain Apple revenue growth and margins",
    "Compare Microsoft profit guidance last quarter",
    "Outline Nvidia expansion strategy in data centers",
    "What restructuring plan is management proposing?"
]
y_train = ["risk","risk","performance","performance","strategy","strategy"]
_intent_clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("lr", LogisticRegression(max_iter=300, class_weight="balanced", multi_class="ovr"))
]).fit(X_train, y_train)

RISK_KW = {"risk","risk factor","risk factors","uncertainty","cyber","cybersecurity","breach","litigation","security"}
PERF_KW = {"revenue","growth","margin","profit","loss","guidance","results","compare","last quarter","quarterly"}
STRAT_KW= {"strategy","plan","roadmap","expansion","acquisition","restructuring","capex","data center","data centers"}

def _kw_score(t: str, kws: set[str]) -> int:
    return sum(1 for k in kws if k in t)

def classify_intent(text: str) -> Tuple[str, float]:
    tx = normalize(text)
    proba = _intent_clf.predict_proba([tx])[0].tolist()
    k_r = _kw_score(tx, RISK_KW); k_p = _kw_score(tx, PERF_KW); k_s = _kw_score(tx, STRAT_KW)
    k_sum = max(1, (k_r + k_p + k_s))
    priors = [k_r/k_sum, k_p/k_sum, k_s/k_sum]
    alpha, beta = 0.6, 0.4
    blended = [alpha*proba[i] + beta*priors[i] for i in range(3)]
    s = sum(blended) or 1.0
    blended = [b/s for b in blended]
    idx = max(range(3), key=lambda i: blended[i])
    return INTENT_LABELS[idx], float(blended[idx])

def expand_query(query: str) -> dict:
    norm = normalize(query)
    toks = simple_tokenize(norm)
    lex_ex = keyword_expand(toks)
    paras = t5_paraphrases_safe(norm, num_return=5, max_new_tokens=48) if USE_FLAN_T5 else []
    para_tokens = []
    for p in paras:
        para_tokens.extend(simple_tokenize(p))
    para_tokens = list(dict.fromkeys(para_tokens))
    para_ex = keyword_expand(para_tokens) if para_tokens else []
    expansions = []
    for lst in (lex_ex, para_ex):
        for w in lst:
            if w not in expansions:
                expansions.append(w)
    return {
        "normalized": norm,
        "tokens": toks,
        "expansions": expansions,
        "paraphrases": paras,
        "keywords": build_keywords(toks, expansions)
    }

@dataclass
class QueryProcessorConfig:
    labels: List[str] = field(default_factory=lambda: ["risk","performance","strategy"])

class QueryProcessor:
    def __init__(self, config: QueryProcessorConfig = QueryProcessorConfig()):
        self.config = config
    def process(self, query: str) -> Dict[str, Any]:
        raw = query
        ex  = expand_query(query)
        ents = extract_entities(raw)
        label, conf = classify_intent(ex["normalized"])
        emb = sbert_embed(ex["normalized"]) if USE_SBERT else None
        return {
            "normalized": ex["normalized"],
            "label": label,
            "confidence": conf,
            "expansions": ex["expansions"],
            "paraphrases": ex["paraphrases"],
            "keywords": ex["keywords"],
            "entities": ents,
            "filters": ents.copy(),
            "embedding": emb
        }
