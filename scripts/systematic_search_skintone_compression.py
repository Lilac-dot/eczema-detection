"""
Systematic literature search for the skin-tone compression audit: has anyone measured how
standard model compression (quantization / pruning) changes a classifier's performance
across skin tones or other demographic subgroups -- especially in dermatology?

Protocol (PRISMA-style, recorded so it can be re-run and reported):
  1. Identification -- keyword queries (compression term x fairness term x domain term) against
     the Semantic Scholar Graph API and the arXiv API, plus FORWARD CITATIONS (papers citing)
     of the key known papers (Hooker 2019/2020, Iofinova 2023, Stoychev & Gunes 2022, FairPrune
     2022, FairQuantize 2024, FairQuant 2026).
  2. De-duplication -- by DOI / arXiv id / normalised title.
  3. Automated screening on title + abstract: the record must mention
       (a) a compression term, AND (b) a fairness/subgroup term, AND
       (c) a medical, dermatology or skin-tone term.
     Records passing (a)+(b) but not (c) are kept in a separate "general compression-fairness"
     list, since they are background rather than direct competitors.
  4. Eligibility -- done by hand from the shortlist (see the write-up in docs/).

Writes docs/lit_search_2026-09-24/{records_all.csv, screened_direct.csv,
screened_general.csv, search_log.json}.
"""
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from paths import ROOT

OUT = ROOT / "docs" / "lit_search_2026-09-24"
S2 = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,abstract,venue,externalIds,citationCount,url"
UA = {"User-Agent": "honors-lit-search/1.0 (academic research)"}

COMPRESSION = ["quantization", "quantized", "pruning", "pruned", "model compression", "compressed model",
               "int8", "8-bit", "low-bit", "mixed-precision", "sparsity", "knowledge distillation"]
FAIRNESS = ["fairness", "fair ", "bias", "disparit", "subgroup", "demographic", "skin tone", "skin type",
            "fitzpatrick", "race", "ethnic", "worst-group", "underrepresented", "under-represented", "equit"]
DOMAIN = ["dermatolog", "skin", "lesion", "eczema", "dermatitis", "medical", "clinical", "health",
          "fitzpatrick", "melanoma", "chest x-ray", "radiograph"]

QUERIES = [
    "quantization fairness skin tone dermatology",
    "pruning fairness skin disease diagnosis",
    "model compression fairness dermatology",
    "quantization bias Fitzpatrick skin type",
    "compressed neural network skin lesion fairness",
    "post-training quantization subgroup performance medical imaging",
    "pruning disparate impact medical image classification",
    "model compression demographic bias medical imaging",
    "edge deployment fairness skin cancer classification",
    "on-device skin disease classification skin tone bias",
    "quantization fairness computer vision demographic",
    "pruning bias underrepresented groups neural network",
    "compression identified exemplars fairness",
    "calibration data post-training quantization bias",
    "eczema atopic dermatitis deep learning skin of color",
    "inflammatory skin disease classification darker skin performance",
]
ARXIV_QUERIES = [
    'abs:quantization AND abs:fairness AND abs:skin',
    'abs:pruning AND abs:fairness AND abs:skin',
    'abs:compression AND abs:fairness AND abs:dermatology',
    'abs:quantization AND abs:bias AND abs:medical',
    'abs:pruning AND abs:bias AND abs:medical',
    'abs:quantization AND abs:Fitzpatrick',
    'abs:pruning AND abs:Fitzpatrick',
]
SEED_PAPERS = {  # forward-citation seeds (arXiv ids)
    "Hooker 2019 What do compressed DNNs forget": "arXiv:1911.05248",
    "Hooker 2020 Characterising bias in compressed models": "arXiv:2010.03058",
    "Iofinova 2023 Bias in pruned vision models": "arXiv:2304.12622",
    "Stoychev & Gunes 2022 compression fairness FER": "arXiv:2201.01709",
    "FairPrune 2022": "arXiv:2203.02110",
    "FairQuant 2026": "arXiv:2602.23192",
    "Groh 2021 Fitzpatrick17k": "arXiv:2104.09957",
}


CACHE_PATH = OUT / "api_cache.json"
CACHE = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
FAILED = []  # URLs that never succeeded -- reported in the log, never silently counted as 0


def get_json(url, tries=10):
    """Successful responses are cached, so re-runs only retry what failed."""
    if url in CACHE:
        return CACHE[url]
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                CACHE[url] = json.load(r)
                with open(CACHE_PATH, "w") as f:
                    json.dump(CACHE, f)
                return CACHE[url]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(60, 5 * 2 ** i) if e.code == 429 else 5)  # rate limited: back off
        except Exception:
            time.sleep(5)
    FAILED.append(url)
    return None


def s2_search(q, limit=100):
    url = f"{S2}/paper/search?query={urllib.parse.quote(q)}&limit={limit}&fields={FIELDS}"
    cached = url in CACHE
    d = get_json(url)
    if not cached:
        time.sleep(3)
    return None if d is None else d.get("data", [])


def s2_citations(paper_id, pages=5):
    out = []
    for off in range(0, pages * 100, 100):
        d = get_json(f"{S2}/paper/{paper_id}/citations?fields={FIELDS}&limit=100&offset={off}")
        time.sleep(1.2)
        if not d or not d.get("data"):
            break
        out += [c["citingPaper"] for c in d["data"] if c.get("citingPaper")]
        if len(d["data"]) < 100:
            break
    return out


def arxiv_search(q, n=100):
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q)
           + f"&start=0&max_results={n}")
    for _ in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                root = ET.fromstring(r.read())
            break
        except Exception:
            time.sleep(5)
    else:
        return []
    time.sleep(3)  # arXiv asks for >= 3 s between calls
    ns = {"a": "http://www.w3.org/2005/Atom"}
    recs = []
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/", 1)[-1].split("v")[0]
        recs.append(dict(title=" ".join(e.find("a:title", ns).text.split()),
                         abstract=" ".join(e.find("a:summary", ns).text.split()),
                         year=int(e.find("a:published", ns).text[:4]), venue="arXiv",
                         externalIds={"ArXiv": aid}, citationCount=None,
                         url=f"https://arxiv.org/abs/{aid}"))
    return recs


def key(r):
    ids = r.get("externalIds") or {}
    if ids.get("DOI"):
        return "doi:" + ids["DOI"].lower()
    if ids.get("ArXiv"):
        return "arxiv:" + ids["ArXiv"]
    return "t:" + re.sub(r"[^a-z0-9]", "", (r.get("title") or "").lower())


def has(text, terms):
    return any(t in text for t in terms)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"s2_queries": {}, "arxiv_queries": {}, "citation_seeds": {}}
    records = {}

    def add(recs, origin):
        for r in recs:
            if not r or not r.get("title"):
                continue
            k = key(r)
            if k not in records:
                records[k] = dict(r, origins=[origin])
            else:
                records[k]["origins"].append(origin)

    for q in QUERIES:
        res = s2_search(q)
        log["s2_queries"][q] = "FAILED" if res is None else len(res)
        add(res or [], f"s2:{q}")
        print(f"S2 '{q}': {log['s2_queries'][q]}", flush=True)
    for q in ARXIV_QUERIES:
        res = arxiv_search(q)
        log["arxiv_queries"][q] = len(res)
        add(res, f"arxiv:{q}")
        print(f"arXiv '{q}': {len(res)}", flush=True)
    for name, pid in SEED_PAPERS.items():
        res = s2_citations(pid)
        log["citation_seeds"][name] = len(res)
        add(res, f"cites:{name}")
        print(f"citations of {name}: {len(res)}", flush=True)

    rows = []
    for k, r in records.items():
        text = ((r.get("title") or "") + " " + (r.get("abstract") or "")).lower()
        a, b, c = has(text, COMPRESSION), has(text, FAIRNESS), has(text, DOMAIN)
        ids = r.get("externalIds") or {}
        rows.append(dict(key=k, title=r.get("title"), year=r.get("year"), venue=r.get("venue"),
                         citations=r.get("citationCount"), url=r.get("url"),
                         arxiv=ids.get("ArXiv"), doi=ids.get("DOI"),
                         compression=a, fairness=b, domain=c,
                         n_origins=len(r["origins"]), origins=" | ".join(sorted(set(r["origins"]))),
                         abstract=(r.get("abstract") or "")[:2000]))
    fields = list(rows[0].keys())

    def write(name, subset):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(subset, key=lambda r: -(r["year"] or 0)))

    direct = [r for r in rows if r["compression"] and r["fairness"] and r["domain"]]
    general = [r for r in rows if r["compression"] and r["fairness"] and not r["domain"]]
    write("records_all.csv", rows)
    write("screened_direct.csv", direct)
    write("screened_general.csv", general)
    log["failed_requests"] = FAILED
    log.update(n_identified=sum(len(r["origins"]) for r in records.values()),
               n_unique=len(rows), n_direct=len(direct), n_general=len(general),
               date="2026-09-24")
    with open(OUT / "search_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps({k: v for k, v in log.items() if k.startswith("n_")}, indent=2))


if __name__ == "__main__":
    main()
