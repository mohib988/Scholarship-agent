"""
Scholarship Agent: a LangGraph pipeline where every stage is an LLM agent with live web search.

  cv_intake <-> ask_user
     -> source_agent      1. finds the best OFFICIAL sites (government + collaborative, home -> target)
     -> active_agent         reads those sites, lists scholarships that are open now
     -> field_agent       2. searches scholarships for the user's field / scheme / degree, merges the lists
     -> apply_agent          how to apply, preferred apply date, study plan (course, IELTS, watch-outs, extras)
     -> professor_agent      PhD (or on request): fields to target, supervisors, how to approach them
     -> visa_agent        4. visa rules, timing, extra documents for home -> target
     -> planner_agent     3. when to start, dated tasks, weekly routine

Every agent's output is passed through a fixed schema (shape()): the UI always gets the same keys,
and anything extra the model returns is kept under "extra" instead of being thrown away.

Needs:  pip install openai langgraph streamlit pypdf python-docx python-dotenv pandas
Env:    OPENAI_API_KEY  (and optionally OPENAI_MODEL, default gpt-4o; gpt-4o-mini is cheaper but weaker)
Run:    python scholarship_agent.py --cv my_cv.pdf      or      streamlit run app.py
"""
import os, sys, io, json, re, argparse, operator
from datetime import date, timedelta
from typing import TypedDict, Annotated

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
import dotenv

dotenv.load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SEARCH_TOOL = os.getenv("OPENAI_SEARCH_TOOL", "web_search_preview")  # or "web_search" on newer accounts
_client = None

DOC_OPTIONS = ["cv", "transcripts", "degree_certificate", "passport", "ielts", "recommendation_letters",
               "sop", "research_proposal", "photos"]


# =============================================================== AGENT RUNTIME
def client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def run_agent_llm(role, task, search=True):
    """One agent call: role prompt + task (dict/str) -> parsed JSON dict (+ citation urls).
    With search=True the model can browse the web by itself (built-in OpenAI web search)."""
    instructions = (f"{role}\n\nToday's date is {date.today().isoformat()}. Rely only on what you read in sources. "
                    "Use null / [] when something is not stated. Never invent URLs. Prefer official government, "
                    "agency and university pages. Reply with ONE JSON object only, no markdown, no commentary.")
    kwargs = dict(model=MODEL, instructions=instructions,
                  input=task if isinstance(task, str) else json.dumps(task, default=str))
    if search:
        kwargs["tools"] = [{"type": SEARCH_TOOL}]
    r = client().responses.create(**kwargs)
    cites = []
    for item in r.output:
        if getattr(item, "type", "") == "message":
            for part in item.content:
                for a in getattr(part, "annotations", None) or []:
                    if getattr(a, "type", "") == "url_citation" and a.url not in cites:
                        cites.append(a.url)
    try:
        data = parse_json(r.output_text)
    except Exception:  # one repair attempt without search
        fix = client().responses.create(model=MODEL, input="Convert to valid JSON only, keep all content:\n" + r.output_text)
        data = parse_json(fix.output_text)
    return (data if isinstance(data, dict) else {}), cites


# =============================================================== HELPERS
def norm_degree(d):
    d = (d or "").strip().lower()
    if d.startswith("ph") or "doctor" in d:
        return "PhD"
    if d.startswith("m"):
        return "MS"
    return None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_date(v):
    try:
        return date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def as_list(v):
    if not v:
        return []
    if isinstance(v, (str, dict)):
        return [v]
    return list(v)


def has_doc(required, held):
    r = str(required).lower()
    return any(h.replace("_", " ") in r or r in h.replace("_", " ") for h in held)


def name_key(v):
    return re.sub(r"\W+", "", str(v or "").lower())


def norm_ielts(v):
    """Collapse whatever the model says into: yes | no | waivable | unknown."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v or "").strip().lower()
    if not s or s in ("null", "none", "unknown"):
        return "unknown"
    if any(w in s for w in ("waiv", "altern", "exempt", "optional", "medium of instruction", "moi", "or toefl")):
        return "waivable"
    if s.startswith(("no", "not", "false")):
        return "no"
    if s.startswith(("yes", "true", "required", "mandatory")) or "required" in s:
        return "yes"
    return "unknown"


def apply_by(deadline, suggested=None, buffer_days=21):
    """Preferred submission date: the model's suggestion if sane, else deadline - 3 weeks; never in the past."""
    today, d, s = date.today(), parse_date(deadline), parse_date(suggested)
    if d:
        if not s or s > d:
            s = d - timedelta(days=buffer_days)
        s = min(max(s, today), d)
    elif s:
        s = max(s, today)
    return str(s) if s else None


def shape(obj, schema):
    """Force a dict into a fixed schema. Missing keys get defaults, list fields become lists,
    and unknown keys are kept under 'extra' so the output stays flexible but predictable."""
    obj = obj if isinstance(obj, dict) else {}
    out = {}
    for k, default in schema.items():
        v = obj.get(k)
        if isinstance(default, dict):
            v = shape(v, default)
        elif isinstance(default, list):
            v = as_list(v)
        elif v in (None, ""):
            v = default
        out[k] = v
    extra = {k: v for k, v in obj.items() if k not in schema and v not in (None, "", [], {})}
    if extra:
        out["extra"] = extra
    return out


def read_cv(path):
    with open(path, "rb") as f:
        return read_cv_bytes(path, f.read())


def read_cv_bytes(filename, data):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def brief(p):
    """Profile fields the agents need."""
    keys = ("degree", "field", "interests", "gpa_4", "ielts", "nationality", "target_country",
            "publications", "experience_years", "docs_held", "scheme", "intake_preference")
    return {k: p.get(k) for k in keys if p.get(k) not in (None, "", [])}


# =============================================================== OUTPUT SCHEMAS
SCHOLARSHIP_SCHEMA = {"name": None, "provider": None, "kind": "other", "degree_levels": [], "fields": [],
                      "funding": "unknown", "min_gpa_4": None, "min_ielts": None, "ielts_required": "unknown",
                      "deadline": None, "intake": None, "required_documents": [], "url": None,
                      "description": "", "relevance": 0}

STUDY_PLAN_SCHEMA = {"programme": None, "universities": [], "duration": None, "language_requirement": None,
                     "ielts_required": "unknown", "watch_out": [], "extras": []}

GUIDE_SCHEMA = {"name": None, "apply_url": None, "deadline": None, "preferred_apply_date": None, "steps": [],
                "required_documents": [], "missing_documents": [], "gaps": [], "tips": [],
                "study_plan": STUDY_PLAN_SCHEMA}

PROF_SCHEMA = {"name": None, "university": None, "department": None, "url": None, "contact": None,
               "research_focus": None, "recent_papers": [], "funding_note": None,
               "how_to_approach": None, "suggestion": None}

VISA_SCHEMA = {"visa_type": None, "official_url": None, "when_to_apply": None, "processing_time": None,
               "required_documents": [], "extra_documents": [], "fees": None, "notes": []}

SCHOLARSHIP_JSON = ('{"scholarships":[{"name","provider","kind":"government"|"collaborative"|"university"|"other",'
                    '"degree_levels" (list of "MS"/"PhD"),"fields" (list),"funding" ("full"|"partial"|"unknown"),'
                    '"min_gpa_4","min_ielts","ielts_required" ("yes"|"no"|"waivable"|"unknown"),'
                    '"deadline" ("YYYY-MM-DD"|null),"intake","required_documents" (list),"url",'
                    '"description" (max 40 words),"relevance" (0-10 for this applicant)}]}')


# =============================================================== STATE
REQUIRED = ["degree", "field", "gpa_4", "nationality", "target_country"]
PROMPTS = {"degree": "Degree (MS or PhD)?", "field": "Field of study?", "gpa_4": "GPA on a 4.0 scale?",
           "nationality": "Your country (where you apply from)?", "target_country": "Country to study in?"}


class State(TypedDict, total=False):
    cv_text: str
    interactive: bool
    profile: dict
    missing: list
    sources: list
    active: list
    found: list
    ranked: list
    guides: list
    prof_fields: list
    professors: list
    visa: dict
    plan: list
    start_by: str
    routine: dict
    notes: Annotated[list, operator.add]
    citations: Annotated[list, operator.add]


# =============================================================== AGENT 0: CV -> profile
CV_ROLE = ('You extract applicant data from a CV. Return JSON with keys: name, email, degree ("MS" or "PhD" they '
           "want to apply for, null if unknown), field, interests (list of research keywords), gpa_4 (number on a "
           "4.0 scale, convert if needed), ielts (number), nationality, target_country, publications (integer), "
           f"experience_years (number), docs_held (list from: {', '.join(DOC_OPTIONS)}).")


def parse_cv_profile(cv_text):
    """Used by the UI to pre-fill the form from a CV."""
    data, _ = run_agent_llm(CV_ROLE, cv_text[:12000], search=False)
    return {k: v for k, v in data.items() if v not in (None, "", [])}


def cv_intake(s):
    profile = dict(s.get("profile") or {})
    if s.get("cv_text") and not profile.get("_parsed"):
        print("[cv_intake] reading CV ...")
        for k, v in parse_cv_profile(s["cv_text"]).items():  # form values win over CV guesses
            if not profile.get(k):
                profile[k] = v
        profile["_parsed"] = True
    return {"profile": profile, "missing": [f for f in REQUIRED if not profile.get(f)]}


def ask_user(s):
    p = dict(s["profile"])
    p["_asks"] = p.get("_asks", 0) + 1
    for f in s["missing"]:
        v = input(f"  {PROMPTS[f]} ").strip()
        p[f] = num(v) if f == "gpa_4" else v
    return {"profile": p}


# =============================================================== AGENT 1: where to look
SOURCE_ROLE = (
    "You are a scholarship-sources research agent. Use web search to find the best OFFICIAL websites where a "
    "student from the home country can find funded graduate scholarships in the target country. Cover: "
    "(a) government scholarships of the target country open to international students, "
    "(b) collaborative / bilateral schemes between the home and target country (for example the home country's "
    "higher-education agency, joint programmes, Erasmus Mundus, Commonwealth, Fulbright, etc. where relevant), "
    "(c) national or university portals if the first two are thin. Skip aggregator spam. "
    'Return {"sources":[{"name","url","kind":"government"|"collaborative"|"university"|"other","why"}]} (max 8).')


def source_agent(s):
    p = s["profile"]
    scheme = (p.get("scheme") or "any")
    task = {"home_country": p.get("nationality"), "target_country": p.get("target_country"),
            "degree": p.get("degree"), "field": p.get("field"), "scheme_wanted": scheme}
    data, cites = run_agent_llm(SOURCE_ROLE, task)
    src = [shape(x, {"name": None, "url": None, "kind": "other", "why": None}) for x in as_list(data.get("sources"))]
    if scheme in ("government", "collaborative", "university"):
        src = [x for x in src if x.get("kind") == scheme] or src
    print(f"[source_agent] {len(src)} sources")
    return {"sources": src, "citations": cites}


# =============================================================== AGENT 2a: active list from those sources
ACTIVE_ROLE = (
    "You are a scholarship-scanning agent. Visit the given official portals (search the web and read their pages) "
    "and list scholarships for the requested degree level that are OPEN NOW or have an upcoming deadline after "
    "today's date. Skip closed ones. Include government and collaborative schemes. For ielts_required say whether "
    "IELTS is compulsory, not needed, or can be replaced (TOEFL/PTE/Duolingo/English-medium certificate). "
    f"Return {SCHOLARSHIP_JSON} (max 12).")


def active_agent(s):
    task = {"applicant": brief(s["profile"]), "portals": s.get("sources", [])}
    data, cites = run_agent_llm(ACTIVE_ROLE, task)
    items = [shape(x, SCHOLARSHIP_SCHEMA) for x in as_list(data.get("scholarships"))]
    print(f"[active_agent] {len(items)} scholarships")
    return {"active": items, "citations": cites}


# =============================================================== AGENT 2b: field + scheme search
FIELD_ROLE = (
    "You are a field-specific scholarship search agent. Use web search to find currently open (deadline after "
    "today) funded scholarships, fellowships and funded programmes in the target country that fit the applicant's "
    "field, research interests and degree level (MS and PhD are equally important; for PhD include funded PhD "
    "positions and doctoral scholarships). Respect the applicant's nationality eligibility. "
    f"Same JSON format: {SCHOLARSHIP_JSON} (max 12).")


def field_agent(s):
    p = s["profile"]
    task = {"applicant": brief(p), "already_found": [x.get("name") for x in s.get("active", [])]}
    data, cites = run_agent_llm(FIELD_ROLE, task)
    items = [shape(x, SCHOLARSHIP_SCHEMA) for x in as_list(data.get("scholarships"))]
    print(f"[field_agent] {len(items)} scholarships")

    # merge, dedupe, then plain-code guards (closed deadline, wrong level, GPA / IELTS too low)
    seen, merged, today = set(), [], date.today()
    deg = norm_degree(p.get("degree"))
    for x in s.get("active", []) + items:
        key = name_key(x.get("name"))
        if not key or key in seen:
            continue
        seen.add(key)
        levels = [norm_degree(d) for d in as_list(x.get("degree_levels"))]
        d = parse_date(x.get("deadline"))
        if d and d < today:
            continue
        if levels and deg and deg not in levels:
            continue
        if num(x.get("min_gpa_4")) and num(p.get("gpa_4")) and num(p["gpa_4"]) < num(x["min_gpa_4"]):
            continue
        if num(x.get("min_ielts")) and num(p.get("ielts")) and num(p["ielts"]) < num(x["min_ielts"]):
            continue
        x["ielts_required"] = norm_ielts(x.get("ielts_required"))
        x["rel"] = num(x.get("relevance")) or 0
        x["rank_score"] = round(x["rel"] + (2 if x.get("funding") == "full" else 0), 1)
        x["apply_by"] = apply_by(x.get("deadline"))
        merged.append(x)
    ranked = sorted(merged, key=lambda x: x["rank_score"], reverse=True)[:10]
    print(f"[field_agent] {len(ranked)} after merge and eligibility filter")
    return {"found": items, "ranked": ranked, "citations": cites}


# =============================================================== AGENT 3: how to apply + study plan
APPLY_ROLE = (
    "You are an application coach. For each scholarship, use web search on its official page to confirm how to "
    "apply, then write a concrete, ordered application guide for THIS applicant: where to apply, a preferred date "
    "to submit (safe margin before the deadline, usually 2-4 weeks; account for nomination or attestation steps), "
    "required documents, which ones the applicant already has (docs_held) and which are missing, requirement gaps "
    "(GPA, IELTS, experience, publications), 2-3 tips, and a short study plan. Return "
    '{"guides":[{"name","apply_url","deadline","preferred_apply_date":"YYYY-MM-DD","steps" (list),'
    '"required_documents" (list),"missing_documents" (list),"gaps" (list),"tips" (list),'
    '"study_plan":{"programme" (course / programme to apply to that fits the applicant),'
    '"universities" (list, eligible or recommended), "duration" (length + intake),'
    '"language_requirement" (exact English rule, e.g. "IELTS 6.5, no band below 6" or "English-medium '
    'certificate accepted"), "ielts_required" ("yes"|"no"|"waivable"|"unknown"),'
    '"watch_out" (list: return/bond clauses, age limits, home-agency nomination, quotas, unfunded fees, etc.),'
    '"extras" (list: stipend amount, application fee, interview/test, anything else that matters)}}]}.')


def apply_agent(s):
    top = [{k: x.get(k) for k in ("name", "url", "deadline", "required_documents", "min_gpa_4", "min_ielts",
                                   "ielts_required")}
           for x in s.get("ranked", [])[:5]]
    if not top:
        return {"guides": []}
    data, cites = run_agent_llm(APPLY_ROLE, {"applicant": brief(s["profile"]), "scholarships": top})
    guides = [shape(g, GUIDE_SCHEMA) for g in as_list(data.get("guides"))]
    held = {d.lower() for d in as_list(s["profile"].get("docs_held"))}
    by_name = {name_key(x.get("name")): x for x in s.get("ranked", [])}
    for g in guides:
        src = by_name.get(name_key(g.get("name")), {})
        g["deadline"] = g.get("deadline") or src.get("deadline")
        g["apply_url"] = g.get("apply_url") or src.get("url")
        g["preferred_apply_date"] = apply_by(g["deadline"], g.get("preferred_apply_date"))
        sp = g["study_plan"]
        sp["ielts_required"] = norm_ielts(sp.get("ielts_required") if sp.get("ielts_required") != "unknown"
                                          else src.get("ielts_required"))
        if g.get("required_documents"):  # recompute in code so it matches the user's checklist
            g["missing_documents"] = [d for d in g["required_documents"] if not has_doc(d, held)]
        if src:  # push the better dates back into the ranked table
            src["apply_by"] = g["preferred_apply_date"]
            if sp["ielts_required"] != "unknown":
                src["ielts_required"] = sp["ielts_required"]
    print(f"[apply_agent] {len(guides)} guides")
    return {"guides": guides, "ranked": list(s.get("ranked", [])), "citations": cites}


# =============================================================== AGENT 4: professors
def wants_research(s):
    p = s["profile"]
    return norm_degree(p.get("degree")) == "PhD" or bool(p.get("want_professor"))


PROF_ROLE = (
    "You are a research-supervisor search agent. First list 3-5 research fields / sub-areas in the target "
    "country that best match the applicant (where funding and active groups exist). Then use web search to find "
    "5 professors / research groups who work on those areas, are active recently, and ideally have funded "
    "PhD/MS openings or take scholarship students. Use university pages, Google Scholar-like pages or lab sites. "
    'Return {"target_fields":[{"field","why"}],"professors":[{"name","university","department","url",'
    '"contact" (public email or contact page, null if not shown),"research_focus","recent_papers" (list of titles),'
    '"funding_note","how_to_approach" (one sentence: best channel and what to send),'
    '"suggestion" (two sentences: a research angle this applicant could pitch, grounded only in the professor\'s '
    "work and the applicant profile)}]}.")


def professor_agent(s):
    data, cites = run_agent_llm(PROF_ROLE, {"applicant": brief(s["profile"])})
    fields = [shape(f if isinstance(f, dict) else {"field": f}, {"field": None, "why": None})
              for f in as_list(data.get("target_fields"))]
    profs = [shape(x, PROF_SCHEMA) for x in as_list(data.get("professors"))]
    print(f"[professor_agent] {len(fields)} fields, {len(profs)} professors")
    return {"prof_fields": fields, "professors": profs, "citations": cites}


# =============================================================== AGENT 5: visa
VISA_ROLE = (
    "You are a student-visa research agent. Using OFFICIAL embassy / immigration pages found via web search, "
    "explain what a citizen of the home country needs to study in the target country. Return "
    '{"visa_type","official_url","when_to_apply" (timing relative to admission / course start),"processing_time",'
    '"required_documents" (list),"extra_documents" (list of less obvious items: attested degrees, medical, police '
    'certificate, financial proof, insurance, etc),"fees","notes" (list)}.')


def visa_agent(s):
    p = s["profile"]
    data, cites = run_agent_llm(VISA_ROLE, {"home_country": p.get("nationality"),
                                            "target_country": p.get("target_country"), "degree": p.get("degree")})
    data = shape(data, VISA_SCHEMA)
    held = {d.lower() for d in as_list(p.get("docs_held"))}
    data["missing"] = [d for d in data["required_documents"] if not has_doc(d, held)]
    print(f"[visa_agent] {data.get('visa_type')}")
    return {"visa": data, "citations": cites}


# =============================================================== AGENT 6: planner
PLAN_ROLE = (
    "You are an application planning agent. Build a dated plan for this applicant. Work backwards from the "
    "preferred apply dates / deadlines (anchor on the nearest realistic one; if none is known say so and assume "
    "~120 days), include: scholarship research, English test booking/taking (only if IELTS is required or the "
    "applicant has none), recommendation letters, SOP / research proposal, document attestation, supervisor "
    "outreach (if PhD), submission, then visa steps after the expected result, using the visa timing given. "
    "If time is short, compress and say so. Do not schedule anything in the past. "
    'Return {"start_by":"YYYY-MM-DD","summary":"2 sentences","tasks":[{"task","due":"YYYY-MM-DD","why"}],'
    '"routine":{"Mon":"...","Wed":"...","Fri":"...","Sun":"..."}} (weekly routine for the student).')


def planner_agent(s):
    task = {"applicant": brief(s["profile"]),
            "scholarships": [{"name": g.get("name"), "deadline": g.get("deadline"),
                              "preferred_apply_date": g.get("preferred_apply_date"),
                              "ielts_required": g["study_plan"].get("ielts_required"),
                              "missing": g.get("missing_documents")}
                             for g in s.get("guides", [])],
            "visa": {k: (s.get("visa") or {}).get(k) for k in ("when_to_apply", "processing_time", "missing")},
            "wants_professor_outreach": wants_research(s)}
    data, _ = run_agent_llm(PLAN_ROLE, task, search=False)
    today = date.today()
    plan = []
    for t in as_list(data.get("tasks")):
        t = shape(t, {"task": None, "due": None, "why": None})
        d = parse_date(t.get("due"))
        if d and t.get("task"):
            t["due"] = str(max(d, today))  # never in the past
            plan.append(t)
    plan.sort(key=lambda t: t["due"])
    notes = [data["summary"]] if data.get("summary") else []
    start = parse_date(data.get("start_by"))
    print(f"[planner_agent] {len(plan)} tasks")
    return {"plan": plan, "start_by": str(max(start, today)) if start else None,
            "routine": data.get("routine") if isinstance(data.get("routine"), dict) else {}, "notes": notes}


# =============================================================== GRAPH
def after_intake(s):
    if s["missing"] and s.get("interactive") and s["profile"].get("_asks", 0) < 2:
        return "ask_user"
    return "source_agent"


def after_apply(s):
    return "professor_agent" if wants_research(s) else "visa_agent"


NODES = [("cv_intake", cv_intake), ("ask_user", ask_user), ("source_agent", source_agent),
         ("active_agent", active_agent), ("field_agent", field_agent), ("apply_agent", apply_agent),
         ("professor_agent", professor_agent), ("visa_agent", visa_agent), ("planner_agent", planner_agent)]


def build_graph():
    g = StateGraph(State)
    for name, fn in NODES:
        g.add_node(name, fn)
    g.add_edge(START, "cv_intake")
    g.add_conditional_edges("cv_intake", after_intake, {"ask_user": "ask_user", "source_agent": "source_agent"})
    g.add_edge("ask_user", "cv_intake")
    g.add_edge("source_agent", "active_agent")
    g.add_edge("active_agent", "field_agent")
    g.add_edge("field_agent", "apply_agent")
    g.add_conditional_edges("apply_agent", after_apply, {"professor_agent": "professor_agent", "visa_agent": "visa_agent"})
    g.add_edge("professor_agent", "visa_agent")
    g.add_edge("visa_agent", "planner_agent")
    g.add_edge("planner_agent", END)
    return g.compile()


app = build_graph()


def stream_agent(profile=None, cv_text="", interactive=False):
    """Yield (node_name, partial_state) as each agent finishes. Used by the UI for live progress."""
    init = {"cv_text": cv_text, "profile": profile or {}, "interactive": interactive, "notes": [], "citations": []}
    for step in app.stream(init, stream_mode="updates"):
        for node, update in step.items():
            yield node, update or {}


def merge_update(result, upd):
    for k, v in upd.items():
        if k in ("notes", "citations"):
            result[k] = result.get(k, []) + [x for x in v if x not in result.get(k, [])]
        else:
            result[k] = v
    return result


def run_agent(profile=None, cv_text="", interactive=False):
    result = {"notes": [], "citations": []}
    for _, upd in stream_agent(profile, cv_text, interactive):
        merge_update(result, upd)
    return result


# =============================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", help="path to CV (.pdf, .docx or .txt)")
    ap.add_argument("--out", default="result.json")
    a = ap.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    r = run_agent(cv_text=read_cv(a.cv) if a.cv else "", interactive=True)

    print("\n===== SCHOLARSHIPS =====")
    for x in r.get("ranked", [])[:8]:
        print(f"- [{x.get('kind')}] {x.get('name')} | {x.get('funding')} | deadline {x.get('deadline') or 'check site'}"
              f" | apply by {x.get('apply_by') or '-'} | IELTS {x.get('ielts_required')} | {x.get('url')}")
    print("\n===== HOW TO APPLY =====")
    for g in r.get("guides", []):
        sp = g["study_plan"]
        print(f"\n{g.get('name')}  ({g.get('apply_url')})  apply by {g.get('preferred_apply_date')}")
        print(f"  Programme: {sp.get('programme')} | IELTS: {sp.get('ielts_required')} | {sp.get('language_requirement')}")
        for i, st in enumerate(g.get("steps", []), 1):
            print(f"  {i}. {st}")
        if sp.get("watch_out"):
            print("  Watch out:", "; ".join(map(str, sp["watch_out"])))
        if g.get("missing_documents"):
            print("  Missing:", ", ".join(map(str, g["missing_documents"])))
    for f in r.get("prof_fields", []) or []:
        print(f"\nField to target: {f.get('field')} - {f.get('why')}")
    for e in r.get("professors", []) or []:
        print(f"\nProfessor: {e.get('name')} ({e.get('university')}): {e.get('suggestion')}")
    v = r.get("visa", {})
    print(f"\n===== VISA: {v.get('visa_type')} =====\nWhen: {v.get('when_to_apply')}\nMissing: {', '.join(v.get('missing', [])) or 'none'}")
    print("\n===== PLAN (start by %s) =====" % r.get("start_by"))
    for t in r.get("plan", []):
        print(f"{t['due']}  {t['task']}")
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()