"""Streamlit UI for the Scholarship Agent.   Run:  streamlit run app.py"""
import os, json
from datetime import date

import pandas as pd
import streamlit as st

import db
import scholarship_agent as sa

st.set_page_config(page_title="Scholarship Agent", page_icon="🎓", layout="wide")
db.init()

ss = st.session_state
for k, v in dict(profile={}, student_id=None, cv_text="", cv_hash=None, cv_pending=False,
                 result=None, run_id=None).items():
    ss.setdefault(k, v)

STEP_LABELS = {
    "cv_intake": "Checked profile",
    "ask_user": "Asked for missing info",
    "source_agent": "Found official scholarship sources",
    "active_agent": "Scanned sources for open scholarships",
    "field_agent": "Searched by field and filtered eligibility",
    "apply_agent": "Built application guides and study plans",
    "professor_agent": "Found research fields and professors",
    "visa_agent": "Checked visa requirements",
    "planner_agent": "Built dated plan",
}
IELTS_LABEL = {"yes": "🔴 Required", "no": "🟢 Not required", "waivable": "🟡 Waivable / alternatives",
               "unknown": "⚪ Check site"}
SCHEMES = ["any", "government", "collaborative", "university"]


# =============================================================== helpers
def days_left(d):
    d = sa.parse_date(d)
    return (d - date.today()).days if d else None


def bullets(items, empty="—"):
    items = [str(i) for i in sa.as_list(items) if i]
    return "\n".join(f"- {i}" for i in items) if items else empty


def load_student(sid):
    rec = db.get_student(sid)
    if not rec:
        return
    ss.student_id, ss.profile, ss.cv_text = sid, rec["profile"], rec.get("cv_text") or ""
    runs = db.list_runs(sid, limit=1)
    ss.result = db.get_run(runs[0]["id"])["result"] if runs else None
    ss.run_id = runs[0]["id"] if runs else None
    ss.cv_pending = False


def new_student():
    for k in ("profile", "student_id", "cv_text", "cv_hash", "result", "run_id"):
        ss[k] = {} if k == "profile" else ("" if k == "cv_text" else None)
    ss.cv_pending = False


def on_pick():
    choice = ss.picker
    if choice == "new":
        new_student()
    else:
        load_student(choice)


# =============================================================== sidebar: student records
with st.sidebar:
    st.header("🎓 Student records")
    students = db.list_students()
    ids = ["new"] + [s["id"] for s in students]
    names = {s["id"]: f"{s['name'] or 'Unnamed'} · {s['degree'] or '?'} → {s['target_country'] or '?'}"
             for s in students}
    idx = ids.index(ss.student_id) if ss.student_id in ids else 0
    st.selectbox("Student", ids, index=idx, key="picker", on_change=on_pick,
                 format_func=lambda i: "➕ New student" if i == "new" else names[i])
    if ss.student_id:
        rec = db.get_student(ss.student_id)
        st.caption(f"{rec['email']}  \nUpdated {rec['updated_at'].replace('T', ' ')}  \n"
                   f"CV stored: {'yes' if rec.get('cv_text') else 'no'}")
        runs = db.list_runs(ss.student_id)
        if runs:
            run_ids = [r["id"] for r in runs]
            pick = st.selectbox("Past runs", run_ids,
                                index=run_ids.index(ss.run_id) if ss.run_id in run_ids else 0,
                                format_func=lambda i: next(f"{r['created_at'][:16].replace('T', ' ')} · "
                                                           f"{r['n_scholarships']} found" for r in runs if r["id"] == i))
            if pick != ss.run_id and st.button("Open this run"):
                ss.run_id, ss.result = pick, db.get_run(pick)["result"]
                st.rerun()
        if st.button("🗑️ Delete student", type="secondary"):
            db.delete_student(ss.student_id)
            new_student()
            st.rerun()
    st.divider()
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set: searches will fail.")
    st.caption(f"Model: {sa.MODEL}")

st.title("Scholarship Agent")
st.caption("Finds open MS / PhD scholarships, how to apply, visa needs and a dated plan. "
           "Always confirm deadlines on the official page.")

tab_profile, tab_results = st.tabs(["1 · Student profile", "2 · Results"])

# =============================================================== tab 1: profile
with tab_profile:
    st.subheader("CV")
    up = st.file_uploader("Upload CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    if up is not None:
        raw = up.getvalue()
        h = db.cv_hash(raw)
        if ss.cv_hash != h:
            ss.cv_hash = h
            saved = db.find_by_cv_hash(h)
            if saved:
                load_student(saved["id"])
                ss.cv_hash = h
                st.success(f"This CV is already saved for **{saved['name'] or saved['email']}**: "
                           "using the stored profile (no re-reading needed).")
            else:
                ss.cv_text = sa.read_cv_bytes(up.name, raw)
                ss.cv_pending = True
        if ss.cv_pending:
            c1, c2 = st.columns([1, 3])
            if c1.button("📄 Read CV and fill the form", type="primary"):
                with st.spinner("Reading CV ..."):
                    parsed = sa.parse_cv_profile(ss.cv_text)
                for k, v in parsed.items():
                    if not ss.profile.get(k):
                        ss.profile[k] = v
                ss.cv_pending = False
                st.rerun()
            c2.caption("New CV: the AI will extract your details once, then they are saved with your record.")
    elif ss.cv_text:
        st.info("Using the CV stored with this student record.")

    p = ss.profile
    st.subheader("Details")
    with st.form("profile_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Full name", p.get("name") or "")
        email = c2.text_input("Email (record key) *", p.get("email") or "")
        deg_now = sa.norm_degree(p.get("degree")) or "MS"
        degree = c3.selectbox("Degree *", ["MS", "PhD"], index=["MS", "PhD"].index(deg_now))

        c1, c2, c3 = st.columns(3)
        field = c1.text_input("Field of study *", p.get("field") or "")
        nationality = c2.text_input("Your country (apply from) *", p.get("nationality") or "")
        target = c3.text_input("Country to study in *", p.get("target_country") or "")

        interests = st.text_input("Research interests (comma separated)",
                                  ", ".join(map(str, sa.as_list(p.get("interests")))))

        c1, c2, c3, c4 = st.columns(4)
        gpa = c1.number_input("GPA (4.0 scale) *", 0.0, 4.0, float(sa.num(p.get("gpa_4")) or 0.0), 0.01)
        ielts = c2.number_input("IELTS (0 = not taken)", 0.0, 9.0, float(sa.num(p.get("ielts")) or 0.0), 0.5)
        pubs = c3.number_input("Publications", 0, 100, int(sa.num(p.get("publications")) or 0))
        exp = c4.number_input("Experience (years)", 0.0, 50.0, float(sa.num(p.get("experience_years")) or 0.0), 0.5)

        c1, c2 = st.columns(2)
        scheme = c1.selectbox("Scholarship type", SCHEMES,
                              index=SCHEMES.index(p.get("scheme")) if p.get("scheme") in SCHEMES else 0)
        intake = c2.text_input("Preferred intake (optional)", p.get("intake_preference") or "",
                               placeholder="e.g. Fall 2027")

        held = [d for d in sa.as_list(p.get("docs_held")) if d in sa.DOC_OPTIONS]
        docs = st.multiselect("Documents you already have", sa.DOC_OPTIONS, default=held or ["cv"])
        want_prof = st.checkbox("Find professors to approach (always on for PhD)",
                                value=bool(p.get("want_professor")) or degree == "PhD")

        b1, b2, _ = st.columns([1, 1, 3])
        save_only = b1.form_submit_button("💾 Save")
        run = b2.form_submit_button("🔎 Save & find scholarships", type="primary")

    if save_only or run:
        new = {**p, "name": name.strip() or None, "email": email.strip().lower(), "degree": degree,
               "field": field.strip(), "nationality": nationality.strip(), "target_country": target.strip(),
               "interests": [i.strip() for i in interests.split(",") if i.strip()],
               "gpa_4": gpa or None, "ielts": ielts or None, "publications": int(pubs),
               "experience_years": exp, "scheme": scheme, "intake_preference": intake.strip() or None,
               "docs_held": docs, "want_professor": want_prof, "_parsed": True}
        missing = [lbl for k, lbl in [("email", "email"), ("field", "field"), ("nationality", "your country"),
                                      ("target_country", "target country"), ("gpa_4", "GPA")] if not new.get(k)]
        if missing:
            st.error("Please fill: " + ", ".join(missing))
        else:
            ss.student_id = db.upsert_student(new, cv_text=ss.cv_text or None, cv_hash=ss.cv_hash)
            ss.profile = new
            st.success("Student record saved.")
            if run:
                result = {"notes": [], "citations": []}
                with st.status("Agents are searching official sources ... (2–5 minutes)", expanded=True) as status:
                    try:
                        for node, upd in sa.stream_agent(profile=new, cv_text="", interactive=False):
                            sa.merge_update(result, upd)
                            extra = ""
                            if node == "field_agent":
                                extra = f" · {len(result.get('ranked') or [])} eligible"
                            elif node == "source_agent":
                                extra = f" · {len(result.get('sources') or [])} sources"
                            status.write(f"✅ {STEP_LABELS.get(node, node)}{extra}")
                        status.update(label="Done. Open the Results tab.", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label="A step failed", state="error")
                        st.exception(e)
                        result = None
                if result:
                    ss.result = result
                    ss.run_id = db.save_run(ss.student_id, result)
                    st.success("Results saved: open **2 · Results**.")

# =============================================================== tab 2: results
with tab_results:
    r = ss.result
    if not r:
        st.info("No results yet. Fill the profile and press **Save & find scholarships**.")
        st.stop()

    rp = r.get("profile") or ss.profile
    ranked, guides = r.get("ranked") or [], r.get("guides") or []
    deadlines = [d for d in (sa.parse_date(x.get("deadline")) for x in ranked) if d]
    missing_all = sorted({str(d) for g in guides for d in g.get("missing_documents", [])})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eligible scholarships", len(ranked))
    m2.metric("Nearest deadline", str(min(deadlines)) if deadlines else "—",
              f"{(min(deadlines) - date.today()).days} days left" if deadlines else None, delta_color="off")
    m3.metric("Start by", r.get("start_by") or "—")
    m4.metric("Missing documents", len(missing_all))
    for n in r.get("notes") or []:
        st.info(n)

    sections = ["Scholarships", "Study plan & how to apply"]
    if r.get("professors") or r.get("prof_fields"):
        sections.append("Professors")
    sections += ["Visa", "Timeline", "Sources"]
    tabs = dict(zip(sections, st.tabs(sections)))

    # ---------- scholarships table
    with tabs["Scholarships"]:
        if not ranked:
            st.warning("No open scholarships passed the eligibility filter. Try scheme = any, or another country.")
        else:
            df = pd.DataFrame([{
                "Scholarship": x.get("name"), "Provider": x.get("provider"), "Type": x.get("kind"),
                "Funding": x.get("funding"), "Last date": x.get("deadline") or "check site",
                "Apply by": x.get("apply_by") or "—", "Days left": days_left(x.get("deadline")),
                "IELTS": IELTS_LABEL.get(sa.norm_ielts(x.get("ielts_required"))),
                "Min GPA": sa.num(x.get("min_gpa_4")), "Min IELTS": sa.num(x.get("min_ielts")),
                "Intake": x.get("intake"), "Match": x.get("rank_score"), "Link": x.get("url"),
            } for x in ranked])
            st.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="open"),
                "Match": st.column_config.ProgressColumn("Match", min_value=0, max_value=12, format="%.1f"),
                "Days left": st.column_config.NumberColumn("Days left", format="%d"),
            })
            st.caption("‘Apply by’ is a safe submission date (≈3 weeks before the last date unless the "
                       "official page suggests otherwise).")
            for x in ranked:
                with st.expander(f"{x.get('name')} — {x.get('provider') or ''}"):
                    st.write(x.get("description") or "")
                    st.markdown(f"**Fields:** {', '.join(map(str, x.get('fields') or [])) or '—'}  \n"
                                f"**Documents:** {', '.join(map(str, x.get('required_documents') or [])) or '—'}")
                    if x.get("extra"):
                        st.json(x["extra"], expanded=False)

    # ---------- study plan (fixed 6 fields) + guide
    with tabs["Study plan & how to apply"]:
        if not guides:
            st.info("No guides were generated.")
        for g in guides:
            sp = g.get("study_plan") or {}
            dl = days_left(g.get("deadline"))
            with st.container(border=True):
                st.markdown(f"### {g.get('name')}")
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**📅 Last date**  \n{g.get('deadline') or 'check site'}"
                            + (f" ({dl} days left)" if dl is not None else ""))
                c2.markdown(f"**✅ Preferred date to apply**  \n{g.get('preferred_apply_date') or '—'}")
                c3.markdown(f"**🔗 Apply at**  \n{g.get('apply_url') or '—'}")

                st.markdown("**Study plan**")
                a, b, c = st.columns(3)
                a.markdown(f"**1. Course / programme**  \n{sp.get('programme') or '—'}")
                b.markdown(f"**2. Universities**  \n{', '.join(map(str, sp.get('universities') or [])) or '—'}")
                c.markdown(f"**3. Duration & intake**  \n{sp.get('duration') or '—'}")
                a, b, c = st.columns(3)
                a.markdown(f"**4. IELTS needed?**  \n{IELTS_LABEL.get(sa.norm_ielts(sp.get('ielts_required')))}  \n"
                           f"{sp.get('language_requirement') or ''}")
                b.markdown(f"**5. Watch out**  \n{bullets(sp.get('watch_out'))}")
                c.markdown(f"**6. Extra things that matter**  \n{bullets(sp.get('extras'))}")

                st.markdown("**How to apply**")
                st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(g.get("steps") or [], 1)) or "—")
                a, b, c = st.columns(3)
                a.markdown(f"**Missing documents**  \n{bullets(g.get('missing_documents'), 'none 🎉')}")
                b.markdown(f"**Gaps**  \n{bullets(g.get('gaps'), 'none')}")
                c.markdown(f"**Tips**  \n{bullets(g.get('tips'))}")

    # ---------- professors
    if "Professors" in tabs:
        with tabs["Professors"]:
            if r.get("prof_fields"):
                st.markdown("**Research fields to target**")
                for f in r["prof_fields"]:
                    st.markdown(f"- **{f.get('field')}** — {f.get('why') or ''}")
            for e in r.get("professors") or []:
                with st.container(border=True):
                    st.markdown(f"#### {e.get('name')}  \n{e.get('department') or ''}, {e.get('university') or ''}")
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Research focus:** {e.get('research_focus') or '—'}  \n"
                                f"**Funding:** {e.get('funding_note') or '—'}  \n"
                                f"**Page:** {e.get('url') or '—'}  \n**Contact:** {e.get('contact') or '—'}")
                    c2.markdown(f"**How to approach:** {e.get('how_to_approach') or '—'}  \n\n"
                                f"**Angle to pitch:** {e.get('suggestion') or '—'}")
                    if e.get("recent_papers"):
                        st.markdown("**Recent papers**\n" + bullets(e["recent_papers"]))

    # ---------- visa
    with tabs["Visa"]:
        v = r.get("visa") or {}
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Visa type**  \n{v.get('visa_type') or '—'}")
        c2.markdown(f"**When to apply**  \n{v.get('when_to_apply') or '—'}")
        c3.markdown(f"**Processing time**  \n{v.get('processing_time') or '—'}")
        st.markdown(f"**Fees:** {v.get('fees') or '—'} · **Official page:** {v.get('official_url') or '—'}")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Required documents**\n" + bullets(v.get("required_documents")))
        c2.markdown("**Easy-to-miss documents**\n" + bullets(v.get("extra_documents")))
        c3.markdown("**You still need**\n" + bullets(v.get("missing"), "none"))
        if v.get("notes"):
            st.markdown("**Notes**\n" + bullets(v["notes"]))

    # ---------- timeline
    with tabs["Timeline"]:
        plan = r.get("plan") or []
        if plan:
            st.dataframe(pd.DataFrame([{"Due": t["due"], "Days from today": days_left(t["due"]),
                                        "Task": t.get("task"), "Why": t.get("why")} for t in plan]),
                         hide_index=True, use_container_width=True)
        routine = r.get("routine") or {}
        if routine:
            st.markdown("**Weekly routine**")
            cols = st.columns(len(routine))
            for col, (day, what) in zip(cols, routine.items()):
                col.markdown(f"**{day}**  \n{what}")

    # ---------- sources
    with tabs["Sources"]:
        for s in r.get("sources") or []:
            st.markdown(f"- **[{s.get('kind')}]** [{s.get('name')}]({s.get('url')}) — {s.get('why') or ''}")
        if r.get("citations"):
            with st.expander(f"All pages the agents cited ({len(r['citations'])})"):
                st.markdown(bullets(r["citations"]))

    st.download_button("⬇️ Download results (JSON)", json.dumps(r, indent=2, default=str),
                       file_name=f"scholarships_{(rp.get('name') or 'student').replace(' ', '_')}.json",
                       mime="application/json")