"""
app.py — LangChain Skills Agent with Token Usage + Dynamic Skill API Keys
"""

import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from llm_service.service import LLMService
from handler.skills_registry import get_registry
from handler.skill_api_keys  import get_keys_for_skill

import streamlit as st
import logging
sys.path.insert(0, str(Path(__file__).parent))

import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(filename="run.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()

st.set_page_config(
    page_title="LangChain Skills Agent",
    page_icon="🧠",
    layout="wide",
)

from agent.skill_agent import run_agent, reload_tools
from handler.dynamic_skillcreator import create_skill_programmatic

@st.cache_resource
def call_llmservice(option,api_key):
    llm_service = LLMService(service_name=option,api_key=api_key)
    model = llm_service.get_llm_model()
    return model 

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  — initialise everything before any rendering
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "chat_messages":      [],
    "last_created_skill": None,
    "creation_result":    None,
    "token_history":      [],
    "_pending_rerun":     False,
    "skill_keys":         {},    # { "SERPAPI_API_KEY": "value", ... }
    "yt_cookies_path":    None,  # path to cookies.txt written on disk this session
    "yt_cookies_bytes":   None,  # raw bytes of cookies.txt — survives all reruns
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def _apply_skill_keys():
    """Push all user-entered skill keys from session_state into os.environ."""
    for env_var, val in st.session_state["skill_keys"].items():
        if val:
            os.environ[env_var] = val

# Apply on every script run so os.environ is always up-to-date
_apply_skill_keys()

# ── Pending rerun (must fire BEFORE any rendering) ───────────────────────────
if st.session_state.get("_pending_rerun"):
    st.session_state["_pending_rerun"] = False
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TOKEN HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def _safe_usage(u) -> Dict:
    if not isinstance(u, dict):
        return dict(_EMPTY_USAGE)
    return {k: int(u.get(k, 0) or 0) for k in _EMPTY_USAGE}

def _fmt(n: int) -> str:
    return f"{n:,}"

def _token_badge(usage: Dict):
    u = _safe_usage(usage)
    if u["total_tokens"] == 0:
        return
    st.caption(
        f"🔢 ⬆ {_fmt(u['input_tokens'])} in  ·  "
        f"⬇ {_fmt(u['output_tokens'])} out  ·  "
        f"**Σ {_fmt(u['total_tokens'])}**"
    )

def _record_tokens(activity: str, label: str, usage: Dict):
    u = _safe_usage(usage)
    if u["total_tokens"] == 0:
        return
    st.session_state["token_history"].append({
        "ts":       datetime.now().strftime("%H:%M:%S"),
        "activity": activity,
        "label":    label[:60],
        "input":    u["input_tokens"],
        "output":   u["output_tokens"],
        "total":    u["total_tokens"],
    })
    st.session_state["_pending_rerun"] = True

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Configuration")
    option = st.sidebar.selectbox(
            "Select LLM Service ",("GROQ","Openai","Google","Ollama"),
            index=None,
            placeholder="Select LLM Service...", 
            )
    api_key = st.sidebar.text_input("API_KEY", type="password", key="password")
    if option is not None:
        print(f"Selected LLM Service: {option}")
        llm = call_llmservice(option,api_key)

    # 1. skill registry 
    _reg = get_registry()
    _skills_with_keys = {
        name: get_keys_for_skill(name)
        for name in _reg
        if get_keys_for_skill(name)
    }
    if _skills_with_keys:
        st.divider()
        st.subheader("🔑 Skill API Keys")
        st.caption("Stored in session memory only — never written to disk.")
        for _skill_name, _specs in _skills_with_keys.items():
            for _spec in _specs:
                _wk = f"sk_{_spec['env_var']}"   # widget key
                if _wk not in st.session_state:
                    # initialise from os.environ if already set (e.g. from .env file)
                    st.session_state[_wk] = os.environ.get(_spec["env_var"], "")
        # Sync widget values → os.environ on every render pass
        for _spec in _specs:
                _wk  = f"sk_{_spec['env_var']}"
                _val = st.session_state.get(_wk, "")
                if _val:
                    os.environ[_spec["env_var"]] = _val
                elif _spec["env_var"] in os.environ:
                    del os.environ[_spec["env_var"]]

        # Status badge
        _n_required = sum(1 for s in _specs if s["required"])
        _n_filled   = sum(
                1 for s in _specs
                if s["required"] and st.session_state.get(f"sk_{s['env_var']}", "").strip()
            )
        _all_ok = (_n_filled >= _n_required)
        _icon   = "✅" if _all_ok else "⚠️"

        with st.expander(f"{_icon} {_skill_name}", expanded=not _all_ok):
            if _all_ok:
                st.success("All required keys are set.", icon="✅")
            else:
                st.warning(
                    f"{_n_required - _n_filled} required key(s) missing — "
                    "paste them below to enable this skill.",
                    icon="⚠️",
                )

            for _spec in _specs:
                _env    = _spec["env_var"]
                _label  = _spec["label"]
                _help   = _spec["help"]
                _req    = _spec["required"]
                _ispass = _spec["is_password"]
                _badge  = " *" if _req else " (optional)"

                # key=f"sk_{_env}" means Streamlit stores the live value at
                # st.session_state["sk_SERPAPI_API_KEY"] etc. automatically.
                # We pre-seeded it above, so it always reflects current state.
                st.text_input(
                    f"{_label}{_badge}",
                    type="password" if _ispass else "default",
                    help=_help,
                    key=f"sk_{_env}",
                    placeholder="Paste key here…",
                )


    # ── 1. Loaded skills ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("📦 Loaded Skills")
    if _reg:
        for _name, _skill in _reg.items():
            _badge = "🆕 " if _name == st.session_state.get("last_created_skill") else "🔧 "
            with st.expander(f"{_badge}{_name}"):
                st.caption(_skill["description"])
    else:
        st.warning("No skills loaded.")
    st.caption(f"**{len(_reg)}** skill(s) loaded")
    if st.button("🔄 Refresh"):
        st.rerun()


    # ── 2 Session tokens ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔢 Session Tokens")
    _hist = st.session_state["token_history"]
    _tt   = sum(r["total"]  for r in _hist)
    _ti   = sum(r["input"]  for r in _hist)
    _to   = sum(r["output"] for r in _hist)
    _nc   = len(_hist)
    sb1, sb2 = st.columns(2)
    sb1.metric("Total",    _fmt(_tt))
    sb2.metric("In / Out", f"{_fmt(_ti)} / {_fmt(_to)}")
    st.caption(f"{_nc} LLM call{'s' if _nc != 1 else ''} this session")
    if st.button("🗑️ Reset tokens", use_container_width=True):
        st.session_state["token_history"] = []
        st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_chat, tab_create, tab_library, tab_tokens = st.tabs(["💬 Chat", "🛠️ Create Skill", "📦 Skill Library", "📊 Token Usage",])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 Chat with your Skills")
    st.caption("Ask questions, run tools, and have a conversation!")
     # ── Missing-key warning banner ────────────────────────────────────────────
    _reg_now = get_registry()
    _blocked = []
    for _sname in _reg_now:
        _specs = get_keys_for_skill(_sname)
        for _sp in _specs:
            if _sp["required"]:
                _v = (
                    st.session_state["skill_keys"].get(_sp["env_var"])
                    or os.environ.get(_sp["env_var"], "")
                )
                if not _v:
                    _blocked.append(_sname)
                    break
    if _blocked:
        st.warning(
            f"🔑 **Missing API keys** for: {', '.join(f'`{n}`' for n in _blocked)}. "
            "Add them in the sidebar under **🔑 Skill API Keys** — "
            "the expander auto-opens when keys are missing.",
            icon="⚠️",
        )
    st.divider()

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                ic = st.columns([2, 2, 4])
                if msg.get("skill"):  ic[0].caption(f"🔧 `{msg['skill']}`")
                if msg.get("tools"):  ic[1].caption(f"🛠️ `{', '.join(msg['tools'])}`")
                _token_badge(msg.get("token_usage", {}))
    
    prefill    = st.session_state.pop("prefill_query", "")
    user_input = st.chat_input("Ask anything...")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = run_agent(llm,user_query=user_input,verbose=False,registry = get_registry())
                        response       = result["response"]  # always a clean str
                        
                        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                        selected_skill = result.get("selected_skill")
                        tools_called   = result.get("tools_called", [])
                        tool_results   = result.get("tool_results", [])
                        usage          = _safe_usage(result.get("token_usage", {}))
                        # ── Fallback: if LLM returned empty response but tools ran, ──
                        # show the tool output directly so the user always sees content
                        if not response.strip() and tool_results:
                            fallback_parts = []
                            for tr in tool_results:
                                tool_name = tr.get("tool",'')
                                if tool_name in ("read_skill_instructions", "list_available_skills"):
                                    continue
                                preview = tr.get("result_full") or tr.get("result_preview", "")
                                if not preview:
                                    continue
                                try:
                                    import json as _json
                                    data = _json.loads(preview)
                                    if isinstance(data, dict):
                                        # prefer transcript > formatted_with_timestamps > result > summary
                                        for key in ("transcript", "formatted_with_timestamps", "summary", "result", "text", "content"):
                                            if data.get(key):
                                                fallback_parts.append(
                                                    f"### 📄 {tool_name.replace('_', ' ').title()}\n\n"
                                                    f"{data[key]}"
                                                )
                                                break
                                except Exception:
                                    print("tool result preview is not valid JSON, skipping fallback preview")
                    
                        st.markdown(response)
                        ic = st.columns([2, 2, 4])
                        if selected_skill: ic[0].caption(f"🔧 `{selected_skill}`")
                        if tools_called:   ic[1].caption(f"🛠️ `{', '.join(tools_called)}`")
                        _token_badge(usage)

                        st.session_state["chat_messages"].append({
                            "role": "assistant", "content": response,
                            "skill": selected_skill, "tools": tools_called,
                            "token_usage": usage,
                            "ts": datetime.now().strftime("%H:%M:%S"),
                        })
                        _record_tokens("💬 Chat", user_input[:50], usage)
                        st.rerun()
                    except Exception as e:
                        err = f"❌ **Error:** {e}"
                        st.error(err)
                        st.session_state["chat_messages"].append({"role": "assistant", "content": err})

    if st.session_state["chat_messages"]:
        if st.button("🗑️ Clear chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CREATE SKILL
# ══════════════════════════════════════════════════════════════════════════════

with tab_create:
    st.header("🛠️ Create a New Skill")
    with st.expander("ℹ️ Pipeline steps"):
        st.markdown("""
        | Step | Action |
        |------|--------|
        | 1 | Extract structured brief |
        | 2 | Generate SKILL.md |
        | 3 | Generate implementation script |
        | 4 | Generate @tool wrapper |
        | 5 | Write to `skills/<n>/` |
        | 6 | Inject @tool into skill_agent.py |
        | 7 | Routing self-test |
        """)
    
    with st.form("skill_form"):
        skill_desc = st.text_area(
            "Describe your skill",
            placeholder="e.g. 'Extract text from PDF files'",
            height=100,
        )
        c1, c2 = st.columns([1, 3])
        submitted = c1.form_submit_button("🚀 Create", use_container_width=True, type="primary")
        c2.markdown(
            "<small style='color:grey'>~30–60 s · token usage shown after</small>",
            unsafe_allow_html=True,
        )
    if submitted:
        st.session_state["creation_result"]    = None
        st.session_state["last_created_skill"] = None
        log_box, log_lines = st.empty(), []
        def ui_log(msg):
            log_lines.append(msg)
            log_box.markdown("\n\n".join(log_lines))
        with st.spinner("Running pipeline..."):
            try:
                result = create_skill_programmatic(llm,skill_desc.strip(), log=ui_log)
                st.session_state["creation_result"]    = result
                st.session_state["last_created_skill"] = result["skill_name"]
                usage = _safe_usage(result.get("token_usage", {}))
                _record_tokens("🛠️ Skill Creation", f"Created: {result['skill_name']}", usage)
                try:
                    reload_tools()
                    ui_log("🔄 Tools reloaded — live in Chat tab.")
                except Exception as re_err:
                    ui_log(f"⚠️ Hot-reload failed ({re_err}). Restart to activate.")
            except Exception as e:
                st.error(f"❌ Failed: {e}")
                st.stop()
        st.session_state["_pending_rerun"] = False
        st.rerun()

    result = st.session_state.get("creation_result")
    if result:
        st.success(f"✅ Skill **{result['skill_name']}** created successfully!")
        ## tool usage panel
        usage = _safe_usage(result.get("token_usage", {}))
        st.subheader("🔢 Token Usage — This Creation")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total",          _fmt(usage["total_tokens"]))
        t2.metric("Input (prompt)", _fmt(usage["input_tokens"]))
        t3.metric("Output (gen)",   _fmt(usage["output_tokens"]))
        t4.metric("LLM Calls", "5", help="Brief+SKILL.md+Script+@tool+Routing")

        st.divider()
        rc1, rc2 = st.columns(2)
        with rc1:
            st.subheader("📄 SKILL.md")
            st.code(result["skill_md"], language="markdown")
        with rc2:
            st.subheader("🐍 Script")
            st.code(result["script_code"], language="python")

        st.subheader("🔧 @tool Stub")
        st.code(result["tool_stub"], language="python")

        st.subheader("📁 Files")
        for f in sorted(Path(result["skill_dir"]).rglob("*")):
            if f.is_file():
                st.caption(f"📄 `{f.relative_to(Path(result['skill_dir']).parent.parent)}`")

        if result.get("test_passed"):
            st.success(f"🧪 Routing passed — {result.get('test_reason','')}")
        else:
            st.warning(f"⚠️ Routing: {result.get('test_reason','')}")

        if result.get("python_libraries"):
            st.info(f"📦 `pip install {' '.join(result['python_libraries'])}`")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SKILL LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

with tab_library:
    st.header("📦 Skill Library")
    lib_reg = get_registry()
    if not lib_reg:
        st.warning("No skills yet.")
    else:
        search = st.text_input("🔍 Filter", placeholder="name or description")
        for name, skill in lib_reg.items():
            if search and search.lower() not in name.lower() \
                      and search.lower() not in skill["description"].lower():
                continue
            badge = "🆕 " if name == st.session_state.get("last_created_skill") else ""
            with st.expander(
                f"{badge}🔧 {name}",
                expanded=(name == st.session_state.get("last_created_skill")),
            ):
                st.markdown(f"**{skill['description']}**")

                # Show required API keys for this skill
                _sk = get_keys_for_skill(name)
                if _sk:
                    _req_keys = [s for s in _sk if s["required"]]
                    _opt_keys = [s for s in _sk if not s["required"]]
                    _parts = []
                    if _req_keys:
                        _parts.append("**Required:** " + ", ".join(f"`{s['env_var']}`" for s in _req_keys))
                    if _opt_keys:
                        _parts.append("**Optional:** " + ", ".join(f"`{s['env_var']}`" for s in _opt_keys))
                    st.caption("🔑 " + "  ·  ".join(_parts))

                lc1, lc2 = st.columns(2)
                with lc1:
                    md_path = skill.get("skill_md_path")
                    if md_path and Path(md_path).exists():
                        st.code(Path(md_path).read_text(encoding="utf-8"), language="markdown")
                with lc2:
                    sd = skill.get("scripts_dir")
                    if sd and Path(sd).exists():
                        for s in Path(sd).glob("*.py"):
                            st.caption(f"`{s.name}`")
                            st.code(s.read_text(encoding="utf-8"), language="python")
                if st.button("💬 Test in Chat", key=f"t_{name}"):
                    st.session_state["prefill_query"] = f"Help me use the {name} skill"
                    st.info("Switch to 💬 Chat tab.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TOKEN USAGE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_tokens:
    st.header("📊 Token Usage Dashboard")
    st.caption("Model: `gemini-3-pro-preview`")

    hist = st.session_state["token_history"]
    tt   = sum(r["total"]  for r in hist)
    ti   = sum(r["input"]  for r in hist)
    to_  = sum(r["output"] for r in hist)

    st.subheader("🔢 Session Totals")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tokens",  _fmt(tt))
    m2.metric("Input Tokens",  _fmt(ti))
    m3.metric("Output Tokens", _fmt(to_))
    m4.metric("LLM Calls",     str(len(hist)))

    st.divider()

    st.subheader("📂 By Activity")
    groups: Dict[str, Dict] = {}
    for row in hist:
        g = groups.setdefault(row["activity"], {"input":0,"output":0,"total":0,"calls":0})
        g["input"] += row["input"]; g["output"] += row["output"]
        g["total"] += row["total"]; g["calls"]  += 1

    if groups:
        gcols = st.columns(max(len(groups), 1))
        for i, (act, s) in enumerate(groups.items()):
            with gcols[i]:
                st.markdown(f"**{act}**")
                st.metric("Total",  _fmt(s["total"]))
                st.metric("Input",  _fmt(s["input"]))
                st.metric("Output", _fmt(s["output"]))
                st.caption(f"{s['calls']} call{'s' if s['calls']!=1 else ''}")
    else:
        st.info("No LLM calls yet.")

    st.divider()

    st.subheader("📋 Call History")
    if hist:
        hdr = st.columns([1, 2, 4, 2, 2, 2])
        for col, lbl in zip(hdr, ["Time","Activity","Label","In ⬆","Out ⬇","Total Σ"]):
            col.markdown(f"**{lbl}**")
        st.divider()
        for row in reversed(hist):
            rc = st.columns([1, 2, 4, 2, 2, 2])
            rc[0].caption(row["ts"])
            rc[1].caption(row["activity"])
            rc[2].caption(row["label"])
            rc[3].metric("i", _fmt(row["input"]),  label_visibility="collapsed")
            rc[4].metric("o", _fmt(row["output"]), label_visibility="collapsed")
            rc[5].metric("t", _fmt(row["total"]),  label_visibility="collapsed")
            st.divider()

        st.subheader("📈 Usage Over Time")
        try:
            import pandas as pd
            df = pd.DataFrame([
                {"Call": f"{r['ts']} {r['activity']}", "Input": r["input"], "Output": r["output"]}
                for r in hist
            ])
            st.bar_chart(df.set_index("Call"), color=["#4e8df5", "#f5a142"])
            st.caption("🔵 Input  🟠 Output")
        except ImportError:
            st.info("`pip install pandas` for chart.")

        st.divider()
        lines = ["time,activity,label,input_tokens,output_tokens,total_tokens"]
        for r in hist:
            lines.append(
                f"{r['ts']},{r['activity']},{r['label'].replace(',',';')},"
                f"{r['input']},{r['output']},{r['total']}"
            )
        st.download_button(
            "📥 Download CSV",
            data="\n".join(lines),
            file_name=f"tokens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No history yet.")

    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["token_history"] = []
        st.rerun()
