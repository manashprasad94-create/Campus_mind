"""
app.py — CampusMind Streamlit Frontend
Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import os

from api import (
    get_faq_answer,
    process_image,
    process_pdf,
    create_notes_vectorstore,
    get_notes_answer,
    summarize_notes,
    markdown_to_pdf,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CampusMind",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #1c2330;
    --border:    #30363d;
    --gold:      #e8a020;
    --gold-dim:  #b07a18;
    --teal:      #39c5bb;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --danger:    #f85149;
    --radius:    12px;
}

/* ── Global reset ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-logo h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--gold) !important;
    margin: 0;
    letter-spacing: -0.5px;
}
.sidebar-logo p {
    font-size: 0.78rem;
    color: var(--muted) !important;
    margin: 0.25rem 0 0;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── Nav radio ── */
[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted) !important;
    margin-bottom: 0.5rem;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.2rem 0;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.95rem;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: var(--surface2);
    border-color: var(--border);
}

/* ── Page header ── */
.page-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.page-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--text);
    margin: 0 0 0.3rem;
}
.page-header p {
    color: var(--muted);
    margin: 0;
    font-size: 0.95rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-gold {
    border-color: var(--gold-dim);
    background: linear-gradient(135deg, #1c2330 0%, #1a1c14 100%);
}

/* ── Chat messages ── */
.chat-msg {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    animation: fadeUp 0.3s ease;
}
.chat-msg.user { flex-direction: row-reverse; }

.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar-user { background: var(--gold); color: #000; }
.avatar-bot  { background: var(--teal); color: #000; }

.chat-bubble {
    max-width: 75%;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    font-size: 0.94rem;
    line-height: 1.55;
}
.bubble-user {
    background: var(--gold);
    color: #000;
    border-bottom-right-radius: 4px;
}
.bubble-bot {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-left-radius: 4px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gold) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.5rem 1.25rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button:disabled { opacity: 0.4 !important; }

/* Secondary button override */
.btn-secondary > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

/* ── Text inputs ── */
.stTextInput > div > input,
.stTextArea textarea,
.stChatInputContainer textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > input:focus,
.stChatInputContainer textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(232,160,32,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.25rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: var(--teal) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Expander ── */
.stExpander {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.stExpander summary { color: var(--text) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-gold  { background: rgba(232,160,32,0.15); color: var(--gold); border: 1px solid var(--gold-dim); }
.badge-teal  { background: rgba(57,197,187,0.15); color: var(--teal); border: 1px solid rgba(57,197,187,0.4); }
.badge-muted { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 0.75rem;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeUp 0.4s ease; }

/* ── Info/warning boxes ── */
.info-box {
    background: rgba(57,197,187,0.08);
    border: 1px solid rgba(57,197,187,0.3);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: var(--teal);
    margin-bottom: 1rem;
}
.warn-box {
    background: rgba(232,160,32,0.08);
    border: 1px solid rgba(232,160,32,0.3);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: var(--gold);
    margin-bottom: 1rem;
}

/* ── Markdown rendered text ── */
.notes-content {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    font-size: 0.92rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "faq_history": [],          # list of {"role": ..., "content": ...}
        "notes_markdown": None,     # processed markdown notes
        "notes_recommendations": None,
        "notes_vectorstore": None,  # FAISS vectorstore
        "notes_qa_history": [],     # list of {"role": ..., "content": ...}
        "notes_summary": None,
        "notes_pdf_bytes": None,
        "notes_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1>🎓 CampusMind</h1>
        <p>Your AI Study Companion</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["💬  FAQ Assistant", "📝  Notes Assistant"],
        label_visibility="visible",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">About</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:var(--muted); line-height:1.7;">
    CampusMind uses <b style="color:var(--text)">LLM's</b>
    and RAG to power intelligent Q&A over your college FAQ and 
    personal handwritten notes.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if page == "📝  Notes Assistant" and st.session_state.notes_markdown:
        if st.button("🗑️ Clear Notes Session", use_container_width=True):
            st.session_state.notes_markdown = None
            st.session_state.notes_recommendations = None
            st.session_state.notes_vectorstore = None
            st.session_state.notes_qa_history = []
            st.session_state.notes_summary = None
            st.session_state.notes_pdf_bytes = None
            st.session_state.notes_filename = None
            st.rerun()


# ─────────────────────────────────────────────
# ██████████  FAQ ASSISTANT PAGE
# ─────────────────────────────────────────────
if page == "💬  FAQ Assistant":

    st.markdown("""
    <div class="page-header fade-in">
        <h2>FAQ Assistant</h2>
        <p>Ask any question about the college — admissions, courses, facilities, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat history display ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.faq_history:
            st.markdown("""
            <div class="info-box fade-in">
                👋 Hello! I'm <b>CampusMind</b>, your college FAQ guide.
                Ask me anything about admissions, courses, campus life, or facilities!
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.faq_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-msg user fade-in">
                    <div class="chat-avatar avatar-user">🧑</div>
                    <div class="chat-bubble bubble-user">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-msg fade-in">
                    <div class="chat-avatar avatar-bot">🤖</div>
                    <div class="chat-bubble bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Input ──
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("faq_form", clear_on_submit=True):
        col_input, col_btn, col_clear = st.columns([7, 1.2, 1.2])
        with col_input:
            user_q = st.text_input(
                "Ask a question",
                placeholder="e.g. What are the admission requirements?",
                label_visibility="collapsed",
                key="faq_input",
            )
        with col_btn:
            ask_btn = st.form_submit_button("Send ➤", use_container_width=True)
        with col_clear:
            clear_btn = st.form_submit_button("Clear", use_container_width=True)

    # ── Handle question ──
    if clear_btn:
        st.session_state.faq_history = []
        st.rerun()

    if ask_btn and user_q.strip():
        st.session_state.faq_history.append({"role": "user", "content": user_q.strip()})
        with st.spinner("Searching FAQ..."):
            result = get_faq_answer(user_q.strip())          # ← renamed to result

        # Extract answer text
        answer_text = result["answer"]

        # Optional: append source badges to the bubble
        if result["sources"]:
            source_tags = " ".join([
                f'<span class="badge badge-muted">{s["category"]} Q.{s["q_num"]}</span>'
                for s in result["sources"]
            ])
            answer_text += f'<br><br><div style="margin-top:0.5rem">{source_tags}</div>'

        st.session_state.faq_history.append({"role": "assistant", "content": answer_text})
        st.rerun()

    elif ask_btn and not user_q.strip():
        st.markdown('<div class="warn-box">⚠️ Please type a question first.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ██████████  NOTES ASSISTANT PAGE
# ─────────────────────────────────────────────
elif page == "📝  Notes Assistant":

    st.markdown("""
    <div class="page-header fade-in">
        <h2>Notes Assistant</h2>
        <p>Upload handwritten notes — we'll transcribe, fact-check, and let you chat with them.</p>
    </div>
    """, unsafe_allow_html=True)

    # ────────────────────────────────────────
    # STEP 1 — Upload (only shown if no notes yet)
    # ────────────────────────────────────────
    if st.session_state.notes_markdown is None:

        st.markdown('<p class="section-label">Step 1 — Upload Your Notes</p>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload handwritten notes",
            type=["jpg", "jpeg", "png", "webp", "pdf"],
            label_visibility="collapsed",
            help="Supported: JPG, PNG, WEBP, PDF",
            accept_multiple_files=True
        )

        if uploaded_files:
            # Preview grid for all uploaded files
            st.markdown('<p class="section-label">Uploaded Files</p>', unsafe_allow_html=True)
            cols = st.columns(min(len(uploaded_files), 4))

            for i, uploaded in enumerate(uploaded_files):
                with cols[i % 4]:
                    if uploaded.type.startswith("image/"):
                        st.image(uploaded, caption=uploaded.name, width=150)
                    else:
                        st.markdown(f"""
                        <div class="card fade-in" style="text-align:center; padding: 0.75rem;">
                            <div style="font-size:1.8rem">📄</div>
                            <div style="margin-top:0.3rem; color:var(--text); font-size:0.75rem; 
                                word-break:break-all;">{uploaded.name}</div>
                            <div style="color:var(--muted); font-size:0.72rem">{uploaded.size // 1024} KB · PDF</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Summary info card
            total_size = sum(f.size for f in uploaded_files)
            st.markdown(f"""
            <div class="card card-gold fade-in">
                <p class="section-label">Batch Summary</p>
                <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
                    <tr>
                        <td style="color:var(--muted); padding:0.35rem 0">Files</td>
                        <td style="color:var(--text)">{len(uploaded_files)} file(s)</td>
                    </tr>
                    <tr>
                        <td style="color:var(--muted); padding:0.35rem 0">Total Size</td>
                        <td style="color:var(--text)">{total_size // 1024} KB</td>
                    </tr>
                    <tr>
                        <td style="color:var(--muted); padding:0.35rem 0">Types</td>
                        <td style="color:var(--text)">{", ".join(set(f.type for f in uploaded_files))}</td>
                    </tr>
                </table>
                <br>
                <div class="info-box" style="margin:0">
                    ✨ Meta Vision will transcribe all notes and check for errors.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Process Notes", use_container_width=False):
                all_notes = []
                all_recs = []

                progress = st.progress(0, text="Starting...")

                for idx, uploaded in enumerate(uploaded_files):
                    ext = uploaded.name.rsplit(".", 1)[-1].lower()
                    suffix = f".{ext}"
                    progress.progress((idx) / len(uploaded_files), text=f"Processing {uploaded.name}...")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getbuffer())
                        tmp_path = tmp.name

                    try:
                        with st.spinner(f"🤖 Gemini Vision is reading: {uploaded.name}"):
                            if ext == "pdf":
                                notes, recs = process_pdf(tmp_path)
                            else:
                                notes, recs = process_image(tmp_path)

                        if notes.startswith("Error"):
                            st.error(f"Failed to process {uploaded.name}: {notes}")
                        else:
                            all_notes.append(f"## {uploaded.name}\n\n{notes}")
                            all_recs.extend(recs if isinstance(recs, list) else [recs])

                    finally:
                        os.unlink(tmp_path)

                progress.progress(1.0, text="Done!")

                if all_notes:
                    combined_notes = "\n\n---\n\n".join(all_notes)
                    combined_recs = "\n".join(all_recs)

                    st.session_state.notes_markdown = combined_notes
                    st.session_state.notes_recommendations = combined_recs
                    st.session_state.notes_filename = f"{len(uploaded_files)} files"

                    with st.spinner("🧠 Building notes index..."):
                        vs = create_notes_vectorstore(combined_notes)
                        st.session_state.notes_vectorstore = vs

                    st.success(f"✅ {len(all_notes)} file(s) processed successfully!")
                    st.rerun()

        else:
            st.markdown("""
            <div class="warn-box">
                📤 Drag and drop files above, or click to browse.
                Supports <b>JPG, PNG, WEBP</b> images and <b>PDF</b> files.
                You can select <b>multiple files</b> at once.
            </div>
            """, unsafe_allow_html=True)


    # ────────────────────────────────────────
    # STEP 2 — Notes processed: show results
    # ────────────────────────────────────────
    else:
        # Status bar
        col_s1, col_s2, col_s3 = st.columns([3, 2, 2])
        with col_s1:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.5rem">
                <span class="badge badge-teal">✅ Processed</span>
                <span style="color:var(--muted); font-size:0.85rem">
                    {st.session_state.notes_filename}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Tabs: Notes | Recommendations | Summary | Q&A ──
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄  Transcribed Notes",
            "💡  AI Recommendations",
            "⚡  Summary",
            "💬  Chat with Notes",
        ])

        # ── TAB 1: Transcribed Notes ──
        with tab1:
            col_notes, col_actions = st.columns([3, 1])

            with col_notes:
                st.markdown('<p class="section-label">Clean Markdown Notes</p>', unsafe_allow_html=True)
                with st.expander("📋 View Notes", expanded=True):
                    st.markdown(st.session_state.notes_markdown)

            with col_actions:
                st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)

                # Generate PDF button
                if st.session_state.notes_pdf_bytes is None:
                    if st.button("📄 Generate PDF", use_container_width=True):
                        with st.spinner("Creating PDF..."):
                            pdf = markdown_to_pdf(
                                st.session_state.notes_markdown,
                                st.session_state.notes_recommendations,
                            )
                            st.session_state.notes_pdf_bytes = pdf
                        st.rerun()
                else:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=st.session_state.notes_pdf_bytes,
                        file_name="campusmind_notes.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.markdown('<p style="color:var(--teal); font-size:0.8rem; margin-top:0.4rem">PDF ready!</p>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-label">Raw Markdown</p>', unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Download .md",
                    data=st.session_state.notes_markdown,
                    file_name="campusmind_notes.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        # ── TAB 2: AI Recommendations ──
        with tab2:
            if st.session_state.notes_recommendations:
                st.markdown("""
                <div class="warn-box">
                    🤖 These are AI-generated suggestions. Always verify with your textbook or professor.
                </div>
                """, unsafe_allow_html=True)
                st.markdown(st.session_state.notes_recommendations)
            else:
                st.markdown('<div class="info-box">No recommendations available for these notes.</div>', unsafe_allow_html=True)

        # ── TAB 3: Summary ──
        with tab3:
            if st.session_state.notes_summary is None:
                st.markdown("""
                <div class="card fade-in" style="text-align:center; padding:2.5rem">
                    <div style="font-size:2.5rem; margin-bottom:0.75rem">⚡</div>
                    <p style="color:var(--muted); margin-bottom:1.5rem">
                        Generate a concise bullet-point summary of your notes — 
                        perfect for quick revision.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("⚡ Generate Summary", use_container_width=False):
                    with st.spinner("Summarizing..."):
                        summary = summarize_notes(st.session_state.notes_markdown)
                        st.session_state.notes_summary = summary
                    st.rerun()
            else:
                st.markdown('<p class="section-label">Key Takeaways</p>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card fade-in">
                    {st.session_state.notes_summary}
                </div>
                """, unsafe_allow_html=True)

                col_regen, _ = st.columns([1, 4])
                with col_regen:
                    if st.button("🔄 Regenerate"):
                        with st.spinner("Regenerating..."):
                            st.session_state.notes_summary = summarize_notes(
                                st.session_state.notes_markdown
                            )
                        st.rerun()

        # ── TAB 4: Chat with Notes ──
        with tab4:
            st.markdown("""
            <div class="info-box">
                💬 Ask questions based on your notes. The AI will answer
                using only what's in your uploaded content.
            </div>
            """, unsafe_allow_html=True)

            # Chat history
            for msg in st.session_state.notes_qa_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-msg user fade-in">
                        <div class="chat-avatar avatar-user">🧑</div>
                        <div class="chat-bubble bubble-user">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-msg fade-in">
                        <div class="chat-avatar avatar-bot">🤖</div>
                        <div class="chat-bubble bubble-bot">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("notes_qa_form", clear_on_submit=True):
                col_q, col_ask, col_clr = st.columns([7, 1.2, 1.2])
                with col_q:
                    notes_q = st.text_input(
                        "Notes question",
                        placeholder="e.g. What is the main formula discussed?",
                        label_visibility="collapsed",
                        key="notes_input",
                    )
                with col_ask:
                    notes_ask = st.form_submit_button("Ask ➤", use_container_width=True)
                with col_clr:
                    notes_clear = st.form_submit_button("Clear", use_container_width=True)

            if notes_clear:
                st.session_state.notes_qa_history = []
                st.rerun()

            if notes_ask and notes_q.strip():
                if st.session_state.notes_vectorstore is None:
                    st.error("Vectorstore not ready. Please re-process your notes.")
                else:
                    st.session_state.notes_qa_history.append(
                        {"role": "user", "content": notes_q.strip()}
                    )
                    with st.spinner("Searching your notes..."):
                        ans = get_notes_answer(
                            notes_q.strip(),
                            st.session_state.notes_vectorstore,
                        )
                    st.session_state.notes_qa_history.append(
                        {"role": "assistant", "content": ans}
                    )
                    st.rerun()

            elif notes_ask and not notes_q.strip():
                st.markdown('<div class="warn-box">⚠️ Please type a question first.</div>', unsafe_allow_html=True)