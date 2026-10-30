"""Part 9: the user interface, with Streamlit.
Run it from the main folder of the project (with Redis running in Ubuntu):

    streamlit run streamlit_app.py

    sidebar: upload files, add links, see and remove the sources
    main:    a conversation. Every answer shows if it was accepted, its sources, and every round of the loop.
"""
import html
import uuid

import streamlit as st

from src.system import EvaluatorGeneratorSystem

st.set_page_config(page_title="Evaluator-Generator RAG", layout="wide")


@st.cache_resource(show_spinner="Loading the models (only the first time) ...")
def load_system():
    """Built once for the whole app: the models, the knowledge, the memories, the cache and the workflow."""
    return EvaluatorGeneratorSystem()


def start_conversation():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def show_text(text):
    """dir="auto": the browser shows Arabic right to left and English left to right.
    html.escape: the text is shown as it is, even if it has HTML in it."""
    st.markdown(f'<div dir="auto" style="white-space: pre-wrap">{html.escape(text)}</div>', unsafe_allow_html=True)


def show_message(message):
    with st.chat_message(message["role"]):
        show_text(message["content"])
        result = message.get("result")
        if not result:
            return
        if result["from_cache"]:
            st.caption("⚡ From the cache: a question with the same meaning was answered before.")
        elif result["validated"]:
            rounds = result["improvements"]
            st.caption("✅ Accepted by the Evaluator " +
                       ("on the first answer." if rounds == 0 else f"after {rounds} improvement(s)."))
        else:
            st.caption("⚠️ Not fully validated after 4 improvements.")
        with st.expander("Sources used"):
            st.text(result["context"] or "No sources (the knowledge is empty).")
        with st.expander("How the answer was reached"):
            for round_info in result["rounds"]:
                st.markdown(f"**Round {round_info['attempt']}: {round_info['decision']}**")
                show_text(round_info["answer"])
                st.caption("Scores: " + ", ".join(f"{name} {score}/5"
                                                  for name, score in round_info["scores"].items()))
                if round_info["feedback"]:
                    st.caption(f"Feedback: {round_info['feedback']}")


system = load_system()
if "session_id" not in st.session_state:
    start_conversation()

# ---------- Sidebar: the knowledge ----------
with st.sidebar:
    st.header("Knowledge")

    with st.form("files", clear_on_submit=True):
        files = st.file_uploader("Files", accept_multiple_files=True,
                                 type=[extension.lstrip(".") for extension in system.supported_extensions()])
        if st.form_submit_button("Add the files") and files:
            for file in files:
                with st.spinner(f"Reading {file.name} ..."):
                    try:
                        st.success(f"{file.name}: {system.add_file(file.name, file.getvalue())} chunk(s)")
                    except ValueError as error:
                        st.error(f"{file.name}: {error}")

    with st.form("link", clear_on_submit=True):
        url = st.text_input("Web page or Wikipedia link", placeholder="https://en.wikipedia.org/wiki/...")
        if st.form_submit_button("Add the link") and url.strip():
            with st.spinner("Reading the page ..."):
                try:
                    st.success(f"Added: {system.add_url(url.strip())} chunk(s)")
                except Exception as error:   # a wrong link, a page that is not found, no internet ...
                    st.error(f"The link was not added: {error}")

    st.subheader("Sources")
    sources = system.list_sources()
    if not sources:
        st.caption("No sources yet.")
    for info in sources:
        name_column, button_column = st.columns([5, 1])
        name_column.caption(f"**{info['source']}**  \n{info['type']} · {info['chunks']} chunk(s)")
        if button_column.button("", icon=":material/delete:", key=f"remove {info['source']}",
                                help="Remove this source"):
            system.remove_source(info["source"])
            st.rerun()

    status = system.status()
    st.divider()
    st.caption(f"Knowledge version: {status['knowledge_version']}  \n"
               f"Cache (Redis): {'on' if status['cache_on'] else 'off'}  \n"
               f"LangSmith: {'on' if status['langsmith_on'] else 'off'}")

# ---------- Main: the conversation ----------
title_column, button_column = st.columns([5, 1])
title_column.title("Evaluator-Generator platform")
if button_column.button("New conversation"):
    start_conversation()
    st.rerun()
st.caption("The Generator answers only from your sources, and the Evaluator checks every answer "
           "(up to 4 improvement rounds).")
if not sources:
    st.info("Add files or links in the sidebar first.")

for message in st.session_state.messages:
    show_message(message)

question = st.chat_input("Ask a question about your sources")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    show_message(st.session_state.messages[-1])
    with st.spinner("The Generator is writing, and the Evaluator is checking ..."):
        try:
            result = system.ask(question, st.session_state.session_id)
            answer = {"role": "assistant", "content": result["answer"], "result": result}
        except (ValueError, RuntimeError) as error:   # a bad question, or the model did not answer
            answer = {"role": "assistant", "content": f"⚠️ {error}"}
    st.session_state.messages.append(answer)
    st.rerun()
