import streamlit as st
import asyncio
from dotenv import load_dotenv

import ingest
import search_agent
import logs

load_dotenv()

REPO_OWNER = "luongnv89"
REPO_NAME = "claude-howto"


@st.cache_resource
def init_agent():
    index, docs = ingest.index_data(
        REPO_OWNER,
        REPO_NAME,
        filter_func=None,
        chunk=True,
        chunking_params={"size": 2000, "step": 1000},
    )

    agent = search_agent.init_agent(index, docs, REPO_OWNER, REPO_NAME)
    return agent, len(docs)


st.set_page_config(
    page_title="GitHub Doc Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("GitHub Doc Assistant")
st.caption(f"Ask me anything about {REPO_OWNER}/{REPO_NAME}")

with st.spinner("Indexing repository and building hybrid search..."):
    agent, n_docs = init_agent()

st.success("Hybrid search is ready!")
st.caption(f"Indexed records: {n_docs}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(agent.run(user_prompt=prompt))
            answer = response.output
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    logs.log_interaction_to_file(agent, response.new_messages(), source="user")