from backend.core import run_llm

import streamlit as st

st.header("LangChain Course - LLM with Vector DB")

prompt = st.text_input("Prompt", placeholder="What is LangChain?")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )
        formatted_response = f"**Response:** {generated_response['result']}\n\n**Sources:** {', '.join(sources)}"
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
