from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


def build_agent(llm, vectorstore):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ==========================
    # TOOL
    # ==========================
    @tool
    def search_autosar_docs(query: str) -> str:
        """Search AUTOSAR specifications."""
        docs = retriever.invoke(query)

        results = []

        for d in docs:
            meta = d.metadata
            text = d.page_content[:500]

            results.append(
                f"Document: {meta.get('source')}\n"
                f"Page: {meta.get('page')}\n"
                f"{text}"
            )

        return "\n\n".join(results)

    tools = [search_autosar_docs]

    # ==========================
    # LangGraph Agent
    # ==========================
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="""
You are an AUTOSAR expert assistant.

When answering:
1. Give feature overview
2. List configuration parameters
3. List APIs
4. Explain relations with other modules
5. Cite document + page
"""
    )

    return agent