import os
from operator import itemgetter

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()


print("Initializing components...")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4.1-nano")

vector_store = PineconeVectorStore(embedding=embeddings, index_name=os.environ["INDEX_NAME"])

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on the following context:
    
    {context}
    
    Question: {question}
    Please provide the detailed answer.    
    """
)

def format_docs(docs):
    """Format docs to join the chunks into one string"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_chain_with_lcel():
    print(f"=" * 70)
    print("LCEL Invocation - RAG")
    print(f"=" * 70)

    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

def retrieve_chain_without_lcel(query: str):
    print(f"=" * 70)
    print("Raw Invocation - Without LCEL")
    print(f"=" * 70)

    chunks = retriever.invoke(query)

    context = format_docs(chunks)

    messages = prompt_template.format(context=context, question=query)

    llm_response = llm.invoke(messages)

    return llm_response.content


if __name__ == '__main__':
    query = "When Harry first visits Ollivanders to get his wand, what are the specific details of the wand that ends up choosing him — including its wood, core, length, and any notable quality mentioned by Ollivander?"

    #Raw invocation - Without RAG
    print(f"=" * 70)
    print("Raw Invocation - Without RAG")
    print(f"=" * 70)
    response = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(response.content)

    print("\n\n")
    print(retrieve_chain_without_lcel(query))

    print("\n\n")
    retriever = retrieve_chain_with_lcel()
    lcel_response = retriever.invoke({"question": query})

    print(lcel_response)




