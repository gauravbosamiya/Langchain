from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of india and pakistan from the perspective of chinese"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n ----- Result {i+1} --------")
    print(f"content:\n{doc.page_content}....")