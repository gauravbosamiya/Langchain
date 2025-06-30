from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""
text_splitter = SemanticChunker(
    embedding, breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)
docs = text_splitter.split_text(sample)
print(docs)
