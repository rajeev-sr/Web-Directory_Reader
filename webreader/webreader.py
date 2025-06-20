import warnings

# Disable all UserWarnings (including the CUDA one)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable CUDA device query warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disable Deprecation warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.gemini import Gemini
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

# Load your keys
load_dotenv()

# Use latest GoogleGenAI/Gemini LLM
gemini_llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

Settings.llm = gemini_llm
# Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model=GeminiEmbedding(model_name="models/embedding-001")

def main(url: str) -> None:
    doc = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=doc)
    query_engine = index.as_query_engine()
    res = query_engine.query("Tell me about Gen AI")
    print(res)

if __name__ == "__main__":
    main(url="https://joinseven.medium.com/blog-series-genai-a-brief-introduction-in-generative-ai-4e11154df3f2")
