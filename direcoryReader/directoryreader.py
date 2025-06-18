
import warnings

# Disable all UserWarnings (including the CUDA one)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable CUDA device query warnings
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disable Deprecation warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)


from llama_index.core import SimpleDirectoryReader
import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

gemini_llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

Settings.llm = gemini_llm
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def main(url:str)-> None :
    doc=SimpleDirectoryReader(url).load_data()
    # print(f"Loaded {len(doc)} docs")
    # print(doc[0].text_resource.text)
    index=VectorStoreIndex.from_documents(documents=doc)
    query_engine=index.as_query_engine()
    res=query_engine.query("tell me some aws command")
    print(res)

if __name__=="__main__":
    main(url=r"/home/rajeev-kumar/command/aws")
