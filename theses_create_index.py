import pandas as pd

from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  
# MODEL_NAME = "BAAI/bge-small-en"  
# MODEL_NAME = "BAAI/bge-base-en" 
# MODEL_NAME = "BAAI/bge-large-en"  

INDEX_NAME = "sapi_theses"
# INDEX_NAME = "sapi_theses_bge_small_en"
# INDEX_NAME = "sapi_theses_bge_base_en"
# INDEX_NAME = "sapi_theses_bge_large_en"


def extract_name_from_filename(filename):
    # Remove the trailing .pdf
    name_part = filename[:-4]
    if "Diplomadolgozat" in name_part:
        name_part = name_part.split("Diplomadolgozat")[0]
    # Split the name part by underscores and join with spaces
    name = ' '.join(name_part.split('_'))
    return name


def add_theses_to_index_from_file(year, program, filename):
   
    documents = [] 
    df = pd.read_csv(filename)
    print(df.shape)
    for index, row in df.iterrows():
        author = row['author']
        title = row['title']
        abstract = row['abstract']
        if pd.isna(row['abstract']):
            continue
        try:
            document = Document(
                page_content=abstract,
                metadata={"author": author, "title": title, "study_program": program, "year": year}
            )
            documents.append(document)
            print(f"Author: {author}, Title: {title}")
            # es_store.add_documents([document])
        except Exception as e:
            print("Exception: ", e)
            break
        
    print(f"Number of documents: {len(documents)}")
    es_store.add_documents(documents)
    print(f"Added {len(documents)} documents to the index {INDEX_NAME}.")


es_store = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name=INDEX_NAME,
        # embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    )

if __name__ == "__main__":
    # Add theses to the index
  
    # add_theses_to_index_from_file(2021, "INF", "theses/BSC_2021_INF_abstracts.csv")
    add_theses_to_index_from_file(2022, "INF", "theses/BSC_2022_INF_abstracts.csv")
    add_theses_to_index_from_file(2023, "INF", "theses/BSC_2023_INF_abstracts.csv")
    add_theses_to_index_from_file(2024, "INF", "theses/BSC_2024_INF_abstracts.csv")

    add_theses_to_index_from_file(2021, "CALC", "theses/BSC_2021_CALC_abstracts.csv")
    add_theses_to_index_from_file(2022, "CALC", "theses/BSC_2022_CALC_abstracts.csv")
    add_theses_to_index_from_file(2023, "CALC", "theses/BSC_2023_CALC_abstracts.csv")
    add_theses_to_index_from_file(2023, "SD",  "theses/MSC_2023_SD_abstracts.csv")
    count = es_store.client.count(index=INDEX_NAME)['count']
    print(f"Document count: {count}")

