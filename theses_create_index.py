import pandas as pd

from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


def extract_name_from_filename(filename):
    # Remove the trailing .pdf
    name_part = filename[:-4]
    if "Diplomadolgozat" in name_part:
        name_part = name_part.split("Diplomadolgozat")[0]
    # Split the name part by underscores and join with spaces
    name = ' '.join(name_part.split('_'))
    return name


def add_theses_to_index_from_file(year, program, filename):
    es_store = ElasticsearchStore(
        # es_url="http://elasticsearch:9200",
        es_url="http://localhost:9200",
        index_name="sapi_theses1",
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    )
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
        except:
            print("Exception: ", abstract)
    es_store.add_documents(documents)


# Add theses to the index
add_theses_to_index_from_file(2021, "INF", "theses/BSC_2021_INF_abstracts.csv")
add_theses_to_index_from_file(2022, "INF", "theses/BSC_2022_INF_abstracts.csv")
add_theses_to_index_from_file(2023, "INF", "theses/BSC_2023_INF_abstracts.csv")
add_theses_to_index_from_file(2024, "INF", "theses/BSC_2024_INF_abstracts.csv")

