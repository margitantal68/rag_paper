import keyword
import re
import pandas as pd
from sympy import use
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings

# INDEX_NAME = "sapi_theses"
# INDEX_NAME = "sapi_theses_bge_small_en"
# INDEX_NAME = "sapi_theses_bge_base_en"
INDEX_NAME = "sapi_theses_bge_large_en"

# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# EMBEDDING_MODEL_NAME = "BAAI/bge-small-en" 
# EMBEDDING_MODEL_NAME = "BAAI/bge-base-en"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"

# Initialize ElasticsearchStore and embedding model
es_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name =INDEX_NAME,
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
)

embedding_function = HuggingFaceEmbeddings(model_name =EMBEDDING_MODEL_NAME)


def keyword_search_index(es_store, query_text):
    search_body = {
        "_source": {
            "excludes": ["vector"]  # Exclude the "vector" field from the results
        },
        "query": {
            "match": {
                "text": query_text
            }
        }
    }

    # Execute the search
    response = es_store.client.search(index=INDEX_NAME, body=search_body)
    
    # Return the hits from the response
    return response["hits"]["hits"]



def vector_search_index(es_store, query_text, field="vector", k=10):
    # Generate the embedding vector for the query text
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    query_vector = embedding_model.embed_query(query_text)

    # Define the vector search body using `knn`
    search_body = {
        "_source": {
            "excludes": ["vector"]  # Exclude any additional vectors in the results
        },
        "knn": {
            "field": field,
            "query_vector": query_vector,
            "k": k
        }
    }

    # Execute the search
    response = es_store.client.search(index=INDEX_NAME, body=search_body)

    # Return the hits from the response
    return response["hits"]["hits"]


def print_results(results, print_text=False):
    for document in results:
        # Extract the relevant information
        author = document['_source']['metadata']['author']
        title = document['_source']['metadata']['title']
        study_program = document['_source']['metadata']['study_program']
        year = document['_source']['metadata']['year']

        # Display the extracted information
        print("Author:", author, ", Title:", title, ", Study Program:", study_program, ", Year:", year)
        if print_text:
            print("**********Text:", document['_source']['text'])



# Keyword search
def keyword_search(query, print_text=False):
    print("Keyword search: ", query)
    results = keyword_search_index(es_store, query)
    print(type)
    print_results(results, print_text)


# Vector Search
def vector_search(query, print_text=False):
    print("Vector search: ", query)
    results = vector_search_index(es_store, query)
    print("#results: ", len(results))
    print_results(results, print_text)


def evaluate_retrieval():
    folder = "theses/TESTSET/"
    filename = f"test_dataset_classified.csv"
    ofilename = f"test_dataset_retrieval_results_" + INDEX_NAME + ".csv"

    data = pd.read_csv(folder + filename)
    question_type_column = data['question_type']

    # for each row in the dataframe
    keyword_results_posistions = []
    vector_results_positions = []
    for index, row in data.iterrows():
        user_input = row['user_input']
        reference_context = row['reference_contexts'][2:-2]

        print("\n**********")
        print(f"user input: {user_input}")

        # Locate the abstract in the Elasticsearch index
        # Get the first document from the reference context 
        # and extract the author and title
        vector_result = vector_search_index(es_store, reference_context)
        author = vector_result[0]['_source']['metadata']['author']
        title = vector_result[0]['_source']['metadata']['title']
            
        print(f"Author: {author}, Title: {title}")

        # Find the most relevant documents for the user's query, then compute 
        # the rank of the abstract in the results using keyword search.
        keyword_results = keyword_search_index(es_store, user_input)
        idx = 1
        for result in keyword_results:
            author_result = result['_source']['metadata']['author']
            title_result  = result['_source']['metadata']['title']
            if( author == author_result and title == title_result):
                print(f"Keyword search result {idx}: {author_result}, {title_result}")
                break
            idx += 1
        keyword_results_posistions.append(idx)
        

        # Find the most relevant documents for the user's query, then compute 
        # the rank of the abstract in the results using vector search.
        vector_results = vector_search_index(es_store, user_input)
        idx = 1
        for result in vector_results:
            author_result = result['_source']['metadata']['author']
            title_result  = result['_source']['metadata']['title']
            if( author == author_result and title == title_result):
                print(f"Vector search result {idx}: {author_result}, {title_result}")
                break
            idx += 1
        vector_results_positions.append(idx)


    outdata = pd.DataFrame()
    outdata['keyword_results'] = keyword_results_posistions
    outdata['vector_results'] = vector_results_positions
    outdata['question_type'] = question_type_column
    outdata.to_csv(folder + ofilename, index=False)

   

def retrieval_results_by_question_type():
    folder = "theses/TESTSET/"
    filename = f"test_dataset_retrieval_results_" + INDEX_NAME + ".csv"

    data = pd.read_csv(folder + filename)
    mrr_keyword = sum([1/x for x in data['keyword_results']])/len(data['keyword_results'])
    mrr_vector  = sum([1/x for x in data['vector_results']])/len(data['vector_results'])
    print(f"Mean Reciprocal Rank for keyword search: {mrr_keyword: .2f}")
    print(f"Mean Reciprocal Rank for vector search: {mrr_vector: .2f}")

    # compute Retrieval@1 metric for keyword and vector search
    print("Retrieval@1 results:")
    retrieval_at_1_keyword = sum([1 for x in data['keyword_results'] if x == 1])/len(data['keyword_results'])
    retrieval_at_1_vector = sum([1 for x in data['vector_results'] if x == 1])/len(data['vector_results'])
    print(f"Retrieval@1 for keyword search: {retrieval_at_1_keyword: .2f}")
    print(f"Retrieval@1 for vector search: {retrieval_at_1_vector: .2f}")

   # compute Retrieval@3 metric for keyword and vector search
    print("Retrieval@5 results:")
    retrieval_at_5_keyword = sum([1 for x in data['keyword_results'] if x <= 5])/len(data['keyword_results'])
    retrieval_at_5_vector = sum([1 for x in data['vector_results'] if x <= 5])/len(data['vector_results'])
    print(f"Retrieval@5 for keyword search: {retrieval_at_5_keyword: .2f}")
    print(f"Retrieval@5 for vector search: {retrieval_at_5_vector: .2f}")

    # compute Retrieval@5 metric for keyword and vector search
    print("Retrieval@5 results:")
    retrieval_at_5_keyword = sum([1 for x in data['keyword_results'] if x <= 5])/len(data['keyword_results'])
    retrieval_at_5_vector = sum([1 for x in data['vector_results'] if x <= 5])/len(data['vector_results'])
    print(f"Retrieval@5 for keyword search: {retrieval_at_5_keyword: .2f}")
    print(f"Retrieval@5 for vector search: {retrieval_at_5_vector: .2f}")


    print("Retrieval results by question_type:")
    question_types = data['question_type'].unique()

    grouped = data.groupby('question_type')
    for question_type in question_types:
        print(f'Evaluation Results for the question type {question_type}:')
        df_question_type = grouped.get_group(question_type)

        mrr_keyword = sum([1/x for x in df_question_type['keyword_results']])/len(df_question_type['keyword_results'])
        mrr_vector = sum([1/x for x in df_question_type['vector_results']])/len(df_question_type['vector_results'])
        print(f"\tMean Reciprocal Rank for keyword search: {mrr_keyword: .2f}")
        print(f"\tMean Reciprocal Rank for vector search: {mrr_vector: .2f}")

        retrieval_at_1_keyword = sum([1 for x in df_question_type['keyword_results'] if x == 1])/len(df_question_type['keyword_results'])
        retrieval_at_1_vector = sum([1 for x in df_question_type['vector_results'] if x == 1])/len(df_question_type['vector_results'])
        print(f"\tRetrieval@1 for keyword search: {retrieval_at_1_keyword: .2f}")
        print(f"\tRetrieval@1 for vector search: {retrieval_at_1_vector: .2f}")

        retrieval_at_5_keyword = sum([1 for x in df_question_type['keyword_results'] if x <= 5])/len(df_question_type['keyword_results'])
        retrieval_at_5_vector = sum([1 for x in df_question_type['vector_results'] if x <= 5])/len(df_question_type['vector_results'])  
        print(f"\tRetrieval@5 for keyword search: {retrieval_at_5_keyword: .2f}")
        print(f"\tRetrieval@5 for vector search: {retrieval_at_5_vector: .2f}")


if __name__ == "__main__":
    evaluate_retrieval()
    retrieval_results_by_question_type()