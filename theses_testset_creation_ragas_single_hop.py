import os
import csv
import asyncio

from langchain_core.documents import Document
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset import TestsetGenerator

from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

load_dotenv()  # Load environment variables from .env file

def extract_chunks_from_csv(file_path):
    chunks = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                document = Document(
                    page_content=row['abstract'],
                    metadata={
                        "author": row.get("author", ""),
                        "title": row.get("title", "")
                    }
                )
                chunks.append(document)
            print(f"Successfully read {len(chunks)} chunks from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return chunks


generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

personas = [
    Persona(
        name="curious student",
        role_description="A student who is curious about the world and wants to learn more about different topics.",
    ),
]

# transforms = [HeadlineSplitter(), NERExtractor()]
transforms = [ NERExtractor()]

generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas
)

distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

abstracts = extract_chunks_from_csv('theses/BSC_2024_INF_abstracts_temp.csv')

async def main():
    print("Generating testset")

    print("len(abstracts): ", len(abstracts))
    dataset = generator.generate_with_langchain_docs(
        abstracts[:],
        testset_size=len(abstracts) + 2,
        transforms=transforms,
        query_distribution=distribution,
    )
    print("Eval dataset:")
    eval_dataset = dataset.to_evaluation_dataset()
    print("Query:", eval_dataset[0].user_input)
    print("Reference:", eval_dataset[0].reference)
    dataset.to_csv('theses/TESTSET/testset_2024_INF_temp.csv')

asyncio.run(main())