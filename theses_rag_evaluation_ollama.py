import os
import ast
from pyexpat import model
from numpy import std
import ollama
import pandas as pd

from openai import OpenAI
from datasets import Dataset

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_similarity,
    answer_correctness,
)
from ragas import evaluate
# from utils import LLM_MODEL



# DeepSeek response is different
def extract_answer_from_deepseek_response(response):
    # the response is a text having the following format: <think> ...</think> Plain answer.
    # extract the plain answer
    answer = response.split("</think>")[1].strip()
    return answer


def get_answer_from_ollama(user_input, reference_contexts):
    context = "\n".join(reference_contexts)
    # full_prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}"
    full_prompt = (
        f"You are a helpful assistant. Using the following reference contexts:\n{context}\n\n"
        f"Please answer the following question:\n{user_input}\n"
        f"I need a short and concise answer based on the provided reference contexts."
        f"If you can not provide an answer based on the provided reference context, say 'I don't know"
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": full_prompt,
            }
        ])
    return response.message.content


def main(input_filename):
    # Load the dataset
    data = pd.read_csv(input_filename + '.csv')

    # Ensure reference_contexts is treated as a list of strings
    data['reference_contexts'] = data['reference_contexts'].apply(eval)

    # # Generate answers for each user_input
    answers = []
    for index, row in data.iterrows():
        user_input = row['user_input']
        reference_contexts = row['reference_contexts']
        print(f"Processing row {index+1}/{len(data)}: {user_input}")
        answer = get_answer_from_ollama(user_input, reference_contexts)
        if OLLAMA_MODEL == "deepseek-r1":
            answer = extract_answer_from_deepseek_response(answer)
        print(f"\tAnswer: {answer}")
        answers.append(answer)

    # # Add the answers as a new column
    data['answer'] = answers

    # # Save the updated dataset
    # data.to_csv(input_filename+ '_with_answers_' + LLM_MODEL + '.csv', index=False)
    # print("Answers have been added and saved ")

    data.rename(columns={"reference_contexts": "retrieved_contexts"}, inplace=True)
    dataset = Dataset.from_pandas(data)

    # Evaluate the dataset
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_similarity,
            answer_correctness,
        ],
    )

    df = result.to_pandas()
    print(df.head())

    # Save evaluation results
    df.to_csv(input_filename +'_' + OLLAMA_MODEL + '.csv', index=False)

    average_faithfulness = df['faithfulness'].mean()
    std_faithfulness = df['faithfulness'].std()

    average_answer_relevancy = df['answer_relevancy'].mean()
    std_answer_relevancy = df['answer_relevancy'].std()

    average_answer_similarity = df['semantic_similarity'].mean()
    std_answer_similarity = df['semantic_similarity'].std()

    average_answer_correctness = df['answer_correctness'].mean()
    std_answer_correctness = df['answer_correctness'].std()

    print(f'Evaluation Results for the dataset {input_filename}:')
    print(f'\tFaithfulness avg: {average_faithfulness: .2f}, std: {std_faithfulness: .2f}')
    print(f'\tAnswer Relevancy avg: {average_answer_relevancy: .2f}, std: {std_answer_relevancy: .2f}')
    print(f'\tSemantic Similarity avg: {average_answer_similarity: .2f}, std: {std_answer_similarity: .2f}')
    print(f'\tAnswer Correctness avg: {average_answer_correctness: .2f}, std: {std_answer_correctness: .2f}')


def compute_metrics_question_types(llm_model):
    input_filename = 'theses/TESTSET/test_dataset_classified.csv'
    data = pd.read_csv(input_filename)
    question_type_column = data['question_type']

    filename = 'theses/TESTSET/test_dataset_' + llm_model + '.csv'
    df = pd.read_csv(filename)
    df['question_type'] = question_type_column

    print(df.columns)
     # Calculate the average scores with standard deviation for each metric and each question type
    question_types = df['question_type'].unique()
    
    print(f'Question Types: {question_types}')

    grouped = df.groupby('question_type')
    for question_type in question_types:
        print(f'Evaluation Results for the question type {question_type}:')
        df_question_type = grouped.get_group(question_type)

        average_faithfulness = df_question_type['faithfulness'].mean()
        std_faithfulness = df_question_type['faithfulness'].std()

        average_answer_relevancy = df_question_type['answer_relevancy'].mean()
        std_answer_relevancy = df_question_type['answer_relevancy'].std()

        average_answer_similarity = df_question_type['semantic_similarity'].mean()
        std_answer_similarity = df_question_type['semantic_similarity'].std()

        average_answer_correctness = df_question_type['answer_correctness'].mean()
        std_answer_correctness = df_question_type['answer_correctness'].std()

        print(f'\tFaithfulness avg: {average_faithfulness: .2f}, std: {std_faithfulness: .2f}')
        print(f'\tAnswer Relevancy avg: {average_answer_relevancy: .2f}, std: {std_answer_relevancy: .2f}')
        print(f'\tSemantic Similarity avg: {average_answer_similarity: .2f}, std: {std_answer_similarity: .2f}')
        print(f'\tAnswer Correctness avg: {average_answer_correctness: .2f}, std: {std_answer_correctness: .2f}')


if __name__ == "__main__":
    # Local models
    # OLLAMA_MODEL = "llama3"
    # OLLAMA_MODEL = "deepseek-r1"
    OLLAMA_MODEL = "deepseek-r1:32b"
    # OLLAMA_MODEL = "mistral" 
    # OLLAMA_MODEL = "mistral:7b-instruct-v0.3-fp16"
    # OLLAMA_MODEL = "gemma2:9b-instruct-fp16"

    # OLLAMA_BASE_URL = 'http://192.168.11.102:11500'		
		
    # OLLAMA_MODEL = 'deepseek-r1:32b'	
    # OLLAMA_MODEL = 'chatgpt-4o-latest'
    # OLLAMA_MODEL = 'o1-preview'

    input_filename = 'theses/TESTSET/test_dataset'
    print(f"Running evaluation for {OLLAMA_MODEL} model")
    # main(input_filename)
    compute_metrics_question_types(OLLAMA_MODEL)
