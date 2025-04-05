import os
import ast
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset


load_dotenv()  # Load environment variables from .env file


def extract_question_type_from_answer(answer):
    """Extract the question type from the answer."""
    if "reasoning" in answer:
        return "reasoning"
    elif "fact_single" in answer:
        return "fact_single"
    elif "summary" in answer:
        return "summary"
    elif "unanswerable" in answer:
        return "unanswerable"
    else:
        return "unknown"
    

def get_answer_from_openai(user_input, reference_contexts):
    """Fetch answer from OpenAI given the user input and reference contexts."""
    context = "\n".join(reference_contexts)
    prompt = (
        f"You are a helpful assistant. Please classify the following question which have to be answered based on a given context "
        f"into one of the following two categories.\n\n"
        f"Question: {user_input}\n"
        f"Context: {reference_contexts}\n\n"
        f"Categories: [1] fact_single, [2] reasoning, [3] summary, [4] unanswearable\n"
        f"Examples:\n"
        f"QUESTION_TYPE: reasoning"
        f"EXPLANATION: Answer is not explicitly mentioned in the context but can be inferred from it via simple reasoning."
        f"EXAMPLE: An ESG report section on a company’s electricity usage. Has there been a net increase in consumption over 5 years?"
        f"QUESTION_TYPE: fact_single"
        f"EXPLANATION: Answer is present in the context. It has one unit of information and cannot be partially correct."
        f"EXAMPLE: A table of a sensor’s electrical properties. What supply voltage should I use? "
        f"QUESTION_TYPE: summary"
        f"EXPLANATION: The conclusion section of a paper."
        f"QUESTION_TYPE: unanswerable"
        f"EXPLANATION: Answer is neither present in the context nor can be inferred from it"
    )

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    print(response.choices[0].message.content)
    answer = response.choices[0].message.content
    return answer



def main():
    # Load the dataset
    input_filename = 'theses/TESTSET/test_dataset.csv'
    data = pd.read_csv(input_filename)

    # Ensure reference_contexts is treated as a list of strings
    data['reference_contexts'] = data['reference_contexts'].apply(eval)

    # # Generate answers for each user_input
    answers = []
    for index, row in data.iterrows():
        user_input = row['user_input']
        reference_contexts = row['reference_contexts']
       
        answer_from_llm = get_answer_from_openai(user_input, reference_contexts)
        answer = extract_question_type_from_answer(answer_from_llm)
        
        print(f"Processing row {index+1}/{len(data)}: {user_input} : {answer}")
        print("-------------------")
        answers.append(answer)
        

    # # Add the answers as a new column
    data['answer'] = answers

    # # Save the updated dataset
    # FILE = 'theses/TESTSET/question_classification_' + DATA + '.csv'
    FILE = 'theses/TESTSET/test_dataset_classified.csv'
    data.to_csv( FILE, index=False)

    print("Answers have been added and saved to " + FILE)


if __name__ == "__main__":
    main()
