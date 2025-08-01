# RAG Evaluation using RAGAS
This is a repository for paper: `Evaluating Open-Source LLMs in RAG Systems: A Benchmark on Diploma Theses Abstracts Using RAGAS`

📚 [Antal, M., Buza, K. Evaluating Open-Source LLMs in RAG Systems: A Benchmark on Diploma Theses Abstracts Using Ragas. Acta Univ. Sapientiae Inform. 17, 5 (2025). https://doi.org/10.1007/s44427-025-00006-3](https://link.springer.com/article/10.1007/s44427-025-00006-3)

🎯 [Presentation Slides](docs/rag_paper.md)

🎯 [Presentation Slides (PDF)](docs/rag_paper.pdf)

## Installation

### Prerequisites

1. Python 3.11 or higher
1. Git
1. OpenAI API key

### Steps
1. Clone the repository
    ```bash
    git clone https://github.com/margitantal68/rag_paper
    ```

1. Navigate to the project directory
    ```bash
    cd rag_paper
    ```

1. Create and activate a virtual environment
    * On Linux/macOS:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

    * On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
1. Set Up Elasticsearch
    * Install Elasticsearch using Docker:
        ```bash
        docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.9.0
        ```
1. Set Up Ollama
    * Install Ollama and pull the required models
    
1. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```


## Usage

This project requires an OpenAI API key. Follow these steps to set it up:

1. Obtain your OpenAI API key from [OpenAI's website](https://platform.openai.com/docs/overview).
1. Copy the **.env.example** file in the project directory:
    ```bash
    cp .env.example .env
    ```

1. Set the API key in the **.env** file:
    ```bash
    OPENAI_API_KEY=your_api_key_here
    ```

1. Run the scripts in the following order:
- Create the Elasticsearch index
    ```bash
    python theses_create_index.py
    ```
- Evaluate the Retriever
    ```bash
    python theses_retrieval_evaluation.py
    ```
- Evaluate the Generation
    ```bash
    python theses_rag_evaluation.py
    ```

⚠️ Do not run the script for testset creation `theses_testset_creation_ragas_single_hop.py` as it is not needed for the evaluation. The testset is already created and included in the repository `theses\TESTSET\test_dataset.csv`.

⚠️ Do not run the script for question classification `theses_testset_question_classification.py`  as it is not needed for the evaluation. The classification is already done and included in the repository `theses\TESTSET\test_dataset.csv`. 