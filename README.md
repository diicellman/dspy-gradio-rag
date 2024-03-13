# DSPy RAG Demo Application Template

## Description 
This project serves as a comprehensive example and demo template for building Retrieval-Augmented Generation (RAG) applications. Designed to showcase the integration of RAG technology with a [FastAPI](https://github.com/tiangolo/fastapi) backend, [DSPy](https://github.com/stanfordnlp/dspy) for data processing, [Ollama](https://github.com/ollama/ollama) for localization, and a [Gradio](https://github.com/gradio-app/gradio) interface, it offers a practical reference for developers, researchers, and AI enthusiasts. Whether you're looking to understand RAG systems better or seeking a solid foundation to build your own context-aware, AI-driven applications, this template provides the necessary tools and examples to get started.

## Features
- **Educational RAG Example**: Demonstrates how to implement and utilize Retrieval-Augmented Generation technology within applications.
- **FastAPI Backend**: Offers a template for setting up an efficient and scalable backend suitable for AI-driven applications.
- **DSPy Integration**: Provides examples of data processing and manipulation, crucial for the preprocessing and handling of retrieval data in RAG systems.
- **Ollama Localization**: Showcases how to make applications globally accessible and user-friendly across different languages.
- **Gradio Interface**: Includes a simple yet powerful interface example for interacting with RAG applications, making it easy to test and demonstrate the technology.

## Installation

### Prerequisites

- Docker and Docker-Compose
- Git (optional, for cloning the repository)
- Ollama,  follow the [readme](https://github.com/ollama/ollama) to set up and run a local Ollama instance.

### Clone the Repository

First, clone the repository to your local machine (skip this step if you have the project files already).

```bash
git clone https://github.com/diicellman/dspy-gradio-rag.git
cd dspy-gradio-rag
```
### Getting Started with Local Development

First, navigate to the backend directory:
```bash
cd backend/
```

Second, setup the environment:

```bash
poetry config virtualenvs.in-project true
poetry install
poetry shell
```
Specify your environment variables in an .env file in backend directory.
Example .env file:
```yml
ENVIRONMENT=<your_environment_value>
INSTRUMENT_DSPY=<true or false>
COLLECTOR_ENDPOINT=<your_arize_phoenix_endpoint>
OLLAMA_BASE_URL=<your_ollama_instance_endpoint>
```
Third, run this command to create embeddings of data located in data/example folder:
```bash
python app/utils/load.py
```

Then run this command to start the FastAPI server:
```bash
python main.py
```

### Getting Started with Docker-Compose
This project now supports Docker Compose for easier setup and deployment, including backend services and Arize Phoenix for query tracing. 

1. Configure your environment variables in the .env file or modify the compose file directly.
2. Ensure that Docker is installed and running.
3. Run the command `docker-compose -f compose.yml up` to spin up services for the backend, and Phoenix.
4. Backend docs can be viewed using the [OpenAPI](http://0.0.0.0:8000/docs).
5. Frontend can be viewed using [Gradio](http://0.0.0.0:8000/gradio)
6. Traces can be viewed using the [Phoenix UI](http://0.0.0.0:6006).
7. When you're finished, run `docker compose down` to spin down the services.

## Usage

The FastAPI and Gradio integration allows for seamless interaction between the user and the NLP backend. Utilize the FastAPI endpoints for NLP tasks and visualize results and interact with the system through the Gradio frontend.


## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs, questions, and suggestions.