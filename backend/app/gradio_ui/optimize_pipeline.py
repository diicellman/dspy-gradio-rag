import gradio as gr
import pandas as pd

from app.utils.rag_functions import get_list_ollama_models, compile_rag
from app.utils.models import QAItem

with gr.Blocks(title="Optimize Pipeline") as compile_pipeline:
    gr.Markdown(
        """
# Using DSPy Optimizers with Small Training Sets

DSPy, a powerful framework developed by StanfordNLP, provides a suite of tools and functionalities for natural language processing tasks. One of its capabilities includes optimizing models based on a variety of input data. This page focuses on how to use DSPy optimizers with a small dataset, specifically a CSV file containing pairs of questions and answers.

## Preparing Your Data

Your dataset should be in a CSV file with two columns: `question` and `answer`. Here's an example of what your data might look like:

```csv
question,answer
"What is the capital of France?","Paris"
"Who wrote 'To Kill a Mockingbird'?","Harper Lee"
"What is the chemical symbol for water?","H2O"
"Who painted the Mona Lisa?","Leonardo da Vinci"
"What year did the Titanic sink?","1912"
"""
    )
    file = gr.File(label="CSV file", file_types=[".csv"])
    all_data = gr.Dataframe(label="All data")
    # upload_button = gr.UploadButton("Click to Upload a File", file_types=[".csv"], file_count="single")

    with gr.Accordion("Model parameters", open=True):
        with gr.Row():
            model_name = gr.Dropdown(
                get_list_ollama_models(),
                value=get_list_ollama_models()[0],
                label="Ollama model",
                info="List of models available on your machine.",
            )

            max_tokens = gr.Slider(
                minimum=128, maximum=2048, value=150, label="max-tokens"
            )

        with gr.Row():
            temperature = gr.Slider(
                minimum=0, maximum=2, step=0.1, value=0.1, label="temperature"
            )
            top_p = gr.Slider(minimum=0, maximum=1, step=0.1, value=1, label="top-p")

    with gr.Row():
        clear = gr.ClearButton([file, all_data], value="Clear data")
        submit_button = gr.Button("Compile")

    def upload_file(csv_file):
        table = pd.read_csv(csv_file.name, delimiter=",")
        table = table.astype(str)

        return table

    def proccess_data(
        csv_file,
        ollama_model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        table = pd.read_csv(csv_file.name, delimiter=",")
        table_dict = table.to_dict(orient="records")
        table = table.astype(str)

        items_list = []
        for item in table_dict:
            try:
                qa_item = QAItem(**item)
                items_list.append(qa_item)
            except Exception as e:
                print("Error during validation:", e)

        try:
            compile_response = compile_rag(
                items=items_list,
                ollama_model_name=ollama_model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return gr.Info("Successfully compiled RAG program!")
        except Exception as e:
            return gr.Warning(f"An error occurred: {e}")

    file.upload(upload_file, file, all_data)
    submit_button.click(
        proccess_data, [file, model_name, temperature, top_p, max_tokens], None
    )
