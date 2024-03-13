import gradio as gr

from app.utils.rag_functions import get_list_ollama_models, get_zero_shot_query

with gr.Blocks(title="Zero shot query") as zero_shot:
    chatbot = gr.Chatbot(label="Zero shot query", show_copy_button=True)

    with gr.Row():
        msg = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )
        submit_button = gr.Button("Submit")
        clear = gr.ClearButton([msg, chatbot], value="Clear chat")

    with gr.Accordion("Model parameters", open=False):
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

    def respond(
        message: str,
        chat_history: list,
        model_name_str: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        bot_message = get_zero_shot_query(
            query=message,
            ollama_model_name=model_name_str,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ).answer

        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(
        respond,
        [msg, chatbot, model_name, temperature, top_p, max_tokens],
        [msg, chatbot],
    )
    submit_button.click(
        respond,
        [msg, chatbot, model_name, temperature, top_p, max_tokens],
        [msg, chatbot],
    )
