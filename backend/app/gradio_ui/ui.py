import gradio as gr
from app.gradio_ui.zero_shot_ui import zero_shot
from app.gradio_ui.compiled_ui import compiled
from app.gradio_ui.optimize_pipeline import compile_pipeline

# hello_world = gr.Interface(lambda name: "Hello " + name, "text", "text")
# bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")

gradio_iface = gr.TabbedInterface(
    [zero_shot, compiled, compile_pipeline],
    ["Zero Shot Query", "Compiled Query", "Optimize Pipeline"],
    title="DSPy",
)
