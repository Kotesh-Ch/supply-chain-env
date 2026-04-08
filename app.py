import gradio as gr

def demo(x):
    return "Output: " + x

gr.Interface(fn=demo, inputs="text", outputs="text").launch()