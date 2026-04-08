import gradio as gr

def demo(x):
    return "Output: " + x

iface = gr.Interface(fn=demo, inputs="text", outputs="text")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)