import gradio as gr
from transformers import pipeline

# Load a pre-trained language model
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt):
    return generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]

# Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="LLM Text Generator",
    description="Generate text using a pre-trained GPT-2 model."
)

if __name__ == "__main__":
    iface.launch()