import gradio as gr
from search_engine import SearchEngine

engine = SearchEngine()

def search(text_query, image_query):
    if text_query:
        results = engine.search_text(text_query)
    elif image_query:
        results = engine.search_image(image_query)
    else:
        return []

    gallery = []
    for meta, score in results:
        gallery.append((meta["image_path"], f"{meta['caption']} ({score:.2f})"))
    return gallery

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Multimodal Search Engine")
    text_input = gr.Textbox(label="Text Query")
    image_input = gr.Image(type="filepath", label="Image Query")
    search_button = gr.Button("Search")
    output = gr.Gallery(label="Results", columns=3, height="auto")

    search_button.click(fn=search, inputs=[text_input, image_input], outputs=output)

demo.launch()
