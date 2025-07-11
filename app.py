from __future__ import annotations

import gradio as gr
import spaces
from PIL import Image
import torch
from run_SAP_flux import parse_input_arguments, LLM_SAP, generate_models_params, load_model
from llm_interface.llm_SAP import load_Zephyr_pipeline
import re

gr.HTML("""
<style>
#result-column {
    display: flex;
    align-items: center;
    justify-content: center;
    height: auto;
    min-height: 512px;
}

#result-image {
    aspect-ratio: 1 / 1;
    max-width: 100%;
    height: auto;
    object-fit: contain;
    border: 1px solid #ccc;
    border-radius: 8px;
    background-color: #f8f8f8;
}
#flux-output-img img,
#sap-output-img img {
    width: 384px;
    height: 384px;
    object-fit: contain;
    border: 1px solid #ccc;
    border-radius: 8px;
    background-color: #f8f8f8;
    display: block;
    margin: auto;
}
</style>
""")


DESCRIPTION = '''# Image Generation from Contextually-Contradictory Prompts
This demo accompanies our [paper](https://tdpc2025.github.io/SAP/) on **Image Generation from Contextually-Contradictory Prompts**. The source code is available on [GitHub](https://github.com/TDPC2025/SAP). 
Our **SAP (Stage Aware Prompting)** method supports multiple diffusion models and can be paired with various large language models (LLMs). This interface allows you to generate images using:

- **FLUX.dev**: Baseline image generation using the unmodified FLUX model.
- **SAP with zephyr-7b-beta**: SAP applied to FLUX with zephyr-7b-beta as the LLM.
- **SAP with GPT-4o**: SAP applied to FLUX with GPT-4o as the LLM *(requires an OpenAI API key)*.

For best results, we recommend using **SAP with GPT-4o**, which delivers the best implementation of our method.

**Note:** When using **SAP with zephyr-7b-beta**, the model may take a few seconds to load on the first run, as the LLM is initialized. Subsequent generations will be faster.
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_cache = {}
llm_cache = {}

def toggle_api_visibility(choice):
    return gr.update(visible=(choice == "SAP with GPT-4o"))

@spaces.GPU
def main_pipeline(
    prompt: str,
    seed: int,
    model_choice: str,
    api_key: str):
    
    res_image = run_demo(prompt, seed, model_choice, api_key)

    return res_image

# Function to load pregenerated SAP-GPT image
def load_static_result(path):
    import os
    if not os.path.isfile(path):
        # fallback if current dir is different — try relative to script
        path = os.path.join(os.path.dirname(__file__), path)
    return Image.open(path)

def on_example_select(row):
    if row is None or len(row) < 2:
        return None
    return load_static_result(row[1])

def handle_dataset_selection(index):
    try:
        row = example_data[index]
        print(f"row: {row}")
        image = load_static_result(row["img"])
        return image, row["prompt"]
    except Exception as e:
        print(f"Error: {e}")
        return None, ""

def handle_example_compare(index):
    try:
        row = example_data[index]
        flux_image = load_static_result(row["flux_img"])
        sap_image = load_static_result(row["sap_img"])
        return flux_image, sap_image
    except Exception as e:
        print(f"Error loading images for index {index}: {e}")
        return None, None


def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]+', '_', text.lower()).strip('_')

@torch.inference_mode()
def run_demo(prompt, seed, model_choice=None, api_key="API_KEY"):
    # Align CLI args
    args = parse_input_arguments()
    args.prompt = prompt
    args.seeds_list = [seed]

    # ------------------------------
    # FLUX MODE: No LLM, just base model
    # ------------------------------
    if model_choice == 'FLUX':
        SAP_prompts = {"prompts_list": [prompt], "switch_prompts_steps": []}    
    # ------------------------------
    # SAP MODE: LLM + Prompt Decomposition
    # ------------------------------
    else:
        # Decide on which LLM to use
        llm_type = 'Zephyr' if "SAP with zephyr-7b-beta" in model_choice else 'GPT'

        # Load or cache LLM (optional but smart if it's large)
        if llm_type == 'Zephyr':
            if llm_type not in llm_cache:
                llm_cache[llm_type] = load_Zephyr_pipeline()
            llm_model = llm_cache[llm_type]
        else:
            llm_model = None

        # Prompt decomposition
        SAP_prompts = LLM_SAP(prompt, llm=llm_type, key=api_key, llm_model=llm_model)[0]

    # Load SAPFlux
    if "SAPFlux" not in model_cache:
        model_cache["SAPFlux"] = load_model()
    model = model_cache["SAPFlux"]

    # Generate model params with decomposed prompts
    params = generate_models_params(args, SAP_prompts)

    # ------------------------------
    # Run the model
    # ------------------------------
    image = model(**params).images[0]
    return image

def warmup_models():
    print("Background warmup started...")

    if "SAPFlux" not in model_cache:
        print("Loading SAPFlux model...")
        model_cache["SAPFlux"] = load_model()

        model = model_cache["SAPFlux"]
        try:
            _ = model(
                sap_prompts={"prompts_list": ["A robot walking a dog"], "switch_prompts_steps": []},
                height=512,
                width=512,
                num_inference_steps=3,
                guidance_scale=3.5,
                generator=[torch.Generator().manual_seed(42)],
                num_images_per_prompt=1
            )
            print("SAPFlux warmup complete.")
        except Exception as e:
            print(f"Warmup error: {e}")

    # Mark warmup done
    return gr.update(interactive=True), True, gr.update(value="✅ Ready!")

with gr.Blocks(css='app/style.css') as demo:
    warmup_done = gr.State(value=False)

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():

            model_choice = gr.Radio(
                choices=["FLUX", "SAP with zephyr-7b-beta", "SAP with GPT-4o"],
                label="Model Selection",
                value="FLUX"
            )

            api_key = gr.Textbox(
                label="OpenAI API Key (for GPT)",
                placeholder="sk-...",
                visible=False
            )

            model_choice.change(
                fn=toggle_api_visibility,
                inputs=model_choice,
                outputs=api_key)

            prompt = gr.Text(
                label='Prompt',
                max_lines=1,
                placeholder='A bear is performing a handstand in the park',
            )

            seed = gr.Slider(
                label='Seed',
                minimum=0,
                maximum=16*1024,
                value=30498,
                step=1
            )
     
            # run_button = gr.Button('Generate')
            run_button = gr.Button('Generate', interactive=False)
            status_text = gr.Markdown("🚀 Loading models... Please wait.")
        with gr.Column(scale=1, elem_id="result-column"):
            # result = gr.Gallery(label='Result')
            result = gr.Image(
                label="Result",
                type="pil",
                elem_id="result-image"
            )
    with gr.Row():
        gr.Markdown("### ✨ SAP + GPT-4o Examples")
    with gr.Row():

        example_data = [
            {
                "prompt": "A camping tent is inside a bedroom.",
                "flux_img": "images/flux_tent.jpg",
                "sap_img": "images/sap_tent.jpg"
            },
            {
                "prompt": "An eagle is swimming under-water.",
                "flux_img": "images/flux_eagle.jpg",
                "sap_img": "images/sap_eagle.jpg"
            },
            {
                "prompt": "Shrek is blue.",
                "flux_img": "images/flux_shrek.jpg",
                "sap_img": "images/sap_shrek.jpg"
            },
            {
                "prompt": "A man giving a piggyback ride to an elephant.",
                "flux_img": "images/flux_elephant.jpg",
                "sap_img": "images/sap_elephant.jpg"
            },
            {
                "prompt": "A knight in chess is a unicorn.",
                "flux_img": "images/flux_chess.jpg",
                "sap_img": "images/sap_chess.jpg"
            },
            {
                "prompt": "A bear is perfroming a handstand in the park.",
                "flux_img": "images/flux_bear.jpg",
                "sap_img": "images/sap_bear.jpg"
            },
            ]

        flux_out = gr.Image(
            label="FLUX Output",
            type="pil",
            elem_id="flux-output-img"
        )
        sap_out = gr.Image(
            label="SAP + GPT-4o Output",
            type="pil",
            elem_id="sap-output-img"
        )
    # --- Spacer ---
    
    gr.Markdown("Click a row to compare FLUX vs SAP")

    # --- Dataset Table ---
    dataset = gr.Dataset(
        components=[
            gr.Textbox(visible=False),  # prompt (optional)
            gr.Image(type="filepath", height=64, width=64, visible=False),
            gr.Image(type="filepath", height=64, width=64, visible=False)
        ],
        headers=["Prompt", "FLUX Preview", "SAP Preview"],
        samples=[
            [ex["prompt"], ex["flux_img"], ex["sap_img"]] for ex in example_data
        ],
        type="index",
        label=None
    )

    # --- Logic: Load outputs on click ---
    dataset.select(
        fn=handle_example_compare,
        inputs=[dataset],
        outputs=[flux_out, sap_out]
    )
            
    
    inputs = [
        prompt,
        seed,
        model_choice,
        api_key
    ]
    outputs = [
        result
    ]
    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)
    
    demo.load(fn=warmup_models, inputs=[], outputs=[run_button, warmup_done, status_text])


demo.queue(max_size=50)