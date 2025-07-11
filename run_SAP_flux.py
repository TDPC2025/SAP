import os
import torch
import argparse
from pathlib import Path
from SAP_pipeline_flux import SapFlux
from llm_interface.llm_SAP import LLM_SAP
BASE_FOLDER = os.getcwd()

################################
API_KEY = "YOUR_API_KEY"
################################

def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=1024, help="define the generated image height")
    parser.add_argument('--width', type=int, default=1024, help="define the generated image width")
    parser.add_argument('--seeds_list', nargs='+', type=int, default=[30498], help="define the list of seeds for the prompt generated images")
    parser.add_argument('--prompt', type=str, default="A bear is performing a handstand in the park")
    parser.add_argument('--llm', type=str, default="GPT", help="define the llm to be used, support GPT and Zephyr")
    args = parser.parse_args()
    return args

def load_model():
    model = SapFlux.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    model.enable_model_cpu_offload()
    return model

def save_results(images, prompt, seeds_list):
    prompt_model_path = os.path.join(BASE_FOLDER, "results", prompt)
    Path(prompt_model_path).mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(seeds_list):
        images[i].save(os.path.join(prompt_model_path, f"Seed{seed}.png"))

def generate_models_params(args, SAP_prompts):
    generators_lst = []
    for seed in args.seeds_list:
        generator = torch.Generator()
        generator.manual_seed(seed)
        generators_lst.append(generator)
    params = {"height": args.height, 
              "width": args.width,
              "num_inference_steps": 50,
              "generator": generators_lst,
              "num_images_per_prompt": len(generators_lst),
              "guidance_scale": 3.5, 
              "sap_prompts": SAP_prompts}
    return params

def run(args):
    # generate prompt decomposition
    SAP_prompts = LLM_SAP(args.prompt, llm=args.llm, key=API_KEY)[0] # using [0] because of a single prompt decomposition
    params = generate_models_params(args, SAP_prompts)
    # Load model
    model = load_model()
    # Run model
    images = model(**params).images
    # Save results
    save_results(images, args.prompt, args.seeds_list)

def main():
    args = parse_input_arguments()
    # pass update args with defualts
    run(args)
    
if __name__ == "__main__":
    main()