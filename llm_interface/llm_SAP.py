import requests
import json
import re
import ast


def LLM_SAP(prompts_list, llm='GPT', key='', llm_model=None):
    if isinstance(prompts_list, str):
        prompts_list = [prompts_list]
    if llm == 'Zephyr':
        result = LLM_SAP_batch_Zephyr(prompts_list, llm_model)
    elif llm == 'GPT':
        result = LLM_SAP_batch_gpt(prompts_list, key)
    return result

# Load the Zephyr model once and reuse it
def load_Zephyr_pipeline():
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    model_id = "HuggingFaceH4/zephyr-7b-beta"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Zephyr prefers specific generation parameters to stay aligned
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=512,  # you can tune this
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    return pipe
    

def LLM_SAP_batch_Zephyr(prompts_list, llm_model):
    print("### run LLM_SAP_batch with zephyr-7b-beta###")

    # Load templates
    with open('llm_interface/template/template_SAP_system_short.txt', 'r') as f:
        template_system = ' '.join(f.readlines())

    with open('llm_interface/template/template_SAP_user.txt', 'r') as f:
        template_user = ' '.join(f.readlines())

    numbered_prompts = [f"### Input {i + 1}: {p}\n### Output:" for i, p in enumerate(prompts_list)]
    prompt_user = template_user + "\n\n" + "\n\n".join(numbered_prompts)
    full_prompt = template_system + "\n\n" + prompt_user

    # Load Zephyr
    if llm_model is None:
        pipe = load_Zephyr_pipeline()
    else: 
        pipe = llm_model

    # zephyr
    # Run inference
    output = pipe(
        full_prompt,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        return_full_text=False
    )[0]["generated_text"]
    
    # Parse output
    print(f"output: {output}")
    parsed_outputs = parse_batched_llm_output(output, prompts_list)
    return parsed_outputs

def LLM_SAP_batch_gpt(prompts_list, key):
    print("### run LLM_SAP_batch with gpt-4o ###")

    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    with open('llm_interface/template/template_SAP_system.txt', 'r') as f:
        template_system=f.readlines()
        prompt_system=' '.join(template_system)

    with open('llm_interface/template/template_SAP_user.txt', 'r') as f:
        template_user=f.readlines()
        template_user=' '.join(template_user)

    numbered_prompts = [f"### Input {i + 1}: {p}\n### Output:" for i, p in enumerate(prompts_list)]
    prompt_user = template_user + "\n\n" + "\n\n".join(numbered_prompts)
    payload = json.dumps({
    "model": "gpt-4o", 
    "messages": [
        {
            "role": "system",
            "content": prompt_system
        },
        {
            "role": "user",
            "content": prompt_user
        }
    ]
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    
    text=obj['choices'][0]['message']['content']
    print(f"text: {text}")
    parsed_outputs = parse_batched_llm_output(text, prompts_list)

    return parsed_outputs


def parse_batched_llm_output(llm_output_text, original_prompts):
    """
    llm_output_text: raw string returned by the llm for multiple prompts
    original_prompts: list of the multiple original input strings
    """
    outputs = re.split(r"### Input \d+: ", llm_output_text)
    results = []

    for i in range(len(original_prompts)):
        out = outputs[i]
        cleaned = out.strip()
        print(f"original_prompts: {original_prompts}")
        try:
            result = get_params_dict_SAP(cleaned)
            results.append(result)
        except Exception as e:
            print(f"Failed to parse prompt {i+1}: {e}")
            results.append(None)
    return results


def get_params_dict_SAP(response):
    """
    Parses the LLM output from SAP-style few-shot prompts.
    Cleans up Markdown-style code fences and returns a dict.
    """
    try:
        # Extract explanation
        explanation = response.split("a. Explanation:")[1].split("b. Final dictionary:")[0].strip()

        # Extract and clean dictionary string
        dict_block = response.split("b. Final dictionary:")[1].strip()

        # Remove ```python and ``` if present
        # dict_str = re.sub(r"```(?:python)?", "", dict_block).replace("```", "").strip()
        dict_str = re.sub(r"```[^\n]*\n?", "", dict_block).replace("```", "").strip()

        # Parse dictionary safely
        final_dict = ast.literal_eval(dict_str)

        return {
            "explanation": explanation,
            "prompts_list": final_dict["prompts_list"],
            "switch_prompts_steps": final_dict["switch_prompts_steps"]
        }

    except Exception as e:
        print("Parsing failed:", e)
        return None