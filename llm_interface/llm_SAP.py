import requests
import json
import re
import ast

def LLM_SAP(prompts_list, key):
    if isinstance(prompts_list, str):
        prompts_list = [prompts_list]
    result = LLM_SAP_batch(prompts_list, key)
    
    return result

def LLM_SAP_batch(prompts_list, key):
    print("### run LLM_SAP_batch ###")

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

    parsed_outputs = parse_batched_gpt_output(text, prompts_list)

    return parsed_outputs


def parse_batched_gpt_output(gpt_output_text, original_prompts):
    """
    gpt_output_text: raw string returned by GPT-4o for multiple prompts
    original_prompts: list of the multiple original input strings
    """
    outputs = re.split(r"### Input \d+: ", gpt_output_text)
    results = []

    for i, out in enumerate(outputs):
        cleaned = out.strip()
        prompt_text = original_prompts[i]
        try:
            result = get_params_dict_SAP(cleaned, prompt_text)
            results.append(result)
        except Exception as e:
            print(f"Failed to parse prompt {i+1}: {e}")
            results.append(None)
    return results


def get_params_dict_SAP(response, prompt):
    """
    Parses the LLM output from PromptFlow-style few-shot prompts.
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
            # "prompt": prompt,
            "explanation": explanation,
            "prompts_list": final_dict["prompts_list"],
            "switch_prompts_steps": final_dict["switch_prompts_steps"]
        }

    except Exception as e:
        print("Parsing failed:", e)
        return None