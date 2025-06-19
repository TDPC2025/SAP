import base64
import requests

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_image_with_gpt(image_path, prompt, key):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    # GPT PROMPT

    eval_prompt = f"""You are an assistant evaluating an image on two **independent** aspects: \
(1) how well it aligns with the meaning of a given text prompt, and \
(2) its visual quality.

The text prompt is: \"{prompt}\"

---

**PART 1: PROMPT ALIGNMENT (Semantic Fidelity)**  
Evaluate only the *meaning* conveyed by the image — ignore visual artifacts.  
Focus on:
- Are the correct objects present and depicted in a way that clearly demonstrates their intended roles and actions from the prompt?
- Does the scene illustrate the intended situation or use-case in a concrete and functional way, rather than through symbolic, metaphorical, or hybrid representation?
- If the described usage or interaction is missing or unclear, alignment should be penalized.
- Focus strictly on the presence, roles, and relationships of the described elements — not on rendering quality.


Score from 1 to 5:
5: Fully conveys the prompt's meaning with correct elements
4: Mostly accurate — main elements are correct, with minor conceptual or contextual issues
3: Main subjects are present but important attributes or actions are missing or wrong
2: Some relevant components are present, but key elements or intent are significantly misrepresented
1: Does not reflect the prompt at all

---

**PART 2: VISUAL QUALITY (Rendering Fidelity)**  
Now focus only on how the image looks visually — ignore whether it matches the prompt.  
Focus on:
- Are there rendering artifacts, distortions, or broken elements?

- Are complex areas like faces, hands, and shaped objects well-formed and visually coherent?
- Are complex areas like faces, hands, limbs, and object grips well-formed and anatomically correct?

- Is lighting, texture, and perspective consistent across the scene?
- Do elements appear physically coherent — i.e., do objects connect naturally (no floating tools, clipped limbs, or merged shapes)?
- Distortion, warping, or implausible blending of objects (e.g. melted features, fused geometry) should reduce the score.
- Unusual or surreal objects are acceptable **if** they are clearly rendered and visually deliberate.

Score from 1 to 5:
5: Clean, realistic, and fully coherent — no visible flaws
4: Mostly clean with minor visual issues or stiffness  
3: Noticeable visual flaws (e.g. broken grips, distorted anatomy), but the image is still readable  
2: Major visual issues — warped or broken key elements disrupt coherence  
1: Severe rendering failure — image appears nonsensical or corrupted

---

Respond using this format:
### ALIGNMENT SCORE: score
### ALIGNMENT EXPLANATION: explanation
### QUALITY SCORE: score
### QUALITY EXPLANATION: explanation"""

    # Getting the base64 string
    base64_image = encode_image(image_path)

    payload = {
      "model": "gpt-4o-2024-05-13",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": eval_prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 4096
    }

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    print('waiting for GPT-4 response')
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output=response.json()
    
    text=output['choices'][0]['message']['content']
    print(text)
    
    alignment_score = int(text.split("### ALIGNMENT SCORE:")[1].split("\n")[0].strip())
    alignment_explanation = text.split("### ALIGNMENT EXPLANATION:")[1].split("### QUALITY SCORE:")[0].strip()
    quality_score = int(text.split("### QUALITY SCORE:")[1].split("\n")[0].strip())
    quality_explanation = text.split("### QUALITY EXPLANATION:")[1].strip()

    output_dict =  {'alignment score': alignment_score, 
                    'alignment explanation': alignment_explanation,
                    'quality score': quality_score,
                    'quality explanation': quality_explanation}
    return output_dict