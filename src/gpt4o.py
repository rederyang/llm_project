from openai import OpenAI
import json
from tqdm import tqdm
from boolean import BooleanAlgebra
from pyeda.inter import expr

algebra = BooleanAlgebra()

client = OpenAI(api_key="your/api/key")

prompt = """Given the circuit diagram below, write the corresponding Boolean expression. Format your expression in standard Boolean notation. For example: ~ (B & A) | (B | A). Make sure to use only the operators &, |, ~, and parentheses. You do not need to simplify the expression. Begin your final answer with the keyword “ANSWER:”. 
"""
num_samples = 240
responses = []

def map_boolean_operators(expr_str):
    # expr_str = expr_str.lower()
    expr_str = expr_str.replace("not", "~")
    expr_str = expr_str.replace("and", "&")
    expr_str = expr_str.replace("xor", "^")
    expr_str = expr_str.replace("or", "|")
    return expr_str

correct = 0

for i in tqdm(range(num_samples)):
    circuit_id = f"circuit_{i + 1}"
    with open(f"data/{circuit_id}.json", 'r') as file:
        data = json.load(file)
    gt = data["expression"]
    gt = map_boolean_operators(gt)
    gt_expr = expr(gt)

    success = False
    while not success:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"https://huggingface.co/datasets/user3984/circuit-cot/resolve/main/{circuit_id}.png",
                                }
                            },
                        ],
                    }
                ],
            )
            response = completion.choices[0].message.content
            answer = response.split("ANSWER:")[-1]
            answer_expr = expr(answer)
            success = True
        except:
            print(answer)
            print("An error occurred. Regenerating")

    equivalent = gt_expr.equivalent(answer_expr)
    if equivalent:
        correct += 1
    
    responses.append({"circuit_id": circuit_id,
                      "response": response,
                      "answer": answer,
                      "gt": gt,
                      'equivalent': equivalent})

with open("gpt4o_responses.json", "w") as file:
    json.dump(responses, file, indent=4)

print(f"Accuracy: {correct / num_samples}")
