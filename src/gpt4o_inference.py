#%%
import os
import json
import glob
import base64
from mimetypes import guess_type

from tqdm import tqdm
from openai import OpenAI
# from pyeda.inter import expr
expr = lambda x: x


def map_boolean_operators(expr_str):
    # expr_str = expr_str.lower()
    expr_str = expr_str.replace("not", "~")
    expr_str = expr_str.replace("and", "&")
    expr_str = expr_str.replace("xor", "^")
    expr_str = expr_str.replace("or", "|")
    return expr_str

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

#%%
data_dir = "../data/circuit2/circuits_output2_bbox"
png_file_paths = glob.glob(os.path.join(data_dir, "*.png"))

client = OpenAI(api_key="")

prompt = """Given the circuit diagram below, write the corresponding Boolean expression. Format your expression in standard Boolean notation. For example: ~ (B & A) | (B | A). Make sure to use only the operators &, |, ~, and parentheses. You do not need to simplify the expression. Begin your final answer with the keyword “ANSWER:”. """

#%%
responses = []
correct = 0
for png_file_path in tqdm(png_file_paths):
    circuit_id = png_file_path.split('/')[-1].split('.')[-2]

    # read meta data
    with open(os.path.join(data_dir, f"{circuit_id}.json"), 'r') as file:
        data = json.load(file)
    gt = data["expression"]
    gt = map_boolean_operators(gt)
    gt_expr = expr(gt)

    # convert to base64 string
    base64_string = local_image_to_data_url(png_file_path) 
    
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
                                    "url": base64_string,
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

    equivalent = False
    # equivalent = gt_expr.equivalent(answer_expr)
    # if equivalent:
    #     correct += 1
    
    responses.append({"circuit_id": circuit_id,
                      "response": response,
                      "answer": answer,
                      "gt": gt,
                      'equivalent': equivalent})

with open("gpt4o_responses.json", "w") as file:
    json.dump(responses, file, indent=4)

print(f"Accuracy: {correct / len(responses)}")

# %%
