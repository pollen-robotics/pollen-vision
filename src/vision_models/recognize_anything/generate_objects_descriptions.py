"""Utility script to generate descriptions for a list of objects using OpenAI's GPT-3.5 model."""

import json

from openai import OpenAI  # Needs to be installed with `pip install openai`

# Change the list of objects here
objects = [
    "mug",
    "bottle",
    "fork",
    "drawer",
    "laptop",
    "chair",
    "lamp",
    "sponge",
    "sink",
    "dishwasher",
    "oven",
    "phone",
    "robot",
]
client = OpenAI()

prompt = """
Generate 10 descriptions for each of the following objects :
{}
""".format(
    "\n".join(objects)
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": """
            You are a helpful assistant designed to output JSON.
            You will return a json object of the form :
            [{'object': ['description 1', 'description 2', ...]}, {'object2': ['description 1', 'description 2', ...]}, ...]
            """,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
)

json_output = json.loads(str(completion.choices[0].message.content))
final_output = []
for obj_name, descriptions in json_output.items():
    final_output.append({obj_name: descriptions})


json.dump(final_output, open("objects_descriptions.json", "w"))
print(final_output)
