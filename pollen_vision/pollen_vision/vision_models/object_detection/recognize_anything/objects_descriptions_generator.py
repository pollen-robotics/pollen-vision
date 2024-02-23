"""Objects descriptions generator using OpenAI's GPT-3.5 model.

This module provides a class to generate descriptions for a list of objects using OpenAI's GPT-3.5 model.
The descriptions are saved to a JSON file in the objects_descriptions directory.
The descriptions can then be used with the recognize_anything model (RAM) to recognize whether objects
are in a scene or not.

Usage:
    In CLI:
    python objects_descriptions_generator.py -o 'robot' 'mug' -n 10 -f objects_descriptions

    In code:
    generator = ObjectDescriptionGenerator()
    descriptions = generator.generate_descriptions(["robot", "mug"], 10)
    generator.save_descriptions(descriptions, "my_objects_descriptions")
"""
import json
import os

from openai import OpenAI


class ObjectDescriptionGenerator:
    def __init__(self, api_key: 'str | None' = None) -> None:
        """
        Args:
            - api_key: the API key to use to authenticate to the OpenAI API. If None, the API key is read from the
                OPENAI_API_KEY environment variable.
        """
        self._openai_client = OpenAI(api_key=api_key)

    def generate_descriptions(
        self,
        objects: list[str],
        generation_nb_per_object: int = 10
            ) -> dict[str, str]:
        """Returns a dictionary containing the descriptions of the objects."""

        prompt = """
            Generate {} descriptions for each of the following objects :
            {}
            """.format(
                generation_nb_per_object,
                "\n".join(objects)
            )

        completion = self._openai_client.chat.completions.create(
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
        return completion.choices[0].message.content

    def save_descriptions(self, descriptions: dict[str, str], descriptor_file_name: str) -> None:
        """Saves the descriptions to a JSON file."""
        json_output = json.loads(descriptions)
        final_output = []
        for obj_name, descriptions in json_output.items():
            final_output.append({obj_name: descriptions})

        file_path = f"{os.path.dirname(__file__)}/objects_descriptions/{descriptor_file_name}.json"

        json.dump(final_output, open(file_path, "w"))
        print(f"Descriptions saved to {file_path}")


if __name__ == "__main__":
    import argparse

    argParser = argparse.ArgumentParser(description="Generate descriptions for a list of objects using OpenAI's GPT-3.5 model.")
    argParser.add_argument(
        "-o",
        "--objects",
        nargs="+",
        required=True,
        help="List of objects to generate descriptions for. Example: --objects 'robot' 'mug'.",
    )
    argParser.add_argument(
        "-n",
        "--number",
        type=int,
        default=10,
        help="Number of descriptions to generate for each object",
    )
    argParser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Name of the file to save the descriptions to",
    )
    args = argParser.parse_args()

    print(f"Generating descriptions for {args.objects} with {args.number} descriptions per object...")
    generator = ObjectDescriptionGenerator()
    descriptions = generator.generate_descriptions(args.objects, args.number)
    print("Descriptions generated!")
    generator.save_descriptions(descriptions, args.file)
