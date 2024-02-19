# Recognize anything (RAM++) Wrapper

https://github.com/xinyu1205/recognize-anything

## Usage

To use OpenAI's API to generate objects descriptions:

```bash
$ pip install openai
```

You also need to set the environment variable `OPENAI_API_KEY` to your API key.


First, generate a `descriptions.json` file. Edit the `objects` list in `generate_objects_descriptions.py`, then run 
```bash
$ python3 generate_objects_descriptions.py
```

Use the wrapper in your code :

```python

from vision_models.recognize_anything.RAM_wrapper import RAM_wrapper

ram = RAM_wrapper(
    objects_descriptions_file_path=<...>
)

predictions = ram.infer(<image>)
```



