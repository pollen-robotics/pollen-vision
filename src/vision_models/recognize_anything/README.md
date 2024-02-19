# Recognize anything (RAM++) Wrapper

https://github.com/xinyu1205/recognize-anything

## Usage

To use OpenAI's API to generate objects descriptions:

```bash
$ pip install openai
```

You also need to set the environment variable `OPENAI_API_KEY` to your API key.


First, generate a `descriptions.json` file. Edit the `objects` list in `generate_objects_descriptions.py`, then run 
```bash
$ python3 generate_objects_descriptions.py
```

Then run the example script, for example

```bash
$ python3 infer_live.py --config CONFIG_OAK_D_PRO --objects_descriptions descriptions.json
```









