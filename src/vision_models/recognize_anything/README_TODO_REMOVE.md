# Recognize anything (RAM++)

https://github.com/xinyu1205/recognize-anything

## Installation

Install RAM++

```bash
$ pip install git+https://github.com/xinyu1205/recognize-anything.git
```

Download checkpoint and place it into the `checkpoints` directory 

https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth

To use OpenAI's API to generate objects descriptions:

```bash
$ pip install openai
```

You also need to set the environment variable `OPENAI_API_KEY` to your API key.

## Usage

First, generate a `descriptions.json` file. Edit the `objects` list in `generate_objects_descriptions.py`, then run 
```bash
$ python3 generate_objects_descriptions.py
```

Then run the example script, for example

```bash
$ python3 infer_live.py --config CONFIG_OAK_D_PRO --objects_descriptions descriptions.json
```









