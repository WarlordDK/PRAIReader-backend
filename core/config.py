import os
from dotenv import load_dotenv


load_dotenv()
HUGGINGFACE_HUB_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

llm_models_list = [{'id' : 1, 'model_name' : 'IlyaGusev/saiga_llama3_8b', 'dev_level' : 'hard'},
               {'id' : 2, 'model_name' : 'distilgpt2', 'dev_level' : 'light'}]

vlm_models_list = [{'id' : 1, 'model_name' : 'Salesforce/blip2-flan-t5-xl', 'dev_level' : 'light'},
                   {'id' : 2, 'model_name' : 'microsoft/Florence-2-large', 'dev_level' : 'medium'},
                   {'id' : 3, 'model_name' : 'Qwen/Qwen2-VL-7B-Instruct', 'dev_level' : 'medium'}]


def get_hf_token():
    return HUGGINGFACE_HUB_TOKEN

def get_qdrant_url():
    return QDRANT_URL

def get_qdrant_api_key():
    return QDRANT_API_KEY

def get_llm_models_list():
    return llm_models_list

def get_vlm_models_list():
    return vlm_models_list