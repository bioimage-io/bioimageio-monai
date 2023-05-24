from monai.bundle import ConfigParser
import json
import yaml
import os

GOOD = '\033[92mGOOD\033[0m'
WARNING = '\033[93mWARNING\033[0m'
ERROR = '\033[91mERROR\033[0m'

def load_weights_path(inference_parser, inference_keys):
    
    if 'load_path' in inference_parser['handlers'][0].keys(): # handlers is always in inference_keys
        model_weights_path = inference_parser.get_parsed_content('handlers#0#load_path', instantiate=True)
    elif 'ckpt_path' in inference_keys:
        model_weights_path = inference_parser.get_parsed_content('ckpt_path', instantiate=True)
    elif 'modelname' in inference_keys:
        model_weights_path = inference_parser.get_parsed_content('modelname', instantiate=True)
    elif 'ckpt' in inference_keys:
        model_weights_path = inference_parser.get_parsed_content('ckpt', instantiate=True)
    else:
        raise RuntimeError(WARNING + ' - Does not have a path to the weights')  
        
    return model_weights_path

def load_metadata_parser(model_path):
    
    # Now lets check that metadata file exists in all of them
    metadata_parser = ConfigParser()
    if os.path.exists(os.path.join(model_path, 'configs', 'metadata.json')):
        print('Metadata: uses a json file.')
        metadata_file = 'metadata.json'
        metadata_keys = list(json.load(open(os.path.join(model_path, 'configs', metadata_file))).keys())
    elif os.path.exists(os.path.join(model_path, 'configs', 'metadata.yaml')):
        print('Metadata: uses a yaml file.')
        metadata_file = 'metadata.yaml'
        metadata_keys = list(yaml.safe_load(open(os.path.join(model_path, 'configs', metadata_file))).keys())
    else:
        raise RuntimeError(ERROR + ' - not json neither yaml file for metadata.')  

    metadata_parser.read_config(os.path.join(model_path, 'configs', metadata_file))
    return metadata_parser, metadata_keys

def load_inference_parser(model_path):
    
    inference_parser = ConfigParser()
    if os.path.exists(os.path.join(model_path, 'configs', 'inference.json')):
        print('Inference: uses a json file.')
        inference_file = 'inference.json'
        inference_keys = list(json.load(open(os.path.join(model_path, 'configs', inference_file))).keys())
    elif os.path.exists(os.path.join(model_path, 'configs', 'inference.yaml')):
        print('Inference: uses a yaml file.')
        inference_file = 'inference.yaml'
        inference_keys = list(yaml.safe_load(open(os.path.join(model_path, 'configs', inference_file))).keys())
    else:
        raise RuntimeError(ERROR + ' - not json neither yaml file for inference.')  
        
    inference_parser.read_config(os.path.join(model_path, 'configs', inference_file))
    return inference_parser, inference_keys