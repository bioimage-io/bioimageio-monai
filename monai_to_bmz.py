from bioimageio.core.build_spec import build_model
from bioimageio.core.resource_tests import test_model
import traceback
import os

import torch
import numpy as np

GOOD = '\033[92mGOOD\033[0m'
WARNING = '\033[93mWARNING\033[0m'
ERROR = '\033[91mERROR\033[0m'

from transformations import monai_to_bmz_preprocessing, monai_to_bmz_postprocessing, convert_to_tensor
from loaders import load_weights_path, load_metadata_parser, load_inference_parser
from utils import obtain_metadata, select_input_axes, select_output_axes

#####
# Main function - convierte MONAI model a BMZ format 
#####

def convert_model(model_path, input_img, bioimage_save_path):
        
    # First, the model will be loaded with the weights and the prediction of the input image will be done

    # Load the parsers for both inferece and metadata files
    inference_parser, inference_keys = load_inference_parser(model_path)
    metadata_parser, metadata_keys = load_metadata_parser(model_path)
     
    # Change the bundle root to be where the model is located and executed
    inference_parser['bundle_root'] = model_path

    # Load the dive (cuda or cpu)
    device = inference_parser.get_parsed_content('device', instantiate=True)

    # Load the name of the model
    name = metadata_parser.get_parsed_content('name')

    # Load the network
    try:
        network_def = inference_parser.get_parsed_content('network_def', instantiate=True)
        print('\033[92mGOOD\033[0m - Good loading network_def')
    except Exception as e:
        raise RuntimeError(ERROR + ' - Loading network_def gives error')

    # Load the weights that will be uploaded to the model
    model_original_weights_path = load_weights_path(inference_parser, inference_keys)

    # Load the weights into the model
    network = network_def.to(device)
    model_weigths = torch.load(model_original_weights_path, map_location=device)
    
    if name == 'Whole brain large UNEST segmentation':
        # Special case where the model weights are saved in a dictionary
        network.load_state_dict(model_weigths['model'])
    else:
        network.load_state_dict(model_weigths)
    network.eval()
    
    #####
    # Preprocessing
    #####
    
    given_image = input_img[None,...] # As only one image is given, we create a batch size of 1
    preprocessed_img = given_image[0] # and to process we take the image (we want the batch size for the model building)
    
    print('Read image shape: {}'.format(preprocessed_img.shape))

    if 'preprocessing' in inference_keys:
        # Load the preprocessing functions
        bmz_preprocessing = monai_to_bmz_preprocessing(inference_parser, name)
        
        for preproc_funct in bmz_preprocessing:
            preprocessed_img = preproc_funct(preprocessed_img)
            print('\t{}'.format(preproc_funct))
            print('\tPreprocessed image: {} - {}'.format(preprocessed_img.shape, preprocessed_img.dtype))
    else:
        preprocessed_img = convert_to_tensor(preprocessed_img, **{'dtype': torch.float32})

    print('Input image shape: {} - {}'.format(preprocessed_img.shape, preprocessed_img.dtype))

    #####
    # Apply the prediction of the model
    #####
    try:
        with torch.no_grad():
            prediction = network(preprocessed_img.to(device))
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(ERROR + ' - Prediction was done bad')

    print('I prediction a dict? {}'.format(type(prediction) is dict))
    print('Prediction shape: {}'.format(prediction.shape))

    #####
    # Postprocessing
    #####
    
    if 'postprocessing' in inference_keys:
        monai_postprocessing = inference_parser['postprocessing']['transforms']
        bmz_postprocessing = monai_to_bmz_postprocessing(monai_postprocessing, name)
        
        # Postprocess the prediction    
        postprocessed_img = prediction
        for postcproc_funct in bmz_postprocessing:
            postprocessed_img = postcproc_funct(postprocessed_img)
            print('\t{}'.format(preproc_funct))
            print('Preprocessed image: {} - {}'.format(postprocessed_img.shape, postprocessed_img.dtype))
    else:
        postprocessed_img = prediction

    print('Postprocessed image shape: {}'.format(postprocessed_img.shape))
    single_input = given_image
    single_prediction = postprocessed_img
    print('Single input shape: {}'.format(single_input.shape))
    print('Single prediction shape: {}'.format(single_prediction.shape))

    ###
    # Now, metadata from the model will be created
    ###
    
    Trained_model_name, Trained_model_description, authors, citations = obtain_metadata(inference_parser, metadata_parser, metadata_keys)

    ###
    # Create the output folder and a markdown README file
    ###

    output_root = os.path.join(bioimage_save_path, Trained_model_name + '.bioimage.io.model')
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{Trained_model_name}.zip")

    # Create a markdown readme with information
    readme_path = os.path.join(output_root, "README.md")
    with open(readme_path, "w") as f:
        f.write("Visit https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki")
    
    ###
    # The input and output numpy files will be created and saved
    ###

    test_img = np.copy(single_input)
    test_in_path = os.path.join(output_root, "test_input.npy")
    np.save(test_in_path, test_img)
    
    test_prediction = np.copy(single_prediction)
    test_out_path = os.path.join(output_root, "test_output.npy")
    np.save(test_out_path, test_prediction)

    # The model will be saved in a TorchScript format
    weight_path = os.path.join(output_root, "model.pt")
    model_scripted = torch.jit.script(network) # Export to TorchScript
    model_scripted.save(weight_path) # Save

    print('test_img.shape: {}'.format(test_img.shape))
    print('test_prediction.shape: {}'.format(test_prediction.shape))
    
    ###
    # export the model with Pytorch weihgts
    ###

    try:
        build_model(
            weight_uri=weight_path,
            test_inputs=[test_in_path],
            test_outputs=[test_out_path],
            input_axes=select_input_axes(len(test_img.shape)),
            output_axes=select_output_axes(len(test_prediction.shape)),
            covers=['/home/cocomputer/Documentos/AI4LIFE/MONAI/cover.png'],
            name=Trained_model_name,
            description=Trained_model_description,
            authors=authors,
            attachments=None,
            tags=['electron-microscopy', 'pytorch', 'brain'],
            license= 'CC-BY-4.0',#license,
            documentation=readme_path,
            cite=citations,
            output_path=output_path,
            weight_type='torchscript',
            pytorch_version=torch.__version__,
            add_deepimagej_config=False, # Because if not sigmoid cannot be used
            preprocessing=None,
            postprocessing=None
        )
        print('{} - Model has been built'.format(GOOD))  

        f = open("model_passed.txt", "a")
        f.write(Trained_model_name + '\n')
        f.close() 
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(ERROR + ' - Building of the model was done bad')
    
    res = test_model(output_path)

    if res[-1]['error'] is not None:
        raise RuntimeError('{} - {}'.format(ERROR, res[-1]['error']))
    else:
        print('{} - Model has been succesfully saved in: {}'.format(GOOD, output_path))

def run_monai_to_bmz():

    import nibabel as nib
    import argparse
    from skimage import io
    from monai.transforms.compose import Compose
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('image_path')
    parser.add_argument('saving_path')

    args = parser.parse_args()
    extension = os.path.splitext(args.image_path)[1]
    
    if args.image_path != "":
        if not os.path.exists(args.image_path):
            print(ERROR + ' - Given image path is bad.')

        if extension == '.npy':
            img = np.load(args.image_path, allow_pickle=True)
        elif extension == '.gz':
            nii_img = nib.load(args.image_path)
            img = nii_img.get_fdata()
        else:
            img = io.imread(args.image_path)

        if len(img.shape) > 0:
            print(GOOD + ' - Input image loaded.')
            convert_model(args.model_path, img, args.saving_path)
        else:
            print(ERROR + ' - Image not loaded.')


if __name__ == '__main__':
    
    run_monai_to_bmz()