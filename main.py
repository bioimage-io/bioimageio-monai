import os
import subprocess
import argparse

import utils
import monai_to_bmz

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('image_path')
    parser.add_argument('saving_path')
    args = parser.parse_args()

    root_path = args.model_path
    input_img_path = args.image_path
    saving_path = args.saving_path

    # Read the input image
    if input_img_path != "":
        img = utils.read_input_img(input_img_path)
        if len(img.shape) > 0:
            print(utils.GOOD + ' - Input image loaded.')
            monai_to_bmz.convert_model(root_path, img, saving_path)
        else:
            print(utils.ERROR + ' - Image not loaded.')


if __name__ == '__main__':
    # The inputs that the model should receive
    zip_file_path = '/home/cocomputer/Documentos/AI4LIFE/MONAI/MONAI_zip_models/brats_mri_segmentation_v0.4.3.zip'
    input_img_path = '/home/cocomputer/Documentos/AI4LIFE/MONAI/MONAI_preprocess_data/brats_mri_segmentation/Brats18_CBICA_AAM_1/input_img.tif'
    saving_path = './result'

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('image_path')
    parser.add_argument('saving_path')
    args = parser.parse_args()

    zip_file_path = args.model_path
    input_img_path = args.image_path
    saving_path = args.saving_path
    '''

    # Create a folder where the BMZ format moodel will be saved
    os.makedirs(saving_path, exist_ok=True)

    # Extract the model in a tmp folder
    root_path = utils.zip_extracor(zip_file_path)

    print(f'Temporal folder: {root_path}')

    subprocess.run(['python', root_path+'/run_conversion.py', root_path, input_img_path, saving_path])


    