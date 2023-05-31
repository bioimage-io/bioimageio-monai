import tempfile
import zipfile
import os
import numpy as np
from skimage import io
import nibabel as nib


GOOD = '\033[92mGOOD\033[0m'
WARNING = '\033[93mWARNING\033[0m'
ERROR = '\033[91mERROR\033[0m'

def zip_extracor(zip_file_path):
    file = zipfile.ZipFile(zip_file_path)

    tmpdir = tempfile.mkdtemp()

    file.extractall(path=tmpdir)

    root_path = os.path.join(tmpdir, os.listdir(tmpdir)[0]) if len(os.listdir(tmpdir)) == 1 else tmpdir

    with open(os.path.join(root_path, 'run_conversion.py'), 'w') as f:
        f.write('import sys\n')
        f.write(f'sys.path.append(\'{os.path.abspath(".")}\')\n')
        f.write('from monai_to_bmz import run_monai_to_bmz\n')
        f.write('run_monai_to_bmz()\n')

    return root_path

def read_input_img(input_img_path):
    extension = os.path.splitext(input_img_path)[1]

    if not os.path.exists(input_img_path):
        print(ERROR + ' - Given image path is bad.')

    if extension == '.npy':
        img = np.load(input_img_path, allow_pickle=True)
    elif extension == '.gz':
        nii_img = nib.load(input_img_path)
        img = nii_img.get_fdata()
    else:
        img = io.imread(input_img_path)

    return img

def obtain_metadata(inference_parser, metadata_parser, metadata_keys):
    # Load the metadata from the network
    net_name = inference_parser.get_parsed_content('network_def#_target_')
    name = metadata_parser.get_parsed_content('name')
    Trained_model_name = '{}-{}'.format(net_name, name)
    Trained_model_description = metadata_parser.get_parsed_content('description')
    Trained_model_license = metadata_parser.get_parsed_content('copyright')
    
    # Load the authors input spec
    Trained_model_authors = metadata_parser.get_parsed_content('authors')
    #Trained_model_authors_affiliation = ['']*len(Trained_model_authors)
    auth_names = Trained_model_authors.split(",")
    auth_affs = ['']*len(auth_names) # Trained_model_authors_affiliation.split(",")
    assert len(auth_names) == len(auth_affs)
    authors = [{"name": auth_name, "affiliation": auth_aff} for auth_name, auth_aff in zip(auth_names, auth_affs)]

    # Load the citation input spec
    Trained_model_references = metadata_parser.get_parsed_content('references') if 'references' in metadata_keys else []
    Trained_model_DOI = ['']*len(Trained_model_references)
    assert len(Trained_model_DOI) == len(Trained_model_references)
    citations = [{'text': text, 'doi': doi} for text, doi in zip(Trained_model_references, Trained_model_DOI)]

    # If no training_data is provided, please provide the URL tot he data and a short description
    training_data_source = metadata_parser.get_parsed_content('data_source') if 'data_source' in metadata_keys else ''
    training_data_description = metadata_parser.get_parsed_content('data_type') if 'data_type' in metadata_keys else ''

    return Trained_model_name, Trained_model_description, authors, citations

def select_input_axes(num_axes):
    if num_axes == 5:
        return ['bzyxc']
    elif num_axes == 4:
        return ['bcyx']
    elif num_axes == 3:
        return ['byx']
    elif num_axes == 2:
        return ['bc']

def select_output_axes(num_axes):
    if num_axes == 5:
        return ['bcyxz']
    elif num_axes == 4:
        return ['bcyx']
    elif num_axes == 3:
        return ['byx']
    elif num_axes == 2:
        return ['bc']