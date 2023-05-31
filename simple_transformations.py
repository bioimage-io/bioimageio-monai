import numpy as np
import torch

#####
# TRANSFORMATIONS
#####

def convert_to_tensor(img, dtype=None):
    if img.dtype == np.uint16:
        img = img.astype(np.uint8)
    return torch.as_tensor(img, dtype=dtype)

# Already done ZeroMeanUnitVariance
class NormalizeIntensity:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.nonzero = kwargs['nonzero'] if 'nonzero' in kwargs else None
        self.channel_wise = kwargs['channel_wise'] if 'channel_wise' in kwargs else None
        self.subtrahend = kwargs['subtrahend'] if 'subtrahend' in kwargs else None
        self.divisor = kwargs['divisor'] if 'divisor' in kwargs else None

    @staticmethod
    def _mean(x):
        return np.mean(x)

    @staticmethod
    def _std(x):
        return np.std(x)
    
    def normalize(self, arr, sub=None, div=None):
        
        if self.nonzero:
            slices = arr != 0
        else:
            slices = np.ones_like(arr, dtype=bool)
        if not slices.any():
            return arr

        _sub = sub if sub is not None else self._mean(arr[slices])
        if isinstance(_sub, np.ndarray):
            _sub.astype(arr.dtype)
            _sub = _sub[slices]

        _div = div if div is not None else self._std(arr[slices])
        if np.isscalar(_div):
            if _div == 0.0:
                _div = 1.0
        elif isinstance(_div, np.ndarray):
            _div.astype(arr.dtype)
            _div = _div[slices]
            _div[_div == 0.0] = 1.0

        arr[slices] = (arr[slices] - _sub) / _div + 1.0e-6
        return arr
    
    def __call__(self, arr):
        dtype = arr.dtype
        print(arr[0, 100:110,100:110, 103])
        if self.channel_wise:
            for i, d in enumerate(arr):
                arr[1] = self.normalize(d,
                         sub=self.subtrahend[i] if self.subtrahend is not None else None,
                         div=self.divisor[i] if self.divisor is not None else None,
                )
        else:
            arr = self.normalize(arr, sub=self.subtrahend, div=self.divisor)
        print(arr[0, 100:110,100:110, 103])
        return arr.astype(dtype)

    def bmz_dict(self):
        if self.subtrahend is None and self.divisor is None:
            mode = "per_sample"
        elif self.subtrahend is not None and self.divisor is not None:
            mode = "fixed"
        
        return {"name": "zero_mean_unit_variance", "kwargs": {"mode": mode}}# , "mean": self.subtrahend,"std": self.divisor}}

def monai_to_bmz_preprocessing(inference_parser, name):

    monai_preproc_functions = {'AddChanneld': None, 
                               'AddClickSignalsd': None, 
                               'AddLabelAsGuidanced': None, 
                               'AsChannelFirstd': None,
                               'CastToTyped': None, 
                               'DiscardAddGuidanced': None, 
                               'EnsureChannelFirstd': None, 
                               'EnsureType': None, 
                               'EnsureTyped': None, 
                               'LoadImage': None,
                               'LoadImaged': None, 
                               'NormalizeIntensityd': None, #NormalizeIntensity, 
                               'Orientationd': None, 
                               'Resized': None, 
                               'ScaleIntensity': None, 
                               'ScaleIntensityRanged': None, 
                               'ScaleIntensityd': None, 
                               'Spacingd': None, 
                               'SqueezeDimd': None, 
                               'ToTensord': None, 
                               'Transpose':None}
    
    monai_preprocessing_list = []
    bmz_preprocessing_list = []

    # Read from the parser all the preprocessing transformations
    monai_transforms = inference_parser['preprocessing']['transforms']

    for i, transform_function in enumerate(monai_transforms):
        # First the name of the function are taken to select the function with the dictionary
        preproc_function_name = inference_parser.get_parsed_content(f'preprocessing#transforms#{i}#_target_')
        
        # Then the arguments of the function are taken
        function_args = transform_function
        function_args.pop('_target_')
        print(function_args)
        
        # Some of these parameters needs to be obtained from the parser to initialze them
        parsed_function_args = {}
        for k in function_args.keys():
            value = inference_parser.get_parsed_content(f'preprocessing#transforms#{i}#{k}')
            if k == 'dtype' and isinstance(value, torch.dtype):
                value = torch.tensor(0, dtype=torch.float32).numpy().dtype
            parsed_function_args[k] = value
        print(parsed_function_args)
        
        # Finally read and load both monai and bmz preprocessing functions
        monai_function = monai_preproc_functions[preproc_function_name]
        if monai_function is not None:
            initialized_monai_function = monai_function(**parsed_function_args)
            bmz_function = initialized_monai_function.bmz_dict()

            monai_preprocessing_list.append(initialized_monai_function)
            bmz_preprocessing_list.append([bmz_function])

    # In order to be readed by the model, the input will be converted to pytorch tensor with a batch axis
    monai_preprocessing_list.append(lambda x: convert_to_tensor(x[None,...], dtype=torch.float32))

    return monai_preprocessing_list, bmz_preprocessing_list

#######################################
    
def monai_to_bmz_postprocessing(inference_parser, name):

    monai_postproc_functions = {'Activationsd': None, 
                               'AffineBoxToWorldCoordinated': None, 
                               'AsDiscreted': None, 
                               'ClipBoxToImaged': None,
                               'ConvertBoxModed': None, 
                               'DeleteItemsd': None, 
                               'EnsureTyped': None, 
                               'FlattenSubKeysd': None,
                               'FromMetaTensord': None, 
                               'HoVerNetInstanceMapPostProcessingd': None, 
                               'HoVerNetNuclearTypePostProcessingd': None, 
                               'Invertd': None, 
                               'KeepLargestConnectedComponentd': None, 
                               'Lambdad': None, 
                               'SaveImaged': None, 
                               'SqueezeDimd': None, 
                               'ToNumpyd': None,
                               'ToTensord': None, 
                               'scripts.valve_landmarks.NpySaverd': None}
    
    monai_postprocessing_list = []
    bmz_postprocessing_list = []

    # Read from the parser all the postprocessing transformations
    monai_transforms = inference_parser['postprocessing']['transforms']

    for i, transform_function in enumerate(monai_transforms):
        # First the name of the function are taken to select the function with the dictionary
        postproc_function_name = inference_parser.get_parsed_content(f'postprocessing#transforms#{i}#_target_')
        print(postproc_function_name)
        
        # Then the arguments of the function are taken
        function_args = transform_function
        function_args.pop('_target_')
        print(function_args)
        
        # Some of these parameters needs to be obtained from the parser to initialze them
        parsed_function_args = {}
        for k in function_args.keys():
            value = inference_parser.get_parsed_content(f'postprocessing#transforms#{i}#{k}')
            if k == 'dtype' and isinstance(value, torch.dtype):
                value = torch.tensor(0, dtype=torch.float32).numpy().dtype
            parsed_function_args[k] = value
        
        print(parsed_function_args)
        
        # Finally read and load both monai and bmz preprocessing functions
        monai_function = monai_postproc_functions[postproc_function_name]
        if monai_function is not None:
            initialized_monai_function = monai_function(**parsed_function_args)
            bmz_function = initialized_monai_function.bmz_dict()

            monai_postprocessing_list.append(initialized_monai_function)
            bmz_postprocessing_list.append([bmz_function])

    monai_postprocessing_list.append(lambda x: x.detach().numpy())

    return monai_postprocessing_list, bmz_postprocessing_list
