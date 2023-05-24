import numpy as np
import torch
import nibabel as nib

#####
# TRANSFORMATIONS
#####

def convert_to_tensor(img, dtype=None):
    if img.dtype == np.uint16:
        img = img.astype(np.uint8)
    return torch.as_tensor(img, dtype=dtype)

class AddChannel:
    def _init__(self, **kwargs):
        pass

    def __call__(self, arr):
        return arr[None, ...]

# TODO
class AddLabelAsGuidance:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.source = kwargs['source'] if 'source' in kwargs else None

    def __call__(self, arr):
        d = dict(arr)
        for key in self.keys:
            image = d[key]
            label = d[self.source]

            label = label > 0
            if len(label.shape) < len(image.shape):
                label = label[None]
                
            image = np.cat([image, label.type(image.dtype)], dim=len(label.shape) - 3)
            d[key] = image
        return d
    
class AsChannelFirst:
    def __init__(self, **kwargs):
        self.channel_dim = kwargs['channel_dim'] if 'channel_dim' in kwargs else -1

    def __call__(self, arr):
        if isinstance(arr, np.ndarray):
            return np.moveaxis(arr, self.channel_dim, 0)
        elif torch.is_tensor(arr):
            return torch.movedim(arr, self.channel_dim, 0)

# TODO
class AddClickSignals:
    def __init__(self, **kwargs):
        self.image = kwargs['image'] if 'image' in kwargs else None
        self.foreground = kwargs['foreground'] if 'foreground' in kwargs else None
        self.gaussian = kwargs['gaussian'] if 'gaussian' in kwargs else None

    def __call__(self, arr):
        return arr

# Already implemented EnsureDtype
class CastToType:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else None

    def __call__(self, arr):
        return arr.astype(self.dtype)

class DiscardAddGuidance:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys']
        self.label_names = kwargs['label_names'] if 'label_names' in kwargs else []
        self.number_intensity_ch = kwargs['number_intensity_ch'] if 'number_intensity_ch' in kwargs else 1
        self.discard_probability = kwargs['discard_probability'] if 'discard_probability' in kwargs else 1.0

    def __call__(self, arr):
        if self.discard_probability >= 1.0 or np.random.choice(
            [True, False], p=[self.discard_probability, 1 - self.discard_probability]
        ):
            signal = np.zeros(
                (len(self.label_names), arr.shape[-3], arr.shape[-2], arr.shape[-1]), dtype=np.float32
            )
            if arr.shape[0] == self.number_intensity_ch + len(self.label_names):
                arr[self.number_intensity_ch :, ...] = signal
            else:
                arr = np.concatenate([arr, signal], axis=0)
        return arr

# Already implemented EnsureDtype
class EnsureType:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else None
        self.device = kwargs['device'] if 'device' in kwargs else None

    def __call__(self, arr):
        return arr.astype(self.dtype) if self.dtype is not None else arr

class EnsureChannelFirst:
    def __init__(self, **kwargs):
        self.channel_dim = kwargs['channel_dim'] if 'channel_dim' in kwargs else -1
        self.meta_key_postfix = kwargs['meta_key_postfix'] if 'meta_key_postfix' in kwargs else None

    def __call__(self, arr):
        return np.moveaxis(arr, self.channel_dim, 0)
    
class LoadImage:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.image_only = kwargs['image_only'] if 'image_only' in kwargs else None
        self.meta_key_postfix = kwargs['meta_key_postfix'] if 'meta_key_postfix' in kwargs else None
        self._disabled_ = kwargs['_disabled_'] if '_disabled_' in kwargs else None
        self.reader = kwargs['reader'] if 'reader' in kwargs else None
        self.affine_lps_to_ras = kwargs['affine_lps_to_ras'] if 'affine_lps_to_ras' in kwargs else None
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else None
        self.converter = kwargs['converter'] if 'converter' in kwargs else None

    def __call__(self, arr):
        return arr

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

        arr[slices] = (arr[slices] - _sub) / _div
        return arr

    def __call__(self, arr):
        dtype = arr.dtype

        if self.channel_wise:
            for i, d in enumerate(arr):
                arr[1] = self.normalize(d,
                         sub=self.subtrahend[i] if self.subtrahend is not None else None,
                         div=self.divisor[i] if self.divisor is not None else None,
                )
        else:
            arr = self.normalize(arr, sub=self.subtrahend, div=self.divisor)

        return arr.astype(dtype)

# TODO
class Resize:
    def __init__(self, **kwargs):
        self.spatial_size = kwargs['spatial_size'] if 'spatial_size' in kwargs else None
        self.mode = kwargs['mode'] if 'mode' in kwargs else 'area'
        self.mode = self.mode[0] if isinstance(self.mode, list) else self.mode

    def __call__(self, arr):
        arr = torch.as_tensor(arr)
        interpolated =  torch.nn.functional.interpolate(input=arr.unsqueeze(0), size=self.spatial_size, mode=self.mode)[0]
        return interpolated.numpy()
 
class ScaleIntensity:
    def __init__(self, **kwargs):
        self.minv = kwargs['minv'] if 'minv' in kwargs else None
        self.maxv = kwargs['maxv'] if 'maxv' in kwargs else None

    def __call__(self, arr):
        mina = arr.min()
        maxa = arr.max()

        if mina == maxa:
            return arr * self.minv if self.minv is not None else arr

        norm = (arr - mina) / (maxa - mina)  # normalize the array first
        if (self.minv is None) or (self.maxv is None):
            return norm
        
        return (norm * (self.maxv - self.minv)) + self.minv  # rescale by minv and maxv, which is the normalized array by default

# TODO
class Spacing:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.pixdim = kwargs['pixdim'] if 'pixdim' in kwargs else None
        self._disabled_ = kwargs['_disabled_'] if '_disabled_' in kwargs else None
        self.mode = kwargs['mode'] if 'mode' in kwargs else None

    def __call__(self, arr):
        return arr

class SqueezeDim:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.dim = kwargs['dim'] if 'dim' in kwargs else 0

    def __call__(self, arr):
        dim = (self.dim + len(arr.shape)) if self.dim < 0 else self.dim
        return np.squeeze(arr, dim)
    
class ScaleIntensityRange:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.a_min = kwargs['a_min'] if 'a_min' in kwargs else None
        self.a_max = kwargs['a_max'] if 'a_max' in kwargs else None
        self.b_min = kwargs['b_min'] if 'b_min' in kwargs else None
        self.b_max = kwargs['b_max'] if 'b_max' in kwargs else None
        self.clip = kwargs['clip'] if 'clip' in kwargs else None

    def __call__(self, arr):
        
        if self.a_max - self.a_min == 0.0:
            if self.b_min is None:
                return arr - self.a_min
            return arr - self.a_min + self.b_min

        arr = (arr - self.a_min) / (self.a_max - self.a_min)
        if (self.b_min is not None) and (self.b_max is not None):
            arr = arr * (self.b_max - self.b_min) + self.b_min

        if self.clip:
            arr = np.clip(arr, self.b_min, self.b_max)

        return arr

class Transpose:
    def __init__(self, **kwargs):
        self.indices = kwargs['indices'] if 'indices' in kwargs else None

    def __call__(self, arr):
        arr = convert_to_tensor(arr)
        return arr.permute(self.indices or tuple(range(arr.ndim)[::-1]))
    
class ToTensor:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None

    def __call__(self, arr):
        return torch.as_tensor(arr)

# TODO uses nibabel
class Orientation:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.axcodes = kwargs['axcodes'] if 'axcodes' in kwargs else None

    def orientation(self, img, spatial_ornt):
        spatial_shape = img.shape[1:]

        spatial_ornt[:, 0] += 1  # skip channel dim
        spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
        axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
        full_transpose = np.arange(len(spatial_shape) + 1)  # channel-first array
        full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])

        out = img
        if axes:
            out = np.flip(out, dims=axes)
        if not np.all(full_transpose == np.arange(len(out.shape))):
            out = out.permute(full_transpose.tolist())
        return out

    def __call__(self, arr):
        spatial_shape = arr.shape[1:]
        sr = len(spatial_shape)
        affine_ = np.eye(sr + 1, dtype=np.float64)
        src = nib.io_orientation(affine_)
        dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
        spatial_ornt = nib.orientations.ornt_transform(src, dst)

        return self.orientation(arr, spatial_ornt)

def monai_to_bmz_preprocessing(inference_parser, name):

    monai_preproc_functions = {'AddChanneld': AddChannel, 
                               'AddClickSignalsd': AddClickSignals, 
                               'AddLabelAsGuidanced': AddLabelAsGuidance, 
                               'AsChannelFirstd': AsChannelFirst,
                               'CastToTyped': EnsureType, 
                               'DiscardAddGuidanced': DiscardAddGuidance, 
                               'EnsureChannelFirstd': EnsureChannelFirst, 
                               'EnsureType': EnsureType, 
                               'EnsureTyped': EnsureType, 
                               'LoadImage': LoadImage,
                               'LoadImaged': LoadImage, 
                               'NormalizeIntensityd': NormalizeIntensity, 
                               'Orientationd': Orientation, 
                               'Resized': Resize, 
                               'ScaleIntensity': ScaleIntensity, 
                               'ScaleIntensityRanged': ScaleIntensityRange, 
                               'ScaleIntensityd': ScaleIntensity, 
                               'Spacingd': Spacing, 
                               'SqueezeDimd': SqueezeDim, 
                               'ToTensord': ToTensor, 
                               'Transpose':Transpose}
    
    bmz_preprocessing = []
    #bmz_preprocessing = [lambda x:x[0] if len(x.shape) >= 4 else x]

    if name == "MedNIST registration":
        
        for i, preproc_function in enumerate(inference_parser['image_load'] ):
            preproc_function_name = inference_parser.get_parsed_content(f'image_load#{i}#_target_')
            # preproc_function_name = preproc_function.__class__.__name__
            print(preproc_function_name)
            
            function_args = preproc_function
            function_args.pop('_target_')
            print(function_args)
            # function_args = preproc_function
            
            parsed_function_args = {}

            for k in function_args.keys():
                value = inference_parser.get_parsed_content(f'image_load#{i}#{k}')
                if k == 'dtype' and isinstance(value, torch.dtype):
                    value = torch.tensor(0, dtype=torch.float32).numpy().dtype
                parsed_function_args[k] = value
            
            print(parsed_function_args)
            
            # bmz_preprocessing.append(preproc_function_name)
            bmz_preprocessing.append(monai_preproc_functions[preproc_function_name](**parsed_function_args))

        for i, preproc_function in enumerate(inference_parser['image_aug'] ):
            preproc_function_name = inference_parser.get_parsed_content(f'image_aug#{i}#_target_')
            # preproc_function_name = preproc_function.__class__.__name__
            print(preproc_function_name)
            
            function_args = preproc_function
            function_args.pop('_target_')
            print(function_args)
            # function_args = preproc_function
            
            parsed_function_args = {}

            for k in function_args.keys():
                value = inference_parser.get_parsed_content(f'image_aug#{i}#{k}')
                if k == 'dtype' and isinstance(value, torch.dtype):
                    value = torch.tensor(0, dtype=torch.float32).numpy().dtype
                parsed_function_args[k] = value
            
            print(parsed_function_args)
            
            # bmz_preprocessing.append(preproc_function_name)
            bmz_preprocessing.append(monai_preproc_functions[preproc_function_name](**parsed_function_args))

    else:
        monai_preprocessing = inference_parser['preprocessing']['transforms']
        # monai_preprocessing = inference_parser.get_parsed_content(f'preprocessing').transforms

        for i, preproc_function in enumerate(monai_preprocessing):
            preproc_function_name = inference_parser.get_parsed_content(f'preprocessing#transforms#{i}#_target_')
            # preproc_function_name = preproc_function.__class__.__name__
            
            function_args = preproc_function
            function_args.pop('_target_')
            print(function_args)
            # function_args = preproc_function
            
            parsed_function_args = {}

            for k in function_args.keys():
                value = inference_parser.get_parsed_content(f'preprocessing#transforms#{i}#{k}')
                if k == 'dtype' and isinstance(value, torch.dtype):
                    value = torch.tensor(0, dtype=torch.float32).numpy().dtype
                parsed_function_args[k] = value
            
            print(parsed_function_args)
            
            # bmz_preprocessing.append(preproc_function_name)
            bmz_preprocessing.append(monai_preproc_functions[preproc_function_name](**parsed_function_args))

    bmz_preprocessing.append(lambda x: x[None,...])
    bmz_preprocessing.append(lambda x: convert_to_tensor(x, dtype=torch.float32))
    #bmz_preprocessing.append(lambda x: convert_to_tensor(x))

    return bmz_preprocessing

#######################################

class Activationsd:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.keys = kwargs['sigmoid'] if 'sigmoid' in kwargs else False
        self.keys = kwargs['softmax'] if 'softmax' in kwargs else False

    def __call__(self, arr):
        if self.sigmoid:
            return 1.0 / (1.0 + np.exp(-arr))
        if self.softmax:
            return (np.exp(arr - np.max(arr)) / np.exp(arr - np.max(arr)).sum())

class AffineBoxToWorldCoordinated:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['box_keys'] if 'box_keys' in kwargs else None
        self.box_ref_image_keys = kwargs['box_ref_image_keys'] if 'box_ref_image_keys' in kwargs else None 
        self.image_meta_key_postfix = kwargs['image_meta_key_postfix'] if 'image_meta_key_postfix' in kwargs else False
        self.affine_lps_to_ras = kwargs['affine_lps_to_ras'] if 'affine_lps_to_ras' in kwargs else False
    
    def __call__(self, arr):
        # TODO
        '''
         # convert numpy to tensor if needed
        boxes_t, *_ = convert_data_type(boxes, torch.Tensor)

        # some operation does not support torch.float16
        # convert to float32

        boxes_t = boxes_t.to(dtype=COMPUTE_DTYPE)
        affine_t, *_ = convert_to_dst_type(src=affine, dst=boxes_t)

        spatial_dims = get_spatial_dims(boxes=boxes_t)

        # affine transform left top and bottom right points
        # might flipped, thus lt may not be left top any more
        lt: torch.Tensor = _apply_affine_to_points(boxes_t[:, :spatial_dims], affine_t, include_shift=True)
        rb: torch.Tensor = _apply_affine_to_points(boxes_t[:, spatial_dims:], affine_t, include_shift=True)

        # make sure lt_new is left top, and rb_new is bottom right
        lt_new, _ = torch.min(torch.stack([lt, rb], dim=2), dim=2)
        rb_new, _ = torch.max(torch.stack([lt, rb], dim=2), dim=2)

        boxes_t_affine = torch.cat([lt_new, rb_new], dim=1)

        # convert tensor back to numpy if needed
        boxes_affine: NdarrayOrTensor
        boxes_affine, *_ = convert_to_dst_type(src=boxes_t_affine, dst=boxes)
        return boxes_affine  # type: ignore[return-value]
        '''

class AsDiscreted:
    def __init__(self, **kwargs):
        self.keys = kwargs['keys'] if 'keys' in kwargs else None
        self.argmax = kwargs['argmax'] if 'argmax' in kwargs else False
        self.to_onehot = kwargs['to_onehot'] if 'to_onehot' in kwargs else None
        self.threshold = kwargs['threshold'] if 'threshold' in kwargs else None
        self.kwargs = kwargs

    def np_one_hot(labels, num_classes, dtype = np.float32, dim = 1):
        # if `dim` is bigger, add singleton dim at the end
        if labels.ndim < dim + 1:
            shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
            labels = np.reshape(labels, shape)

        sh = list(labels.shape)

        if sh[dim] != 1:
            raise AssertionError("labels should have a channel with length equal to one.")

        sh[dim] = num_classes

        o = np.zeros(sh, dtype=dtype)
        np.put_along_axis(o, np.int64(labels), 1, axis=dim)

        return o

    def __call__(self, arr):
        if self.argmax:
            arr = np.argmax(arr, axis=self.kwargs.get("dim", 0), keepdim=self.kwargs.get("keepdim", True))
        if self.to_onehot is not None:
            img_t = self.one_hot(img_t, num_classes=self.to_onehot, dim=self.kwargs.get("dim", 0), dtype=self.kwargs.get("dtype", torch.float))

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img_t = img_t >= threshold

        return img_t


class ClipBoxToImaged:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['box_keys'] if 'box_keys' in kwargs else None
        self.label_keys = kwargs['label_keys'] if 'label_keys' in kwargs else False
        self.box_ref_image_keys = kwargs['box_ref_image_keys'] if 'box_ref_image_keys' in kwargs else False
        self.remove_empty = kwargs['remove_empty'] if 'remove_empty' in kwargs else True

    # TODO
    def get_spatial_dims(self, boxes):
        pass

    def ensure_tuple_rep(self, spatial_size, spatial_dims):
        pass

    def clip_boxes_to_image(boxes, spatial_size, remove_empty):
        pass

    def select_labels(labels, keep):
        pass

    def __call__(self, arr):
        d = dict(arr)
        spatial_size = d[self.box_ref_image_keys].shape[1:]
        labels = [d[label_key] for label_key in self.label_keys]  # could be multiple arrays

        boxes = d[self.box_keys]
        labels = labels
        spatial_size = spatial_size

        spatial_dims = self.get_spatial_dims(boxes=boxes)
        spatial_size = self.ensure_tuple_rep(spatial_size, spatial_dims)  # match the spatial image dim

        boxes_clip, keep = self.clip_boxes_to_image(boxes, spatial_size, self.remove_empty)
        
        d[self.box_keys] =  boxes_clip
        clipped_labels = self.select_labels(labels, keep)

        for label_key, clipped_labels_i in zip(self.label_keys, clipped_labels):
            d[label_key] = clipped_labels_i
        return d        

class ConvertBoxModed:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr

class DeleteItemsd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class FlattenSubKeysd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class FromMetaTensord:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class HoVerNetInstanceMapPostProcessingd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class HoVerNetNuclearTypePostProcessingd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class Invertd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class KeepLargestConnectedComponentd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class Lambdad:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class SaveImaged:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class ToNumpyd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
class NpySaverd:
    def __init__(self, **kwargs):
        self.box_keys = kwargs['keys'] if 'keys' in kwargs else None

    # TODO
    def __call__(self, arr):
        return arr
    
def monai_to_bmz_postprocessing(inference_parser, name):

    monai_postproc_functions = {'Activationsd': Activationsd, 
                               'AffineBoxToWorldCoordinated': AffineBoxToWorldCoordinated, 
                               'AsDiscreted': AsDiscreted, 
                               'ClipBoxToImaged': ClipBoxToImaged,
                               'ConvertBoxModed': ConvertBoxModed, 
                               'DeleteItemsd': DeleteItemsd, 
                               'EnsureTyped': EnsureType, 
                               'FlattenSubKeysd': FlattenSubKeysd,
                               'FromMetaTensord': FromMetaTensord, 
                               'HoVerNetInstanceMapPostProcessingd': HoVerNetInstanceMapPostProcessingd, 
                               'HoVerNetNuclearTypePostProcessingd': HoVerNetNuclearTypePostProcessingd, 
                               'Invertd': Invertd, 
                               'KeepLargestConnectedComponentd': KeepLargestConnectedComponentd, 
                               'Lambdad': Lambdad, 
                               'SaveImaged': SaveImaged, 
                               'SqueezeDimd': SqueezeDim, 
                               'ToNumpyd': ToNumpyd,
                               'ToTensord': ToTensor, 
                               'scripts.valve_landmarks.NpySaverd': NpySaverd}
    
    bmz_postprocessing = []

    monai_postprocessing = inference_parser['postprocessing']['transforms']
    # monai_postprocessing = inference_parser.get_parsed_content(f'postprocessing').transforms

    for i, postproc_function in enumerate(monai_postprocessing):
        postproc_function_name = inference_parser.get_parsed_content(f'postprocessing#transforms#{i}#_target_')
        # postproc_function_name = preproc_function.__class__.__name__
        print(postproc_function_name)
        
        function_args = postproc_function
        function_args.pop('_target_')
        print(function_args)
        # function_args = preproc_function
        
        parsed_function_args = {}

        for k in function_args.keys():
            value = inference_parser.get_parsed_content(f'postprocessing#transforms#{i}#{k}')
            if k == 'dtype' and isinstance(value, torch.dtype):
                value = torch.tensor(0, dtype=torch.float32).numpy().dtype
            parsed_function_args[k] = value
        
        print(parsed_function_args)
        
        # bmz_postprocessing.append(preproc_function_name)
        bmz_postprocessing.append(monai_postproc_functions[postproc_function_name](**parsed_function_args))

    bmz_postprocessing.append(lambda x: x[None,...])
    bmz_postprocessing.append(lambda x: convert_to_tensor(x, dtype=torch.float32))
    #bmz_postprocessing.append(lambda x: convert_to_tensor(x))

    return [lambda x: x.detach().numpy()]
