import warnings

import h5py
import numpy as np
import torch


def clean_return(ret, keep_array=False):
    "Clean return output for improved parsing"
    if not keep_array and len(ret) == 1: return ret[0]
    return ret


def is_list_like(l):
    "Test if `l` has the `__len__` method`"
    try:
        len(l)
        assert not isinstance(l, str)
        return True
    except:
        return False


def list_crawl(l, f):
    "Crawl through the *potential* list l, applying f"
    try:
        return [f(l)]
    except:
        try:
            return sum([list_crawl(le, f) for le in l], [])
        except: return []


def gen_tolist(ar):
    "Convert tensor or numpy array to list"
    try: ar = ar.detach().cpu().tolist()
    except:
        try:
            ar = ar.tolist()
        except: pass
        
    return ar


def dict_entry(base, add):
    """
    Add `add` entries to `base`, not tolerant of duplicates
    """
    for k, v in add.items():
        if k not in base:
            base[k] = v
        else:
            # Assumes any overlaps are also dicts
            dict_entry(base[k], v)
    
    return base


def rolled_index(arrays, idx):
    """
    Recursive function to return result of indexing into multiple lists as if they were
    sub-arrays

    Example:
    for i in range(3 * 2 * 4):
        print(rolled_index([[1, 2, 3], [4, 5], [6, 7, 8, 9]], i))

    > [1, 4, 6]
    > [1, 4, 7]
    > ...
    > [3, 5, 9]
    """
    if len(arrays) == 0: return []
    return rolled_index(arrays[:-1], idx//len(arrays[-1])) + [arrays[-1][idx%len(arrays[-1])]]


def slicify_array(array):
    """
    Convert np.array to contiguous slices, i.e.
    [0, 1, 2, 4, 5, 6, 8] -> [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]

    Example:
    a = np.array([0, 1, 2, 4, 5, 6, 8])
    slices = slicify_array(a)
    for sl in slices: print(a[sl])

    > [0 1 2]
    > [4 5 6]
    > [8]
    """
    anomalies = np.argwhere(~(array[1:]-1 == array[:-1])).flatten()
    endpoints = np.concat((anomalies+1, [len(array)]))
    startpoints = np.concat(([0], anomalies+1))
    slices = [slice(st, en) for st, en in zip(startpoints, endpoints)]
    return slices


def h5_tree(val, pre=''):
    "Show the structure of an h5 file as a tree"
    # https://stackoverflow.com/a/73686304
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


def print_cuda_memory(peak=True):
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()
    print(f'{mem / 1024**3:.2f} GB')


def overwrite_dict(original, modification, copy=True):
    "Overwrite dictionary values based on provided modification dictionary"

    # Copy if requested
    new = original.copy() if copy else original

    # Write over old keys/Add new keys
    for k, v in modification.items():
        new[k] = v

    return new


def get_policy_state(policy):
    return {
        'policy': policy.state_dict(),
        'optimizer': policy.optimizer.state_dict(),
        'scheduler': policy.scheduler.state_dict(),
    }


def set_policy_state(policy, state_dicts):
    policy.load_state_dict(state_dicts['policy'])
    policy.optimizer.load_state_dict(state_dicts['optimizer'])
    policy.scheduler.load_state_dict(state_dicts['scheduler'])


def set_device_recursive(state_dict, device):
    # https://github.com/pytorch/pytorch/issues/1442#issue-225795077
    for key in state_dict:
        if isinstance(state_dict[key], dict):
            state_dict[key] = set_device_recursive(state_dict[key], device)
        else:
            try:
                state_dict[key] = state_dict[key].to(device)
            except:
                pass
    return state_dict


def get_s3_handler_with_access(fname):
    # Get s3 handler
    import s3fs
    try:
        s3 = s3fs.S3FileSystem()
        try: s3.ls(fname.split('/')[2])
        except: s3.ls('/'.join(fname.split('/')[2:4]))
    except:
        warnings.warn('No suitable credentials found for s3 '
                      'access, using anonymous mode')
        s3 = s3fs.S3FileSystem(anon=True)
        try: s3.ls(fname.split('/')[2])
        except: s3.ls('/'.join(fname.split('/')[2:4]))

    return s3


def padded_stack(arrays, method='zero'):
    batch_num = len(arrays)
    shape = np.array([arr.shape for arr in arrays])
    max_dims = shape.max(axis=0)
    new_shape = (batch_num, *max_dims)
    if method == 'zero': return_matrix = np.zeros(new_shape)
    elif method == 'ones': return_matrix = np.ones(new_shape)
    elif method == 'neg_ones': return_matrix = -np.ones(new_shape)
    elif method == 'false': return_matrix = np.zeros(new_shape, dtype=bool)
    elif method == 'true': return_matrix = np.ones(new_shape, dtype=bool)
    else: raise ValueError(f'No method `{method}` found')
    for i, arr in enumerate(arrays):
        return_matrix[(i, *[slice(dim) for dim in arr.shape])] = arr
    return return_matrix
