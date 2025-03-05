import h5py
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
