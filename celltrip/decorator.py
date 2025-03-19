import cProfile
import json
import sys
import time
import traceback
import tracemalloc

import torch

from . import utility as _utility


def try_catch(_func=None, show_traceback=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try: return func(*args, **kwargs)
            except Exception as err:
                if show_traceback: print(traceback.format_exc(), end='')
                else: print(err)
        return wrapper
    
    if _func is None: return decorator
    else: return decorator(_func)


def profile(_func=None, fname=None):
    # Adapted from https://stackoverflow.com/a/5376616
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal fname
            if fname is None: fname = func.__name__ + '.prof'
            prof = cProfile.Profile()
            ret = prof.runcall(func, *args, **kwargs)
            prof.dump_stats(fname)
            return ret
        return wrapper
    
    if _func is None: return decorator
    else: return decorator(_func)


def metrics(_func=None, append_to_dict=False):
    def decorator(func):
        "Add `return_metrics` argument to function, and return memory/VRAM usage if True"
        # NOTE: Might be incorrect if two running on the same kernel
        def wrapper(*args, return_metrics=False, **kwargs):
            # Parameters
            nonlocal append_to_dict

            # Record initial
            start_time = time.perf_counter()
            if return_metrics:
                tracemalloc.start()
                base_memory = tracemalloc.get_traced_memory()[1]
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    base_vram = torch.cuda.max_memory_allocated()

            ret = func(*args, **kwargs)

            # Record final
            metrics = {}
            metrics['Time'] = time.perf_counter() - start_time
            if return_metrics:
                # Get extra memory from args
                # NOTE: Assumes all on CPU
                arg_sizes = [sys.getsizeof(obj) for obj in args+tuple(kwargs.values())]
                np_arg_sizes = _utility.general.list_crawl(
                    args+tuple(kwargs.values()),
                    lambda x: x.nbytes)
                
                arg_mem = sum(arg_sizes+np_arg_sizes)

                
                metrics['memory'] = tracemalloc.get_traced_memory()[1] - base_memory + arg_mem  # Memory usage
                if torch.cuda.is_available():
                    metrics['VRAM'] = torch.cuda.max_memory_allocated() - base_vram  # VRAM usage
                else: metrics['VRAM'] = 0

            if append_to_dict:
                ret.update(metrics)
                return ret
            return ret, metrics
        
        return wrapper
    
    if _func is None: return decorator
    else: return decorator(_func)
            
def call_on_exit(func):
    "Add `exit_hook` argument to function, which runs on KeyboardInterrupt"
    def wrapper(*args, exit_hook=None, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            if exit_hook is not None: exit_hook()
    
    return wrapper


def print_ret(_func=None):
    def decorator(func):
        "Add argument to print returned metrics, True by default"
        def wrapper(*args, verbose_return=True, **kwargs):
            ret = func(*args, **kwargs)
            if verbose_return:
                print(ret)
                # print(json.dumps(ret, indent=2, sort_keys=False))
            return ret
        
        return wrapper
    
    if _func is None: return decorator
    else: return decorator(_func)
