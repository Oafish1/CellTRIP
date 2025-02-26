import cProfile
import traceback
import tracemalloc

import torch


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


def metrics(func):
    "Add `return_metrics` argument to function, and return memory/VRAM usage if True"
    # NOTE: Might be incorrect if two running on the same kernel
    def wrapper(*args, return_metrics=False, **kwargs):
        if return_metrics:
            tracemalloc.start()
            base_memory = tracemalloc.get_traced_memory()[1]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                base_vram = torch.cuda.max_memory_allocated()

        ret = func(*args, **kwargs)

        if return_metrics:
            metrics = []
            metrics += [tracemalloc.get_traced_memory()[1] - base_memory]  # Memory usage
            if torch.cuda.is_available():
                metrics += [torch.cuda.max_memory_allocated() - base_vram]  # VRAM usage
            else: metrics += [0]
        else: metrics = None

        return ret, metrics
    
    return wrapper
            
def call_on_exit(func):
    "Add `exit_hook` argument to function, which runs on KeyboardInterrupt"
    def wrapper(*args, exit_hook=None, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            if exit_hook is not None: exit_hook()
    
    return wrapper