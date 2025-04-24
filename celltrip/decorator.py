import cProfile
import line_profiler
import sys
import time
import traceback
import tracemalloc

import torch


def try_catch(_func=None, show_traceback=False, fallback_ret=None):
    def try_catch_decorator(func):
        def try_catch_wrapper(*args, **kwargs):
            try: return func(*args, **kwargs)
            except Exception as err:
                if show_traceback: print(traceback.format_exc(), end='')
                else: print(err)
                return fallback_ret
        return try_catch_wrapper
    
    if _func is None: return try_catch_decorator
    else: return try_catch_decorator(_func)


def profile(_func=None, fname=None, time_annotation=False):
    # Adapted from https://stackoverflow.com/a/5376616
    def profile_decorator(func):
        def profile_wrapper(*args, **kwargs):
            nonlocal fname
            nonlocal time_annotation
            if fname is None: fname_use = func.__name__ + '.prof'
            else: fname_use = fname
            if time_annotation: fname_use += '.' + str(int(time.perf_counter()))
            prof = cProfile.Profile()
            ret = prof.runcall(func, *args, **kwargs)
            prof.dump_stats(fname_use)
            return ret
        return profile_wrapper
    
    if _func is None: return profile_decorator
    else: return profile_decorator(_func)


def line_profile(_func=None, signatures=[]):
    # Adapted from https://stackoverflow.com/a/5376616
    def profile_decorator(func):
        def profile_wrapper(*args, **kwargs):
            prof = line_profiler.LineProfiler(func, *signatures)
            ret = prof.runcall(func, *args, **kwargs)
            prof.print_stats(output_unit=1)
            return ret
        return profile_wrapper
    
    if _func is None: return profile_decorator
    else: return profile_decorator(_func)


def metrics(_func=None, append_to_dict=False, dict_index=None):
    def metrics_decorator(func):
        "Add `return_metrics` argument to function, and return memory/VRAM usage if True"
        # NOTE: Might be incorrect if two running on the same kernel
        def metrics_wrapper(*args, return_metrics=False, **kwargs):
            # Parameters
            nonlocal append_to_dict
            nonlocal dict_index

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
                # Get memory from np args
                # NOTE: Causes segfault in docker shm, unfortunately
                # TODO: Find workaround
                # np_arg_sizes = _utility.general.list_crawl(
                #     args+tuple(kwargs.values()),
                #     lambda x: x.nbytes)
                np_arg_sizes = []
                
                arg_mem = sum(arg_sizes+np_arg_sizes)

                # Synchronize CUDA
                if torch.cuda.is_available(): torch.cuda.synchronize()
                
                metrics['memory'] = tracemalloc.get_traced_memory()[1] - base_memory + arg_mem  # Memory usage
                if torch.cuda.is_available():
                    metrics['VRAM'] = torch.cuda.max_memory_allocated() - base_vram  # VRAM usage
                else: metrics['VRAM'] = 0

            # if inspect.iscoroutinefunction(func):
            #     async def ret_func():
            #         ret = await ret
            #         if append_to_dict:
            #             if ret is None: ret = metrics
            #             else: ret.update(metrics)
            #             return ret
            #         return ret, metrics
            #     return ret_func()

            if append_to_dict:
                if ret is None: ret = metrics
                else:
                    if dict_index is not None: ret[dict_index].update(metrics)
                    else: ret.update(metrics)
                return ret
            return ret, metrics
        
        return metrics_wrapper
    
    if _func is None: return metrics_decorator
    else: return metrics_decorator(_func)

            
def call_on_exit(func):
    "Add `exit_hook` argument to function, which runs on KeyboardInterrupt"
    def call_on_exit_wrapper(*args, exit_hook=None, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            if exit_hook is not None: exit_hook()
    
    return call_on_exit_wrapper


def print_ret(_func=None):
    def print_ret_decorator(func):
        "Add argument to print returned metrics, True by default"
        def print_ret_wrapper(*args, verbose_return=True, **kwargs):
            ret = func(*args, **kwargs)
            if verbose_return:
                print(ret)
                # print(json.dumps(ret, indent=2, sort_keys=False))
            return ret
        
        return print_ret_wrapper
    
    if _func is None: return print_ret_decorator
    else: return print_ret_decorator(_func)
