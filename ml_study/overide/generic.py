# -*- coding: utf-8 -*-
"""
# ######################################################################################
# 文件名称：generic.py
# 摘   要：
# 作   者：hello-hzb
# 日   期：11/6/19
# 备   注：
#
# 算法知识点：
# 1.
# 2.
#
# python知识点：
# 1.
# 2.
# ######################################################################################
"""

from __future__ import absolute_import

try:
    from decorator import decorate
except ImportError as err_msg:
    # Allow decorator to be missing in runtime
    raise err_msg

schedule_target = None


class Target(object):
    def __init__(self, target):
        global schedule_target
        schedule_target = target

    def __enter__(self):
        print("进入enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("退出exit")
        print(exc_type, exc_value)


def create_target(target):
    global schedule_target
    schedule_target = target


def generic_func(fdefault):
    """Wrap a target generic function.

    Generic function allows registeration of further functions
    that can be dispatched on current target context.
    If no registered dispatch is matched, the fdefault will be called.

    Parameters
    ----------
    fdefault : function
        The default function.

    Returns
    -------
    fgeneric : function
        A wrapped generic function.

    Example
    -------
    .. code-block:: python

      import tvm
      # wrap function as target generic
      @tvm.target.generic_func
      def my_func(a):
          return a + 1
      # register specialization of my_func under target cuda
      @my_func.register("cuda")
      def my_func_cuda(a):
          return a + 2
      # displays 3, because my_func is called
      print(my_func(2))
      # displays 4, because my_func_cuda is called
      with tvm.target.cuda():
          print(my_func(2))
    """
    dispatch_dict = {}
    func_name = fdefault.__name__

    def register(key, func=None, override=False):
        """Register function to be the dispatch function.

        Parameters
        ----------
        key : str or list of str
            The key to be registered.

        func : function
            The function to be registered.

        override : bool
            Whether override existing registeration.

        Returns
        -------
        The register function is necessary.
        """
        def _do_reg(myf):
            key_list = [key] if isinstance(key, str) else key
            for k in key_list:
                if k in dispatch_dict and not override:
                    raise ValueError(
                        "Key is already registered for %s" % func_name)
                dispatch_dict[k] = myf
            return myf
        if func:
            return _do_reg(func)
        return _do_reg

    def dispatch_func(func, *args, **kwargs):
        """The wrapped dispath function"""
        # target = current_target()
        target = [schedule_target]
        if target is None:
            return func(*args, **kwargs)
        for k in target:
            if k in dispatch_dict:
                return dispatch_dict[k](*args, **kwargs)
        return func(*args, **kwargs)
    fdecorate = decorate(fdefault, dispatch_func)
    fdecorate.register = register
    fdecorate.fdefault = fdefault
    return fdecorate

