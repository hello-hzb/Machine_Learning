# -*- coding: utf-8 -*-
"""
# ######################################################################################
# 文件名称：func_overide.py
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

import generic


@generic.generic_func
def my_func(a):
    return a + 1

# register specialization of my_func under target cuda
@my_func.register("cuda")
def my_func_cuda(a):
    print("cuda")
    return a + 2


# @my_func.register("dsp")
# def my_func_dsp(a):
#     print("dsp")
#     return a + 3

# displays 3, because my_func is called
print(my_func(2))
# displays 4, because my_func_cuda is called


with generic.Target("dsp"):
    print(my_func(2))
    print("hello")