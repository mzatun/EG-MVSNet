# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging # logging是python内置标准模块
import os
import sys
import time
import numpy as np
import torch
from os.path import join
import cv2

# fixme: 创建logger日期文件
def setup_logger(name, save_dir, prefix="", timestamp=True):
    logger = logging.getLogger(name) # 创建logger
    logger.setLevel(logging.INFO) # 设置logger级别为info

    # 创建SteamHandler能够打印信息到终端
    ch = logging.StreamHandler(stream=sys.stdout) # 创建SteamHandler，能够将信息打到终端上
    ch.setLevel(logging.INFO) # 设置StreamHandler Level为Info
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s") # 创建打印格式formatter
    ch.setFormatter(formatter) # 配置ch
    logger.addHandler(ch) # 配置logger

    # 创建日志文件写入Handler
    if save_dir:
        timestamp = time.strftime(".%m_%d_%H_%M_%S") if timestamp else "" # 时间格式
        prefix = "." + prefix if prefix else ""
        log_file = os.path.join(save_dir, "log{}.txt".format(prefix + timestamp)) # log保存目录
        fh = logging.FileHandler(log_file) # 创建FileHandler日志写入器
        fh.setLevel(logging.INFO) # 设置写入info等级
        fh.setFormatter(formatter) # 设置写入格式
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def shutdown_logger(logger):
    logger.handlers = []

