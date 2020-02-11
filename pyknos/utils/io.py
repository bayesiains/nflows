"""Utility functions for Input/Output."""

import os
import socket
import time

ON_CLUSTER = True if "charles" in socket.gethostname() else False


def is_on_cluster():
    return True if "charles" in socket.gethostname() else False


def get_timestamp():
    formatted_time = time.strftime("%d-%b-%y||%H:%M:%S")
    return formatted_time


def get_project_root():
    if ON_CLUSTER:
        return "/home/s1638128/deployment/lfi"
    else:
        return os.environ["LFI_PROJECT_DIR"]


def get_log_root():
    if ON_CLUSTER:
        return "/home/s1638128/tmp/lfi/log"
    else:
        return os.path.join(get_project_root(), "log")


def get_data_root():
    if ON_CLUSTER:
        return os.path.join(get_project_root(), "data")
    else:
        return os.path.join(get_project_root(), "data")


def get_output_root():
    if ON_CLUSTER:
        return "/home/s1638128/tmp/lfi/out"
    else:
        return os.path.join(get_project_root(), "out")


def get_checkpoint_root():
    if ON_CLUSTER:
        return "/home/s1638128/tmp/lfi/checkpoint"
    else:
        return os.path.join(get_project_root(), "checkpoint")
