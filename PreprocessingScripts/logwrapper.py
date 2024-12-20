"""
Provides logging to a JSON for the homebrew pipeline.
Should not needed to be modified.

Author: raphael.wurm@gmail.com 
Date: Jul-2023
"""

#  LOGGING the functions called and all arguments passed, storing in a JSON file ###
import os
import functools
import datetime
import logging
import json

log_records = []


def log_execution(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        
        # Log the execution
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_records.append({"timestamp": timestamp, "function_name": f.__name__,
                            "message": f"Executing function {f.__name__} with arguments: {args}, {kwargs}"})
                            
        result = f(*args, **kwargs)
        
        return result                    


    return wrapper



if __name__ == "__main__":
    pass
