import sys

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message.
    
    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module for extracting traceback information.

    Returns:
        str: A detailed error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )
    
    return error_message
    
    
class CustomException(Exception):
    """
    Custom exception class for handling exceptions in the project.
    
    Args:
        Exception (Exception): The base exception class.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
