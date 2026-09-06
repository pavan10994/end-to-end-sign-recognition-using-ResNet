import sys
from src.logger.custom_logger import logger


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts filename, line number, and exact error message from sys execution info.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script name [{file_name}] "
            f"line number [{line_number}] error message [{str(error)}]"
        )
    else:
        error_message = f"Error message [{str(error)}]"

    return error_message


class CustomException(Exception):
    """
    Custom Exception handler for the signature recognition pipeline.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(
            error=error_message, error_detail=error_detail
        )
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message