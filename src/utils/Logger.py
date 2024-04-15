import logging

def setup_logger(log_path: str = "logs/BenchmarkingLogs.log"):
    """
    Set up and configure the logger for the application.

    Returns:
        logger (logging.Logger): The configured logger object.
    """
    logger = logging.getLogger('MyAppLogger')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Check if handler already exists to avoid duplicate logs
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == fh.baseFilename for h in logger.handlers):
        logger.addHandler(fh)

    return logger
