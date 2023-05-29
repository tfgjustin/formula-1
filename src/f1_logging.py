import logging


def init_logging(log_tag, loglevel=logging.INFO):
    _FORMAT = '%%(asctime)s %s %%(message)s' % log_tag
    logging.basicConfig(format=_FORMAT)
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    return logger


def open_file_with_suffix(base_output_filename, suffix, mode='w'):
    if base_output_filename is None:
        return None
    elif base_output_filename.startswith('/dev/'):
        return open(base_output_filename, mode)
    else:
        return open(base_output_filename + suffix, mode)
