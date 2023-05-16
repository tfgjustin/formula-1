def open_file_with_suffix(base_output_filename, suffix, mode='w'):
    if base_output_filename is None:
        return None
    elif base_output_filename.startswith('/dev/'):
        return open(base_output_filename, mode)
    else:
        return open(base_output_filename + suffix, mode)
