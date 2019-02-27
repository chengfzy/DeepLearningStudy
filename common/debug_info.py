"""
Some Function for Debug
"""


def section(title):
    """
    Obtain section string
    :param title: title string
    :return: section string
    """
    space = '=' * int((100 - len(title)) / 2)
    return space + ' ' + title + ' ' + space


def sub_section(title):
    """
        Obtain sub section string
        :param title: title string
        :return: sub-section string
        """
    space = '-' * int((100 - len(title)) / 2)
    return space + ' ' + title + ' ' + space
