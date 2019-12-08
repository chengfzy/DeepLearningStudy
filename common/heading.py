import enum


class HeadingType(enum.Enum):
    """ Heading type """
    Title = 1
    Section = 2
    SubSection = 3
    Paragraph = 4


class Heading:
    """
    Heading class, used for print some heading information
    """

    def __init__(self, text: str, break_line: bool, heading_type: HeadingType, sec_len: int = 100):
        """
        Init Heading with text, break line, type and paragraph length
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        :param heading_type: Heading type
        :param sec_len: Print length for section and subsection type, and the length of paragraph will be SecLen / 1.5
        """
        self.__text = text
        self.__break_line = break_line
        self.__heading_type = heading_type
        self.__sec_len = sec_len

    def __str__(self):
        info_str = '\n' if self.__break_line else ''
        info_len = self.__sec_len
        fill_char = ''
        if self.__heading_type == HeadingType.Title:
            fill_char = '='
        elif self.__heading_type == HeadingType.Section:
            fill_char = '='
        elif self.__heading_type == HeadingType.SubSection:
            fill_char = '*'
        elif self.__heading_type == HeadingType.Paragraph:
            info_len = int(self.__sec_len / 1.5)
            fill_char = '-'

        if self.__heading_type == HeadingType.Title:
            fill_char2 = '|'
            half_len = int((info_len - len(self.__text) - 2) / 2)
            info_len = 2 * half_len + len(self.__text) + 2
            fill_str1 = info_len * fill_char
            fill_str2 = (info_len - 2) * ' '
            fill_str3 = half_len * ' '
            info_str += fill_str1 + '\n'
            info_str += fill_char2 + fill_str2 + fill_char2 + '\n'
            info_str += fill_char2 + fill_str2 + fill_char2 + '\n'
            info_str += fill_char2 + fill_str3 + self.__text + fill_str3 + fill_char2 + '\n'
            info_str += fill_char2 + fill_str2 + fill_char2 + '\n'
            info_str += fill_char2 + fill_str2 + fill_char2 + '\n'
            info_str += fill_str1
        elif len(self.__text) == 0:
            info_str += info_len * fill_char
        else:
            fill_str = max(5, int((info_len - len(self.__text) - 1) / 2)) * fill_char
            info_str += f'{fill_str} {self.__text} {fill_str}'

        info_str += '\n'
        return info_str


class Title(Heading):
    """ Title heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Title)


class Section(Heading):
    """ Section heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Section)


class SubSection(Heading):
    """ SubSection heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.SubSection)


class Paragraph(Heading):
    """ Paragraph heading """

    def __init__(self, text: str = "", break_line: bool = False):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Paragraph)
