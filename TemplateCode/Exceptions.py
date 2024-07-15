# contains all exceptions used in the project


# this exception is used when incompatible options are used
class IncompatibleOptions(Exception):
    def __init__(self, opt1, opt2):
        self.message = "IncompatibleOptions: " + opt1 + " and " + opt2
        super().__init__(self.message)


# this exception is used when an unknown option is used
class UnknownOption(Exception):
    def __init__(self, msg):
        self.message = "UnknownOption: " + msg
        super().__init__(self.message)
