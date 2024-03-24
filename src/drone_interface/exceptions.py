class AirSimConnectionError(Exception):
    def __init__(self, message):
        super(AirSimConnectionError, self).__init__(message)
