class ObservationError(ValueError):
    """A client observation doesn't match the served policy's expectations.

    Message contract: ONE line, actionable, names the offending key and what
    was expected. Recoverable — the server keeps the connection open.
    """
