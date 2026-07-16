class ObservationError(ValueError):
    """A client observation doesn't match the served policy's expectations.

    Message contract: ONE line, actionable, names the offending key and what
    was expected. Recoverable — the server keeps the connection open.
    """


# An observation that passes ficelle's own validation can still trip an
# upstream lerobot assertion deep in inference (e.g. degenerate generation).
# Both transports (serve.py, iroh_transport.py) catch AssertionError
# separately from generic internal errors and reply with this one-liner
# instead of "internal server error (AssertionError)" — same wording on both
# transports, defined once here so they can't drift.
POLICY_GENERATION_ERROR = (
    "ficelle: the policy failed to generate actions for this observation "
    "(upstream model error; see server log for the traceback)"
)
