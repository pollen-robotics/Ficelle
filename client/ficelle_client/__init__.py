from ficelle_client.iroh_client import IrohPolicyClient
from ficelle_client.policy_client import PolicyClient, PolicyServerError

__all__ = ["IrohPolicyClient", "PolicyClient", "PolicyServerError", "open_client"]


def open_client(addr: str, **kw):
    """Dispatch on the address shape: an iroh EndpointTicket string (starts
    with "endpoint") gets an IrohPolicyClient; anything else is a websocket
    host[:port] for PolicyClient — which already parses "host:port" strings
    itself, so no separate split is needed here."""
    if addr.startswith("endpoint"):
        return IrohPolicyClient(addr, **kw)
    return PolicyClient(addr, **kw)
