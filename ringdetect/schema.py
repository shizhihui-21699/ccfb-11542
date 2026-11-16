"""Entity types, value normalization, and tunable defaults.

The graph is a typed entity graph (a bipartite "account <-> resource" graph).
Every record in a daily file contributes one *account* node and up to four
resource nodes (device / card / address / ip).

Normalization matters more than it looks: two spellings of the same address
that fail to merge are two different pieces of evidence, and a raw PAN in the
state directory is a compliance incident.  Both are handled here.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import unicodedata

ACCOUNT, DEVICE, CARD, ADDRESS, IP = range(5)

TYPE_NAMES = ("account", "device", "card", "address", "ip")
TYPE_BY_NAME = {name: idx for idx, name in enumerate(TYPE_NAMES)}

#: Resource types that take part in the graph (accounts are implicit).
RESOURCE_TYPES = (DEVICE, CARD, ADDRESS, IP)

#: A record field may be missing or null; this is the field -> type mapping used
#: unless the caller overrides it (``--field device=device_id`` etc.).
DEFAULT_FIELD_MAP = {t: t for t in TYPE_NAMES}

#: Relative trust in each resource type.  A shared device/card is a strong
#: signal that the same human controls both accounts; a shared address is weak
#: (co-tenancy is normal) and a shared IP is weaker still (NAT, public wifi).
DEFAULT_TYPE_WEIGHT = {
    DEVICE: 1.0,
    CARD: 1.0,
    ADDRESS: 0.55,
    IP: 0.40,
}

PAN_RE = re.compile(r"^[0-9]{13,19}$")
_WS_RE = re.compile(r"\s+")


class ConfigError(ValueError):
    """Raised for an invalid or unsafe configuration."""


def _clean(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return _WS_RE.sub(" ", text)


def _hash_card(pan: str, salt: bytes) -> str:
    digest = hmac.new(salt, pan.encode("utf-8"), hashlib.sha256).hexdigest()
    return "pan:" + digest[:32]


def normalize_value(etype: int, value: object, *, ip_prefix: int = 0,
                    card_salt: bytes | None = None) -> str | None:
    """Normalize one entity value.  Returns ``None`` for empty values.

    ``card_salt`` is mandatory for anything that looks like a raw PAN; we refuse
    to persist primary account numbers in cleartext.
    """
    if value is None:
        return None
    text = _clean(value)
    if not text:
        return None

    if etype == ACCOUNT:
        return text
    if etype == DEVICE:
        return text
    if etype == CARD:
        if PAN_RE.match(text):
            if not card_salt:
                raise ConfigError(
                    "a raw card number was supplied but no hashing salt is configured; "
                    "set RINGDETECT_CARD_SALT (or --card-salt-env) or pass tokenized cards"
                )
            return _hash_card(text, card_salt)
        return text
    if etype == ADDRESS:
        # Drop trailing punctuation noise, keep the address token itself.
        return text.strip(" ,;")
    if etype == IP:
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        try:
            addr = ipaddress.ip_address(text)
        except ValueError:
            return text
        if ip_prefix and addr.version == 4 and ip_prefix < 32:
            return str(ipaddress.ip_network(f"{addr}/{ip_prefix}", strict=False))
        return str(addr)
    raise ConfigError(f"unknown entity type {etype!r}")


def entity_key(etype: int, value: str) -> str:
    """Stable, type-scoped key used for interning."""
    return f"{TYPE_NAMES[etype]}\x1f{value}"


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
