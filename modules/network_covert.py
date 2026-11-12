"""Covert channels in packet captures (classic pcap and pcapng), pure Python.

Signals:
- **DNS tunnelling** (iodine, dnscat2 and similar): per base domain, many
  queries with long, unique, high-entropy subdomains, or NULL/PRIVATE query
  types that ordinary resolvers do not send.
- **ICMP tunnelling**: echo payloads that are neither the standard Linux/BSD
  (timestamp + byte ramp) nor the Windows (alphabet) ping patterns, and that
  are high-entropy and vary across packets.
- **Header fields**: the IPv4 reserved ("evil") bit, TCP reserved bits, and a
  TCP urgent pointer without the URG flag. None of these is set by normal
  stacks.

Captures are read with packet-count and byte limits. Nothing is reassembled,
replayed or sent.
"""

from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

MAX_BYTES = 256 * 1024 * 1024
MAX_PACKETS = 2_000_000
ETHERNET, RAW, LINUX_SLL, RAW_IPV4, RAW_IPV6 = 1, 101, 113, 228, 229
DNS_PORT = 53
TUNNEL_QTYPES = {10: "NULL", 65399: "PRIVATE"}
MIN_DNS_QUERIES = 20
MIN_ICMP_PACKETS = 5
PCAP_MAGICS = (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d")


@dataclass
class Capture:
    """Per-capture aggregates gathered in one pass."""

    packets: int = 0
    dns: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    dns_qtypes: Counter = field(default_factory=Counter)
    icmp_payloads: list[bytes] = field(default_factory=list)
    evil_bit: int = 0
    tcp_reserved: int = 0
    urgent_without_flag: int = 0


def entropy(data: bytes | str) -> float:
    if not data:
        return 0.0
    total = len(data)
    return -sum(c / total * math.log2(c / total) for c in Counter(data).values())


def frames(data: bytes) -> Iterator[tuple[int, bytes]]:
    """(link type, frame) pairs of a classic pcap or pcapng capture."""
    magic = data[:4]
    if magic in PCAP_MAGICS:
        endian = "<" if magic[0] in (0xD4, 0x4D) else ">"
        link = struct.unpack(endian + "I", data[20:24])[0] & 0xFFFF
        pos = 24
        while pos + 16 <= len(data):
            length = struct.unpack(endian + "I", data[pos + 8 : pos + 12])[0]
            yield link, data[pos + 16 : pos + 16 + length]
            pos += 16 + length
    elif magic == b"\x0a\x0d\x0d\x0a" and len(data) >= 12:
        endian = "<" if data[8:12] == b"\x4d\x3c\x2b\x1a" else ">"
        links: list[int] = []
        pos = 0
        while pos + 12 <= len(data):
            kind, length = struct.unpack(endian + "II", data[pos : pos + 8])
            if length < 12:
                return
            body = data[pos + 8 : pos + length - 4]
            if kind == 1 and len(body) >= 2:
                links.append(struct.unpack(endian + "H", body[:2])[0])
            elif kind == 6 and len(body) >= 20:
                iface, _, _, captured = struct.unpack(endian + "IIII", body[:16])
                if iface < len(links):
                    yield links[iface], body[20 : 20 + captured]
            elif kind == 3 and links and len(body) >= 4:
                yield links[0], body[4:]
            pos += length


def ip_packet(link: int, frame: bytes) -> bytes | None:
    if link == ETHERNET:
        offset, kind = 14, frame[12:14]
        while kind in (b"\x81\x00", b"\x88\xa8") and len(frame) >= offset + 4:
            kind, offset = frame[offset + 2 : offset + 4], offset + 4
        return frame[offset:] if kind in (b"\x08\x00", b"\x86\xdd") else None
    if link == LINUX_SLL:
        return frame[16:] if frame[14:16] in (b"\x08\x00", b"\x86\xdd") else None
    if link in (RAW, RAW_IPV4, RAW_IPV6):
        return frame
    return None


def dns_query(payload: bytes) -> tuple[str, int] | None:
    """First question name and type of a DNS query message."""
    if len(payload) < 12 or payload[2] & 0x80 or struct.unpack(">H", payload[4:6])[0] == 0:
        return None
    labels: list[str] = []
    pos = 12
    while pos < len(payload):
        size = payload[pos]
        if size == 0:
            break
        if size & 0xC0 or pos + 1 + size > len(payload) or len(labels) > 127:
            return None
        labels.append(payload[pos + 1 : pos + 1 + size].decode("latin-1"))
        pos += 1 + size
    if pos + 3 > len(payload) - 2 or not labels:
        return None
    return ".".join(labels).lower(), struct.unpack(">H", payload[pos + 1 : pos + 3])[0]


def inspect(link: int, frame: bytes, capture: Capture) -> None:
    packet = ip_packet(link, frame)
    if not packet or len(packet) < 20:
        return
    if packet[0] >> 4 == 4:
        header = (packet[0] & 0x0F) * 4
        capture.evil_bit += bool(packet[6] & 0x80)
        total = int.from_bytes(packet[2:4], "big")  # drop link-layer padding
        proto, body = packet[9], packet[header:total]
    elif packet[0] >> 4 == 6 and len(packet) >= 40:
        proto, body = packet[6], packet[40 : 40 + int.from_bytes(packet[4:6], "big")]
    else:
        return
    if proto in (1, 58) and len(body) >= 8 and body[0] in (8, 0, 128, 129):
        capture.icmp_payloads.append(body[8:])
    elif proto == 6 and len(body) >= 20:
        capture.tcp_reserved += bool(body[12] & 0x0E)
        capture.urgent_without_flag += not body[13] & 0x20 and body[18:20] != b"\x00\x00"
    elif proto == 17 and len(body) >= 8 and DNS_PORT in struct.unpack(">HH", body[:4]):
        query = dns_query(body[8:])
        if query:
            name, qtype = query
            parts = name.split(".")
            base = ".".join(parts[-2:])
            capture.dns[base].append(".".join(parts[:-2]))
            capture.dns_qtypes[qtype] += 1


def standard_ping(payload: bytes) -> bool:
    """Ping payloads of common stacks: byte ramps (Linux/BSD, optionally after a
    timestamp), cyclic alphabets (Windows a..w, other tools a..z / A..W) or all zeros."""
    if not payload.strip(b"\x00"):
        return True
    for first in (b"a"[0], b"A"[0]):
        for period in (23, 26):
            cycle = bytes(first + i % period for i in range(len(payload)))
            if payload == cycle:
                return True
    for start in (0, 8, 16):
        tail = payload[start:]
        if tail and all(b == (tail[0] + i) & 0xFF for i, b in enumerate(tail)):
            return True
    return False


def signals(capture: Capture) -> list[Signal]:
    out = []
    for base, subs in capture.dns.items():
        if len(subs) < MIN_DNS_QUERIES:
            continue
        joined = "".join(s.replace(".", "") for s in subs)
        mean_len = len(joined) / len(subs)
        unique = len(set(subs)) / len(subs)
        bits = entropy(joined)
        if mean_len >= 24 and unique >= 0.8 and bits >= 3.5:
            out.append(Signal("dns_tunnel_suspected", 85,
                              f"{len(subs)} queries under {base!r}: mean subdomain {mean_len:.0f} "
                              f"chars, {unique:.0%} unique, {bits:.2f} bits/char",
                              category="network_covert", evidence="strong"))
    rare = {name: capture.dns_qtypes[code] for code, name in TUNNEL_QTYPES.items()
            if capture.dns_qtypes[code]}
    if rare:
        out.append(Signal("dns_tunnel_query_type", 75,
                          "queries of type " + ", ".join(f"{k}×{v}" for k, v in rare.items()),
                          category="network_covert", evidence="strong"))
    # Echo payloads that no ping implementation sends and that look like data.
    odd = [p for p in capture.icmp_payloads
           if len(p) >= 16 and entropy(p) >= 4.0 and not standard_ping(p)]
    if len(odd) >= MIN_ICMP_PACKETS:
        distinct = len(set(odd)) / len(odd)
        bits = sum(entropy(p) for p in odd) / len(odd)
        if distinct >= 0.5:
            out.append(Signal("icmp_tunnel_suspected", 80,
                              f"{len(odd)} echo payloads outside standard ping patterns, "
                              f"{distinct:.0%} distinct, {bits:.2f} bits/byte",
                              category="network_covert", evidence="strong"))
    for name, count, label in (("ipv4_reserved_bit_set", capture.evil_bit, "IPv4 reserved bit"),
                               ("tcp_reserved_bits_set", capture.tcp_reserved,
                                "TCP reserved bits"),
                               ("tcp_urgent_pointer_without_urg", capture.urgent_without_flag,
                                "TCP urgent pointer without URG")):
        if count:
            out.append(Signal(name, 70 if count >= 5 else 45, f"{label} in {count} packet(s)",
                              category="network_header_channel",
                              evidence="strong" if count >= 5 else "heuristic"))
    return out


class NetworkCovertAnalyzer(Analyzer):
    """Flag DNS/ICMP tunnelling and header-field covert channels in captures."""

    name = "network_covert"

    def analyze(self, src: Path) -> AnalysisResult:
        if src.suffix.lower() not in (".pcap", ".pcapng", ".cap") or src.stat().st_size > MAX_BYTES:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        capture = Capture()
        for link, frame in frames(src.read_bytes()):
            capture.packets += 1
            if capture.packets > MAX_PACKETS:
                break
            inspect(link, frame, capture)
        found = signals(capture)
        return AnalysisResult(self.name, max((s.score for s in found), default=0),
                              tuple(found), None)
