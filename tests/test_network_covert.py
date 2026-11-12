"""Network covert-channel rules on generated captures (no traffic is sent)."""

import base64
import struct

import numpy as np
import pytest

from modules import network_covert as nc
from modules.network_covert import NetworkCovertAnalyzer

RNG = np.random.default_rng(0)
BENIGN = ["www.google.com", "mail.google.com", "github.com", "api.github.com", "www.wikipedia.org",
          "en.wikipedia.org", "cdn.jsdelivr.net", "fonts.gstatic.com", "www.youtube.com",
          "i.ytimg.com", "outlook.office365.com", "login.microsoftonline.com"]


def ipv4(proto, body, flags=0x40):
    total = 20 + len(body)
    return struct.pack(">BBHHBBBBH4s4s", 0x45, 0, total, 1, flags, 0, 64, proto, 0,
                       b"\x0a\x00\x00\x01", b"\x0a\x00\x00\x02") + body


def ether(packet, vlan=False):
    head = b"\x00" * 12 + (b"\x81\x00\x00\x01" if vlan else b"")
    return head + b"\x08\x00" + packet


def dns(name, qtype=1):
    question = b"".join(bytes([len(p)]) + p.encode() for p in name.split(".")) + b"\x00"
    header = struct.pack(">HHHHHH", 1, 0x0100, 1, 0, 0, 0)
    message = header + question + struct.pack(">HH", qtype, 1)
    return ipv4(17, struct.pack(">HHHH", 40000, 53, 8 + len(message), 0) + message)


def icmp(payload, kind=8):
    return ipv4(1, bytes([kind, 0, 0, 0, 0, 1, 0, 1]) + payload)


def tcp(flags=0x10, urgent=0, reserved=0):
    header = struct.pack(">HHIIBBHHH", 1234, 80, 1, 1, (5 << 4) | reserved, flags, 1024, 0, urgent)
    return ipv4(6, header)


def pcap(packets, link=1, big_endian=False):
    e = ">" if big_endian else "<"
    out = struct.pack(e + "IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, link)
    for p in packets:
        out += struct.pack(e + "IIII", 0, 0, len(p), len(p)) + p
    return out


def pcapng(packets, link=1, simple=False):
    def block(kind, body):
        body += b"\x00" * (-len(body) % 4)
        return struct.pack("<II", kind, len(body) + 12) + body + struct.pack("<I", len(body) + 12)

    out = block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    out += block(1, struct.pack("<HHI", link, 0, 65535))
    for p in packets:
        if simple:
            out += block(3, struct.pack("<I", len(p)) + p)
        else:
            out += block(6, struct.pack("<IIIII", 0, 0, 0, len(p), len(p)) + p)
    return out


def analyze(tmp_path, data, suffix=".pcap"):
    path = tmp_path / f"c{suffix}"
    path.write_bytes(data)
    result = NetworkCovertAnalyzer().analyze(path)
    return result.status, {s.name: s for s in result.signals}


def benign():
    packets = [ether(dns(BENIGN[i % len(BENIGN)])) for i in range(200)]
    ramp = struct.pack("<Q", 1700000000) + bytes(range(0x10, 0x38))
    packets += [ether(icmp(ramp)) for _ in range(20)]
    packets += [ether(icmp(b"abcdefghijklmnopqrstuvwabcdefghi", 0)) for _ in range(20)]
    packets += [ether(tcp()) for _ in range(50)]
    return packets


def iodine_like(n=80):
    labels = [base64.b32encode(RNG.bytes(40)).decode().strip("=").lower() for _ in range(n)]
    return [ether(dns(f"{label[:60]}.{label[60:]}.t.example.com", 10 if i % 3 == 0 else 1))
            for i, label in enumerate(labels)]


def dnscat_like(n=80):
    return [ether(dns(f"{RNG.bytes(30).hex()}.tunnel.example.org", 16)) for _ in range(n)]


@pytest.mark.parametrize("writer", [pcap, pcapng])
def test_benign_capture_is_clean(tmp_path, writer):
    status, found = analyze(tmp_path, writer(benign()))
    assert status == "ok" and found == {}


def test_dns_tunnels_are_flagged(tmp_path):
    found = analyze(tmp_path, pcap(benign() + iodine_like()))[1]
    assert found["dns_tunnel_suspected"].evidence == "strong"
    assert "NULL" in found["dns_tunnel_query_type"].detail
    found = analyze(tmp_path, pcapng(benign() + dnscat_like()), ".pcapng")[1]
    assert "dns_tunnel_suspected" in found and "dns_tunnel_query_type" not in found
    simple = analyze(tmp_path, pcapng(dnscat_like(), simple=True), ".pcapng")[1]
    assert "dns_tunnel_suspected" in simple


def test_icmp_tunnel_is_flagged(tmp_path):
    packets = benign() + [ether(icmp(RNG.bytes(64))) for _ in range(10)]
    found = analyze(tmp_path, pcap(packets))[1]
    assert found["icmp_tunnel_suspected"].score == 80
    padded = [ether(icmp(b"")) + b"\x00" * 18 for _ in range(10)]  # Ethernet minimum-size padding
    assert analyze(tmp_path, pcap(padded))[1] == {}
    few = benign() + [ether(icmp(RNG.bytes(64))) for _ in range(3)]
    assert analyze(tmp_path, pcap(few))[1] == {}


def test_header_channels(tmp_path):
    packets = [ether(tcp(urgent=0x4142)) for _ in range(6)]
    packets += [ether(tcp(reserved=0x2)) for _ in range(2)]
    packets += [ether(ipv4(6, tcp()[20:], flags=0xC0))]
    found = analyze(tmp_path, pcap(packets, big_endian=True))[1]
    assert found["tcp_urgent_pointer_without_urg"].evidence == "strong"
    assert found["tcp_reserved_bits_set"].score == 45
    assert found["ipv4_reserved_bit_set"].evidence == "heuristic"
    assert analyze(tmp_path, pcap([ether(tcp(flags=0x30, urgent=5))]))[1] == {}


def test_link_types_and_malformed_input(tmp_path):
    raw = [dns(f"{RNG.bytes(30).hex()}.x.example.net") for _ in range(30)]
    assert "dns_tunnel_suspected" in analyze(tmp_path, pcap(raw, link=101))[1]
    sll = [b"\x00" * 14 + b"\x08\x00" + p for p in raw]
    assert "dns_tunnel_suspected" in analyze(tmp_path, pcap(sll, link=113))[1]
    vlan = [ether(p, vlan=True) for p in raw]
    assert "dns_tunnel_suspected" in analyze(tmp_path, pcap(vlan))[1]
    echo6 = bytes([128, 0, 0, 0, 0, 1, 0, 1])
    frames6 = [b"\x00" * 12 + b"\x86\xdd" + b"\x60\x00\x00\x00" + (48).to_bytes(2, "big")
               + b"\x3a\x40" + b"\x00" * 32 + echo6 + RNG.bytes(40) for _ in range(6)]
    assert "icmp_tunnel_suspected" in analyze(tmp_path, pcap(frames6))[1]
    junk = [ether(b"\x45" + b"\x00" * 5), b"\x00" * 12 + b"\x08\x06" + b"\x00" * 28,
            ether(ipv4(17, struct.pack(">HHHH", 1, 53, 8, 0) + b"\x00" * 5)),
            ether(b"\x50" + b"\x00" * 30)]
    assert analyze(tmp_path, pcap(junk, link=1))[1] == {}
    assert analyze(tmp_path, pcap(raw, link=999))[1] == {}
    assert analyze(tmp_path, b"\x0a\x0d\x0d\x0a" + b"\x00" * 4 + b"\x4d\x3c\x2b\x1a")[1] == {}
    assert analyze(tmp_path, b"not a capture")[1] == {}
    short_block = pcapng([])[:28] + struct.pack("<III", 6, 4, 0)
    assert analyze(tmp_path, short_block, ".pcapng")[1] == {}


def test_dns_parser_edges():
    msg = struct.pack(">HHHHHH", 1, 0x8100, 1, 0, 0, 0)
    assert nc.dns_query(msg + b"\x03abc\x00\x00\x01\x00\x01") is None  # response bit
    assert nc.dns_query(struct.pack(">HHHHHH", 1, 0, 1, 0, 0, 0) + b"\xc0\x0c") is None
    assert nc.dns_query(struct.pack(">HHHHHH", 1, 0, 1, 0, 0, 0) + b"\x00\x00") is None
    assert nc.dns_query(b"\x00" * 4) is None
    assert nc.standard_ping(b"") and not nc.standard_ping(b"\x01\x05\x09")
    assert nc.entropy(b"") == 0.0


def test_scope_limits(tmp_path, monkeypatch):
    assert analyze(tmp_path, pcap(benign()), ".txt")[0] == "unsupported"
    monkeypatch.setattr(nc, "MAX_PACKETS", 5)
    assert "dns_tunnel_suspected" not in analyze(tmp_path, pcap(iodine_like()))[1]
    monkeypatch.setattr(nc, "MAX_BYTES", 10)
    assert analyze(tmp_path, pcap(benign()))[0] == "unsupported"
