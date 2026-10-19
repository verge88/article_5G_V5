"""Authenticated experimental KIE over the real SCP -> NF HTTP/2 path."""
import hashlib
import hmac
import json
import secrets
import socket
import time
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import DataReceived, ResponseReceived, StreamEnded, StreamReset

NFS = {"SMF": "127.0.0.4", "AMF": "127.0.0.5", "AUSF": "127.0.0.11",
       "UDM": "127.0.0.12", "PCF": "127.0.0.13", "UDR": "127.0.0.20"}


def key_for(master, ip):
    return hmac.digest(hashlib.sha256(master.encode()).digest(), ip.encode(), "sha256")


def decode_response(body, master, ip, seq, nonce):
    envelope = json.loads(body)
    payload = envelope["payload"]
    mac = hmac.new(key_for(master, ip), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, envelope["mac"]):
        raise ValueError("KIE authentication failed")
    parts = payload.split("|")
    if len(parts) != 9 or parts[:3] != [ip, str(seq), nonce]:
        raise ValueError("KIE identity, sequence or nonce mismatch")
    start, end, requests, responses, errors, service_us = map(int, parts[3:])
    if end <= start or min(requests, responses, errors, service_us) < 0:
        raise ValueError("Invalid KIE interval or counters")
    if abs(time.time() * 1e6 - end) > 5e6:
        raise ValueError("Stale KIE timestamp")
    return dict(begin_us=start, end_us=end, reported_requests=requests,
                reported_responses=responses, reported_errors=errors,
                reported_service_us=service_us, authenticated=True)


def probe(master, role, seq, timeout=1.0):
    ip = NFS[role]
    nonce = secrets.token_hex(16)
    auth = hmac.new(key_for(master, ip), f"{ip}|{seq}|{nonce}".encode(), hashlib.sha256).hexdigest()
    conn = H2Connection(H2Configuration(client_side=True, header_encoding="utf-8"))
    started = time.perf_counter()
    wire_bytes = 0
    body = bytearray()
    status = None
    with socket.create_connection(("127.0.0.200", 7777), timeout) as sock:
        conn.initiate_connection()
        conn.send_headers(1, [(":method", "GET"), (":scheme", "http"),
                             (":authority", "127.0.0.200:7777"), (":path", "/_kie/v1/report"),
                             ("user-agent", "NWDAF-research"),
                             ("3gpp-sbi-target-apiroot", f"http://{ip}:7777"),
                             ("x-kie-seq", str(seq)), ("x-kie-nonce", nonce), ("x-kie-auth", auth)], end_stream=True)
        data = conn.data_to_send()
        sock.sendall(data)
        wire_bytes += len(data)
        while time.perf_counter() - started < timeout:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("HTTP/2 peer closed before END_STREAM")
            wire_bytes += len(chunk)
            for event in conn.receive_data(chunk):
                if isinstance(event, ResponseReceived):
                    status = dict(event.headers).get(":status")
                elif isinstance(event, DataReceived):
                    body.extend(event.data)
                    conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                elif isinstance(event, StreamReset):
                    raise ConnectionError(f"HTTP/2 stream reset: {event.error_code}")
                elif isinstance(event, StreamEnded):
                    if status != "200":
                        raise ValueError(f"KIE HTTP status {status}: {body[:150]!r}")
                    row = decode_response(body, master, ip, seq, nonce)
                    row.update(role=role, ip=ip, sequence=seq,
                               kie_rtt_ms=1000*(time.perf_counter()-started),
                               kie_bytes=wire_bytes, envelope=json.loads(body))
                    return row
            data = conn.data_to_send()
            if data:
                sock.sendall(data)
                wire_bytes += len(data)
    raise TimeoutError("KIE response timed out")
