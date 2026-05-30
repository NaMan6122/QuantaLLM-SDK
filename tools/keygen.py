#!/usr/bin/env python3
"""
QuantaLLM SDK License Key Generator

Generates signed JWT license keys for customers.
The HMAC secret must match the one embedded in the SDK.

Usage:
    python keygen.py --pkg com.customer.app --expiry 2027-01-01
    python keygen.py --pkg com.customer.app --expiry 2027-01-01 --tier pro
"""

import argparse
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone


SECRET_HASH = bytes([
    0x71, 0x75, 0x61, 0x6e, 0x74, 0x61, 0x6c, 0x6c,
    0x6d, 0x2d, 0x73, 0x64, 0x6b, 0x2d, 0x76, 0x31
])
SECRET_SALT = bytes([
    0x51, 0x4c, 0x4d, 0x2d, 0x32, 0x30, 0x32, 0x36,
    0x2d, 0x6c, 0x69, 0x63, 0x65, 0x6e, 0x73, 0x65
])


def derive_secret() -> bytes:
    combined = SECRET_HASH + SECRET_SALT
    return hashlib.sha256(combined).digest()


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_key(pkg: str, expiry: str, tier: str = "standard") -> str:
    exp_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    exp_epoch = int(exp_dt.timestamp())
    iat_epoch = int(time.time())

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "pkg": pkg,
        "exp": exp_epoch,
        "iat": iat_epoch,
        "tier": tier,
    }

    header_b64 = b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = b64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    sign_input = f"{header_b64}.{payload_b64}"
    secret = derive_secret()
    signature = hmac.new(secret, sign_input.encode(), hashlib.sha256).digest()
    signature_b64 = b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def main():
    parser = argparse.ArgumentParser(description="QuantaLLM SDK License Key Generator")
    parser.add_argument("--pkg", required=True, help="Consumer app package name (e.g., com.customer.app)")
    parser.add_argument("--expiry", required=True, help="Expiry date in YYYY-MM-DD format")
    parser.add_argument("--tier", default="standard", choices=["standard", "pro", "enterprise"],
                        help="License tier (default: standard)")
    args = parser.parse_args()

    key = generate_key(args.pkg, args.expiry, args.tier)

    print(f"\n{'=' * 60}")
    print(f"  QuantaLLM SDK License Key")
    print(f"{'=' * 60}")
    print(f"  Package:  {args.pkg}")
    print(f"  Expiry:   {args.expiry}")
    print(f"  Tier:     {args.tier}")
    print(f"{'=' * 60}")
    print(f"\n{key}\n")


if __name__ == "__main__":
    main()
