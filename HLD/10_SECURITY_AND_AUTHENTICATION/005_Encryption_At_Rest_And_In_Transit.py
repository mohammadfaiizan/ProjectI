"""
ENCRYPTION AT REST AND IN TRANSIT
=====================================

Problem Statement:
Data must be protected both when stored (at rest) and when moving over
networks (in transit). Unencrypted data is exposed to storage breaches
and network interception.

Encryption in Transit (TLS/HTTPS):
  TLS 1.3 (2018): faster handshake (1-RTT), stronger ciphers.
  TLS 1.2: widely deployed, acceptable.
  TLS 1.0/1.1: deprecated (POODLE, BEAST vulnerabilities).
  Certificate pinning: client validates cert against known pin.
  mTLS: both client and server present certificates.
  Used by: HTTPS, gRPC, Kafka (TLS), database connections.

TLS Handshake (TLS 1.3):
  1. Client Hello: supported ciphers, key_share (DH public key).
  2. Server Hello: chosen cipher, key_share.
  3. Both derive shared secret via ECDH.
  4. Server sends Certificate + CertificateVerify + Finished.
  5. Client sends Finished. Application data starts.
  Total: 1 RTT (vs 2 RTT in TLS 1.2).

Encryption at Rest:
  Database TDE (Transparent Data Encryption):
    Encrypt at the storage engine level. Transparent to queries.
    PostgreSQL: pgcrypto extension. MySQL: InnoDB TDE. Oracle: TDE.
    Key stored in HSM or KMS.
  Application-level encryption:
    Encrypt before storing. Only application holds keys.
    Protects against DBA snooping or DB breach without key.
  Field-level encryption:
    Encrypt specific sensitive fields (SSN, credit card).
    Different keys per tenant (multi-tenancy isolation).
  File/volume encryption:
    OS-level: dm-crypt/LUKS (Linux), BitLocker (Windows).
    Cloud: AWS EBS encryption, GCP Persistent Disk encryption.

Envelope Encryption (AWS KMS model):
  Two-tier key hierarchy:
    Master Key (KEK): stored in KMS/HSM. Never leaves.
    Data Encryption Key (DEK): generated per record/file.
    DEK encrypted by KEK → stored alongside ciphertext.
  Decrypt: KMS decrypts DEK → DEK decrypts ciphertext.
  Benefit: rotate KEK without re-encrypting all data (just re-wrap DEKs).

Key Management:
  KMS (Key Management Service): AWS KMS, Google Cloud KMS, Vault.
  HSM (Hardware Security Module): physical device for key storage.
  Key rotation: generate new key version. Re-encrypt DEKs.
  Key lifecycle: active → disabled → deleted.

Symmetric vs Asymmetric:
  Symmetric (AES-256-GCM): same key for encrypt/decrypt.
               Fast. Used for data encryption.
  Asymmetric (RSA-2048, EC P-256): public/private key pair.
               Slow. Used for key exchange, digital signatures.
  Hybrid: asymmetric to exchange symmetric key → symmetric for data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import hmac
import os
import secrets
import base64
import json
import time


# ─────────────────────────────────────────────
# SYMMETRIC ENCRYPTION SIMULATION (AES-GCM like)
# ─────────────────────────────────────────────

class AESGCMSimulator:
    """
    Simulates AES-256-GCM: authenticated encryption with associated data (AEAD).
    Key = 32 bytes. Nonce = 12 bytes. Tag = 16 bytes.
    Real production: use cryptography.hazmat.primitives.ciphers.aead.AESGCM
    """

    KEY_SIZE   = 32   # bytes (256 bits)
    NONCE_SIZE = 12   # bytes (96 bits)
    TAG_SIZE   = 16   # bytes (128 bits)

    def generate_key(self) -> bytes:
        return secrets.token_bytes(self.KEY_SIZE)

    def encrypt(self, key: bytes, plaintext: bytes,
                aad: bytes = b"") -> Tuple[bytes, bytes, bytes]:
        """Returns (nonce, ciphertext, tag)."""
        if len(key) != self.KEY_SIZE:
            raise ValueError(f"Key must be {self.KEY_SIZE} bytes")
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        # Simulate encryption: XOR with key-stream derived from key+nonce
        key_stream = self._derive_keystream(key, nonce, len(plaintext))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream))
        # Simulate GCM authentication tag
        tag = self._compute_tag(key, nonce, ciphertext, aad)
        return nonce, ciphertext, tag

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes,
                tag: bytes, aad: bytes = b"") -> bytes:
        """Returns plaintext. Raises ValueError if authentication fails."""
        expected_tag = self._compute_tag(key, nonce, ciphertext, aad)
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Authentication failed: ciphertext tampered")
        key_stream = self._derive_keystream(key, nonce, len(ciphertext))
        return bytes(a ^ b for a, b in zip(ciphertext, key_stream))

    def _derive_keystream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        """Derive key stream via HKDF-like expansion."""
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest()
            stream += block
            counter += 1
        return stream[:length]

    def _compute_tag(self, key: bytes, nonce: bytes,
                      ciphertext: bytes, aad: bytes) -> bytes:
        """HMAC-based authentication tag (simplified GCM)."""
        tag_key = hashlib.sha256(key + b"tag" + nonce).digest()[:self.TAG_SIZE]
        data    = aad + len(aad).to_bytes(8, "big") + ciphertext + len(ciphertext).to_bytes(8, "big")
        return hmac.new(tag_key, data, hashlib.sha256).digest()[:self.TAG_SIZE]


# ─────────────────────────────────────────────
# ENVELOPE ENCRYPTION (KMS model)
# ─────────────────────────────────────────────

@dataclass
class EncryptedRecord:
    """Stores ciphertext + encrypted DEK + metadata for envelope encryption."""
    ciphertext        : bytes
    encrypted_dek     : bytes    # DEK encrypted by KEK
    nonce             : bytes
    tag               : bytes
    kek_version       : int
    algorithm         : str = "AES-256-GCM"
    created_at        : float = field(default_factory=time.time)


class KMSSimulator:
    """
    Simulates AWS KMS: manages KEK (master key), encrypts/decrypts DEKs.
    KEK never leaves the simulated HSM.
    """

    def __init__(self):
        self._kek_versions: Dict[int, bytes] = {}
        self._current_version = 0
        self._generate_kek()

    def _generate_kek(self):
        self._current_version += 1
        kek = secrets.token_bytes(32)
        self._kek_versions[self._current_version] = kek

    def rotate_kek(self) -> int:
        self._generate_kek()
        return self._current_version

    def generate_data_key(self) -> Tuple[bytes, bytes]:
        """Returns (plaintext_dek, encrypted_dek). Plaintext used to encrypt data."""
        dek     = secrets.token_bytes(32)
        enc_dek = self._kek_encrypt(dek)
        return dek, enc_dek

    def decrypt_data_key(self, encrypted_dek: bytes, kek_version: int) -> bytes:
        kek = self._kek_versions.get(kek_version)
        if not kek:
            raise ValueError(f"KEK version {kek_version} not found")
        return self._kek_decrypt(encrypted_dek, kek)

    def _kek_encrypt(self, dek: bytes) -> bytes:
        """Encrypt DEK with current KEK."""
        kek = self._kek_versions[self._current_version]
        # Simple simulation: HMAC-based wrapping
        wrapped = bytes(a ^ b for a, b in zip(
            dek, hashlib.sha256(kek + b"wrap").digest()[:len(dek)]
        ))
        version_prefix = self._current_version.to_bytes(4, "big")
        return version_prefix + wrapped

    def _kek_decrypt(self, encrypted_dek: bytes, kek: bytes) -> bytes:
        wrapped = encrypted_dek[4:]  # skip version prefix
        return bytes(a ^ b for a, b in zip(
            wrapped, hashlib.sha256(kek + b"wrap").digest()[:len(wrapped)]
        ))

    @property
    def current_kek_version(self) -> int:
        return self._current_version


class EnvelopeEncryptor:
    """
    Encrypts data using envelope encryption:
    1. Ask KMS for a DEK (plaintext + wrapped).
    2. Encrypt data with DEK.
    3. Store ciphertext + wrapped DEK.
    4. Discard plaintext DEK from memory.
    """

    def __init__(self, kms: KMSSimulator):
        self._kms  = kms
        self._aes  = AESGCMSimulator()

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> EncryptedRecord:
        plaintext_dek, encrypted_dek = self._kms.generate_data_key()
        nonce, ciphertext, tag = self._aes.encrypt(plaintext_dek, plaintext, aad)
        # DEK discarded after use (out of scope)
        return EncryptedRecord(
            ciphertext=ciphertext, encrypted_dek=encrypted_dek,
            nonce=nonce, tag=tag,
            kek_version=self._kms.current_kek_version,
        )

    def decrypt(self, record: EncryptedRecord, aad: bytes = b"") -> bytes:
        plaintext_dek = self._kms.decrypt_data_key(
            record.encrypted_dek, record.kek_version
        )
        return self._aes.decrypt(plaintext_dek, record.nonce,
                                  record.ciphertext, record.tag, aad)


# ─────────────────────────────────────────────
# FIELD-LEVEL ENCRYPTION
# ─────────────────────────────────────────────

class FieldEncryptor:
    """
    Encrypt specific sensitive fields in a record.
    Different keys per field type or tenant.
    """

    def __init__(self):
        self._aes  = AESGCMSimulator()
        self._keys : Dict[str, bytes] = {}

    def register_key(self, field_name: str, key: bytes = None):
        self._keys[field_name] = key or self._aes.generate_key()

    def encrypt_field(self, field_name: str, value: str) -> str:
        key = self._keys.get(field_name)
        if not key:
            raise KeyError(f"No key for field {field_name!r}")
        nonce, ct, tag = self._aes.encrypt(key, value.encode())
        blob = {"n": nonce.hex(), "c": ct.hex(), "t": tag.hex()}
        return "enc:" + base64.b64encode(json.dumps(blob).encode()).decode()

    def decrypt_field(self, field_name: str, encrypted_value: str) -> str:
        if not encrypted_value.startswith("enc:"):
            return encrypted_value  # not encrypted
        key  = self._keys.get(field_name)
        if not key:
            raise KeyError(f"No key for field {field_name!r}")
        blob = json.loads(base64.b64decode(encrypted_value[4:]))
        return self._aes.decrypt(
            key, bytes.fromhex(blob["n"]),
            bytes.fromhex(blob["c"]), bytes.fromhex(blob["t"])
        ).decode()

    def encrypt_record(self, record: Dict, sensitive_fields: List[str]) -> Dict:
        result = dict(record)
        for f in sensitive_fields:
            if f in result and result[f]:
                result[f] = self.encrypt_field(f, str(result[f]))
        return result

    def decrypt_record(self, record: Dict, sensitive_fields: List[str]) -> Dict:
        result = dict(record)
        for f in sensitive_fields:
            if f in result and isinstance(result[f], str) and result[f].startswith("enc:"):
                result[f] = self.decrypt_field(f, result[f])
        return result


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_encryption():
    print("=" * 65)
    print("ENCRYPTION AT REST AND IN TRANSIT")
    print("=" * 65)

    aes = AESGCMSimulator()

    # ── Symmetric Encryption ──────────────────────
    print("\n[1] AES-256-GCM ENCRYPTION (symmetric)")
    print("─" * 55)

    key       = aes.generate_key()
    plaintext = b"Sensitive user data: SSN=123-45-6789"
    aad       = b"record-type:user"

    nonce, ct, tag = aes.encrypt(key, plaintext, aad)
    print(f"  Plaintext:  {plaintext}")
    print(f"  Ciphertext: {ct.hex()[:40]}...  nonce={nonce.hex()[:16]}...")
    print(f"  Tag:        {tag.hex()}")

    recovered = aes.decrypt(key, nonce, ct, tag, aad)
    print(f"  Decrypted:  {recovered}")
    print(f"  Matches:    {recovered == plaintext}")

    # Tamper detection
    tampered_ct = bytes([ct[0] ^ 0xFF]) + ct[1:]
    try:
        aes.decrypt(key, nonce, tampered_ct, tag, aad)
        print("  Tamper: NOT detected (ERROR)")
    except ValueError as e:
        print(f"  Tamper detected: {e}")

    # ── Envelope Encryption ───────────────────────
    print("\n\n[2] ENVELOPE ENCRYPTION (KMS model)")
    print("─" * 55)

    kms       = KMSSimulator()
    encryptor = EnvelopeEncryptor(kms)

    sensitive = b"Credit card: 4532-1234-5678-9012"
    record    = encryptor.encrypt(sensitive, aad=b"payment-record")
    print(f"  Plaintext:     {sensitive}")
    print(f"  Ciphertext:    {record.ciphertext.hex()[:40]}...")
    print(f"  Encrypted DEK: {record.encrypted_dek.hex()[:20]}...")
    print(f"  KEK version:   {record.kek_version}")

    decrypted = encryptor.decrypt(record, aad=b"payment-record")
    print(f"  Decrypted:     {decrypted}")
    print(f"  Matches:       {decrypted == sensitive}")

    # Key rotation
    new_version = kms.rotate_kek()
    print(f"\n  KEK rotated → version {new_version}")
    # Old record still decryptable (old KEK version stored)
    old_decrypt = encryptor.decrypt(record, aad=b"payment-record")
    print(f"  Old record decryptable after KEK rotation: {old_decrypt == sensitive}")

    # ── Field-Level Encryption ────────────────────
    print("\n\n[3] FIELD-LEVEL ENCRYPTION")
    print("─" * 55)

    fe = FieldEncryptor()
    fe.register_key("ssn")
    fe.register_key("credit_card")
    fe.register_key("email")

    user_record = {
        "id"         : "user-42",
        "name"       : "Alice Smith",
        "email"      : "alice@example.com",
        "ssn"        : "123-45-6789",
        "credit_card": "4532-1234-5678-9012",
    }

    encrypted = fe.encrypt_record(user_record, ["ssn", "credit_card", "email"])
    print(f"  Original record:")
    for k, v in user_record.items():
        print(f"    {k}: {v}")
    print(f"\n  Encrypted record:")
    for k, v in encrypted.items():
        disp = v[:30] + "..." if len(str(v)) > 30 else v
        print(f"    {k}: {disp}")

    decrypted = fe.decrypt_record(encrypted, ["ssn", "credit_card", "email"])
    print(f"\n  Decrypted matches original: {decrypted == user_record}")

    # ── TLS Summary ───────────────────────────────
    print("\n\n[4] TLS / ENCRYPTION IN TRANSIT")
    print("─" * 55)

    tls = [
        ("TLS 1.3 (recommended)", "1-RTT handshake; ECDHE+AES-GCM; no RSA key exchange"),
        ("TLS 1.2 (acceptable)",  "2-RTT handshake; PFS with ECDHE; avoid RSA key exchange"),
        ("TLS 1.0/1.1 (disabled)","POODLE/BEAST vulnerable; banned by RFC 8996"),
        ("Certificate pinning",   "Client pins expected cert hash; prevents MITM with bad CA"),
        ("mTLS",                  "Both sides present certs; service-to-service auth"),
        ("HSTS",                  "Strict-Transport-Security: max-age=31536000; HTTPS only"),
        ("Cipher suites",         "TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384"),
    ]
    for proto, desc in tls:
        print(f"  {proto:<28} {desc}")

    # ── Encryption Design Guide ───────────────────
    print("\n\n[5] ENCRYPTION DESIGN CHECKLIST")
    print("─" * 55)

    checklist = [
        ("At rest: databases",    "TDE + application-level for sensitive fields"),
        ("At rest: files",        "AES-256-GCM with envelope encryption (KMS)"),
        ("In transit: HTTP",      "TLS 1.2+ only; HSTS; HTTPS everywhere"),
        ("In transit: internal",  "mTLS between services; Kafka TLS"),
        ("Key storage",           "HSM or managed KMS (AWS KMS, GCP KMS, Vault)"),
        ("Key rotation",          "Annual rotation minimum; immediate if compromised"),
        ("IV/nonce uniqueness",   "Never reuse nonce with same key (GCM catastrophic)"),
        ("Authenticated enc",     "Always AES-GCM or ChaCha20-Poly1305 (not ECB/CBC)"),
    ]
    for area, guidance in checklist:
        print(f"  {area:<28} {guidance}")


if __name__ == "__main__":
    demonstrate_encryption()
