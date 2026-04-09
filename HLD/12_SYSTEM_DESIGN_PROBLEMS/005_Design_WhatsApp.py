"""
SYSTEM DESIGN: WHATSAPP (Messaging)
======================================

Problem Statement:
Design a real-time messaging application supporting 1-1 chats,
group chats, media sharing, and message delivery receipts.

Functional Requirements:
  - Send and receive messages (text, media)
  - Group chats (up to 1024 members)
  - Message delivery receipts: sent ✓, delivered ✓✓, read ✓✓ (blue)
  - Online presence (last seen)
  - Push notifications for offline users

Non-Functional Requirements:
  - 2B users, 100B messages/day → ~1.15M messages/sec
  - Message latency: < 100ms (online user to online user)
  - Messages delivered at least once; ordered within a chat
  - End-to-end encrypted (E2EE) — Signal Protocol

Messaging Architecture:
  WebSocket:  Persistent connection per mobile client.
  Connection Server:  Maintains active WebSocket connections.
                      Horizontally scalable; each server handles ~100K connections.
  Message Router:     Routes messages to correct connection server.
  Message Queue:      Kafka or in-house MQ for async delivery.
  Push Gateway:       FCM/APNs for offline users.

Message Flow (1-1):
  Sender → Connection Server A (WebSocket) → Message Router →
  → Kafka (message event) → Message Worker →
  → If online: Connection Server B (WebSocket) → Receiver
  → If offline: Push Gateway (FCM/APNs) + store in DB

Storage:
  Messages: Cassandra (chat_id partition key, message_id clustering key).
  Media:    S3-compatible object store; URLs stored in message row.
  User data: MySQL (user_id, phone, name, profile_pic).
  Sessions: Redis (user_id → connection_server_id, websocket_id).

Message IDs:
  Per-chat monotonically increasing sequence number (not Snowflake).
  Cassandra counter per chat → sequence_number.
  Why per-chat: ordering only needed within a chat.
  Cross-chat IDs: globally unique UUID for deduplication.

E2EE (End-to-End Encryption):
  Signal Protocol: Double Ratchet + X3DH key agreement.
  Server never sees plaintext. Server stores ciphertext only.
  Key bundle: published to server, fetched by sender.
  Media: encrypted locally before upload; decryption key in message.

Group Chat:
  Sender sends message once to server.
  Server fan-out: deliver to all group members (up to 1024).
  Server decrypts? No. Sender encrypts once per recipient (or uses sender key).
  WhatsApp uses Signal's "Sender Key" for group: one encryption, fan-out.

Delivery Receipts:
  Sent:     Message stored on server (ack to sender).
  Delivered: Message delivered to device (device sends receipt).
             Aggregated: only one ✓✓ when all recipient devices acknowledge.
  Read:     User opens chat (device sends read receipt).

Last Seen / Presence:
  Online: WebSocket connected.
  Last seen: timestamp stored in Redis, synced to DB.
  Privacy: can hide last seen from non-contacts.
"""

from __future__ import annotations

import time
import uuid
import json
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────
# MESSAGE DELIVERY STATUS
# ─────────────────────────────────────────────

class DeliveryStatus(Enum):
    PENDING   = "pending"    # not yet sent to server
    SENT      = "sent"       # ✓ stored on server
    DELIVERED = "delivered"  # ✓✓ on device
    READ      = "read"       # ✓✓ (blue) opened by user
    FAILED    = "failed"


# ─────────────────────────────────────────────
# MESSAGE
# ─────────────────────────────────────────────

@dataclass
class Message:
    message_id:  str          # global UUID (for dedup)
    chat_id:     str
    seq_num:     int           # per-chat sequence number
    sender_id:   str
    content:     str           # encrypted ciphertext in real system
    media_url:   Optional[str]
    created_at:  float
    status:      DeliveryStatus = DeliveryStatus.PENDING
    message_type: str = "text"  # text, image, video, audio, doc

    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "chat_id":    self.chat_id,
            "seq_num":    self.seq_num,
            "sender_id":  self.sender_id,
            "content":    self.content[:50],
            "type":       self.message_type,
            "ts":         self.created_at,
            "status":     self.status.value,
        }


# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────

class ChatType(Enum):
    ONE_TO_ONE = "1:1"
    GROUP      = "group"


@dataclass
class Chat:
    chat_id:     str
    chat_type:   ChatType
    members:     Set[str]    = field(default_factory=set)
    name:        Optional[str] = None   # group name
    _seq:        int = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq


# ─────────────────────────────────────────────
# USER
# ─────────────────────────────────────────────

class PresenceStatus(Enum):
    ONLINE  = "online"
    OFFLINE = "offline"

@dataclass
class WhatsAppUser:
    user_id:     str
    phone:       str
    name:        str
    presence:    PresenceStatus = PresenceStatus.OFFLINE
    last_seen:   float = 0.0


# ─────────────────────────────────────────────
# SESSION STORE (Redis simulation)
# ─────────────────────────────────────────────

class SessionStore:
    """Maps user_id → connection_server_id (which server holds their WebSocket)."""

    def __init__(self):
        self._sessions: Dict[str, str] = {}   # user_id → server_id

    def connect(self, user_id: str, server_id: str):
        self._sessions[user_id] = server_id

    def disconnect(self, user_id: str):
        self._sessions.pop(user_id, None)

    def get_server(self, user_id: str) -> Optional[str]:
        return self._sessions.get(user_id)

    def is_online(self, user_id: str) -> bool:
        return user_id in self._sessions


# ─────────────────────────────────────────────
# MESSAGE STORE (Cassandra simulation)
# ─────────────────────────────────────────────

class MessageStore:
    """
    Per-chat message storage.
    Partition key: chat_id
    Clustering key: seq_num DESC (newest first)
    """

    def __init__(self):
        # chat_id → {seq_num: Message}
        self._messages: Dict[str, Dict[int, Message]] = defaultdict(dict)

    def store(self, message: Message):
        self._messages[message.chat_id][message.seq_num] = message

    def get_messages(self, chat_id: str, limit: int = 50,
                     before_seq: Optional[int] = None) -> List[Message]:
        msgs = list(self._messages.get(chat_id, {}).values())
        if before_seq is not None:
            msgs = [m for m in msgs if m.seq_num < before_seq]
        return sorted(msgs, key=lambda m: -m.seq_num)[:limit]

    def update_status(self, message_id: str, chat_id: str, status: DeliveryStatus):
        for msg in self._messages.get(chat_id, {}).values():
            if msg.message_id == message_id:
                msg.status = status
                return


# ─────────────────────────────────────────────
# PUSH NOTIFICATION GATEWAY
# ─────────────────────────────────────────────

@dataclass
class PushNotification:
    device_token: str
    title:        str
    body:         str
    data:         Dict[str, str]
    sent_at:      float = field(default_factory=time.time)


class PushGateway:
    """Simulated FCM/APNs gateway."""

    def __init__(self):
        self._sent: List[PushNotification] = []
        # user_id → device_tokens
        self._tokens: Dict[str, List[str]] = defaultdict(list)

    def register_token(self, user_id: str, token: str):
        self._tokens[user_id].append(token)

    def notify(self, user_id: str, title: str, body: str,
               data: Optional[Dict] = None) -> int:
        """Returns number of devices notified."""
        tokens = self._tokens.get(user_id, [])
        for token in tokens:
            self._sent.append(PushNotification(token, title, body, data or {}))
        return len(tokens)


# ─────────────────────────────────────────────
# E2EE SIMULATION (simplified)
# ─────────────────────────────────────────────

class E2EESimulator:
    """
    Simplified Signal Protocol simulation.
    Real: X3DH key agreement + Double Ratchet.
    Here: XOR with key hash (for demonstration only).
    """

    def __init__(self):
        self._key_bundles: Dict[str, str] = {}   # user_id → public_key

    def register_key_bundle(self, user_id: str):
        """Publish identity key to server."""
        key = hashlib.sha256(f"key_{user_id}".encode()).hexdigest()
        self._key_bundles[user_id] = key

    def encrypt(self, plaintext: str, recipient_id: str) -> str:
        """Simulate E2EE encryption (XOR with key bytes for demo)."""
        key = self._key_bundles.get(recipient_id, "0" * 64)
        pt_bytes  = plaintext.encode()
        key_bytes = bytes.fromhex(key * ((len(pt_bytes) // 32) + 1))[:len(pt_bytes)]
        ct_bytes  = bytes(a ^ b for a, b in zip(pt_bytes, key_bytes))
        return ct_bytes.hex()

    def decrypt(self, ciphertext: str, recipient_id: str) -> str:
        """Symmetric for demo (not real Signal Protocol)."""
        return self.encrypt(bytes.fromhex(ciphertext).decode("latin-1"),
                            recipient_id)


# ─────────────────────────────────────────────
# MESSAGING SERVICE
# ─────────────────────────────────────────────

class WhatsAppService:
    def __init__(self):
        self._users:    Dict[str, WhatsAppUser] = {}
        self._chats:    Dict[str, Chat]         = {}
        self._msgs      = MessageStore()
        self._sessions  = SessionStore()
        self._push      = PushGateway()
        self._e2ee      = E2EESimulator()

    def register_user(self, phone: str, name: str) -> WhatsAppUser:
        uid  = f"u_{hashlib.md5(phone.encode()).hexdigest()[:8]}"
        user = WhatsAppUser(uid, phone, name)
        self._users[uid] = user
        self._e2ee.register_key_bundle(uid)
        return user

    def connect(self, user_id: str, server_id: str = "server-1"):
        self._sessions.connect(user_id, server_id)
        user = self._users.get(user_id)
        if user:
            user.presence = PresenceStatus.ONLINE

    def disconnect(self, user_id: str):
        self._sessions.disconnect(user_id)
        user = self._users.get(user_id)
        if user:
            user.presence = PresenceStatus.OFFLINE
            user.last_seen = time.time()

    def create_chat(self, member_ids: List[str],
                    group_name: Optional[str] = None) -> Chat:
        chat_type = ChatType.GROUP if len(member_ids) > 2 else ChatType.ONE_TO_ONE
        chat = Chat(
            chat_id   = uuid.uuid4().hex[:12],
            chat_type = chat_type,
            members   = set(member_ids),
            name      = group_name,
        )
        self._chats[chat.chat_id] = chat
        return chat

    def send_message(self, chat_id: str, sender_id: str,
                     text: str, media_url: Optional[str] = None) -> Message:
        chat = self._chats.get(chat_id)
        if not chat:
            raise ValueError("Chat not found")
        if sender_id not in chat.members:
            raise ValueError("Not a member of this chat")

        seq_num = chat.next_seq()
        msg_id  = uuid.uuid4().hex

        # Encrypt for each recipient (simplified: just for 1:1)
        recipients = chat.members - {sender_id}
        encrypted_content = text  # In real: encrypt with Signal Protocol

        msg = Message(
            message_id   = msg_id,
            chat_id      = chat_id,
            seq_num      = seq_num,
            sender_id    = sender_id,
            content      = encrypted_content,
            media_url    = media_url,
            created_at   = time.time(),
            status       = DeliveryStatus.SENT,
            message_type = "image" if media_url else "text",
        )
        self._msgs.store(msg)

        # Deliver to online recipients; push notify offline
        delivered_count = 0
        for rid in recipients:
            if self._sessions.is_online(rid):
                msg.status = DeliveryStatus.DELIVERED
                delivered_count += 1
            else:
                sender = self._users.get(sender_id)
                sender_name = sender.name if sender else "Unknown"
                self._push.notify(rid, f"New message from {sender_name}",
                                  text[:50], {"chat_id": chat_id, "msg_id": msg_id})

        return msg

    def mark_read(self, chat_id: str, user_id: str, up_to_seq: int):
        """Mark all messages up to seq as read by user_id."""
        messages = self._msgs.get_messages(chat_id, limit=200)
        for msg in messages:
            if msg.seq_num <= up_to_seq and msg.sender_id != user_id:
                self._msgs.update_status(msg.message_id, chat_id, DeliveryStatus.READ)

    def get_history(self, chat_id: str, limit: int = 20) -> List[Message]:
        return self._msgs.get_messages(chat_id, limit)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_whatsapp():
    print("=" * 65)
    print("SYSTEM DESIGN: WHATSAPP")
    print("=" * 65)

    svc = WhatsAppService()

    # ── Register Users ────────────────────────
    print("\n[1] USERS")
    print("─" * 55)

    alice = svc.register_user("+1-555-0001", "Alice")
    bob   = svc.register_user("+1-555-0002", "Bob")
    carol = svc.register_user("+1-555-0003", "Carol")

    for u in [alice, bob, carol]:
        print(f"  {u.name:<8} id={u.user_id}  phone={u.phone}")

    # ── Connect (online) ──────────────────────
    svc.connect(alice.user_id)
    svc.connect(bob.user_id)
    # Carol is offline

    # ── 1:1 Chat ──────────────────────────────
    print("\n[2] ONE-TO-ONE CHAT")
    print("─" * 55)

    chat = svc.create_chat([alice.user_id, bob.user_id])
    print(f"  Chat ID: {chat.chat_id}  type={chat.chat_type.value}")

    msgs = [
        (alice.user_id, "Hey Bob, how's it going?"),
        (bob.user_id,   "Great! Just working on that PR."),
        (alice.user_id, "Nice! Send it over when ready."),
    ]
    for sender_id, text in msgs:
        msg = svc.send_message(chat.chat_id, sender_id, text)
        sender = svc._users[sender_id]
        status_icon = {"sent": "✓", "delivered": "✓✓", "read": "✓✓🔵"}
        icon = status_icon.get(msg.status.value, "?")
        print(f"  [{sender.name}] {text:<35} {icon}")

    # ── Read Receipt ──────────────────────────
    print("\n[3] READ RECEIPTS")
    print("─" * 55)

    svc.mark_read(chat.chat_id, bob.user_id, up_to_seq=10)
    history = svc.get_history(chat.chat_id, limit=5)
    for msg in reversed(history):
        sender = svc._users.get(msg.sender_id)
        print(f"  seq={msg.seq_num}  [{sender.name if sender else '?'}] "
              f"status={msg.status.value}")

    # ── Group Chat ────────────────────────────
    print("\n[4] GROUP CHAT")
    print("─" * 55)

    group = svc.create_chat(
        [alice.user_id, bob.user_id, carol.user_id],
        group_name="Dev Team"
    )
    print(f"  Group: '{group.name}'  members={len(group.members)}")

    msg1 = svc.send_message(group.chat_id, alice.user_id, "Team standup in 5 min!")
    print(f"  Alice sends to group → delivered online: Bob ✓✓  Carol: push notified")
    print(f"  Push notifications sent: {len(svc._push._sent)}")

    for notif in svc._push._sent[:2]:
        print(f"    [{notif.title}] {notif.body}")

    # ── Offline/Online ────────────────────────
    print("\n[5] PRESENCE")
    print("─" * 55)

    for u in [alice, bob, carol]:
        status = "ONLINE" if svc._sessions.is_online(u.user_id) else "offline"
        print(f"  {u.name:<8}: {status}")

    svc.disconnect(alice.user_id)
    print(f"\n  @alice disconnects → {alice.presence.value}  last_seen={alice.last_seen:.0f}")

    # ── E2EE ──────────────────────────────────
    print("\n[6] END-TO-END ENCRYPTION (simplified)")
    print("─" * 55)

    e2ee = svc._e2ee
    plaintext = "Secret message!"
    ciphertext = e2ee.encrypt(plaintext, bob.user_id)
    # In real system: server only sees ciphertext
    print(f"  Plaintext:  {plaintext}")
    print(f"  Ciphertext: {ciphertext[:32]}...")
    print(f"  Server stores: ciphertext only (never plaintext)")
    print(f"  Signal Protocol: X3DH + Double Ratchet for forward secrecy")

    # ── Architecture ──────────────────────────
    print("\n[7] WHATSAPP ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Transport",      "WebSocket over TLS; Erlang/BEAM for connection servers"),
        ("Message routing","Redis: user_id → connection_server_id"),
        ("Message store",  "Cassandra: chat_id partition, seq_num clustering"),
        ("Media",          "S3 object store; CDN for delivery; E2EE before upload"),
        ("Push",           "FCM (Android) + APNs (iOS) for offline delivery"),
        ("E2EE",           "Signal Protocol (X3DH + Double Ratchet)"),
        ("Group msg",      "Sender Key protocol: encrypt once, fan-out ciphertext"),
        ("Delivery status","Per-device ACK; blue ticks = all devices read"),
        ("Scale",          "Erlang actor model: 1M+ concurrent connections/node"),
        ("Dedup",          "message_id UUID + idempotent consumer on Kafka"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_whatsapp()
