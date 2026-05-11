"""
WhatsApp Messaging System - Working Python Implementation
Demonstrates: message delivery flow, offline message queue, message status
              (sent/delivered/read), group messaging fan-out, presence tracking
              with heartbeat, read receipt tracking, TIMEUUID-style ordering,
              unread count management.
No external dependencies — standard library only.
"""

import uuid
import time
import collections
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set, Tuple
from enum import IntEnum


# ---------------------------------------------------------------------------
# Message Status Enum
# ---------------------------------------------------------------------------
class MessageStatus(IntEnum):
    PENDING   = 0    # client queued, not yet sent to server
    SENT      = 1    # server received and stored (single grey tick)
    DELIVERED = 2    # recipient's device received it (double grey tick)
    READ      = 3    # recipient opened the conversation (double blue tick)


# ---------------------------------------------------------------------------
# TIMEUUID-style ID generator
# ---------------------------------------------------------------------------
class TimeUUIDGenerator:
    """
    Generates time-sortable unique message IDs.
    Format: (timestamp_ms, sequence, node_id) encoded as a comparable tuple.
    In production: Cassandra TIMEUUID (UUID v1).
    """
    _sequence = 0
    _last_ms = -1

    @classmethod
    def generate(cls) -> str:
        """Return a TIMEUUID-like string. Sortable lexicographically = chronologically."""
        ms = int(time.time() * 1000)
        if ms == cls._last_ms:
            cls._sequence += 1
        else:
            cls._sequence = 0
            cls._last_ms = ms
        # Format: zero-padded timestamp + sequence for lexicographic sorting
        return f"{ms:016d}-{cls._sequence:06d}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def extract_timestamp_ms(timeuuid: str) -> int:
        """Extract millisecond timestamp from TIMEUUID string."""
        return int(timeuuid.split("-")[0])


# ---------------------------------------------------------------------------
# Presence Service
# ---------------------------------------------------------------------------
class PresenceService:
    """
    Tracks user online/offline status.
    Uses heartbeat model: TTL = 30s, heartbeat every 15s.
    In production: Redis HSET with TTL.
    """

    HEARTBEAT_INTERVAL = 15     # seconds
    PRESENCE_TTL = 30           # seconds

    def __init__(self):
        # { user_id: { status, last_heartbeat, last_seen, server_id } }
        self._presence: Dict[int, dict] = {}

    def user_online(self, user_id: int, server_id: str = "conn_server_1") -> None:
        self._presence[user_id] = {
            "status": "online",
            "last_heartbeat": time.time(),
            "last_seen": datetime.utcnow(),
            "server_id": server_id,
        }

    def heartbeat(self, user_id: int) -> None:
        """Client sends heartbeat every 15s. Refreshes TTL."""
        if user_id in self._presence:
            self._presence[user_id]["last_heartbeat"] = time.time()
            self._presence[user_id]["status"] = "online"

    def user_offline(self, user_id: int) -> None:
        if user_id in self._presence:
            self._presence[user_id]["status"] = "offline"
            self._presence[user_id]["last_seen"] = datetime.utcnow()

    def is_online(self, user_id: int) -> bool:
        """Check if user is online (heartbeat within TTL)."""
        rec = self._presence.get(user_id)
        if not rec:
            return False
        if rec["status"] != "online":
            return False
        # TTL check: if last heartbeat > PRESENCE_TTL ago, consider offline
        age = time.time() - rec["last_heartbeat"]
        return age < self.PRESENCE_TTL

    def get_presence(self, user_id: int) -> dict:
        rec = self._presence.get(user_id)
        if not rec or not self.is_online(user_id):
            last_seen = rec["last_seen"].isoformat() if rec else None
            return {"user_id": user_id, "status": "offline", "last_seen": last_seen}
        return {
            "user_id": user_id,
            "status": "online",
            "last_seen": rec["last_seen"].isoformat(),
            "server_id": rec.get("server_id"),
        }

    def get_connection_server(self, user_id: int) -> Optional[str]:
        """Return which connection server holds this user's WebSocket."""
        if self.is_online(user_id):
            return self._presence[user_id].get("server_id")
        return None


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class Message:
    def __init__(
        self,
        conversation_id: str,
        sender_id: int,
        content: str,                    # In E2EE: this would be ciphertext
        message_type: str = "text",
        media_url: Optional[str] = None,
    ):
        self.message_id = TimeUUIDGenerator.generate()  # TIMEUUID (time-sortable)
        self.conversation_id = conversation_id
        self.sender_id = sender_id
        self.content = content
        self.message_type = message_type
        self.media_url = media_url
        self.status = MessageStatus.SENT      # server stored it
        self.created_at = datetime.utcnow()
        self.is_deleted = False
        # Per-recipient delivery status (for group messages)
        self.recipient_status: Dict[int, MessageStatus] = {}

    def __repr__(self):
        return f"Message({self.message_id[:20]}..., from={self.sender_id}, '{self.content[:30]}')"


class Conversation:
    def __init__(self, conversation_id: str, conversation_type: str = "dm"):
        self.conversation_id = conversation_id
        self.conversation_type = conversation_type  # "dm" or "group"
        self.members: Set[int] = set()
        self.created_at = datetime.utcnow()

    def add_member(self, user_id: int) -> None:
        self.members.add(user_id)

    def remove_member(self, user_id: int) -> None:
        self.members.discard(user_id)


class Group:
    MAX_MEMBERS = 1024

    def __init__(self, group_id: str, name: str, created_by: int):
        self.group_id = group_id
        self.name = name
        self.created_by = created_by
        self.members: Set[int] = {created_by}
        self.admins: Set[int] = {created_by}
        self.conversation_id = f"group_{group_id}"
        self.created_at = datetime.utcnow()

    def add_member(self, user_id: int) -> bool:
        if len(self.members) >= self.MAX_MEMBERS:
            return False
        self.members.add(user_id)
        return True

    def is_member(self, user_id: int) -> bool:
        return user_id in self.members


# ---------------------------------------------------------------------------
# WhatsApp Core System
# ---------------------------------------------------------------------------
class WhatsAppSystem:
    """
    WhatsApp-like messaging system.
    Features:
    - One-to-one messaging with online/offline handling
    - Message status (sent/delivered/read)
    - Group messaging with fan-out
    - Offline message queue (delivered on reconnect)
    - Presence tracking with heartbeat
    - Unread count tracking
    - Read receipts
    """

    def __init__(self):
        # Users: id -> { name, phone }
        self._users: Dict[int, dict] = {}

        # Conversations (DM + group): conv_id -> Conversation
        self._conversations: Dict[str, Conversation] = {}

        # Groups: group_id -> Group
        self._groups: Dict[str, Group] = {}

        # Message store: conv_id -> sorted list of Message (TIMEUUID order)
        # In production: Cassandra table partitioned by conversation_id
        self._messages: Dict[str, List[Message]] = collections.defaultdict(list)

        # Offline message queue: user_id -> deque of (conv_id, message_id)
        # In production: Redis List with TTL
        self._offline_queue: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=10000)
        )

        # Unread counts: (user_id, conv_id) -> count
        self._unread_counts: Dict[Tuple[int, str], int] = collections.defaultdict(int)

        # Last read message per user per conversation
        self._last_read: Dict[Tuple[int, str], str] = {}

        # Presence Service
        self._presence = PresenceService()

        # Message index for quick lookup: message_id -> Message
        self._msg_index: Dict[str, Message] = {}

        # Delivery callback simulation (WebSocket push)
        self._delivery_log: List[str] = []

    # ------------------------------------------------------------------
    # User & Registration
    # ------------------------------------------------------------------

    def register_user(self, user_id: int, name: str, phone: str) -> dict:
        self._users[user_id] = {"name": name, "phone": phone, "id": user_id}
        return {"user_id": user_id, "name": name}

    def _get_or_create_dm_conversation(self, user_a: int, user_b: int) -> str:
        """
        Get or create DM conversation ID for two users.
        Convention: always sort user IDs to get consistent conv_id.
        """
        ids = sorted([user_a, user_b])
        conv_id = f"dm_{ids[0]}_{ids[1]}"
        if conv_id not in self._conversations:
            conv = Conversation(conv_id, "dm")
            conv.add_member(user_a)
            conv.add_member(user_b)
            self._conversations[conv_id] = conv
        return conv_id

    # ------------------------------------------------------------------
    # Presence
    # ------------------------------------------------------------------

    def user_connect(self, user_id: int, server_id: str = "server_1") -> dict:
        """User comes online. Deliver any offline messages."""
        self._presence.user_online(user_id, server_id)
        # Deliver offline queue
        delivered = self._deliver_offline_messages(user_id)
        return {
            "status": "online",
            "offline_messages_delivered": delivered,
        }

    def user_disconnect(self, user_id: int) -> None:
        self._presence.user_offline(user_id)

    def heartbeat(self, user_id: int) -> None:
        self._presence.heartbeat(user_id)

    def get_presence(self, user_id: int) -> dict:
        return self._presence.get_presence(user_id)

    # ------------------------------------------------------------------
    # Send Message (DM)
    # ------------------------------------------------------------------

    def send_message(
        self,
        sender_id: int,
        recipient_id: int,
        content: str,
        message_type: str = "text",
    ) -> dict:
        """
        Send a direct message.
        If recipient is online: push immediately.
        If offline: queue message and send push notification.
        """
        conv_id = self._get_or_create_dm_conversation(sender_id, recipient_id)
        msg = Message(conv_id, sender_id, content, message_type)

        # Store in Cassandra (simulated as sorted list)
        self._messages[conv_id].append(msg)
        self._msg_index[msg.message_id] = msg

        # Initialize delivery status for recipient
        msg.recipient_status[recipient_id] = MessageStatus.SENT

        # Check recipient online status
        if self._presence.is_online(recipient_id):
            # Deliver via WebSocket (simulated)
            self._push_to_client(recipient_id, msg)
            msg.recipient_status[recipient_id] = MessageStatus.DELIVERED
            msg.status = MessageStatus.DELIVERED
            self._delivery_log.append(
                f"PUSH: msg {msg.message_id[:16]} -> user {recipient_id} [online]"
            )
        else:
            # Queue for offline delivery
            self._offline_queue[recipient_id].append((conv_id, msg.message_id))
            self._unread_counts[(recipient_id, conv_id)] += 1
            # Push notification (simulated)
            self._delivery_log.append(
                f"PUSH_NOTIF: msg {msg.message_id[:16]} -> user {recipient_id} [offline]"
            )

        return {
            "message_id": msg.message_id,
            "conversation_id": conv_id,
            "status": msg.status.name,
        }

    def _push_to_client(self, user_id: int, msg: Message) -> None:
        """Simulate pushing message to client via WebSocket."""
        self._unread_counts[(user_id, msg.conversation_id)] += 1

    def _deliver_offline_messages(self, user_id: int) -> int:
        """
        Deliver all queued offline messages to a user who just connected.
        In production: reads from Cassandra `offline_messages` or Redis queue.
        """
        queue = self._offline_queue[user_id]
        count = 0
        while queue:
            conv_id, msg_id = queue.popleft()
            msg = self._msg_index.get(msg_id)
            if msg and not msg.is_deleted:
                msg.recipient_status[user_id] = MessageStatus.DELIVERED
                if msg.status == MessageStatus.SENT:
                    msg.status = MessageStatus.DELIVERED
                # Notify sender of delivery
                self._notify_sender_status(msg, MessageStatus.DELIVERED)
                count += 1
        return count

    def _notify_sender_status(self, msg: Message, status: MessageStatus) -> None:
        """Notify original sender about message delivery/read status."""
        self._delivery_log.append(
            f"STATUS_UPDATE: msg {msg.message_id[:16]}, status={status.name}"
        )

    # ------------------------------------------------------------------
    # Read Receipts
    # ------------------------------------------------------------------

    def mark_as_read(self, user_id: int, conversation_id: str, up_to_message_id: str) -> dict:
        """
        Mark all messages up to `up_to_message_id` as read by user.
        Resets unread count. Notifies senders of read status.
        """
        self._last_read[(user_id, conversation_id)] = up_to_message_id
        self._unread_counts[(user_id, conversation_id)] = 0

        # Update status for messages in conversation
        updated = 0
        for msg in self._messages.get(conversation_id, []):
            # Only update messages up to the specified ID
            if msg.message_id <= up_to_message_id and msg.sender_id != user_id:
                if msg.recipient_status.get(user_id, MessageStatus.SENT) < MessageStatus.READ:
                    msg.recipient_status[user_id] = MessageStatus.READ
                    # Update overall message status if all recipients read it
                    all_read = all(
                        s >= MessageStatus.READ for s in msg.recipient_status.values()
                    )
                    if all_read:
                        msg.status = MessageStatus.READ
                    self._notify_sender_status(msg, MessageStatus.READ)
                    updated += 1

        return {
            "conversation_id": conversation_id,
            "messages_marked_read": updated,
            "unread_count": 0,
        }

    def get_unread_count(self, user_id: int, conversation_id: Optional[str] = None) -> dict:
        """Get unread message count for a user."""
        if conversation_id:
            return {
                "conversation_id": conversation_id,
                "unread_count": self._unread_counts[(user_id, conversation_id)],
            }
        # Total across all conversations
        total = sum(
            count for (uid, cid), count in self._unread_counts.items() if uid == user_id
        )
        return {"user_id": user_id, "total_unread": total}

    # ------------------------------------------------------------------
    # Get Messages (conversation history)
    # ------------------------------------------------------------------

    def get_messages(
        self,
        user_id: int,
        conversation_id: str,
        limit: int = 50,
        before_message_id: Optional[str] = None,
    ) -> dict:
        """
        Fetch message history for a conversation.
        Paginated using TIMEUUID cursor.
        In production: Cassandra query with WHERE message_id < before_id.
        """
        all_msgs = self._messages.get(conversation_id, [])
        # Sort by message_id (TIMEUUID = chronological)
        sorted_msgs = sorted(all_msgs, key=lambda m: m.message_id, reverse=True)

        # Apply cursor
        if before_message_id:
            sorted_msgs = [m for m in sorted_msgs if m.message_id < before_message_id]

        page = sorted_msgs[:limit]

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "message_id": m.message_id,
                    "sender_id": m.sender_id,
                    "content": m.content if not m.is_deleted else "[deleted]",
                    "message_type": m.message_type,
                    "status": m.status.name,
                    "recipient_status": {uid: s.name for uid, s in m.recipient_status.items()},
                    "created_at": m.created_at.isoformat(),
                }
                for m in page
            ],
            "has_more": len(sorted_msgs) > limit,
        }

    # ------------------------------------------------------------------
    # Group Messaging
    # ------------------------------------------------------------------

    def create_group(self, creator_id: int, name: str, member_ids: List[int]) -> dict:
        """Create a group conversation."""
        group_id = str(uuid.uuid4())[:8]
        group = Group(group_id, name, creator_id)

        for uid in member_ids:
            if uid != creator_id:
                if not group.add_member(uid):
                    return {"error": f"group max size {Group.MAX_MEMBERS} reached"}

        self._groups[group_id] = group

        # Create conversation record
        conv = Conversation(group.conversation_id, "group")
        for uid in group.members:
            conv.add_member(uid)
        self._conversations[group.conversation_id] = conv

        return {
            "group_id": group_id,
            "conversation_id": group.conversation_id,
            "name": name,
            "member_count": len(group.members),
        }

    def send_group_message(
        self, sender_id: int, group_id: str, content: str, message_type: str = "text"
    ) -> dict:
        """
        Send message to a group.
        Fan-out: deliver to each online member immediately;
                 queue for offline members.
        Store one message, track per-member delivery status.
        """
        group = self._groups.get(group_id)
        if not group:
            return {"error": "group not found"}
        if not group.is_member(sender_id):
            return {"error": "sender is not a group member"}

        msg = Message(group.conversation_id, sender_id, content, message_type)
        self._messages[group.conversation_id].append(msg)
        self._msg_index[msg.message_id] = msg

        # Fan-out to each member (except sender)
        online_deliveries = 0
        offline_queued = 0
        recipients = group.members - {sender_id}

        for uid in recipients:
            msg.recipient_status[uid] = MessageStatus.SENT
            if self._presence.is_online(uid):
                # Deliver via WebSocket
                self._push_to_client(uid, msg)
                msg.recipient_status[uid] = MessageStatus.DELIVERED
                online_deliveries += 1
                self._delivery_log.append(
                    f"GROUP_PUSH: msg {msg.message_id[:16]} -> user {uid} [online]"
                )
            else:
                # Queue for offline delivery
                self._offline_queue[uid].append((group.conversation_id, msg.message_id))
                self._unread_counts[(uid, group.conversation_id)] += 1
                offline_queued += 1

        # Overall message status: DELIVERED if at least one member received it
        delivered_count = sum(
            1 for s in msg.recipient_status.values() if s >= MessageStatus.DELIVERED
        )
        if delivered_count > 0:
            msg.status = MessageStatus.DELIVERED

        return {
            "message_id": msg.message_id,
            "group_id": group_id,
            "conversation_id": group.conversation_id,
            "online_deliveries": online_deliveries,
            "offline_queued": offline_queued,
            "total_recipients": len(recipients),
        }

    def add_group_member(self, admin_id: int, group_id: str, new_member_id: int) -> dict:
        group = self._groups.get(group_id)
        if not group:
            return {"error": "group not found"}
        if admin_id not in group.admins:
            return {"error": "only admins can add members"}
        if not group.add_member(new_member_id):
            return {"error": f"group is full (max {Group.MAX_MEMBERS})"}
        conv = self._conversations.get(group.conversation_id)
        if conv:
            conv.add_member(new_member_id)
        return {"group_id": group_id, "member_count": len(group.members)}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def delete_message(self, sender_id: int, message_id: str, for_everyone: bool = False) -> dict:
        msg = self._msg_index.get(message_id)
        if not msg:
            return {"error": "message not found"}
        if msg.sender_id != sender_id:
            return {"error": "can only delete own messages"}
        if for_everyone:
            msg.content = ""
            msg.is_deleted = True
            return {"deleted": True, "for_everyone": True}
        else:
            # "Delete for me" — just mark locally (simplified)
            return {"deleted": True, "for_everyone": False}

    def get_conversation_list(self, user_id: int) -> list:
        """Get all conversations for a user, sorted by most recent message."""
        user_convs = []
        for conv_id, conv in self._conversations.items():
            if user_id not in conv.members:
                continue
            msgs = self._messages.get(conv_id, [])
            last_msg = msgs[-1] if msgs else None
            unread = self._unread_counts[(user_id, conv_id)]
            user_convs.append({
                "conversation_id": conv_id,
                "type": conv.conversation_type,
                "unread_count": unread,
                "last_message": last_msg.content[:40] if last_msg and not last_msg.is_deleted else None,
                "last_message_at": last_msg.created_at.isoformat() if last_msg else None,
            })
        user_convs.sort(key=lambda x: x["last_message_at"] or "", reverse=True)
        return user_convs


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------
def run_demo():
    print("=" * 65)
    print("WHATSAPP MESSAGING SYSTEM DEMO")
    print("=" * 65)

    wa = WhatsAppSystem()

    # Register users
    wa.register_user(1, "Alice", "+1-555-0101")
    wa.register_user(2, "Bob", "+1-555-0102")
    wa.register_user(3, "Charlie", "+1-555-0103")
    wa.register_user(4, "Diana", "+1-555-0104")
    print("\n[1] Registered 4 users: Alice, Bob, Charlie, Diana")

    # Users come online
    print("\n[2] Alice and Bob come online; Charlie stays offline")
    r1 = wa.user_connect(1, "conn_server_1")  # Alice
    r2 = wa.user_connect(2, "conn_server_2")  # Bob
    # Charlie stays offline
    print(f"    Alice online: {wa.get_presence(1)['status']}")
    print(f"    Bob online  : {wa.get_presence(2)['status']}")
    print(f"    Charlie     : {wa.get_presence(3)['status']}")

    # Alice sends message to Bob (online -> immediate delivery)
    print("\n[3] Alice -> Bob (online delivery)")
    r = wa.send_message(1, 2, "Hey Bob! How are you?")
    print(f"    Message ID: {r['message_id'][:30]}...")
    print(f"    Status    : {r['status']}")

    r2 = wa.send_message(1, 2, "Are you free for lunch today?")
    print(f"    Status    : {r2['status']}")

    # Alice sends message to Charlie (offline -> queue)
    print("\n[4] Alice -> Charlie (offline — queued)")
    r3 = wa.send_message(1, 3, "Charlie, check your messages!")
    r4 = wa.send_message(1, 3, "Let's catch up this week!")
    print(f"    Status: {r3['status']} (should be SENT — Charlie offline)")
    print(f"    Offline queue size for Charlie: {len(wa._offline_queue[3])}")

    # Bob reads Alice's messages
    print("\n[5] Bob marks Alice's messages as read (blue ticks)")
    conv_id_ab = wa._get_or_create_dm_conversation(1, 2)
    msgs_for_bob = wa.get_messages(2, conv_id_ab)
    if msgs_for_bob["messages"]:
        last_id = msgs_for_bob["messages"][0]["message_id"]  # newest first
        read_result = wa.mark_as_read(2, conv_id_ab, last_id)
        print(f"    Messages marked read: {read_result['messages_marked_read']}")
        print(f"    Bob's unread count  : {read_result['unread_count']}")

    # Show Alice's message statuses after Bob reads
    print("\n[6] Message status after read receipt")
    history = wa.get_messages(1, conv_id_ab)
    for m in history["messages"]:
        print(f"    '{m['content'][:40]}' -> status: {m['status']}")

    # Charlie comes online — offline messages delivered
    print("\n[7] Charlie comes online — receives offline messages")
    charlie_connect = wa.user_connect(3, "conn_server_3")
    print(f"    Offline messages delivered: {charlie_connect['offline_messages_delivered']}")
    print(f"    Charlie is now: {wa.get_presence(3)['status']}")

    # Group messaging
    print("\n[8] Group chat: 'Team Lunch'")
    group = wa.create_group(1, "Team Lunch", [2, 3, 4])
    print(f"    Group created: id={group['group_id']}, members={group['member_count']}")

    # Alice sends message to group
    print("\n[9] Alice sends message to group")
    gr1 = wa.send_group_message(1, group["group_id"], "Hey team, lunch at 12:30 today?")
    print(f"    Online deliveries : {gr1['online_deliveries']}")
    print(f"    Offline queued    : {gr1['offline_queued']}")
    print(f"    Total recipients  : {gr1['total_recipients']}")
    print(f"    Message status    : ", end="")
    gconv_id = group["conversation_id"]
    last_group_msg = wa._messages[gconv_id][-1]
    print({uid: s.name for uid, s in last_group_msg.recipient_status.items()})

    # Diana was offline — check her queue
    print(f"\n    Diana offline queue size: {len(wa._offline_queue[4])}")
    # Diana comes online
    diana_connect = wa.user_connect(4, "conn_server_4")
    print(f"    Diana's offline messages delivered: {diana_connect['offline_messages_delivered']}")

    # Unread counts
    print("\n[10] Unread counts")
    for uid, name in [(1, "Alice"), (2, "Bob"), (3, "Charlie"), (4, "Diana")]:
        total = wa.get_unread_count(uid)["total_unread"]
        print(f"    {name}: {total} unread messages")

    # Conversation list for Alice
    print("\n[11] Alice's conversation list")
    convs = wa.get_conversation_list(1)
    for c in convs:
        print(f"    [{c['type']:5}] {c['conversation_id'][:30]:30} | "
              f"unread={c['unread_count']} | last: {(c['last_message'] or '')[:30]}")

    # Message deletion
    print("\n[12] Message deletion")
    msg_to_delete_id = r['message_id']
    del_result = wa.delete_message(1, msg_to_delete_id, for_everyone=True)
    print(f"    Deleted for everyone: {del_result}")
    # Verify
    hist = wa.get_messages(2, conv_id_ab)
    for m in hist["messages"]:
        if m["message_id"] == msg_to_delete_id:
            print(f"    Content after deletion: '{m['content']}'")

    # TIMEUUID ordering demo
    print("\n[13] TIMEUUID time-ordering")
    ids = [TimeUUIDGenerator.generate() for _ in range(5)]
    ts_values = [TimeUUIDGenerator.extract_timestamp_ms(mid) for mid in ids]
    print(f"    IDs are in ascending order: {ids[0] < ids[1] < ids[2]}")
    print(f"    Timestamps (ms): {ts_values}")

    # Presence heartbeat demo
    print("\n[14] Presence heartbeat (TTL simulation)")
    print(f"    Alice online before heartbeat: {wa.get_presence(1)['status']}")
    wa.heartbeat(1)
    print(f"    After heartbeat: {wa.get_presence(1)['status']}")

    # Delivery log (last 10 entries)
    print("\n[15] Delivery log (last 10 events)")
    for entry in wa._delivery_log[-10:]:
        print(f"    {entry}")

    print("\n" + "=" * 65)
    print("DEMO COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
