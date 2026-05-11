"""
Design Slack — Python Simulation
==================================
Simulates core Slack mechanics:
  - Workspace/Channel/DM model
  - WebSocket connection management with pub/sub routing
  - Message threading (parent-reply relationship)
  - Presence service (heartbeat-based)
  - Unread count tracking (per user/channel)
  - Emoji reactions
  - Full-text search (inverted index)
  - At-least-once delivery with idempotency key
"""

import uuid
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class PresenceStatus(Enum):
    ONLINE = "online"
    AWAY = "away"
    DND = "dnd"
    OFFLINE = "offline"


@dataclass
class Message:
    message_id: str
    channel_id: str
    workspace_id: str
    user_id: str
    content: str
    thread_ts: Optional[str] = None    # Parent message_id if thread reply
    is_edited: bool = False
    is_deleted: bool = False
    edit_history: list = field(default_factory=list)
    reactions: dict = field(default_factory=dict)   # emoji -> set of user_ids
    created_at: float = field(default_factory=time.time)
    idempotency_key: Optional[str] = None


@dataclass
class Channel:
    channel_id: str
    workspace_id: str
    name: str
    is_private: bool = False
    is_dm: bool = False
    created_by: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class Workspace:
    workspace_id: str
    name: str
    slug: str
    owner_id: str
    plan: str = "free"


# ---------------------------------------------------------------------------
# Connection Manager (WebSocket Simulation)
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Simulates WebSocket connection servers with Redis pub/sub for cross-server routing.
    """

    def __init__(self):
        # user_id -> list of subscribed channel_ids
        self._user_channels: dict[str, set[str]] = defaultdict(set)
        # channel_id -> set of user_ids currently "connected"
        self._channel_subscribers: dict[str, set[str]] = defaultdict(set)
        # Delivered messages log: (user_id, message_id)
        self._delivery_log: list[tuple[str, str]] = []
        # Simulated Redis pub/sub: channel_id -> list of messages
        self._pubsub_bus: dict[str, list[dict]] = defaultdict(list)

    def connect_user(self, user_id: str, channel_ids: list[str]):
        """User connects and subscribes to their channels."""
        self._user_channels[user_id].update(channel_ids)
        for ch_id in channel_ids:
            self._channel_subscribers[ch_id].add(user_id)

    def disconnect_user(self, user_id: str):
        for ch_id in self._user_channels.get(user_id, set()):
            self._channel_subscribers[ch_id].discard(user_id)
        self._user_channels.pop(user_id, None)

    def publish(self, channel_id: str, message: dict):
        """Publish to Redis pub/sub channel (simulated)."""
        self._pubsub_bus[channel_id].append(message)
        # Fan out to all subscribers immediately (simulating connection server delivery)
        for user_id in self._channel_subscribers.get(channel_id, set()):
            if user_id != message.get("sender_id"):
                self._delivery_log.append((user_id, message.get("message_id", "")))

    def get_online_users(self, channel_id: str) -> set[str]:
        return self._channel_subscribers.get(channel_id, set()).copy()

    def delivery_stats(self) -> dict:
        return {
            "total_deliveries": len(self._delivery_log),
            "unique_recipients": len(set(uid for uid, _ in self._delivery_log))
        }


# ---------------------------------------------------------------------------
# Presence Service
# ---------------------------------------------------------------------------

class PresenceService:
    """Heartbeat-based presence tracking with TTL simulation."""

    HEARTBEAT_TTL = 60   # seconds before considered offline

    def __init__(self):
        # (user_id, workspace_id) -> (status, last_heartbeat)
        self._state: dict[tuple, tuple[PresenceStatus, float]] = {}
        self._status_history: list[dict] = []

    def update_presence(self, user_id: str, workspace_id: str,
                        status: PresenceStatus = PresenceStatus.ONLINE):
        self._state[(user_id, workspace_id)] = (status, time.time())
        self._status_history.append({
            "user_id": user_id,
            "workspace_id": workspace_id,
            "status": status.value,
            "timestamp": time.time()
        })

    def heartbeat(self, user_id: str, workspace_id: str):
        """Client sends heartbeat every 30 seconds to keep presence alive."""
        key = (user_id, workspace_id)
        if key in self._state:
            status, _ = self._state[key]
            self._state[key] = (status, time.time())
        else:
            self.update_presence(user_id, workspace_id, PresenceStatus.ONLINE)

    def get_presence(self, user_id: str, workspace_id: str) -> str:
        key = (user_id, workspace_id)
        if key not in self._state:
            return PresenceStatus.OFFLINE.value

        status, last_heartbeat = self._state[key]
        if time.time() - last_heartbeat > self.HEARTBEAT_TTL:
            return PresenceStatus.OFFLINE.value
        return status.value

    def get_workspace_presence(self, workspace_id: str, user_ids: list[str]) -> dict:
        return {uid: self.get_presence(uid, workspace_id) for uid in user_ids}


# ---------------------------------------------------------------------------
# Unread Counter
# ---------------------------------------------------------------------------

class UnreadCounter:
    """Per-user, per-channel unread count tracking using Redis-like hash map."""

    def __init__(self):
        # user_id -> {channel_id: count}
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # user_id -> {channel_id: last_read_message_id}
        self._last_read: dict[str, dict[str, str]] = defaultdict(dict)

    def increment(self, channel_id: str, member_ids: list[str],
                  message_id: str, sender_id: str):
        """Called when a new message is sent; increments count for all offline members."""
        for user_id in member_ids:
            if user_id != sender_id:
                self._counts[user_id][channel_id] += 1

    def mark_read(self, user_id: str, channel_id: str, last_message_id: str):
        """User opens channel — reset unread count."""
        self._counts[user_id][channel_id] = 0
        self._last_read[user_id][channel_id] = last_message_id

    def get_unread(self, user_id: str) -> list[dict]:
        return [
            {"channel_id": ch_id, "count": count}
            for ch_id, count in self._counts[user_id].items()
            if count > 0
        ]

    def total_unread(self, user_id: str) -> int:
        return sum(self._counts[user_id].values())


# ---------------------------------------------------------------------------
# Message Store (Simulated Cassandra)
# ---------------------------------------------------------------------------

class MessageStore:
    """Append-only message store. Partitioned by channel_id (Cassandra simulation)."""

    def __init__(self):
        # channel_id -> list of Messages (ordered by created_at)
        self._partitions: dict[str, list[Message]] = defaultdict(list)
        # thread_ts -> list of reply Messages
        self._threads: dict[str, list[Message]] = defaultdict(list)
        # Idempotency key -> message_id (dedup)
        self._idempotency_map: dict[str, str] = {}

    def store_message(self, message: Message) -> dict:
        # Idempotency check
        if message.idempotency_key:
            if message.idempotency_key in self._idempotency_map:
                existing_id = self._idempotency_map[message.idempotency_key]
                return {"status": "duplicate", "message_id": existing_id}
            self._idempotency_map[message.idempotency_key] = message.message_id

        self._partitions[message.channel_id].append(message)
        if message.thread_ts:
            self._threads[message.thread_ts].append(message)

        return {"status": "stored", "message_id": message.message_id}

    def get_channel_history(self, channel_id: str, limit: int = 50,
                            before_ts: float = None) -> list[Message]:
        messages = self._partitions.get(channel_id, [])
        if before_ts:
            messages = [m for m in messages if m.created_at < before_ts]
        return messages[-limit:]

    def get_thread_replies(self, parent_message_id: str) -> list[Message]:
        return self._threads.get(parent_message_id, [])

    def edit_message(self, channel_id: str, message_id: str,
                     new_content: str, user_id: str) -> bool:
        for msg in self._partitions.get(channel_id, []):
            if msg.message_id == message_id and msg.user_id == user_id:
                msg.edit_history.append({"content": msg.content, "edited_at": time.time()})
                msg.content = new_content
                msg.is_edited = True
                return True
        return False

    def add_reaction(self, channel_id: str, message_id: str,
                     emoji: str, user_id: str) -> Optional[dict]:
        for msg in self._partitions.get(channel_id, []):
            if msg.message_id == message_id:
                if emoji not in msg.reactions:
                    msg.reactions[emoji] = set()
                msg.reactions[emoji].add(user_id)
                return {"emoji": emoji, "count": len(msg.reactions[emoji])}
        return None

    def remove_reaction(self, channel_id: str, message_id: str,
                        emoji: str, user_id: str) -> Optional[dict]:
        for msg in self._partitions.get(channel_id, []):
            if msg.message_id == message_id and emoji in msg.reactions:
                msg.reactions[emoji].discard(user_id)
                return {"emoji": emoji, "count": len(msg.reactions[emoji])}
        return None

    def message_count(self, channel_id: str) -> int:
        return len(self._partitions.get(channel_id, []))


# ---------------------------------------------------------------------------
# Message Search (Inverted Index)
# ---------------------------------------------------------------------------

class MessageSearch:
    """Full-text search over messages using an inverted index."""

    def __init__(self):
        # word -> list of (message_id, channel_id, workspace_id, snippet)
        self._index: dict[str, list[dict]] = defaultdict(list)

    def index_message(self, message: Message):
        tokens = self._tokenize(message.content)
        for token in tokens:
            self._index[token].append({
                "message_id": message.message_id,
                "channel_id": message.channel_id,
                "workspace_id": message.workspace_id,
                "user_id": message.user_id,
                "snippet": message.content[:100],
                "timestamp": message.created_at
            })

    def search(self, query: str, workspace_id: str,
               channel_id: str = None) -> list[dict]:
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Intersection of results for all query tokens
        result_sets = []
        for token in tokens:
            hits = self._index.get(token, [])
            filtered = [h for h in hits if h["workspace_id"] == workspace_id]
            if channel_id:
                filtered = [h for h in filtered if h["channel_id"] == channel_id]
            result_sets.append({h["message_id"] for h in filtered})

        if not result_sets:
            return []

        common_ids = result_sets[0]
        for s in result_sets[1:]:
            common_ids &= s

        # Collect results
        results = []
        seen = set()
        for token in tokens:
            for hit in self._index.get(token, []):
                if hit["message_id"] in common_ids and hit["message_id"] not in seen:
                    seen.add(hit["message_id"])
                    results.append(hit)

        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results[:20]

    def _tokenize(self, text: str) -> list[str]:
        return [w.lower().strip(".,!?") for w in text.split() if len(w) > 2]


# ---------------------------------------------------------------------------
# Channel Router
# ---------------------------------------------------------------------------

class ChannelRouter:
    """Routes messages to correct connection servers via pub/sub."""

    def __init__(self, connection_manager: ConnectionManager):
        self.cm = connection_manager
        self.routed_count = 0

    def route_message(self, message: Message) -> int:
        """Returns number of recipients reached."""
        msg_dict = {
            "message_id": message.message_id,
            "channel_id": message.channel_id,
            "user_id": message.user_id,
            "content": message.content,
            "created_at": message.created_at,
            "sender_id": message.user_id
        }
        self.cm.publish(message.channel_id, msg_dict)
        self.routed_count += 1
        online_count = len(self.cm.get_online_users(message.channel_id))
        return online_count


# ---------------------------------------------------------------------------
# Main Slack System
# ---------------------------------------------------------------------------

class SlackSystem:
    """Orchestrates all Slack components."""

    def __init__(self):
        self._workspaces: dict[str, Workspace] = {}
        self._channels: dict[str, Channel] = {}
        self._channel_members: dict[str, set[str]] = defaultdict(set)
        self.message_store = MessageStore()
        self.connection_manager = ConnectionManager()
        self.presence_service = PresenceService()
        self.unread_counter = UnreadCounter()
        self.search_engine = MessageSearch()
        self.channel_router = ChannelRouter(self.connection_manager)

    def create_workspace(self, name: str, slug: str, owner_id: str) -> Workspace:
        ws = Workspace(workspace_id=str(uuid.uuid4()), name=name,
                       slug=slug, owner_id=owner_id)
        self._workspaces[ws.workspace_id] = ws
        return ws

    def create_channel(self, workspace_id: str, name: str,
                       created_by: str, is_private: bool = False) -> Channel:
        ch = Channel(channel_id=str(uuid.uuid4()), workspace_id=workspace_id,
                     name=name, is_private=is_private, created_by=created_by)
        self._channels[ch.channel_id] = ch
        self.join_channel(ch.channel_id, created_by)
        return ch

    def create_dm(self, workspace_id: str, user1: str, user2: str) -> Channel:
        dm = Channel(channel_id=str(uuid.uuid4()), workspace_id=workspace_id,
                     name=f"dm:{user1}:{user2}", is_dm=True)
        self._channels[dm.channel_id] = dm
        self.join_channel(dm.channel_id, user1)
        self.join_channel(dm.channel_id, user2)
        return dm

    def join_channel(self, channel_id: str, user_id: str):
        self._channel_members[channel_id].add(user_id)
        self.connection_manager.connect_user(user_id, [channel_id])

    def send_message(self, channel_id: str, user_id: str, content: str,
                     thread_ts: str = None, idempotency_key: str = None) -> dict:
        msg = Message(
            message_id=str(uuid.uuid4()),
            channel_id=channel_id,
            workspace_id=self._channels[channel_id].workspace_id,
            user_id=user_id,
            content=content,
            thread_ts=thread_ts,
            idempotency_key=idempotency_key or str(uuid.uuid4())
        )

        # Persist
        store_result = self.message_store.store_message(msg)
        if store_result["status"] == "duplicate":
            return {"status": "duplicate", "message_id": store_result["message_id"]}

        # Index for search (async in production, synchronous here)
        self.search_engine.index_message(msg)

        # Update unread counts for all channel members
        members = list(self._channel_members.get(channel_id, set()))
        self.unread_counter.increment(channel_id, members, msg.message_id, user_id)

        # Route via WebSocket
        recipients = self.channel_router.route_message(msg)

        return {
            "status": "sent",
            "message_id": msg.message_id,
            "channel_id": channel_id,
            "online_recipients": recipients,
            "created_at": msg.created_at
        }

    def get_channel_history(self, channel_id: str, limit: int = 50) -> list[dict]:
        messages = self.message_store.get_channel_history(channel_id, limit)
        return [
            {
                "message_id": m.message_id,
                "user_id": m.user_id,
                "content": m.content if not m.is_deleted else "[deleted]",
                "is_edited": m.is_edited,
                "thread_ts": m.thread_ts,
                "reactions": {e: len(u) for e, u in m.reactions.items()},
                "created_at": datetime.fromtimestamp(m.created_at).strftime('%H:%M:%S')
            }
            for m in messages
        ]

    def add_reaction(self, channel_id: str, message_id: str,
                     emoji: str, user_id: str) -> Optional[dict]:
        return self.message_store.add_reaction(channel_id, message_id, emoji, user_id)

    def set_presence(self, user_id: str, workspace_id: str, status: str) -> dict:
        status_enum = PresenceStatus[status.upper()]
        self.presence_service.update_presence(user_id, workspace_id, status_enum)
        return {"user_id": user_id, "status": status}

    def get_unread_count(self, user_id: str) -> list[dict]:
        return self.unread_counter.get_unread(user_id)

    def search_messages(self, query: str, workspace_id: str,
                        channel_id: str = None) -> list[dict]:
        results = self.search_engine.search(query, workspace_id, channel_id)
        return [
            {
                "message_id": r["message_id"],
                "channel_id": r["channel_id"],
                "user_id": r["user_id"],
                "snippet": r["snippet"],
                "timestamp": datetime.fromtimestamp(r["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
            }
            for r in results
        ]


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------

def run_simulation():
    print("=" * 65)
    print("  Slack System Simulation")
    print("=" * 65)

    slack = SlackSystem()

    # Create workspace
    ws = slack.create_workspace("Acme Corp", "acme-corp", "alice")
    print(f"\nWorkspace: {ws.name} (ID: {ws.workspace_id[:8]}...)")

    # Create channels
    general = slack.create_channel(ws.workspace_id, "general", "alice")
    engineering = slack.create_channel(ws.workspace_id, "engineering", "bob")
    print(f"Channel #general    : {general.channel_id[:8]}...")
    print(f"Channel #engineering: {engineering.channel_id[:8]}...")

    # Join users
    slack.join_channel(general.channel_id, "bob")
    slack.join_channel(general.channel_id, "carol")
    slack.join_channel(engineering.channel_id, "alice")
    slack.join_channel(engineering.channel_id, "carol")

    # Create DM
    dm_ch = slack.create_dm(ws.workspace_id, "alice", "bob")
    print(f"DM channel Alice-Bob: {dm_ch.channel_id[:8]}...")

    # Set presence
    print("\n[1] Setting user presence")
    slack.set_presence("alice", ws.workspace_id, "online")
    slack.set_presence("bob", ws.workspace_id, "online")
    slack.set_presence("carol", ws.workspace_id, "away")
    for user in ["alice", "bob", "carol"]:
        status = slack.presence_service.get_presence(user, ws.workspace_id)
        print(f"    {user:<8}: {status}")

    # Send messages
    print("\n[2] Sending messages to #general")
    r1 = slack.send_message(general.channel_id, "alice",
                             "Good morning team! Standup in 10 min.")
    print(f"    Alice: msg_id={r1['message_id'][:8]}... recipients={r1['online_recipients']}")

    r2 = slack.send_message(general.channel_id, "bob",
                             "Good morning! I'm working on the auth service.")
    print(f"    Bob  : msg_id={r2['message_id'][:8]}... recipients={r2['online_recipients']}")

    r3 = slack.send_message(general.channel_id, "carol",
                             "Morning everyone. Checking the deployment status.")
    print(f"    Carol: msg_id={r3['message_id'][:8]}... recipients={r3['online_recipients']}")

    # Thread replies
    print("\n[3] Thread replies")
    parent_id = r1["message_id"]
    t1 = slack.send_message(general.channel_id, "bob",
                             "I might be 2 minutes late to standup",
                             thread_ts=parent_id)
    t2 = slack.send_message(general.channel_id, "carol",
                             "No worries, I'll be there",
                             thread_ts=parent_id)
    thread_replies = slack.message_store.get_thread_replies(parent_id)
    print(f"    Thread on Alice's message: {len(thread_replies)} replies")
    for reply in thread_replies:
        print(f"      [{reply.user_id}]: {reply.content}")

    # Emoji reactions
    print("\n[4] Emoji reactions")
    slack.add_reaction(general.channel_id, r1["message_id"], "thumbsup", "bob")
    slack.add_reaction(general.channel_id, r1["message_id"], "thumbsup", "carol")
    slack.add_reaction(general.channel_id, r1["message_id"], "wave", "bob")
    history = slack.get_channel_history(general.channel_id)
    first_msg = history[0]
    print(f"    Message: '{first_msg['content'][:40]}...'")
    print(f"    Reactions: {first_msg['reactions']}")

    # Idempotency / deduplication
    print("\n[5] Idempotency key deduplication")
    idem_key = "unique-client-key-xyz"
    res_a = slack.send_message(general.channel_id, "alice",
                                "This is a critical announcement!",
                                idempotency_key=idem_key)
    res_b = slack.send_message(general.channel_id, "alice",
                                "This is a critical announcement!",
                                idempotency_key=idem_key)  # retry
    print(f"    First send  : {res_a['status']} | id={res_a['message_id'][:8]}...")
    print(f"    Retry send  : {res_b['status']} | id={res_b['message_id'][:8]}...")
    print(f"    Same ID?    : {res_a['message_id'] == res_b['message_id']}")

    # Unread counts
    print("\n[6] Unread counts (for carol)")
    slack.unread_counter.mark_read("carol", general.channel_id, r3["message_id"])
    unreads = slack.get_unread_count("carol")
    if unreads:
        for u in unreads:
            print(f"    Channel {u['channel_id'][:8]}...: {u['count']} unread")
    else:
        print("    All caught up!")
    total = slack.unread_counter.total_unread("alice")
    print(f"    Alice total unreads: {total}")

    # Message search
    print("\n[7] Full-text search")
    results = slack.search_messages("standup morning", ws.workspace_id)
    print(f"    Query: 'standup morning' -> {len(results)} results")
    for r in results[:3]:
        print(f"      [{r['user_id']}] {r['snippet'][:50]}...")

    results2 = slack.search_messages("deployment status", ws.workspace_id)
    print(f"    Query: 'deployment status' -> {len(results2)} results")

    # Channel history
    print("\n[8] Channel #general history")
    history = slack.get_channel_history(general.channel_id)
    print(f"    Total messages: {slack.message_store.message_count(general.channel_id)}")
    for msg in history[-4:]:
        thread_marker = f" [thread: {msg['thread_ts'][:6]}...]" if msg['thread_ts'] else ""
        print(f"    [{msg['created_at']}] {msg['user_id']:<8}: "
              f"{msg['content'][:40]}{thread_marker}")

    # Delivery stats
    print("\n[9] WebSocket delivery statistics")
    stats = slack.connection_manager.delivery_stats()
    for k, v in stats.items():
        print(f"    {k:<25}: {v}")

    print("\n" + "=" * 65)
    print("  System Summary")
    print("=" * 65)
    total_msgs = sum(
        slack.message_store.message_count(ch_id)
        for ch_id in slack._channels
    )
    print(f"  Total channels     : {len(slack._channels)}")
    print(f"  Total messages     : {total_msgs}")
    print(f"  Messages routed    : {slack.channel_router.routed_count}")
    print(f"  Search index size  : {sum(len(v) for v in slack.search_engine._index.values())} entries")
    print("=" * 65)


if __name__ == "__main__":
    run_simulation()
