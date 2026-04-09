"""
SLACK — Team Messaging Platform
================================

FUNCTIONAL REQUIREMENTS:
- Workspaces, channels (public/private), direct messages
- Threaded replies, reactions, file sharing
- Search across messages
- Real-time delivery via WebSocket
- Message history with infinite scroll

NON-FUNCTIONAL REQUIREMENTS:
- 20 M DAU, 5 B messages/day
- Message delivery < 100 ms (p99)
- 99.99% availability
- Messages stored 10 years

ARCHITECTURE:
  Client ──WebSocket──▶ Gateway Server
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Channel Svc     Message Store    Presence Svc
              │          (Cassandra)          │
        Membership DB    Search Index    Redis (online)
              │          (Elasticsearch)      │
        Notification                   Push Gateway
           Service

KEY DESIGN DECISIONS:
1. WEBSOCKET ROUTING — each Gateway Server maintains an in-memory map of
   {user_id → ws_connection}. Multiple servers need pub/sub (Redis) to route
   messages to the right server.

2. MESSAGE STORAGE — Cassandra with partition key (channel_id) and clustering
   key (message_id DESC, time-ordered). Supports O(1) channel history reads.

3. FANOUT — when user sends to a channel with N members, server looks up all
   online members → publishes to their gateway servers via Redis pub/sub.
   For DMs: two-party fanout only.

4. PRESENCE — heartbeat every 30s; Redis SETEX with 60s TTL.
   "Last seen" stored in DB when user goes offline.

5. SEARCH — Elasticsearch with index per workspace.
   Indexed fields: message text, sender, channel, timestamp, reactions.

6. THREADS — reply_to_id pointer on message; thread_root has reply_count.
   Separate thread timeline alongside channel timeline.

7. UNREAD COUNTS — per-user, per-channel counter in Redis (INCR on new msg,
   reset to 0 on read-ack). Stored in DB for persistence across reconnects.

8. RATE LIMITING — per-user 1 msg/s, per-workspace 100 msg/s (Slack API).
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from collections import defaultdict
import threading
import hashlib


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ChannelType(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    DM = "dm"
    GROUP_DM = "group_dm"


@dataclass
class Workspace:
    workspace_id: str
    name: str
    domain: str   # e.g. "acme"
    created_at: float = field(default_factory=time.time)


@dataclass
class Channel:
    channel_id: str
    workspace_id: str
    name: str
    channel_type: ChannelType
    topic: str = ""
    description: str = ""
    member_ids: Set[str] = field(default_factory=set)
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    is_archived: bool = False

    def is_member(self, user_id: str) -> bool:
        return user_id in self.member_ids


@dataclass
class Message:
    message_id: str
    channel_id: str
    sender_id: str
    text: str
    reply_to_id: Optional[str] = None  # None = top-level
    thread_ts: Optional[str] = None    # root message's message_id
    reply_count: int = 0
    reactions: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    edited: bool = False
    deleted: bool = False
    files: List[str] = field(default_factory=list)   # file IDs
    ts: float = field(default_factory=time.time)

    @property
    def is_reply(self) -> bool:
        return self.reply_to_id is not None


@dataclass
class User:
    user_id: str
    workspace_id: str
    username: str
    display_name: str
    email: str
    is_bot: bool = False
    status: str = ""
    timezone: str = "UTC"


# ---------------------------------------------------------------------------
# Message Store (Cassandra simulation)
# ---------------------------------------------------------------------------

class MessageStore:
    """
    Simulates Cassandra: partition_key=channel_id, clustering_key=ts DESC.
    Supports time-ordered retrieval and cursor-based pagination.
    """

    def __init__(self):
        # channel_id → list of messages (append order = time order)
        self._channel_msgs: Dict[str, List[Message]] = defaultdict(list)
        # message_id → Message (global index)
        self._by_id: Dict[str, Message] = {}
        # thread_root_id → list of reply messages
        self._threads: Dict[str, List[Message]] = defaultdict(list)

    def save(self, msg: Message) -> Message:
        self._channel_msgs[msg.channel_id].append(msg)
        self._by_id[msg.message_id] = msg
        if msg.thread_ts and msg.reply_to_id:
            self._threads[msg.thread_ts].append(msg)
            # Increment reply count on root
            root = self._by_id.get(msg.thread_ts)
            if root:
                root.reply_count += 1
        return msg

    def get(self, message_id: str) -> Optional[Message]:
        return self._by_id.get(message_id)

    def channel_history(self, channel_id: str, before_ts: Optional[float] = None,
                        limit: int = 50) -> List[Message]:
        msgs = self._channel_msgs.get(channel_id, [])
        if before_ts:
            msgs = [m for m in msgs if m.ts < before_ts]
        # Return newest first
        return sorted(msgs, key=lambda m: m.ts, reverse=True)[:limit]

    def thread_replies(self, thread_root_id: str) -> List[Message]:
        return sorted(self._threads.get(thread_root_id, []), key=lambda m: m.ts)

    def edit(self, message_id: str, new_text: str, editor_id: str) -> Optional[Message]:
        msg = self._by_id.get(message_id)
        if msg and msg.sender_id == editor_id and not msg.deleted:
            msg.text = new_text
            msg.edited = True
            return msg
        return None

    def delete(self, message_id: str, actor_id: str) -> bool:
        msg = self._by_id.get(message_id)
        if msg and msg.sender_id == actor_id:
            msg.deleted = True
            msg.text = ""
            return True
        return False

    def add_reaction(self, message_id: str, emoji: str, user_id: str) -> bool:
        msg = self._by_id.get(message_id)
        if msg:
            msg.reactions[emoji].add(user_id)
            return True
        return False

    def remove_reaction(self, message_id: str, emoji: str, user_id: str) -> bool:
        msg = self._by_id.get(message_id)
        if msg and emoji in msg.reactions:
            msg.reactions[emoji].discard(user_id)
            if not msg.reactions[emoji]:
                del msg.reactions[emoji]
            return True
        return False


# ---------------------------------------------------------------------------
# Presence Service
# ---------------------------------------------------------------------------

class PresenceService:
    """Redis-backed presence: heartbeat TTL 60 s."""

    def __init__(self, ttl: float = 60.0):
        self._last_seen: Dict[str, float] = {}
        self._status: Dict[str, str] = {}
        self._ttl = ttl

    def heartbeat(self, user_id: str) -> None:
        self._last_seen[user_id] = time.time()

    def set_status(self, user_id: str, status: str) -> None:
        self._status[user_id] = status
        self.heartbeat(user_id)

    def is_online(self, user_id: str) -> bool:
        last = self._last_seen.get(user_id)
        return last is not None and (time.time() - last) < self._ttl

    def online_members(self, user_ids: List[str]) -> List[str]:
        return [uid for uid in user_ids if self.is_online(uid)]

    def go_offline(self, user_id: str) -> None:
        self._last_seen.pop(user_id, None)


# ---------------------------------------------------------------------------
# Unread Counter
# ---------------------------------------------------------------------------

class UnreadCounter:
    """Per-user, per-channel unread message counts stored in Redis (simulated)."""

    def __init__(self):
        # (user_id, channel_id) → count
        self._counts: Dict[tuple, int] = defaultdict(int)
        # (user_id, channel_id) → last_read_ts
        self._last_read: Dict[tuple, float] = {}

    def increment(self, channel_id: str, member_ids: Set[str], sender_id: str):
        for uid in member_ids:
            if uid != sender_id:
                self._counts[(uid, channel_id)] += 1

    def mark_read(self, user_id: str, channel_id: str) -> None:
        self._counts[(user_id, channel_id)] = 0
        self._last_read[(user_id, channel_id)] = time.time()

    def unread(self, user_id: str, channel_id: str) -> int:
        return self._counts.get((user_id, channel_id), 0)

    def total_unread(self, user_id: str, channel_ids: List[str]) -> int:
        return sum(self.unread(user_id, cid) for cid in channel_ids)


# ---------------------------------------------------------------------------
# Message Bus (Redis pub/sub simulation)
# ---------------------------------------------------------------------------

class MessageBus:
    """Routes real-time events to gateway servers via pub/sub."""

    def __init__(self):
        self._subscribers: Dict[str, List] = defaultdict(list)  # channel → callbacks
        self._event_log: List[Dict] = []

    def subscribe(self, channel: str, callback) -> None:
        self._subscribers[channel].append(callback)

    def publish(self, channel: str, event: Dict) -> int:
        self._event_log.append({"channel": channel, **event})
        count = 0
        for cb in self._subscribers.get(channel, []):
            cb(event)
            count += 1
        return count

    def recent_events(self, limit: int = 10) -> List[Dict]:
        return self._event_log[-limit:]


# ---------------------------------------------------------------------------
# Search Service (Elasticsearch simulation)
# ---------------------------------------------------------------------------

class SearchIndex:
    """Simple keyword search over message text."""

    def __init__(self):
        # term → set of message_ids
        self._inverted: Dict[str, Set[str]] = defaultdict(set)
        self._messages: Dict[str, Message] = {}

    def index(self, msg: Message) -> None:
        self._messages[msg.message_id] = msg
        for token in self._tokenize(msg.text):
            self._inverted[token].add(msg.message_id)

    def search(self, query: str, channel_id: Optional[str] = None,
               sender_id: Optional[str] = None, limit: int = 20) -> List[Message]:
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Intersection of all token matches
        candidates = self._inverted.get(tokens[0], set()).copy()
        for token in tokens[1:]:
            candidates &= self._inverted.get(token, set())

        results = []
        for mid in candidates:
            msg = self._messages.get(mid)
            if msg and not msg.deleted:
                if channel_id and msg.channel_id != channel_id:
                    continue
                if sender_id and msg.sender_id != sender_id:
                    continue
                results.append(msg)

        return sorted(results, key=lambda m: m.ts, reverse=True)[:limit]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower().strip(".,!?") for w in text.split() if len(w) > 2]


# ---------------------------------------------------------------------------
# Slack Service (Orchestrator)
# ---------------------------------------------------------------------------

class SlackService:
    def __init__(self):
        self.store = MessageStore()
        self.presence = PresenceService()
        self.unread = UnreadCounter()
        self.bus = MessageBus()
        self.search = SearchIndex()
        self._workspaces: Dict[str, Workspace] = {}
        self._channels: Dict[str, Channel] = {}
        self._users: Dict[str, User] = {}

    def create_workspace(self, name: str, domain: str) -> Workspace:
        ws = Workspace(workspace_id=str(uuid.uuid4())[:8], name=name, domain=domain)
        self._workspaces[ws.workspace_id] = ws
        return ws

    def create_user(self, workspace_id: str, username: str,
                    display_name: str, email: str) -> User:
        user = User(
            user_id=str(uuid.uuid4())[:8],
            workspace_id=workspace_id,
            username=username,
            display_name=display_name,
            email=email,
        )
        self._users[user.user_id] = user
        return user

    def create_channel(self, workspace_id: str, name: str,
                       channel_type: ChannelType, creator_id: str) -> Channel:
        ch = Channel(
            channel_id=str(uuid.uuid4())[:8],
            workspace_id=workspace_id,
            name=name,
            channel_type=channel_type,
            created_by=creator_id,
            member_ids={creator_id},
        )
        self._channels[ch.channel_id] = ch
        return ch

    def join_channel(self, channel_id: str, user_id: str) -> bool:
        ch = self._channels.get(channel_id)
        if ch and ch.channel_type != ChannelType.PRIVATE:
            ch.member_ids.add(user_id)
            return True
        return False

    def send_message(self, channel_id: str, sender_id: str,
                     text: str, reply_to_id: Optional[str] = None) -> Optional[Message]:
        ch = self._channels.get(channel_id)
        if not ch or not ch.is_member(sender_id):
            return None

        # Determine thread context
        thread_ts = None
        if reply_to_id:
            root = self.store.get(reply_to_id)
            thread_ts = root.thread_ts or root.message_id if root else None

        msg = Message(
            message_id=str(uuid.uuid4())[:12],
            channel_id=channel_id,
            sender_id=sender_id,
            text=text,
            reply_to_id=reply_to_id,
            thread_ts=thread_ts,
        )
        self.store.save(msg)

        # Update unread counters for all members except sender
        self.unread.increment(channel_id, ch.member_ids, sender_id)

        # Index for search
        self.search.index(msg)

        # Broadcast to online members
        online = self.presence.online_members(list(ch.member_ids))
        self.bus.publish(f"channel:{channel_id}", {
            "type": "new_message",
            "message_id": msg.message_id,
            "sender": sender_id,
            "text": text,
            "notified": online,
        })

        return msg

    def get_dm_channel(self, workspace_id: str,
                       user_a: str, user_b: str) -> Channel:
        """Get or create DM channel between two users."""
        # Canonical key: sorted user IDs
        key = "_".join(sorted([user_a, user_b]))
        for ch in self._channels.values():
            if ch.channel_type == ChannelType.DM and ch.name == key:
                return ch
        ch = Channel(
            channel_id=str(uuid.uuid4())[:8],
            workspace_id=workspace_id,
            name=key,
            channel_type=ChannelType.DM,
            member_ids={user_a, user_b},
        )
        self._channels[ch.channel_id] = ch
        return ch


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_workspace_and_channels():
    print("\n=== 1. Workspace, Channels, Members ===")
    slack = SlackService()

    ws = slack.create_workspace("Acme Corp", "acme")
    print(f"Workspace: {ws.name} ({ws.domain})")

    alice = slack.create_user(ws.workspace_id, "alice", "Alice", "alice@acme.com")
    bob = slack.create_user(ws.workspace_id, "bob", "Bob", "bob@acme.com")
    carol = slack.create_user(ws.workspace_id, "carol", "Carol", "carol@acme.com")

    general = slack.create_channel(ws.workspace_id, "general", ChannelType.PUBLIC, alice.user_id)
    slack.join_channel(general.channel_id, bob.user_id)
    slack.join_channel(general.channel_id, carol.user_id)
    print(f"#general members: {len(general.member_ids)}")

    eng = slack.create_channel(ws.workspace_id, "engineering", ChannelType.PUBLIC, alice.user_id)
    slack.join_channel(eng.channel_id, bob.user_id)
    print(f"#engineering members: {len(eng.member_ids)}")

    return slack, ws, alice, bob, carol, general, eng


def demonstrate_2_messaging(slack, ws, alice, bob, carol, general, eng):
    print("\n=== 2. Sending Messages & Real-time Fanout ===")

    # Simulate Alice and Bob online
    slack.presence.heartbeat(alice.user_id)
    slack.presence.heartbeat(bob.user_id)
    # Carol offline

    msg1 = slack.send_message(general.channel_id, alice.user_id,
                               "Good morning team! Ready for the sprint?")
    print(f"Alice sent: '{msg1.text}'")

    events = slack.bus.recent_events(1)
    event = events[0]
    print(f"Broadcast to {len(event['notified'])} online users")
    print(f"Online in #general: {event['notified']}")

    msg2 = slack.send_message(general.channel_id, bob.user_id,
                               "Yes! Let's review the backlog first.")
    print(f"Bob sent: '{msg2.text}'")

    # Carol's unread
    carol_unread = slack.unread.unread(carol.user_id, general.channel_id)
    print(f"Carol's unread count in #general: {carol_unread}")


def demonstrate_3_threads(slack, ws, alice, bob, carol, general, eng):
    print("\n=== 3. Threaded Replies ===")

    root = slack.send_message(general.channel_id, alice.user_id,
                               "What's our deployment plan for Friday?")
    print(f"Root message: '{root.text}' (id={root.message_id})")

    reply1 = slack.send_message(general.channel_id, bob.user_id,
                                 "I'll handle the DB migrations.",
                                 reply_to_id=root.message_id)
    reply2 = slack.send_message(general.channel_id, carol.user_id,
                                 "I'll monitor the dashboards.",
                                 reply_to_id=root.message_id)

    root_refreshed = slack.store.get(root.message_id)
    print(f"Root reply count: {root_refreshed.reply_count}")

    thread = slack.store.thread_replies(root.message_id)
    print(f"Thread replies:")
    for r in thread:
        print(f"  ↳ [{r.sender_id}]: {r.text}")


def demonstrate_4_reactions_and_edit(slack, ws, alice, bob, carol, general, eng):
    print("\n=== 4. Reactions & Message Edit ===")

    msg = slack.send_message(general.channel_id, alice.user_id, "Shipping new feature today!")
    slack.store.add_reaction(msg.message_id, ":rocket:", bob.user_id)
    slack.store.add_reaction(msg.message_id, ":rocket:", carol.user_id)
    slack.store.add_reaction(msg.message_id, ":tada:", alice.user_id)

    m = slack.store.get(msg.message_id)
    print(f"Message: '{m.text}'")
    for emoji, users in m.reactions.items():
        print(f"  {emoji} × {len(users)}: {users}")

    # Edit
    edited = slack.store.edit(msg.message_id, "Shipping new feature today! 🚀", alice.user_id)
    print(f"\nEdited message: '{edited.text}' (edited={edited.edited})")


def demonstrate_5_search(slack, ws, alice, bob, carol, general, eng):
    print("\n=== 5. Message Search ===")

    slack.send_message(general.channel_id, alice.user_id, "deploy the backend service now")
    slack.send_message(general.channel_id, bob.user_id, "backend tests are failing")
    slack.send_message(eng.channel_id, alice.user_id, "backend pipeline broken")

    results = slack.search.search("backend")
    print(f"Search 'backend': {len(results)} results")
    for r in results:
        print(f"  [{r.sender_id} in {r.channel_id}]: {r.text}")

    results_ch = slack.search.search("backend", channel_id=general.channel_id)
    print(f"\nSearch 'backend' in #general only: {len(results_ch)} results")


def demonstrate_6_dm_and_presence(slack, ws, alice, bob, carol, general, eng):
    print("\n=== 6. Direct Messages & Presence ===")

    dm_channel = slack.get_dm_channel(ws.workspace_id, alice.user_id, bob.user_id)
    print(f"DM channel between alice and bob: {dm_channel.name}")

    msg = slack.send_message(dm_channel.channel_id, alice.user_id,
                              "Hey Bob, quick sync?")
    print(f"Alice DM'd Bob: '{msg.text}'")

    # Presence
    slack.presence.heartbeat(alice.user_id)
    slack.presence.heartbeat(bob.user_id)
    # Carol is offline (no heartbeat)

    all_users = [alice.user_id, bob.user_id, carol.user_id]
    online = slack.presence.online_members(all_users)
    print(f"\nOnline users: {online} ({len(online)}/{len(all_users)})")


if __name__ == "__main__":
    slack, ws, alice, bob, carol, general, eng = demonstrate_1_workspace_and_channels()
    demonstrate_2_messaging(slack, ws, alice, bob, carol, general, eng)
    demonstrate_3_threads(slack, ws, alice, bob, carol, general, eng)
    demonstrate_4_reactions_and_edit(slack, ws, alice, bob, carol, general, eng)
    demonstrate_5_search(slack, ws, alice, bob, carol, general, eng)
    demonstrate_6_dm_and_presence(slack, ws, alice, bob, carol, general, eng)
