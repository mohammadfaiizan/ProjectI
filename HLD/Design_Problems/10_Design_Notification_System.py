"""
Notification System Design - Python Implementation
Demonstrates: multi-channel delivery (Push/Email/SMS/InApp), priority queues,
template engine with variable substitution, retry with exponential backoff,
deduplication cache, user preferences, fan-out, delivery tracking.
No external dependencies - standard library only.
"""

import time
import uuid
import math
import random
import hashlib
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class NotificationPriority(Enum):
    CRITICAL = 1       # OTP, fraud alerts — deliver immediately
    TRANSACTIONAL = 2  # Order confirmations, shipping
    SOCIAL = 3         # Likes, follows, comments
    MARKETING = 4      # Promotions, newsletters


class NotificationChannel(Enum):
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"


class DeliveryStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    FAILED = "failed"
    SKIPPED = "skipped"       # User opted out


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class Notification:
    notification_id: str
    user_id: str
    notification_type: str
    priority: NotificationPriority
    title: str
    body: str
    data: dict = field(default_factory=dict)
    channel: Optional[NotificationChannel] = None
    status: DeliveryStatus = DeliveryStatus.PENDING
    retry_count: int = 0
    max_retries: int = 5
    idempotency_key: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    delivered_at: Optional[float] = None
    opened_at: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class NotificationTemplate:
    template_id: str
    name: str
    notification_type: str
    channel: NotificationChannel
    subject_template: str     # For email
    body_template: str
    variables: list           # Required variable names
    locale: str = "en_US"
    version: int = 1


@dataclass
class UserPreferences:
    user_id: str
    # Channel-level opt-in/out
    channel_enabled: dict = field(default_factory=lambda: {
        NotificationChannel.PUSH: True,
        NotificationChannel.EMAIL: True,
        NotificationChannel.SMS: False,
        NotificationChannel.IN_APP: True,
    })
    # Per notification-type preferences
    type_channels: dict = field(default_factory=dict)
    # Quiet hours (local timezone, simplified as hour range)
    quiet_hours_enabled: bool = False
    quiet_start: int = 22   # 10 PM
    quiet_end: int = 8      # 8 AM
    # Max notifications per day per type
    daily_limits: dict = field(default_factory=dict)

    def is_channel_allowed(self, channel: NotificationChannel,
                            notif_type: str) -> bool:
        if not self.channel_enabled.get(channel, False):
            return False
        type_prefs = self.type_channels.get(notif_type)
        if type_prefs is not None:
            return channel in type_prefs
        return True   # Default: allow if channel is enabled

    def is_quiet_hours(self) -> bool:
        if not self.quiet_hours_enabled:
            return False
        hour = int(time.strftime("%H"))
        if self.quiet_start > self.quiet_end:
            return hour >= self.quiet_start or hour < self.quiet_end
        return self.quiet_start <= hour < self.quiet_end


# ─────────────────────────────────────────────
# Abstract Channel
# ─────────────────────────────────────────────

class NotificationChannelHandler(ABC):
    """Abstract base for all delivery channel implementations."""

    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate   # For simulation
        self.sent_count = 0
        self.failed_count = 0

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        pass

    @property
    @abstractmethod
    def channel_type(self) -> NotificationChannel:
        pass


class FCMChannel(NotificationChannelHandler):
    """Firebase Cloud Messaging for Android/Web push notifications."""

    def __init__(self, failure_rate: float = 0.05):
        super().__init__(failure_rate)
        self._device_tokens: dict = {}    # user_id -> [token]

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.PUSH

    def register_token(self, user_id: str, token: str):
        if user_id not in self._device_tokens:
            self._device_tokens[user_id] = []
        self._device_tokens[user_id].append(token)

    def send(self, notification: Notification) -> bool:
        tokens = self._device_tokens.get(notification.user_id, [])
        if not tokens:
            notification.error_message = "No device token registered"
            return False

        # Simulate FCM API call
        if random.random() < self.failure_rate:
            notification.error_message = "FCM service unavailable (503)"
            self.failed_count += 1
            return False

        self.sent_count += 1
        return True


class EmailChannel(NotificationChannelHandler):
    """Email delivery via SendGrid (simulated)."""

    def __init__(self, failure_rate: float = 0.02):
        super().__init__(failure_rate)
        self._email_store: dict = {}   # user_id -> email address

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.EMAIL

    def register_email(self, user_id: str, email: str):
        self._email_store[user_id] = email

    def send(self, notification: Notification) -> bool:
        email = self._email_store.get(notification.user_id)
        if not email:
            notification.error_message = "No email address registered"
            return False

        if random.random() < self.failure_rate:
            notification.error_message = "SendGrid rate limit (429)"
            self.failed_count += 1
            return False

        self.sent_count += 1
        return True


class SMSChannel(NotificationChannelHandler):
    """SMS delivery via Twilio (simulated). Most expensive channel."""

    def __init__(self, failure_rate: float = 0.03):
        super().__init__(failure_rate)
        self._phone_store: dict = {}   # user_id -> phone

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.SMS

    def register_phone(self, user_id: str, phone: str):
        self._phone_store[user_id] = phone

    def send(self, notification: Notification) -> bool:
        phone = self._phone_store.get(notification.user_id)
        if not phone:
            notification.error_message = "No phone number registered"
            return False

        if random.random() < self.failure_rate:
            notification.error_message = "Twilio unavailable"
            self.failed_count += 1
            return False

        self.sent_count += 1
        return True


class InAppChannel(NotificationChannelHandler):
    """In-app notification stored to DB + pushed via WebSocket."""

    def __init__(self):
        super().__init__(failure_rate=0.0)
        self._inbox: dict = defaultdict(list)   # user_id -> [notification]
        self._unread: dict = defaultdict(int)

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.IN_APP

    def send(self, notification: Notification) -> bool:
        self._inbox[notification.user_id].append(notification)
        self._unread[notification.user_id] += 1
        self.sent_count += 1
        return True

    def get_inbox(self, user_id: str, limit: int = 20) -> list:
        messages = self._inbox.get(user_id, [])
        return sorted(messages, key=lambda n: n.created_at, reverse=True)[:limit]

    def get_unread_count(self, user_id: str) -> int:
        return self._unread.get(user_id, 0)

    def mark_read(self, user_id: str, notification_id: str):
        for notif in self._inbox.get(user_id, []):
            if notif.notification_id == notification_id:
                notif.opened_at = time.time()
                notif.status = DeliveryStatus.OPENED
                self._unread[user_id] = max(0, self._unread[user_id] - 1)
                break


# ─────────────────────────────────────────────
# Template Engine
# ─────────────────────────────────────────────

class TemplateEngine:
    """
    Notification template engine with variable substitution.
    Supports {{variable_name}} syntax.
    """

    def __init__(self):
        self._templates: dict = {}    # template_id -> NotificationTemplate

    def register_template(self, template: NotificationTemplate):
        self._templates[template.template_id] = template
        self._templates[template.name] = template   # Index by name too

    def render(self, template_id: str, variables: dict) -> tuple:
        """
        Render template with variables.
        Returns (subject, body) or raises ValueError for missing variables.
        """
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        # Check required variables
        missing = [v for v in template.variables if v not in variables]
        if missing:
            raise ValueError(f"Missing template variables: {missing}")

        subject = self._substitute(template.subject_template, variables)
        body = self._substitute(template.body_template, variables)
        return subject, body

    def _substitute(self, template_str: str, variables: dict) -> str:
        """Replace {{variable}} with actual values."""
        def replacer(match):
            var_name = match.group(1).strip()
            return str(variables.get(var_name, f"{{MISSING:{var_name}}}"))

        return re.sub(r'\{\{(\w+)\}\}', replacer, template_str)

    def create_template(self, name: str, notification_type: str,
                        channel: NotificationChannel, subject: str,
                        body: str, variables: list) -> NotificationTemplate:
        template = NotificationTemplate(
            template_id=str(uuid.uuid4())[:8],
            name=name,
            notification_type=notification_type,
            channel=channel,
            subject_template=subject,
            body_template=body,
            variables=variables
        )
        self.register_template(template)
        return template


# ─────────────────────────────────────────────
# Deduplication Cache
# ─────────────────────────────────────────────

class DeduplicationCache:
    """
    Prevent duplicate notifications using idempotency keys.
    Simulates Redis SETNX with TTL.
    """

    def __init__(self, ttl_seconds: int = 86400):
        self._store: dict = {}      # idempotency_key -> (notification_id, expiry)
        self.ttl = ttl_seconds

    def is_duplicate(self, idempotency_key: str) -> bool:
        entry = self._store.get(idempotency_key)
        if entry is None:
            return False
        notification_id, expiry = entry
        if time.time() > expiry:
            del self._store[idempotency_key]
            return False
        return True

    def register(self, idempotency_key: str, notification_id: str):
        self._store[idempotency_key] = (notification_id, time.time() + self.ttl)

    def cleanup_expired(self):
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]


# ─────────────────────────────────────────────
# Retry Queue with Exponential Backoff
# ─────────────────────────────────────────────

@dataclass(order=True)
class RetryItem:
    retry_at: float                                       # Sort by retry time
    notification: Notification = field(compare=False)


class RetryQueue:
    """
    Min-heap based retry queue.
    Exponential backoff: 5s, 25s, 125s, 625s, 3125s
    With jitter to prevent thundering herd.
    """

    BASE_DELAY = 5.0
    MAX_DELAY = 3600.0

    def __init__(self):
        import heapq
        self._heap: list = []
        self._heapq = heapq
        self.dead_letter: list = []

    def enqueue(self, notification: Notification):
        """Schedule a notification for retry."""
        delay = self._compute_delay(notification.retry_count)
        retry_at = time.time() + delay
        item = RetryItem(retry_at=retry_at, notification=notification)
        self._heapq.heappush(self._heap, item)

    def get_ready(self) -> list:
        """Return all notifications ready for retry (retry_at <= now)."""
        ready = []
        now = time.time()
        while self._heap and self._heap[0].retry_at <= now:
            item = self._heapq.heappop(self._heap)
            ready.append(item.notification)
        return ready

    def move_to_dlq(self, notification: Notification, reason: str):
        notification.error_message = reason
        notification.status = DeliveryStatus.FAILED
        self.dead_letter.append(notification)

    def _compute_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        delay = min(self.BASE_DELAY * (5 ** attempt), self.MAX_DELAY)
        jitter = random.uniform(0, delay * 0.1)   # 10% jitter
        return delay + jitter

    @property
    def pending_count(self) -> int:
        return len(self._heap)


# ─────────────────────────────────────────────
# Priority Queue for Notification Processing
# ─────────────────────────────────────────────

class NotificationPriorityQueue:
    """
    Multi-tier priority queue:
    - CRITICAL: dedicated queue, processed first always
    - TRANSACTIONAL: high priority
    - SOCIAL: medium priority
    - MARKETING: bulk queue, processed in off-peak
    """

    def __init__(self):
        self._queues: dict = {
            NotificationPriority.CRITICAL: deque(),
            NotificationPriority.TRANSACTIONAL: deque(),
            NotificationPriority.SOCIAL: deque(),
            NotificationPriority.MARKETING: deque(),
        }
        self._total = 0

    def enqueue(self, notification: Notification):
        self._queues[notification.priority].append(notification)
        self._total += 1

    def dequeue(self) -> Optional[Notification]:
        """Dequeue from highest priority non-empty queue."""
        for priority in NotificationPriority:
            if self._queues[priority]:
                self._total -= 1
                return self._queues[priority].popleft()
        return None

    @property
    def total(self) -> int:
        return self._total

    def queue_depths(self) -> dict:
        return {p.name: len(q) for p, q in self._queues.items()}


# ─────────────────────────────────────────────
# Main Notification System
# ─────────────────────────────────────────────

class NotificationSystem:
    """
    Full notification system orchestrating all components.
    """

    def __init__(self):
        # Channels
        self.fcm = FCMChannel(failure_rate=0.05)
        self.email = EmailChannel(failure_rate=0.02)
        self.sms = SMSChannel(failure_rate=0.03)
        self.in_app = InAppChannel()
        self._channels: dict = {
            NotificationChannel.PUSH: self.fcm,
            NotificationChannel.EMAIL: self.email,
            NotificationChannel.SMS: self.sms,
            NotificationChannel.IN_APP: self.in_app,
        }

        # Core components
        self.templates = TemplateEngine()
        self.dedup_cache = DeduplicationCache(ttl_seconds=86400)
        self.retry_queue = RetryQueue()
        self.priority_queue = NotificationPriorityQueue()

        # Storage
        self._preferences: dict = {}          # user_id -> UserPreferences
        self._notifications: dict = {}        # notification_id -> Notification
        self._daily_counts: defaultdict = defaultdict(lambda: defaultdict(int))

        # Delivery stats
        self._stats = defaultdict(int)

    # ── User Management ───────────────────────

    def register_user(self, user_id: str,
                      email: str = None,
                      phone: str = None,
                      device_token: str = None) -> UserPreferences:
        prefs = UserPreferences(user_id=user_id)
        self._preferences[user_id] = prefs

        if email:
            self.email.register_email(user_id, email)
        if phone:
            self.sms.register_phone(user_id, phone)
        if device_token:
            self.fcm.register_token(user_id, device_token)

        return prefs

    def update_preferences(self, user_id: str, **kwargs) -> bool:
        prefs = self._preferences.get(user_id)
        if not prefs:
            return False
        for key, value in kwargs.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        return True

    # ── Templates ─────────────────────────────

    def create_template(self, name: str, notification_type: str,
                        channel: NotificationChannel, subject: str,
                        body: str, variables: list) -> NotificationTemplate:
        return self.templates.create_template(
            name, notification_type, channel, subject, body, variables
        )

    def render_template(self, template_name: str, variables: dict) -> tuple:
        return self.templates.render(template_name, variables)

    # ── Send Notification ─────────────────────

    def send_notification(self, user_id: str, notification_type: str,
                          priority: NotificationPriority,
                          title: str, body: str,
                          data: dict = None,
                          channels: list = None,
                          idempotency_key: str = None) -> list:
        """
        Send notification to user via all appropriate channels.
        Returns list of created Notification objects.
        """
        # Deduplication check
        if idempotency_key:
            if self.dedup_cache.is_duplicate(idempotency_key):
                self._stats["deduplicated"] += 1
                print(f"  [Dedup] Duplicate blocked: {idempotency_key}")
                return []

        prefs = self._preferences.get(user_id)
        if not prefs:
            prefs = UserPreferences(user_id=user_id)

        # Quiet hours check for non-critical
        if (prefs.is_quiet_hours()
                and priority not in (NotificationPriority.CRITICAL,)):
            print(f"  [QuietHours] Notification queued for {user_id} until morning")
            self._stats["deferred_quiet_hours"] += 1

        # Determine target channels
        if channels is None:
            channels = [ch for ch in NotificationChannel
                        if prefs.is_channel_allowed(ch, notification_type)]

        created = []
        for channel in channels:
            if not prefs.channel_enabled.get(channel, False):
                continue

            notif = Notification(
                notification_id=str(uuid.uuid4())[:8],
                user_id=user_id,
                notification_type=notification_type,
                priority=priority,
                title=title,
                body=body,
                data=data or {},
                channel=channel,
                status=DeliveryStatus.QUEUED,
                idempotency_key=idempotency_key,
            )

            self._notifications[notif.notification_id] = notif
            self.priority_queue.enqueue(notif)
            created.append(notif)

        # Register idempotency key after all notifications created
        if idempotency_key and created:
            self.dedup_cache.register(idempotency_key, created[0].notification_id)

        return created

    def add_to_queue(self, notification: Notification):
        """Add a pre-built notification to the priority queue."""
        self.priority_queue.enqueue(notification)

    def process_queue(self, max_process: int = 50) -> dict:
        """
        Worker: process notifications from priority queue.
        Returns processing summary.
        """
        processed = 0
        results = defaultdict(int)

        while processed < max_process:
            notif = self.priority_queue.dequeue()
            if notif is None:
                break

            success = self._deliver(notif)
            if success:
                results["delivered"] += 1
            else:
                results["failed"] += 1
                # Schedule retry if under max retries
                notif.retry_count += 1
                if notif.retry_count <= notif.max_retries:
                    self.retry_queue.enqueue(notif)
                    results["retrying"] += 1
                else:
                    self.retry_queue.move_to_dlq(notif, "Max retries exceeded")
                    results["dead_letter"] += 1

            processed += 1

        # Process any ready retries
        retries = self.retry_queue.get_ready()
        for notif in retries:
            success = self._deliver(notif)
            if success:
                results["retry_success"] += 1
            else:
                notif.retry_count += 1
                if notif.retry_count <= notif.max_retries:
                    self.retry_queue.enqueue(notif)
                else:
                    self.retry_queue.move_to_dlq(notif, "Max retries exceeded")

        return dict(results)

    def _deliver(self, notification: Notification) -> bool:
        """Attempt to deliver a notification via its channel."""
        channel_handler = self._channels.get(notification.channel)
        if not channel_handler:
            notification.status = DeliveryStatus.FAILED
            notification.error_message = "Unknown channel"
            return False

        try:
            success = channel_handler.send(notification)
            if success:
                notification.status = DeliveryStatus.SENT
                notification.sent_at = time.time()
                self._stats["total_sent"] += 1
                self._stats[f"sent_{notification.channel.value}"] += 1
            return success
        except Exception as e:
            notification.status = DeliveryStatus.FAILED
            notification.error_message = str(e)
            return False

    # ── Fan-out (Broadcast) ────────────────────

    def broadcast(self, user_ids: list, notification_type: str,
                  priority: NotificationPriority,
                  title: str, body: str, data: dict = None) -> int:
        """
        Send notification to many users (fan-out).
        Rate-limited: 10K/second (simulated).
        """
        total_created = 0
        batch_size = 1000

        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i:i + batch_size]
            for user_id in batch:
                notifications = self.send_notification(
                    user_id=user_id,
                    notification_type=notification_type,
                    priority=priority,
                    title=title,
                    body=body,
                    data=data,
                )
                total_created += len(notifications)

        return total_created

    # ── Notification Center ────────────────────

    def get_notification_history(self, user_id: str,
                                  limit: int = 20) -> list:
        """Retrieve in-app notification history."""
        return self.in_app.get_inbox(user_id, limit)

    def mark_as_read(self, user_id: str, notification_id: str):
        self.in_app.mark_read(user_id, notification_id)

    def get_unread_count(self, user_id: str) -> int:
        return self.in_app.get_unread_count(user_id)

    # ── Stats ──────────────────────────────────

    def get_delivery_stats(self) -> dict:
        return {
            **dict(self._stats),
            "queue_depths": self.priority_queue.queue_depths(),
            "retry_pending": self.retry_queue.pending_count,
            "dead_letter_count": len(self.retry_queue.dead_letter),
            "dedup_cache_size": len(self.dedup_cache._store),
        }

    def get_channel_stats(self) -> dict:
        return {
            "push_fcm": {
                "sent": self.fcm.sent_count,
                "failed": self.fcm.failed_count
            },
            "email": {
                "sent": self.email.sent_count,
                "failed": self.email.failed_count
            },
            "sms": {
                "sent": self.sms.sent_count,
                "failed": self.sms.failed_count
            },
            "in_app": {
                "sent": self.in_app.sent_count,
                "failed": 0
            },
        }


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("NOTIFICATION SYSTEM DESIGN DEMO")
    print("=" * 60)

    ns = NotificationSystem()

    # Register users
    print("\n--- Registering Users ---")
    alice = ns.register_user("alice", email="alice@example.com",
                              phone="+1234567890", device_token="FCM_TOKEN_ALICE")
    bob = ns.register_user("bob", email="bob@example.com",
                            device_token="FCM_TOKEN_BOB")
    carol = ns.register_user("carol", email="carol@example.com")
    print("Registered: alice, bob, carol")

    # Carol opts out of SMS and marketing
    ns.update_preferences("carol", channel_enabled={
        NotificationChannel.PUSH: False,   # No app installed
        NotificationChannel.EMAIL: True,
        NotificationChannel.SMS: False,
        NotificationChannel.IN_APP: False,
    })
    print("Carol preferences: email only")

    # Create templates
    print("\n--- Setting Up Templates ---")
    order_tmpl = ns.create_template(
        name="order_confirmed",
        notification_type="ORDER_CONFIRMED",
        channel=NotificationChannel.EMAIL,
        subject="Your order {{order_id}} is confirmed!",
        body="Hi {{name}}, your order {{order_id}} for {{total}} has been confirmed. "
             "Estimated delivery: {{delivery_date}}.",
        variables=["order_id", "name", "total", "delivery_date"]
    )
    otp_tmpl = ns.create_template(
        name="otp_sms",
        notification_type="OTP",
        channel=NotificationChannel.SMS,
        subject="",
        body="Your verification code is {{otp_code}}. Valid for 5 minutes. "
             "Do not share this code.",
        variables=["otp_code"]
    )
    promo_tmpl = ns.create_template(
        name="promo_summer",
        notification_type="MARKETING",
        channel=NotificationChannel.EMAIL,
        subject="{{discount}} OFF this weekend only! Use code {{promo_code}}",
        body="Hi {{name}}, enjoy {{discount}} off your next purchase! "
             "Use code {{promo_code}} at checkout. Valid until {{expiry}}.",
        variables=["name", "discount", "promo_code", "expiry"]
    )
    print(f"Templates created: {order_tmpl.name}, {otp_tmpl.name}, {promo_tmpl.name}")

    # Render template
    print("\n--- Template Rendering ---")
    subject, body = ns.render_template("order_confirmed", {
        "order_id": "ORD-456789",
        "name": "Alice",
        "total": "$149.99",
        "delivery_date": "May 14, 2026"
    })
    print(f"  Subject: {subject}")
    print(f"  Body:    {body}")

    # Critical notification: OTP
    print("\n--- Critical Notification: OTP ---")
    otp_notifs = ns.send_notification(
        user_id="alice",
        notification_type="OTP",
        priority=NotificationPriority.CRITICAL,
        title="Your verification code",
        body="Your OTP is 847291. Valid for 5 minutes.",
        channels=[NotificationChannel.PUSH, NotificationChannel.SMS],
        idempotency_key="otp-alice-1234567890"
    )
    print(f"  Created {len(otp_notifs)} notifications for alice (push + sms)")

    # Test deduplication
    dup_result = ns.send_notification(
        user_id="alice",
        notification_type="OTP",
        priority=NotificationPriority.CRITICAL,
        title="Your verification code",
        body="Your OTP is 847291. Valid for 5 minutes.",
        channels=[NotificationChannel.PUSH, NotificationChannel.SMS],
        idempotency_key="otp-alice-1234567890"   # Same key!
    )
    print(f"  Duplicate attempt blocked: {len(dup_result)} notifications created")

    # Transactional notifications
    print("\n--- Transactional Notifications: Order Confirmed ---")
    for user_id, order_id in [("alice", "ORD-001"), ("bob", "ORD-002")]:
        notifs = ns.send_notification(
            user_id=user_id,
            notification_type="ORDER_CONFIRMED",
            priority=NotificationPriority.TRANSACTIONAL,
            title="Order Confirmed!",
            body=f"Your order {order_id} has been confirmed.",
            data={"order_id": order_id},
            idempotency_key=f"order-confirm-{order_id}"
        )
        print(f"  {user_id}: {len(notifs)} notifications queued")

    # In-app social notification
    print("\n--- Social Notification: New Follower ---")
    ns.send_notification(
        user_id="alice",
        notification_type="NEW_FOLLOWER",
        priority=NotificationPriority.SOCIAL,
        title="New follower",
        body="Bob started following you!",
        channels=[NotificationChannel.IN_APP, NotificationChannel.PUSH]
    )

    # Marketing
    print("\n--- Marketing Notification ---")
    ns.send_notification(
        user_id="bob",
        notification_type="PROMO",
        priority=NotificationPriority.MARKETING,
        title="30% OFF this weekend!",
        body="Use code SUMMER30 at checkout.",
        channels=[NotificationChannel.EMAIL, NotificationChannel.PUSH]
    )

    # Show queue depths before processing
    print("\n--- Queue Depths (Before Processing) ---")
    print(f"  Queue depths: {ns.priority_queue.queue_depths()}")
    print(f"  Total queued: {ns.priority_queue.total}")

    # Process queue
    print("\n--- Processing Notification Queue ---")
    results = ns.process_queue(max_process=30)
    print(f"  Processing results: {results}")

    # Fan-out / Broadcast
    print("\n--- Fan-out Broadcast (Simulated) ---")
    user_ids = [f"user_{i}" for i in range(50)]
    for uid in user_ids:
        ns.register_user(uid, email=f"{uid}@example.com")
    total = ns.broadcast(
        user_ids=user_ids,
        notification_type="SYSTEM_MAINTENANCE",
        priority=NotificationPriority.TRANSACTIONAL,
        title="Scheduled maintenance",
        body="System maintenance on Sunday 2 AM - 4 AM UTC.",
    )
    print(f"  Broadcast to {len(user_ids)} users -> {total} notifications created")

    # Notification center (in-app inbox)
    print("\n--- Notification Center (In-App Inbox) ---")
    inbox = ns.get_notification_history("alice", limit=5)
    unread = ns.get_unread_count("alice")
    print(f"  Alice's inbox: {len(inbox)} messages, {unread} unread")
    for notif in inbox:
        status = "UNREAD" if not notif.opened_at else "READ"
        print(f"    [{status}] {notif.title} (channel={notif.channel})")

    # Mark first notification as read
    if inbox:
        ns.mark_as_read("alice", inbox[0].notification_id)
        print(f"  After mark-read: {ns.get_unread_count('alice')} unread")

    # Channel stats
    print("\n--- Channel Delivery Stats ---")
    channel_stats = ns.get_channel_stats()
    for channel, stats in channel_stats.items():
        total_c = stats["sent"] + stats["failed"]
        rate = f"{100 * stats['sent'] / total_c:.1f}%" if total_c else "N/A"
        print(f"  {channel:10s}: {stats['sent']} sent, "
              f"{stats['failed']} failed, success_rate={rate}")

    # Overall stats
    print("\n--- Overall System Stats ---")
    stats = ns.get_delivery_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Retry backoff demo
    print("\n--- Exponential Backoff Schedule ---")
    rq = RetryQueue()
    print("  Retry delays for consecutive failures:")
    for attempt in range(6):
        delay = rq._compute_delay(attempt)
        print(f"  Attempt {attempt + 1}: wait ~{delay:.1f}s "
              f"(~{delay/60:.1f} min)")

    # Scale estimates
    print("\n--- Scale Estimates ---")
    stats_list = [
        ("Daily volume",       "1 Billion notifications/day = 12K avg QPS"),
        ("Peak QPS",           "50K notifications/second during events"),
        ("Push share",         "600M FCM calls/day = 7K/sec (well within FCM limits)"),
        ("Fan-out example",    "1 event * 1M subscribers = 1M Kafka messages"),
        ("Dedup cache",        "1B keys/day * 100B = 100GB Redis (with 24h TTL)"),
        ("Retry overhead",     "~2-5% failure rate * 1B = 20-50M retries/day"),
        ("In-app storage",     "Cassandra: 100M in-app * 500B * 30 days = 1.5TB"),
    ]
    for label, value in stats_list:
        print(f"  {label}: {value}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
