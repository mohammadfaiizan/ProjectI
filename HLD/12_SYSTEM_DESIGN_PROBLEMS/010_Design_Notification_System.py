"""
SYSTEM DESIGN: NOTIFICATION SYSTEM
=====================================

Problem Statement:
Design a notification system that sends millions of push notifications,
emails, and SMS messages reliably with low latency.

Functional Requirements:
  - Send notifications via: push (iOS/Android), email, SMS, in-app
  - Support immediate, scheduled, and recurring notifications
  - Template-based notifications with personalization
  - User preferences: opt-out per channel/topic
  - Delivery tracking: sent, delivered, opened

Non-Functional Requirements:
  - 100M notifications/day → ~1150/sec average; burst: 100K/sec
  - Push delivery latency: < 1s for urgent; < 30s for normal
  - Email delivery: < 1 min
  - 99.99% delivery rate (retry on failure)
  - Deduplication (at-least-once delivery without duplicate sends)

Architecture:

  Notification API → Kafka (partitioned by user_id) →
  → Channel Workers (Push/Email/SMS) → 3rd party providers →
  → Delivery Status Tracker → DB

Channel Providers:
  Push: APNs (iOS), FCM (Android/Web)
  Email: SendGrid, Mailgun, SES
  SMS:  Twilio, Nexmo, Plivo

Priority Queues:
  P0 (immediate): OTP, payment confirmations, security alerts
  P1 (fast):      Order shipped, appointment reminder
  P2 (normal):    Marketing, newsletter

Delivery Guarantee:
  Kafka consumer at-least-once + idempotent dedup table.
  Dedup key: notification_id (UUID generated at source).
  If same notification_id seen twice: skip send, return "already sent".

User Preferences:
  Store in Redis (hot) + DB (source of truth).
  Topic subscriptions: {user_id: {channel: {topic: bool}}}.
  Before sending: check if user opted out of this topic/channel.

Retry Logic:
  Exponential backoff: 1s, 2s, 4s, 8s, 16s (max 5 retries).
  After 5 failures: dead letter queue → alert ops team.
  APNs: 400 = bad token (remove from DB); 429 = rate limited (back off).

Throttling:
  Per-user: max 10 notifications/hour per channel.
  Per-provider: respect APNs rate limits (1M per hour per certificate).
  Burst: rate limit by user_id (10 push/hour) to avoid spam.
"""

from __future__ import annotations

import time
import uuid
import json
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# CHANNEL AND PRIORITY
# ─────────────────────────────────────────────

class Channel(Enum):
    PUSH  = "push"
    EMAIL = "email"
    SMS   = "sms"
    INAPP = "in_app"


class Priority(Enum):
    P0_IMMEDIATE = 0   # OTP, security
    P1_FAST      = 1   # transactional
    P2_NORMAL    = 2   # marketing


# ─────────────────────────────────────────────
# NOTIFICATION
# ─────────────────────────────────────────────

@dataclass
class Notification:
    notification_id: str
    user_id:         str
    channel:         Channel
    priority:        Priority
    template_id:     str
    payload:         Dict[str, Any]
    created_at:      float
    scheduled_at:    Optional[float] = None
    sent_at:         Optional[float] = None
    delivered_at:    Optional[float] = None
    opened_at:       Optional[float] = None
    status:          str = "pending"   # pending, sent, delivered, opened, failed
    retry_count:     int = 0


# ─────────────────────────────────────────────
# TEMPLATE ENGINE
# ─────────────────────────────────────────────

@dataclass
class Template:
    template_id:  str
    channel:      Channel
    subject:      Optional[str]   # for email
    body_template: str            # uses {key} substitution

    def render(self, variables: Dict[str, str]) -> str:
        text = self.body_template
        for k, v in variables.items():
            text = text.replace(f"{{{k}}}", str(v))
        return text


class TemplateStore:
    def __init__(self):
        self._templates: Dict[str, Template] = {}

    def register(self, template: Template):
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> Optional[Template]:
        return self._templates.get(template_id)

    def render(self, template_id: str, variables: Dict[str, str]) -> Optional[str]:
        t = self.get(template_id)
        return t.render(variables) if t else None


# ─────────────────────────────────────────────
# USER PREFERENCES
# ─────────────────────────────────────────────

@dataclass
class UserPreference:
    user_id:   str
    opted_out_channels: Set[Channel] = field(default_factory=set)
    opted_out_topics:   Set[str]     = field(default_factory=set)
    # device tokens for push
    push_tokens: List[str] = field(default_factory=list)
    email:       Optional[str] = None
    phone:       Optional[str] = None

    def can_receive(self, channel: Channel, topic: str) -> bool:
        return (channel not in self.opted_out_channels and
                topic not in self.opted_out_topics)


class PreferenceStore:
    def __init__(self):
        self._prefs: Dict[str, UserPreference] = {}

    def set(self, pref: UserPreference):
        self._prefs[pref.user_id] = pref

    def get(self, user_id: str) -> Optional[UserPreference]:
        return self._prefs.get(user_id)

    def opt_out(self, user_id: str, channel: Optional[Channel] = None,
                topic: Optional[str] = None):
        pref = self._prefs.get(user_id)
        if not pref:
            return
        if channel:
            pref.opted_out_channels.add(channel)
        if topic:
            pref.opted_out_topics.add(topic)


# ─────────────────────────────────────────────
# CHANNEL WORKER (simulated)
# ─────────────────────────────────────────────

@dataclass
class DeliveryResult:
    notification_id: str
    status:          str   # delivered, failed, rate_limited
    provider:        str
    error:           Optional[str] = None
    retry_after_s:   Optional[float] = None


class ChannelWorker:
    """Base class for channel-specific delivery workers."""

    def __init__(self, channel: Channel, fail_rate: float = 0.02):
        self.channel    = channel
        self._fail_rate = fail_rate
        self._sent:     List[DeliveryResult] = []

    def send(self, notification: Notification,
             pref: Optional[UserPreference]) -> DeliveryResult:
        raise NotImplementedError

    def _simulate_send(self, notif_id: str, provider: str) -> DeliveryResult:
        r = random.random()
        if r < self._fail_rate:
            return DeliveryResult(notif_id, "failed", provider,
                                  error="provider_error")
        if r < self._fail_rate + 0.01:
            return DeliveryResult(notif_id, "rate_limited", provider,
                                  error="rate_limited", retry_after_s=5.0)
        return DeliveryResult(notif_id, "delivered", provider)


class PushWorker(ChannelWorker):
    def __init__(self):
        super().__init__(Channel.PUSH, fail_rate=0.01)

    def send(self, notification: Notification,
             pref: Optional[UserPreference]) -> DeliveryResult:
        if not pref or not pref.push_tokens:
            return DeliveryResult(notification.notification_id,
                                  "failed", "fcm", "no_device_token")
        # In prod: send to FCM/APNs per token
        provider = "apns" if "ios" in pref.push_tokens[0] else "fcm"
        result = self._simulate_send(notification.notification_id, provider)
        self._sent.append(result)
        return result


class EmailWorker(ChannelWorker):
    def __init__(self):
        super().__init__(Channel.EMAIL, fail_rate=0.02)

    def send(self, notification: Notification,
             pref: Optional[UserPreference]) -> DeliveryResult:
        if not pref or not pref.email:
            return DeliveryResult(notification.notification_id,
                                  "failed", "sendgrid", "no_email")
        result = self._simulate_send(notification.notification_id, "sendgrid")
        self._sent.append(result)
        return result


class SMSWorker(ChannelWorker):
    def __init__(self):
        super().__init__(Channel.SMS, fail_rate=0.03)

    def send(self, notification: Notification,
             pref: Optional[UserPreference]) -> DeliveryResult:
        if not pref or not pref.phone:
            return DeliveryResult(notification.notification_id,
                                  "failed", "twilio", "no_phone")
        result = self._simulate_send(notification.notification_id, "twilio")
        self._sent.append(result)
        return result


# ─────────────────────────────────────────────
# DEDUPLICATION STORE
# ─────────────────────────────────────────────

class DeduplicationStore:
    """
    Tracks already-sent notification_ids.
    In prod: Redis SET with TTL = 7 days.
    """

    def __init__(self, ttl_s: float = 86400 * 7):
        self._sent:   Dict[str, float] = {}   # notif_id → sent_at
        self._ttl     = ttl_s

    def is_duplicate(self, notification_id: str) -> bool:
        sent_at = self._sent.get(notification_id)
        if sent_at is None:
            return False
        if time.time() - sent_at > self._ttl:
            del self._sent[notification_id]
            return False
        return True

    def mark_sent(self, notification_id: str):
        self._sent[notification_id] = time.time()


# ─────────────────────────────────────────────
# RETRY QUEUE
# ─────────────────────────────────────────────

@dataclass
class RetryTask:
    notification: Notification
    next_attempt: float
    attempt_num:  int

    def __lt__(self, other: "RetryTask"):
        return self.next_attempt < other.next_attempt


class RetryQueue:
    """Priority queue for failed notifications to be retried."""

    MAX_ATTEMPTS = 5
    BASE_DELAY   = 1.0   # seconds

    def __init__(self):
        self._queue: List[RetryTask] = []
        self._dlq:   List[Notification] = []   # dead letter queue

    def add(self, notification: Notification, attempt_num: int = 1):
        if attempt_num > self.MAX_ATTEMPTS:
            self._dlq.append(notification)
            return
        delay     = self.BASE_DELAY * (2 ** (attempt_num - 1))
        next_time = time.time() + delay
        self._queue.append(RetryTask(notification, next_time, attempt_num))
        self._queue.sort()

    def pop_ready(self) -> List[RetryTask]:
        now    = time.time()
        ready  = [t for t in self._queue if t.next_attempt <= now]
        self._queue = [t for t in self._queue if t.next_attempt > now]
        return ready

    def dlq_size(self) -> int:
        return len(self._dlq)


# ─────────────────────────────────────────────
# NOTIFICATION SERVICE
# ─────────────────────────────────────────────

class NotificationService:
    def __init__(self):
        self._templates  = TemplateStore()
        self._prefs      = PreferenceStore()
        self._dedup      = DeduplicationStore()
        self._retry_q    = RetryQueue()
        self._workers: Dict[Channel, ChannelWorker] = {
            Channel.PUSH:  PushWorker(),
            Channel.EMAIL: EmailWorker(),
            Channel.SMS:   SMSWorker(),
        }
        self._notifications: List[Notification] = []
        self._stats = defaultdict(int)

    def register_template(self, template: Template):
        self._templates.register(template)

    def register_user(self, user_id: str, push_token: str = None,
                      email: str = None, phone: str = None) -> UserPreference:
        pref = UserPreference(
            user_id     = user_id,
            push_tokens = [push_token] if push_token else [],
            email       = email,
            phone       = phone,
        )
        self._prefs.set(pref)
        return pref

    def send(self, user_id: str, channel: Channel, template_id: str,
             variables: Dict[str, str], priority: Priority = Priority.P1_FAST,
             topic: str = "general",
             idempotency_key: Optional[str] = None) -> Optional[Notification]:

        # Check preferences
        pref = self._prefs.get(user_id)
        if pref and not pref.can_receive(channel, topic):
            self._stats["opted_out"] += 1
            return None

        notif_id = idempotency_key or uuid.uuid4().hex

        # Dedup check
        if self._dedup.is_duplicate(notif_id):
            self._stats["deduplicated"] += 1
            return None

        # Render template
        body = self._templates.render(template_id, variables)

        notif = Notification(
            notification_id = notif_id,
            user_id         = user_id,
            channel         = channel,
            priority        = priority,
            template_id     = template_id,
            payload         = {"body": body, "variables": variables},
            created_at      = time.time(),
        )
        self._notifications.append(notif)
        self._dispatch(notif)
        return notif

    def _dispatch(self, notif: Notification):
        pref   = self._prefs.get(notif.user_id)
        worker = self._workers.get(notif.channel)
        if not worker:
            notif.status = "failed"
            self._stats["no_worker"] += 1
            return

        result = worker.send(notif, pref)
        notif.sent_at = time.time()

        if result.status == "delivered":
            notif.status      = "delivered"
            notif.delivered_at = time.time()
            self._dedup.mark_sent(notif.notification_id)
            self._stats["delivered"] += 1
        else:
            notif.status = "failed"
            self._stats["failed"] += 1
            # Schedule retry
            self._retry_q.add(notif, attempt_num=notif.retry_count + 1)
            notif.retry_count += 1

    def process_retries(self):
        ready = self._retry_q.pop_ready()
        for task in ready:
            self._dispatch(task.notification)

    def stats(self) -> Dict:
        return dict(self._stats)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_notifications():
    print("=" * 65)
    print("SYSTEM DESIGN: NOTIFICATION SYSTEM")
    print("=" * 65)

    random.seed(42)
    svc = NotificationService()

    # ── Register Templates ────────────────────
    print("\n[1] TEMPLATES")
    print("─" * 55)

    templates = [
        Template("otp_sms",      Channel.SMS,   None,
                 "Your OTP is {otp}. Valid for {minutes} minutes."),
        Template("order_push",   Channel.PUSH,  None,
                 "Your order #{order_id} has shipped! Track: {link}"),
        Template("welcome_email",Channel.EMAIL, "Welcome to Example, {name}!",
                 "Hi {name}, welcome aboard! Get started: {link}"),
        Template("promo_push",   Channel.PUSH,  None,
                 "Flash sale! {discount}% off today only. Code: {code}"),
    ]
    for t in templates:
        svc.register_template(t)
        print(f"  [{t.channel.value}] {t.template_id}: {t.body_template[:50]}")

    # ── Register Users ────────────────────────
    print("\n[2] USER REGISTRATION")
    print("─" * 55)

    users = [
        ("user_001", "ios_token_abc123", "alice@example.com", "+1555001"),
        ("user_002", "android_tok_xyz", "bob@example.com",   "+1555002"),
        ("user_003", None,               "carol@example.com", None),
    ]
    for uid, push, email, phone in users:
        pref = svc.register_user(uid, push, email, phone)
        print(f"  {uid}: push={'✓' if push else '✗'}  email={'✓' if email else '✗'}  "
              f"sms={'✓' if phone else '✗'}")

    # ── Send Notifications ────────────────────
    print("\n[3] SENDING NOTIFICATIONS")
    print("─" * 55)

    sends = [
        ("user_001", Channel.SMS,   "otp_sms",      {"otp": "847263", "minutes": "5"},
         Priority.P0_IMMEDIATE, "auth"),
        ("user_001", Channel.PUSH,  "order_push",   {"order_id": "ORD-9876", "link": "https://ex.co/track"},
         Priority.P1_FAST, "orders"),
        ("user_002", Channel.EMAIL, "welcome_email",{"name": "Bob", "link": "https://ex.co/start"},
         Priority.P1_FAST, "account"),
        ("user_003", Channel.PUSH,  "promo_push",   {"discount": "30", "code": "SAVE30"},
         Priority.P2_NORMAL, "marketing"),
    ]

    for uid, channel, tmpl, vars_, prio, topic in sends:
        notif = svc.send(uid, channel, tmpl, vars_, prio, topic)
        if notif:
            print(f"  {uid} [{channel.value}] {tmpl}: status={notif.status}")
            if notif.payload.get("body"):
                print(f"    body: {notif.payload['body'][:60]}")
        else:
            print(f"  {uid} [{channel.value}]: SKIPPED (opted out or dedup)")

    # ── Deduplication ─────────────────────────
    print("\n[4] DEDUPLICATION (idempotency)")
    print("─" * 55)

    idem_key = "notif_" + uuid.uuid4().hex[:8]
    n1 = svc.send("user_001", Channel.PUSH, "order_push",
                   {"order_id": "ORD-TEST", "link": "https://x"}, topic="orders",
                   idempotency_key=idem_key)
    n2 = svc.send("user_001", Channel.PUSH, "order_push",
                   {"order_id": "ORD-TEST", "link": "https://x"}, topic="orders",
                   idempotency_key=idem_key)
    print(f"  First send:  {'sent' if n1 else 'dedup/opt-out'}")
    print(f"  Second send: {'sent' if n2 else 'dedup/opt-out (same idempotency key)'}")

    # ── Opt-out ───────────────────────────────
    print("\n[5] USER OPT-OUT")
    print("─" * 55)

    svc._prefs.opt_out("user_002", topic="marketing")
    n = svc.send("user_002", Channel.EMAIL, "promo_push",
                  {"discount": "50", "code": "HALF"}, topic="marketing")
    print(f"  @user_002 opted out of marketing → send result: {n}")

    svc._prefs.opt_out("user_001", channel=Channel.SMS)
    n = svc.send("user_001", Channel.SMS, "otp_sms",
                  {"otp": "000000", "minutes": "5"}, topic="auth")
    print(f"  @user_001 opted out of SMS → send result: {n}")

    # ── Stats ─────────────────────────────────
    print("\n[6] DELIVERY STATISTICS")
    print("─" * 55)

    stats = svc.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  Total notifications: {len(svc._notifications)}")
    print(f"  Retry queue depth:   {len(svc._retry_q._queue)}")
    print(f"  Dead letter queue:   {svc._retry_q.dlq_size()}")

    # ── Architecture ──────────────────────────
    print("\n[7] NOTIFICATION SYSTEM ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Ingestion",       "REST API → Kafka (partitioned by user_id)"),
        ("Priority",        "3 Kafka topics: P0_immediate, P1_fast, P2_normal"),
        ("Preferences",     "Redis (hot cache) + MySQL (source of truth)"),
        ("Deduplication",   "Redis SET: notification_id → sent_at (7d TTL)"),
        ("Push delivery",   "FCM + APNs; token registration in DB"),
        ("Email",           "SendGrid / SES; DKIM/SPF for deliverability"),
        ("SMS",             "Twilio primary; Nexmo fallback"),
        ("Retry",           "Exponential backoff (1s, 2s, 4s, 8s, 16s); DLQ"),
        ("Analytics",       "Click tracking pixel; open webhook from provider"),
        ("Throttling",      "Max 10 push/hour per user; respect provider limits"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_notifications()
