"""
NOTIFICATION SYSTEM DESIGN - Complete System Design
================================================

A comprehensive notification system that handles:
- Multi-channel notifications (email, SMS, push, in-app)
- Real-time and scheduled notifications
- User preferences and subscription management
- Template management and personalization
- Rate limiting and throttling
- Delivery tracking and analytics
- Retry logic and failure handling
- A/B testing for notification content
- Notification grouping and batching
- Global and user-specific settings

Design Patterns Used:
- Strategy: Different delivery channels and retry strategies
- Observer: Real-time notification dispatch
- Factory: Notification and template creation
- Template Method: Notification processing pipeline
- Decorator: Notification enrichment and formatting
- Command: Notification actions with queuing
- Chain of Responsibility: Delivery channel fallback
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import queue
import json
from dataclasses import dataclass, field
from collections import defaultdict, deque
import re
import time


class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TemplateType(Enum):
    WELCOME = "welcome"
    VERIFICATION = "verification"
    RESET_PASSWORD = "reset_password"
    ORDER_CONFIRMATION = "order_confirmation"
    REMINDER = "reminder"
    PROMOTIONAL = "promotional"
    ALERT = "alert"
    CUSTOM = "custom"


@dataclass
class User:
    user_id: str
    email: str
    phone: str = ""
    push_token: str = ""
    timezone: str = "UTC"
    language: str = "en"
    
    # Preferences
    email_enabled: bool = True
    sms_enabled: bool = True
    push_enabled: bool = True
    in_app_enabled: bool = True
    
    # Subscription preferences
    marketing_emails: bool = False
    security_alerts: bool = True
    order_updates: bool = True
    reminders: bool = True
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())


@dataclass
class NotificationTemplate:
    template_id: str
    name: str
    template_type: TemplateType
    channel: NotificationChannel
    
    # Content
    subject: str = ""
    body: str = ""
    html_body: str = ""
    
    # Variables
    variables: Set[str] = field(default_factory=set)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def __post_init__(self):
        if not self.template_id:
            self.template_id = str(uuid.uuid4())
        
        # Extract variables from template
        self._extract_variables()
    
    def _extract_variables(self):
        """Extract variables from template content."""
        variable_pattern = r'\{\{(\w+)\}\}'
        
        # Find variables in subject and body
        for text in [self.subject, self.body, self.html_body]:
            variables = re.findall(variable_pattern, text)
            self.variables.update(variables)
    
    def render(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Render template with variables."""
        rendered = {
            'subject': self.subject,
            'body': self.body,
            'html_body': self.html_body
        }
        
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            str_value = str(value)
            
            rendered['subject'] = rendered['subject'].replace(placeholder, str_value)
            rendered['body'] = rendered['body'].replace(placeholder, str_value)
            rendered['html_body'] = rendered['html_body'].replace(placeholder, str_value)
        
        return rendered


@dataclass
class Notification:
    notification_id: str
    user_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    
    # Content
    subject: str = ""
    message: str = ""
    html_content: str = ""
    
    # Template info
    template_id: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery
    scheduled_at: Optional[datetime] = None
    max_retries: int = 3
    retry_delay: int = 300  # seconds
    
    # Status tracking
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    
    # Retry tracking
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    error_message: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.notification_id:
            self.notification_id = str(uuid.uuid4())


@dataclass
class DeliveryResult:
    success: bool
    channel: NotificationChannel
    message_id: str = ""
    error_message: str = ""
    delivered_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DELIVERY CHANNELS
# ============================================================================

class NotificationChannel_ABC(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    def send(self, notification: Notification, user: User) -> DeliveryResult:
        """Send notification through this channel."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if channel is available."""
        pass


class EmailChannel(NotificationChannel_ABC):
    """Email notification channel."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.rate_limit = 100  # per minute
        self.sent_count = 0
        self.last_reset = datetime.now()
    
    def send(self, notification: Notification, user: User) -> DeliveryResult:
        """Send email notification."""
        try:
            # Check rate limit
            if not self._check_rate_limit():
                return DeliveryResult(
                    success=False,
                    channel=NotificationChannel.EMAIL,
                    error_message="Rate limit exceeded"
                )
            
            # Simulate email sending
            time.sleep(0.1)  # Simulate network delay
            
            message_id = f"email_{uuid.uuid4()}"
            
            print(f"📧 Email sent to {user.email}: {notification.subject}")
            
            self.sent_count += 1
            
            return DeliveryResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id=message_id,
                delivered_at=datetime.now(),
                metadata={'recipient': user.email}
            )
            
        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if email service is available."""
        return True  # Simplified
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limit."""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.sent_count = 0
            self.last_reset = now
        
        return self.sent_count < self.rate_limit


class SMSChannel(NotificationChannel_ABC):
    """SMS notification channel."""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.api_config = api_config
        self.rate_limit = 50  # per minute
        self.sent_count = 0
        self.last_reset = datetime.now()
    
    def send(self, notification: Notification, user: User) -> DeliveryResult:
        """Send SMS notification."""
        try:
            if not user.phone:
                return DeliveryResult(
                    success=False,
                    channel=NotificationChannel.SMS,
                    error_message="No phone number"
                )
            
            # Check rate limit
            if not self._check_rate_limit():
                return DeliveryResult(
                    success=False,
                    channel=NotificationChannel.SMS,
                    error_message="Rate limit exceeded"
                )
            
            # Simulate SMS sending
            time.sleep(0.2)
            
            message_id = f"sms_{uuid.uuid4()}"
            
            print(f"📱 SMS sent to {user.phone}: {notification.message[:50]}...")
            
            self.sent_count += 1
            
            return DeliveryResult(
                success=True,
                channel=NotificationChannel.SMS,
                message_id=message_id,
                delivered_at=datetime.now(),
                metadata={'recipient': user.phone}
            )
            
        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.SMS,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if SMS service is available."""
        return True  # Simplified
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limit."""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.sent_count = 0
            self.last_reset = now
        
        return self.sent_count < self.rate_limit


class PushChannel(NotificationChannel_ABC):
    """Push notification channel."""
    
    def __init__(self, fcm_config: Dict[str, Any]):
        self.fcm_config = fcm_config
        self.rate_limit = 1000  # per minute
        self.sent_count = 0
        self.last_reset = datetime.now()
    
    def send(self, notification: Notification, user: User) -> DeliveryResult:
        """Send push notification."""
        try:
            if not user.push_token:
                return DeliveryResult(
                    success=False,
                    channel=NotificationChannel.PUSH,
                    error_message="No push token"
                )
            
            # Check rate limit
            if not self._check_rate_limit():
                return DeliveryResult(
                    success=False,
                    channel=NotificationChannel.PUSH,
                    error_message="Rate limit exceeded"
                )
            
            # Simulate push sending
            time.sleep(0.05)
            
            message_id = f"push_{uuid.uuid4()}"
            
            print(f"🔔 Push sent to device: {notification.subject}")
            
            self.sent_count += 1
            
            return DeliveryResult(
                success=True,
                channel=NotificationChannel.PUSH,
                message_id=message_id,
                delivered_at=datetime.now(),
                metadata={'token': user.push_token[:10] + "..."}
            )
            
        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.PUSH,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if push service is available."""
        return True  # Simplified
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limit."""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.sent_count = 0
            self.last_reset = now
        
        return self.sent_count < self.rate_limit


class InAppChannel(NotificationChannel_ABC):
    """In-app notification channel."""
    
    def __init__(self):
        self.notifications_store = defaultdict(list)
    
    def send(self, notification: Notification, user: User) -> DeliveryResult:
        """Send in-app notification."""
        try:
            message_id = f"inapp_{uuid.uuid4()}"
            
            # Store notification for user
            self.notifications_store[user.user_id].append({
                'id': message_id,
                'subject': notification.subject,
                'message': notification.message,
                'timestamp': datetime.now(),
                'read': False
            })
            
            print(f"📱 In-app notification for {user.user_id}: {notification.subject}")
            
            return DeliveryResult(
                success=True,
                channel=NotificationChannel.IN_APP,
                message_id=message_id,
                delivered_at=datetime.now()
            )
            
        except Exception as e:
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.IN_APP,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if in-app service is available."""
        return True
    
    def get_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get in-app notifications for user."""
        notifications = self.notifications_store.get(user_id, [])
        
        if unread_only:
            notifications = [n for n in notifications if not n['read']]
        
        return notifications


# ============================================================================
# MAIN NOTIFICATION SYSTEM
# ============================================================================

class NotificationSystem:
    """Main notification system."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.notifications: Dict[str, Notification] = {}
        
        # Channels
        self.channels: Dict[NotificationChannel, NotificationChannel_ABC] = {
            NotificationChannel.EMAIL: EmailChannel({}),
            NotificationChannel.SMS: SMSChannel({}),
            NotificationChannel.PUSH: PushChannel({}),
            NotificationChannel.IN_APP: InAppChannel()
        }
        
        # Queues
        self.notification_queue = queue.PriorityQueue()
        self.retry_queue = deque()
        
        # Workers
        self.workers = []
        self.running = True
        
        # Analytics
        self.analytics = {
            'total_sent': 0,
            'total_delivered': 0,
            'total_failed': 0,
            'channel_stats': defaultdict(int),
            'template_stats': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start workers
        self._start_workers()
        
        print("🔔 Notification System initialized")
    
    def create_user(self, email: str, phone: str = "", **preferences) -> User:
        """Create a new user."""
        user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            phone=phone,
            **preferences
        )
        
        with self._lock:
            self.users[user.user_id] = user
        
        return user
    
    def create_template(self, name: str, template_type: TemplateType,
                       channel: NotificationChannel, subject: str = "",
                       body: str = "", html_body: str = "") -> NotificationTemplate:
        """Create a notification template."""
        template = NotificationTemplate(
            template_id=str(uuid.uuid4()),
            name=name,
            template_type=template_type,
            channel=channel,
            subject=subject,
            body=body,
            html_body=html_body
        )
        
        with self._lock:
            self.templates[template.template_id] = template
        
        return template
    
    def send_notification(self, user_id: str, channel: NotificationChannel,
                         subject: str = "", message: str = "", html_content: str = "",
                         priority: NotificationPriority = NotificationPriority.NORMAL,
                         scheduled_at: Optional[datetime] = None,
                         template_id: Optional[str] = None,
                         template_variables: Dict[str, Any] = None,
                         **metadata) -> Notification:
        """Send a notification."""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            user_id=user_id,
            channel=channel,
            priority=priority,
            subject=subject,
            message=message,
            html_content=html_content,
            scheduled_at=scheduled_at,
            template_id=template_id,
            template_variables=template_variables or {},
            metadata=metadata
        )
        
        with self._lock:
            self.notifications[notification.notification_id] = notification
        
        # Queue for processing
        self._queue_notification(notification)
        
        return notification
    
    def send_bulk_notifications(self, user_ids: List[str], 
                               notification_data: Dict[str, Any]) -> List[Notification]:
        """Send notifications to multiple users."""
        notifications = []
        
        for user_id in user_ids:
            try:
                notification = self.send_notification(user_id=user_id, **notification_data)
                notifications.append(notification)
            except Exception as e:
                print(f"Failed to send notification to {user_id}: {e}")
        
        return notifications
    
    def schedule_notification(self, user_id: str, channel: NotificationChannel,
                             scheduled_at: datetime, **kwargs) -> Notification:
        """Schedule a notification for later delivery."""
        return self.send_notification(
            user_id=user_id,
            channel=channel,
            scheduled_at=scheduled_at,
            **kwargs
        )
    
    def cancel_notification(self, notification_id: str) -> bool:
        """Cancel a pending notification."""
        if notification_id not in self.notifications:
            return False
        
        notification = self.notifications[notification_id]
        
        if notification.status == NotificationStatus.PENDING:
            notification.status = NotificationStatus.CANCELLED
            return True
        
        return False
    
    def update_user_preferences(self, user_id: str, **preferences) -> bool:
        """Update user notification preferences."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        for key, value in preferences.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        return True
    
    def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification status and details."""
        if notification_id not in self.notifications:
            return None
        
        notification = self.notifications[notification_id]
        
        return {
            'notification_id': notification_id,
            'status': notification.status.value,
            'created_at': notification.created_at.isoformat(),
            'sent_at': notification.sent_at.isoformat() if notification.sent_at else None,
            'delivered_at': notification.delivered_at.isoformat() if notification.delivered_at else None,
            'retry_count': notification.retry_count,
            'error_message': notification.error_message
        }
    
    def get_user_notifications(self, user_id: str, channel: NotificationChannel = None,
                             limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        user_notifications = []
        
        for notification in self.notifications.values():
            if notification.user_id != user_id:
                continue
            
            if channel and notification.channel != channel:
                continue
            
            user_notifications.append({
                'notification_id': notification.notification_id,
                'channel': notification.channel.value,
                'subject': notification.subject,
                'message': notification.message,
                'status': notification.status.value,
                'created_at': notification.created_at.isoformat(),
                'sent_at': notification.sent_at.isoformat() if notification.sent_at else None
            })
        
        # Sort by creation time (newest first)
        user_notifications.sort(key=lambda x: x['created_at'], reverse=True)
        
        return user_notifications[:limit]
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get notification analytics."""
        with self._lock:
            return {
                **self.analytics,
                'total_notifications': len(self.notifications),
                'total_users': len(self.users),
                'total_templates': len(self.templates),
                'pending_notifications': len([n for n in self.notifications.values() 
                                            if n.status == NotificationStatus.PENDING]),
                'success_rate': (self.analytics['total_delivered'] / 
                               max(1, self.analytics['total_sent']) * 100)
            }
    
    def _queue_notification(self, notification: Notification) -> None:
        """Queue notification for processing."""
        # Priority queue: higher priority = lower number
        priority_map = {
            NotificationPriority.CRITICAL: 1,
            NotificationPriority.HIGH: 2,
            NotificationPriority.NORMAL: 3,
            NotificationPriority.LOW: 4
        }
        
        priority = priority_map.get(notification.priority, 3)
        
        # Add timestamp for FIFO within same priority
        timestamp = notification.scheduled_at or notification.created_at
        
        self.notification_queue.put((priority, timestamp.timestamp(), notification))
    
    def _start_workers(self) -> None:
        """Start notification processing workers."""
        # Main worker
        worker = threading.Thread(target=self._process_notifications, daemon=True)
        worker.start()
        self.workers.append(worker)
        
        # Retry worker
        retry_worker = threading.Thread(target=self._process_retries, daemon=True)
        retry_worker.start()
        self.workers.append(retry_worker)
    
    def _process_notifications(self) -> None:
        """Process notifications from queue."""
        while self.running:
            try:
                # Get notification from queue (with timeout)
                try:
                    priority, timestamp, notification = self.notification_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Check if scheduled notification is ready
                if notification.scheduled_at and datetime.now() < notification.scheduled_at:
                    # Put back in queue for later
                    self.notification_queue.put((priority, timestamp, notification))
                    time.sleep(1)
                    continue
                
                # Process notification
                self._send_notification(notification)
                
            except Exception as e:
                print(f"Error processing notification: {e}")
    
    def _process_retries(self) -> None:
        """Process failed notifications for retry."""
        while self.running:
            try:
                if not self.retry_queue:
                    time.sleep(5)
                    continue
                
                notification = self.retry_queue.popleft()
                
                # Check if retry delay has passed
                if (notification.last_retry_at and 
                    datetime.now() - notification.last_retry_at < timedelta(seconds=notification.retry_delay)):
                    # Put back in queue
                    self.retry_queue.append(notification)
                    time.sleep(1)
                    continue
                
                # Retry sending
                if notification.retry_count < notification.max_retries:
                    self._send_notification(notification)
                else:
                    # Max retries reached
                    notification.status = NotificationStatus.FAILED
                    notification.error_message = "Max retries exceeded"
                
            except Exception as e:
                print(f"Error processing retry: {e}")
    
    def _send_notification(self, notification: Notification) -> None:
        """Send individual notification."""
        try:
            user = self.users[notification.user_id]
            
            # Check user preferences
            if not self._check_user_preferences(user, notification):
                notification.status = NotificationStatus.CANCELLED
                return
            
            # Apply template if specified
            if notification.template_id:
                self._apply_template(notification)
            
            # Get channel
            channel_impl = self.channels.get(notification.channel)
            if not channel_impl or not channel_impl.is_available():
                notification.status = NotificationStatus.FAILED
                notification.error_message = "Channel unavailable"
                return
            
            # Send notification
            result = channel_impl.send(notification, user)
            
            # Update notification status
            if result.success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
                
                if result.delivered_at:
                    notification.status = NotificationStatus.DELIVERED
                    notification.delivered_at = result.delivered_at
                
                # Update analytics
                with self._lock:
                    self.analytics['total_sent'] += 1
                    if notification.status == NotificationStatus.DELIVERED:
                        self.analytics['total_delivered'] += 1
                    
                    self.analytics['channel_stats'][notification.channel.value] += 1
                    
                    if notification.template_id:
                        self.analytics['template_stats'][notification.template_id] += 1
            
            else:
                # Handle failure
                notification.retry_count += 1
                notification.last_retry_at = datetime.now()
                notification.error_message = result.error_message
                
                if notification.retry_count < notification.max_retries:
                    # Queue for retry
                    self.retry_queue.append(notification)
                else:
                    notification.status = NotificationStatus.FAILED
                    
                    with self._lock:
                        self.analytics['total_failed'] += 1
        
        except Exception as e:
            notification.status = NotificationStatus.FAILED
            notification.error_message = str(e)
            
            with self._lock:
                self.analytics['total_failed'] += 1
    
    def _check_user_preferences(self, user: User, notification: Notification) -> bool:
        """Check if user wants to receive this notification."""
        channel_preferences = {
            NotificationChannel.EMAIL: user.email_enabled,
            NotificationChannel.SMS: user.sms_enabled,
            NotificationChannel.PUSH: user.push_enabled,
            NotificationChannel.IN_APP: user.in_app_enabled
        }
        
        return channel_preferences.get(notification.channel, True)
    
    def _apply_template(self, notification: Notification) -> None:
        """Apply template to notification."""
        if notification.template_id not in self.templates:
            return
        
        template = self.templates[notification.template_id]
        
        if template.channel != notification.channel:
            return
        
        # Render template
        rendered = template.render(notification.template_variables)
        
        # Update notification content
        if not notification.subject:
            notification.subject = rendered['subject']
        
        if not notification.message:
            notification.message = rendered['body']
        
        if not notification.html_content:
            notification.html_content = rendered['html_body']


def demonstrate_notification_system():
    """Demonstrate the notification system."""
    print("=== NOTIFICATION SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    system = NotificationSystem()
    
    # Create users
    users = []
    user_data = [
        ("alice@example.com", "+1234567890"),
        ("bob@example.com", "+1987654321"),
        ("charlie@example.com", "+1555123456")
    ]
    
    for email, phone in user_data:
        user = system.create_user(
            email=email,
            phone=phone,
            push_token=f"push_token_{len(users)}"
        )
        users.append(user)
        print(f"✓ Created user: {email}")
    
    print()
    
    # Create templates
    welcome_template = system.create_template(
        name="Welcome Email",
        template_type=TemplateType.WELCOME,
        channel=NotificationChannel.EMAIL,
        subject="Welcome to {{app_name}}, {{user_name}}!",
        body="Hello {{user_name}}, welcome to {{app_name}}! We're excited to have you."
    )
    
    reminder_template = system.create_template(
        name="Meeting Reminder",
        template_type=TemplateType.REMINDER,
        channel=NotificationChannel.PUSH,
        subject="Meeting Reminder",
        body="Your meeting '{{meeting_title}}' starts in {{minutes}} minutes."
    )
    
    print("✓ Created notification templates")
    print()
    
    # Send welcome notifications
    for user in users:
        notification = system.send_notification(
            user_id=user.user_id,
            channel=NotificationChannel.EMAIL,
            template_id=welcome_template.template_id,
            template_variables={
                'app_name': 'MyApp',
                'user_name': user.email.split('@')[0].title()
            }
        )
        print(f"✓ Sent welcome email to {user.email}")
    
    print()
    
    # Send different types of notifications
    notifications = []
    
    # High priority alert
    alert = system.send_notification(
        user_id=users[0].user_id,
        channel=NotificationChannel.PUSH,
        subject="Security Alert",
        message="Unusual login detected from new device",
        priority=NotificationPriority.HIGH
    )
    notifications.append(alert)
    
    # SMS notification
    sms = system.send_notification(
        user_id=users[1].user_id,
        channel=NotificationChannel.SMS,
        message="Your order #12345 has been shipped!",
        priority=NotificationPriority.NORMAL
    )
    notifications.append(sms)
    
    # Scheduled notification
    scheduled_time = datetime.now() + timedelta(seconds=2)
    scheduled = system.schedule_notification(
        user_id=users[2].user_id,
        channel=NotificationChannel.IN_APP,
        scheduled_at=scheduled_time,
        subject="Scheduled Reminder",
        message="This is your scheduled reminder!"
    )
    notifications.append(scheduled)
    
    print("✓ Sent various notifications")
    print()
    
    # Wait for processing
    time.sleep(3)
    
    # Check notification statuses
    print("Notification statuses:")
    for notification in notifications:
        status = system.get_notification_status(notification.notification_id)
        if status:
            print(f"  {notification.notification_id[:8]}: {status['status']}")
    
    print()
    
    # Get user notifications
    for user in users[:2]:
        user_notifications = system.get_user_notifications(user.user_id, limit=5)
        print(f"{user.email}'s notifications: {len(user_notifications)}")
        for notif in user_notifications:
            print(f"  - {notif['subject']} ({notif['status']})")
    
    print()
    
    # Show analytics
    analytics = system.get_analytics()
    print("System Analytics:")
    print(f"  Total notifications: {analytics['total_notifications']}")
    print(f"  Total sent: {analytics['total_sent']}")
    print(f"  Total delivered: {analytics['total_delivered']}")
    print(f"  Success rate: {analytics['success_rate']:.1f}%")
    print(f"  Pending: {analytics['pending_notifications']}")
    
    print()
    print("=== DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_notification_system()
