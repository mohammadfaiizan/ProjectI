"""
CHAT APPLICATION DESIGN - Complete System Design
================================================

Problem Statement:
Design a comprehensive chat application system that handles:
- Real-time messaging between users
- Group chats and private conversations
- Message delivery and read receipts
- User presence and online status
- File sharing and media messages
- Message encryption and security
- Push notifications
- Message history and search
- User authentication and profiles
- Moderation and admin features

Requirements:
- Support real-time bidirectional communication
- Handle thousands of concurrent users
- Implement message persistence and history
- Support different message types (text, image, file, emoji)
- Provide end-to-end encryption for security
- Handle user presence and typing indicators
- Support group management (create, join, leave, admin)
- Implement message delivery status tracking
- Provide search functionality across messages
- Support user blocking and reporting

Design Patterns Used:
- Observer: Real-time message notifications
- Strategy: Message encryption strategies
- Command: Message operations
- Factory: Message type creation
- Mediator: Chat room coordination
- State: User presence states
- Decorator: Message formatting and encryption
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time
from dataclasses import dataclass, field
import json
import hashlib
import base64
from collections import defaultdict, deque
import asyncio
import websockets
from cryptography.fernet import Fernet


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    EMOJI = "emoji"
    SYSTEM = "system"
    VOICE = "voice"
    VIDEO = "video"


class MessageStatus(Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


class UserStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    AWAY = "away"
    BUSY = "busy"
    INVISIBLE = "invisible"


class ChatType(Enum):
    PRIVATE = "private"
    GROUP = "group"
    CHANNEL = "channel"


class UserRole(Enum):
    MEMBER = "member"
    ADMIN = "admin"
    MODERATOR = "moderator"
    OWNER = "owner"


@dataclass
class MessageContent:
    """Message content data."""
    text: str = ""
    file_url: str = ""
    file_name: str = ""
    file_size: int = 0
    thumbnail_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeliveryReceipt:
    """Message delivery receipt."""
    message_id: str
    user_id: str
    status: MessageStatus
    timestamp: datetime


# ============================================================================
# ENCRYPTION STRATEGIES
# ============================================================================

class EncryptionStrategy(ABC):
    """Abstract encryption strategy."""
    
    @abstractmethod
    def encrypt(self, message: str) -> str:
        """Encrypt message."""
        pass
    
    @abstractmethod
    def decrypt(self, encrypted_message: str) -> str:
        """Decrypt message."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class NoEncryption(EncryptionStrategy):
    """No encryption strategy."""
    
    def encrypt(self, message: str) -> str:
        """Return message as-is."""
        return message
    
    def decrypt(self, encrypted_message: str) -> str:
        """Return message as-is."""
        return encrypted_message
    
    def get_strategy_name(self) -> str:
        return "No Encryption"


class SimpleEncryption(EncryptionStrategy):
    """Simple encryption using Fernet."""
    
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, message: str) -> str:
        """Encrypt message using Fernet."""
        encrypted_bytes = self.cipher.encrypt(message.encode())
        return base64.b64encode(encrypted_bytes).decode()
    
    def decrypt(self, encrypted_message: str) -> str:
        """Decrypt message using Fernet."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_message.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception:
            return "[Decryption Failed]"
    
    def get_strategy_name(self) -> str:
        return "Simple Encryption"


class EndToEndEncryption(EncryptionStrategy):
    """End-to-end encryption simulation."""
    
    def __init__(self, user_keys: Dict[str, str]):
        self.user_keys = user_keys
        self.current_user = None
    
    def set_current_user(self, user_id: str) -> None:
        """Set current user for encryption context."""
        self.current_user = user_id
    
    def encrypt(self, message: str) -> str:
        """Encrypt message with user's key."""
        if not self.current_user or self.current_user not in self.user_keys:
            return message
        
        # Simulate E2E encryption
        user_key = self.user_keys[self.current_user]
        cipher = Fernet(user_key.encode()[:32].ljust(32, b'0'))
        encrypted_bytes = cipher.encrypt(message.encode())
        return base64.b64encode(encrypted_bytes).decode()
    
    def decrypt(self, encrypted_message: str) -> str:
        """Decrypt message with user's key."""
        if not self.current_user or self.current_user not in self.user_keys:
            return encrypted_message
        
        try:
            user_key = self.user_keys[self.current_user]
            cipher = Fernet(user_key.encode()[:32].ljust(32, b'0'))
            encrypted_bytes = base64.b64decode(encrypted_message.encode())
            decrypted_bytes = cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception:
            return "[E2E Decryption Failed]"
    
    def get_strategy_name(self) -> str:
        return "End-to-End Encryption"


# ============================================================================
# MESSAGE CLASSES
# ============================================================================

class Message:
    """Chat message with metadata."""
    
    def __init__(self, sender_id: str, content: MessageContent, 
                 message_type: MessageType = MessageType.TEXT,
                 chat_id: str = "", reply_to: str = ""):
        self.message_id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.content = content
        self.message_type = message_type
        self.chat_id = chat_id
        self.reply_to = reply_to
        self.timestamp = datetime.now()
        self.edited_at: Optional[datetime] = None
        self.status = MessageStatus.SENT
        self.delivery_receipts: List[DeliveryReceipt] = []
        self.reactions: Dict[str, List[str]] = defaultdict(list)  # emoji -> user_ids
        self.is_deleted = False
        self.is_pinned = False
        
        # Encryption
        self.is_encrypted = False
        self.encrypted_content = ""
    
    def encrypt_message(self, encryption_strategy: EncryptionStrategy) -> None:
        """Encrypt message content."""
        if not self.is_encrypted:
            self.encrypted_content = encryption_strategy.encrypt(self.content.text)
            self.is_encrypted = True
    
    def decrypt_message(self, encryption_strategy: EncryptionStrategy) -> str:
        """Decrypt and return message content."""
        if self.is_encrypted and self.encrypted_content:
            return encryption_strategy.decrypt(self.encrypted_content)
        return self.content.text
    
    def add_reaction(self, user_id: str, emoji: str) -> None:
        """Add reaction to message."""
        if user_id not in self.reactions[emoji]:
            self.reactions[emoji].append(user_id)
    
    def remove_reaction(self, user_id: str, emoji: str) -> None:
        """Remove reaction from message."""
        if user_id in self.reactions[emoji]:
            self.reactions[emoji].remove(user_id)
            if not self.reactions[emoji]:
                del self.reactions[emoji]
    
    def edit_message(self, new_content: MessageContent) -> None:
        """Edit message content."""
        self.content = new_content
        self.edited_at = datetime.now()
    
    def delete_message(self) -> None:
        """Mark message as deleted."""
        self.is_deleted = True
        self.content.text = "[Message deleted]"
    
    def pin_message(self) -> None:
        """Pin message."""
        self.is_pinned = True
    
    def unpin_message(self) -> None:
        """Unpin message."""
        self.is_pinned = False
    
    def add_delivery_receipt(self, user_id: str, status: MessageStatus) -> None:
        """Add delivery receipt."""
        # Remove existing receipt for user
        self.delivery_receipts = [r for r in self.delivery_receipts if r.user_id != user_id]
        
        # Add new receipt
        receipt = DeliveryReceipt(
            message_id=self.message_id,
            user_id=user_id,
            status=status,
            timestamp=datetime.now()
        )
        self.delivery_receipts.append(receipt)
    
    def get_message_info(self, encryption_strategy: Optional[EncryptionStrategy] = None) -> Dict[str, Any]:
        """Get message information."""
        content_text = self.content.text
        if encryption_strategy and self.is_encrypted:
            content_text = self.decrypt_message(encryption_strategy)
        
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'content': {
                'text': content_text if not self.is_deleted else "[Message deleted]",
                'file_url': self.content.file_url,
                'file_name': self.content.file_name,
                'file_size': self.content.file_size,
                'metadata': self.content.metadata
            },
            'message_type': self.message_type.value,
            'chat_id': self.chat_id,
            'reply_to': self.reply_to,
            'timestamp': self.timestamp.isoformat(),
            'edited_at': self.edited_at.isoformat() if self.edited_at else None,
            'status': self.status.value,
            'reactions': dict(self.reactions),
            'is_deleted': self.is_deleted,
            'is_pinned': self.is_pinned,
            'is_encrypted': self.is_encrypted,
            'delivery_count': len(self.delivery_receipts),
            'read_count': len([r for r in self.delivery_receipts if r.status == MessageStatus.READ])
        }
    
    def __str__(self) -> str:
        return f"Message {self.message_id[:8]} from {self.sender_id} at {self.timestamp.strftime('%H:%M')}"


# ============================================================================
# USER CLASSES
# ============================================================================

class User:
    """Chat application user."""
    
    def __init__(self, user_id: str, username: str, email: str):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.display_name = username
        self.avatar_url = ""
        self.bio = ""
        
        # Status and presence
        self.status = UserStatus.OFFLINE
        self.last_seen = datetime.now()
        self.is_typing_in: Set[str] = set()  # chat_ids where user is typing
        
        # Preferences
        self.notification_settings = {
            'push_notifications': True,
            'email_notifications': False,
            'sound_notifications': True,
            'desktop_notifications': True
        }
        
        # Privacy and security
        self.blocked_users: Set[str] = set()
        self.privacy_settings = {
            'show_last_seen': True,
            'show_online_status': True,
            'allow_group_invites': True,
            'read_receipts': True
        }
        
        # Chat memberships
        self.chat_memberships: Dict[str, UserRole] = {}
        
        # Connection info
        self.connection_id: Optional[str] = None
        self.device_info = {}
        
        self._lock = threading.Lock()
    
    def set_status(self, status: UserStatus) -> None:
        """Set user status."""
        with self._lock:
            self.status = status
            if status == UserStatus.OFFLINE:
                self.last_seen = datetime.now()
                self.is_typing_in.clear()
    
    def set_typing(self, chat_id: str, is_typing: bool = True) -> None:
        """Set typing status in a chat."""
        with self._lock:
            if is_typing:
                self.is_typing_in.add(chat_id)
            else:
                self.is_typing_in.discard(chat_id)
    
    def block_user(self, user_id: str) -> None:
        """Block a user."""
        self.blocked_users.add(user_id)
    
    def unblock_user(self, user_id: str) -> None:
        """Unblock a user."""
        self.blocked_users.discard(user_id)
    
    def is_blocked(self, user_id: str) -> bool:
        """Check if user is blocked."""
        return user_id in self.blocked_users
    
    def join_chat(self, chat_id: str, role: UserRole = UserRole.MEMBER) -> None:
        """Join a chat."""
        self.chat_memberships[chat_id] = role
    
    def leave_chat(self, chat_id: str) -> None:
        """Leave a chat."""
        self.chat_memberships.pop(chat_id, None)
        self.is_typing_in.discard(chat_id)
    
    def update_profile(self, display_name: str = None, avatar_url: str = None, bio: str = None) -> None:
        """Update user profile."""
        if display_name:
            self.display_name = display_name
        if avatar_url:
            self.avatar_url = avatar_url
        if bio is not None:
            self.bio = bio
    
    def update_notification_settings(self, settings: Dict[str, bool]) -> None:
        """Update notification settings."""
        self.notification_settings.update(settings)
    
    def update_privacy_settings(self, settings: Dict[str, bool]) -> None:
        """Update privacy settings."""
        self.privacy_settings.update(settings)
    
    def get_user_info(self, requesting_user_id: str = None) -> Dict[str, Any]:
        """Get user information (respecting privacy settings)."""
        info = {
            'user_id': self.user_id,
            'username': self.username,
            'display_name': self.display_name,
            'avatar_url': self.avatar_url,
            'bio': self.bio
        }
        
        # Add status info based on privacy settings
        if self.privacy_settings['show_online_status']:
            info['status'] = self.status.value
        
        if self.privacy_settings['show_last_seen'] and self.status == UserStatus.OFFLINE:
            info['last_seen'] = self.last_seen.isoformat()
        
        # Don't show sensitive info to blocked users
        if requesting_user_id and self.is_blocked(requesting_user_id):
            return {
                'user_id': self.user_id,
                'username': self.username,
                'display_name': self.display_name,
                'status': UserStatus.OFFLINE.value
            }
        
        return info
    
    def __str__(self) -> str:
        return f"User {self.username} ({self.status.value})"


# ============================================================================
# CHAT CLASSES
# ============================================================================

class Chat:
    """Chat conversation (private or group)."""
    
    def __init__(self, chat_id: str, chat_type: ChatType, name: str = "", creator_id: str = ""):
        self.chat_id = chat_id
        self.chat_type = chat_type
        self.name = name
        self.description = ""
        self.avatar_url = ""
        self.creator_id = creator_id
        self.created_at = datetime.now()
        
        # Members and roles
        self.members: Dict[str, UserRole] = {}
        self.banned_users: Set[str] = set()
        
        # Messages
        self.messages: List[Message] = []
        self.pinned_messages: List[str] = []  # message_ids
        
        # Settings
        self.settings = {
            'max_members': 1000 if chat_type == ChatType.GROUP else 2,
            'allow_member_invites': True,
            'message_history_visible': True,
            'read_receipts_enabled': True,
            'typing_indicators_enabled': True
        }
        
        # Statistics
        self.message_count = 0
        self.last_activity = datetime.now()
        
        # Observers for real-time updates
        self.observers: List['ChatObserver'] = []
        
        self._lock = threading.Lock()
    
    def add_observer(self, observer: 'ChatObserver') -> None:
        """Add chat observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer: 'ChatObserver') -> None:
        """Remove chat observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify observers of chat events."""
        for observer in self.observers:
            observer.on_chat_event(self.chat_id, event_type, data)
    
    def add_member(self, user_id: str, role: UserRole = UserRole.MEMBER, 
                   added_by: str = "") -> bool:
        """Add member to chat."""
        with self._lock:
            if user_id in self.banned_users:
                return False
            
            if len(self.members) >= self.settings['max_members']:
                return False
            
            self.members[user_id] = role
            
            # Create system message
            system_message = Message(
                sender_id="system",
                content=MessageContent(text=f"User {user_id} joined the chat"),
                message_type=MessageType.SYSTEM,
                chat_id=self.chat_id
            )
            self.messages.append(system_message)
            self.message_count += 1
            self.last_activity = datetime.now()
            
            # Notify observers
            self.notify_observers("member_added", {
                'user_id': user_id,
                'role': role.value,
                'added_by': added_by
            })
            
            return True
    
    def remove_member(self, user_id: str, removed_by: str = "") -> bool:
        """Remove member from chat."""
        with self._lock:
            if user_id not in self.members:
                return False
            
            # Cannot remove owner unless transferring ownership
            if self.members[user_id] == UserRole.OWNER and user_id != removed_by:
                return False
            
            del self.members[user_id]
            
            # Create system message
            system_message = Message(
                sender_id="system",
                content=MessageContent(text=f"User {user_id} left the chat"),
                message_type=MessageType.SYSTEM,
                chat_id=self.chat_id
            )
            self.messages.append(system_message)
            self.message_count += 1
            self.last_activity = datetime.now()
            
            # Notify observers
            self.notify_observers("member_removed", {
                'user_id': user_id,
                'removed_by': removed_by
            })
            
            return True
    
    def ban_user(self, user_id: str, banned_by: str) -> bool:
        """Ban user from chat."""
        with self._lock:
            # Check permissions
            if (banned_by not in self.members or 
                self.members[banned_by] not in [UserRole.ADMIN, UserRole.OWNER]):
                return False
            
            self.banned_users.add(user_id)
            self.remove_member(user_id, banned_by)
            
            return True
    
    def unban_user(self, user_id: str, unbanned_by: str) -> bool:
        """Unban user from chat."""
        with self._lock:
            # Check permissions
            if (unbanned_by not in self.members or 
                self.members[unbanned_by] not in [UserRole.ADMIN, UserRole.OWNER]):
                return False
            
            self.banned_users.discard(user_id)
            return True
    
    def change_member_role(self, user_id: str, new_role: UserRole, changed_by: str) -> bool:
        """Change member role."""
        with self._lock:
            if user_id not in self.members or changed_by not in self.members:
                return False
            
            # Check permissions
            changer_role = self.members[changed_by]
            current_role = self.members[user_id]
            
            # Only owners can change admin roles, admins can change member roles
            if new_role == UserRole.OWNER or current_role == UserRole.OWNER:
                if changer_role != UserRole.OWNER:
                    return False
            elif new_role == UserRole.ADMIN or current_role == UserRole.ADMIN:
                if changer_role not in [UserRole.OWNER, UserRole.ADMIN]:
                    return False
            
            self.members[user_id] = new_role
            
            # Notify observers
            self.notify_observers("role_changed", {
                'user_id': user_id,
                'new_role': new_role.value,
                'changed_by': changed_by
            })
            
            return True
    
    def add_message(self, message: Message) -> bool:
        """Add message to chat."""
        with self._lock:
            # Check if sender is member
            if message.sender_id != "system" and message.sender_id not in self.members:
                return False
            
            message.chat_id = self.chat_id
            self.messages.append(message)
            self.message_count += 1
            self.last_activity = datetime.now()
            
            # Notify observers
            self.notify_observers("message_added", {
                'message': message.get_message_info()
            })
            
            return True
    
    def edit_message(self, message_id: str, new_content: MessageContent, editor_id: str) -> bool:
        """Edit a message."""
        with self._lock:
            message = self.get_message_by_id(message_id)
            if not message:
                return False
            
            # Check permissions (sender or admin)
            if (message.sender_id != editor_id and 
                self.members.get(editor_id) not in [UserRole.ADMIN, UserRole.OWNER]):
                return False
            
            message.edit_message(new_content)
            
            # Notify observers
            self.notify_observers("message_edited", {
                'message_id': message_id,
                'editor_id': editor_id
            })
            
            return True
    
    def delete_message(self, message_id: str, deleter_id: str) -> bool:
        """Delete a message."""
        with self._lock:
            message = self.get_message_by_id(message_id)
            if not message:
                return False
            
            # Check permissions
            if (message.sender_id != deleter_id and 
                self.members.get(deleter_id) not in [UserRole.ADMIN, UserRole.OWNER]):
                return False
            
            message.delete_message()
            
            # Notify observers
            self.notify_observers("message_deleted", {
                'message_id': message_id,
                'deleter_id': deleter_id
            })
            
            return True
    
    def pin_message(self, message_id: str, pinner_id: str) -> bool:
        """Pin a message."""
        with self._lock:
            message = self.get_message_by_id(message_id)
            if not message:
                return False
            
            # Check permissions
            if self.members.get(pinner_id) not in [UserRole.ADMIN, UserRole.OWNER]:
                return False
            
            if message_id not in self.pinned_messages:
                self.pinned_messages.append(message_id)
                message.pin_message()
            
            return True
    
    def unpin_message(self, message_id: str, unpinner_id: str) -> bool:
        """Unpin a message."""
        with self._lock:
            # Check permissions
            if self.members.get(unpinner_id) not in [UserRole.ADMIN, UserRole.OWNER]:
                return False
            
            if message_id in self.pinned_messages:
                self.pinned_messages.remove(message_id)
                message = self.get_message_by_id(message_id)
                if message:
                    message.unpin_message()
            
            return True
    
    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        for message in self.messages:
            if message.message_id == message_id:
                return message
        return None
    
    def get_recent_messages(self, count: int = 50, before_message_id: str = None) -> List[Message]:
        """Get recent messages."""
        if before_message_id:
            # Find index of before_message_id
            before_index = len(self.messages)
            for i, message in enumerate(self.messages):
                if message.message_id == before_message_id:
                    before_index = i
                    break
            
            return self.messages[max(0, before_index - count):before_index]
        else:
            return self.messages[-count:] if self.messages else []
    
    def search_messages(self, query: str, limit: int = 20) -> List[Message]:
        """Search messages by content."""
        results = []
        query_lower = query.lower()
        
        for message in reversed(self.messages):  # Search from newest
            if len(results) >= limit:
                break
            
            if (not message.is_deleted and 
                query_lower in message.content.text.lower()):
                results.append(message)
        
        return results
    
    def get_chat_info(self) -> Dict[str, Any]:
        """Get chat information."""
        return {
            'chat_id': self.chat_id,
            'chat_type': self.chat_type.value,
            'name': self.name,
            'description': self.description,
            'avatar_url': self.avatar_url,
            'creator_id': self.creator_id,
            'created_at': self.created_at.isoformat(),
            'member_count': len(self.members),
            'members': {user_id: role.value for user_id, role in self.members.items()},
            'message_count': self.message_count,
            'last_activity': self.last_activity.isoformat(),
            'pinned_message_count': len(self.pinned_messages),
            'settings': self.settings
        }
    
    def __str__(self) -> str:
        return f"Chat {self.name or self.chat_id} ({self.chat_type.value}) - {len(self.members)} members"


# ============================================================================
# OBSERVER PATTERN FOR REAL-TIME UPDATES
# ============================================================================

class ChatObserver(ABC):
    """Abstract chat observer."""
    
    @abstractmethod
    def on_chat_event(self, chat_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Handle chat event."""
        pass


class NotificationService(ChatObserver):
    """Notification service for chat events."""
    
    def __init__(self):
        self.notification_queue: deque = deque()
        self.user_connections: Dict[str, List[str]] = defaultdict(list)  # user_id -> connection_ids
    
    def add_user_connection(self, user_id: str, connection_id: str) -> None:
        """Add user connection for notifications."""
        if connection_id not in self.user_connections[user_id]:
            self.user_connections[user_id].append(connection_id)
    
    def remove_user_connection(self, user_id: str, connection_id: str) -> None:
        """Remove user connection."""
        if connection_id in self.user_connections[user_id]:
            self.user_connections[user_id].remove(connection_id)
    
    def on_chat_event(self, chat_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Handle chat event and send notifications."""
        notification = {
            'chat_id': chat_id,
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.notification_queue.append(notification)
        
        # Send real-time notifications
        self._send_real_time_notification(notification)
    
    def _send_real_time_notification(self, notification: Dict[str, Any]) -> None:
        """Send real-time notification to connected users."""
        # In a real implementation, this would use WebSockets
        # For demonstration, we'll just print
        print(f"📱 Real-time notification: {notification['event_type']} in chat {notification['chat_id']}")
    
    def get_notifications_for_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent notifications for user."""
        # In a real implementation, this would filter by user's chat memberships
        return list(self.notification_queue)[-limit:]


# ============================================================================
# CHAT APPLICATION MANAGER
# ============================================================================

class ChatApplication:
    """Main chat application manager."""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.users: Dict[str, User] = {}
        self.chats: Dict[str, Chat] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}  # user_id -> session_info
        
        # Services
        self.notification_service = NotificationService()
        self.encryption_strategy: EncryptionStrategy = SimpleEncryption()
        
        # Statistics
        self.total_messages = 0
        self.total_users = 0
        self.active_users = 0
        
        # Threading
        self._lock = threading.Lock()
        
        print(f"💬 Chat Application '{app_name}' initialized")
    
    def set_encryption_strategy(self, strategy: EncryptionStrategy) -> None:
        """Set encryption strategy."""
        self.encryption_strategy = strategy
        print(f"🔒 Encryption strategy: {strategy.get_strategy_name()}")
    
    def register_user(self, username: str, email: str, password: str) -> Optional[User]:
        """Register a new user."""
        with self._lock:
            # Check if username/email already exists
            for user in self.users.values():
                if user.username == username or user.email == email:
                    return None
            
            user_id = str(uuid.uuid4())
            user = User(user_id, username, email)
            self.users[user_id] = user
            self.total_users += 1
            
            print(f"👤 User registered: {username}")
            return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        # Simplified authentication
        for user in self.users.values():
            if user.username == username:
                session_token = str(uuid.uuid4())
                self.user_sessions[session_token] = {
                    'user_id': user.user_id,
                    'login_time': datetime.now(),
                    'last_activity': datetime.now()
                }
                
                user.set_status(UserStatus.ONLINE)
                self.active_users += 1
                
                print(f"🔑 User authenticated: {username}")
                return session_token
        
        return None
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user."""
        session = self.user_sessions.get(session_token)
        if not session:
            return False
        
        user_id = session['user_id']
        user = self.users.get(user_id)
        
        if user:
            user.set_status(UserStatus.OFFLINE)
            self.active_users = max(0, self.active_users - 1)
        
        del self.user_sessions[session_token]
        return True
    
    def get_user_from_session(self, session_token: str) -> Optional[User]:
        """Get user from session token."""
        session = self.user_sessions.get(session_token)
        if not session:
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        
        return self.users.get(session['user_id'])
    
    def create_private_chat(self, user1_id: str, user2_id: str) -> Optional[Chat]:
        """Create private chat between two users."""
        # Check if chat already exists
        for chat in self.chats.values():
            if (chat.chat_type == ChatType.PRIVATE and 
                len(chat.members) == 2 and
                user1_id in chat.members and user2_id in chat.members):
                return chat
        
        # Create new private chat
        chat_id = f"private_{user1_id}_{user2_id}"
        chat = Chat(chat_id, ChatType.PRIVATE, creator_id=user1_id)
        
        # Add members
        chat.add_member(user1_id, UserRole.MEMBER)
        chat.add_member(user2_id, UserRole.MEMBER)
        
        # Add observer
        chat.add_observer(self.notification_service)
        
        self.chats[chat_id] = chat
        
        # Update user memberships
        self.users[user1_id].join_chat(chat_id)
        self.users[user2_id].join_chat(chat_id)
        
        print(f"💬 Private chat created: {chat_id}")
        return chat
    
    def create_group_chat(self, creator_id: str, name: str, description: str = "") -> Optional[Chat]:
        """Create group chat."""
        chat_id = str(uuid.uuid4())
        chat = Chat(chat_id, ChatType.GROUP, name, creator_id)
        chat.description = description
        
        # Add creator as owner
        chat.add_member(creator_id, UserRole.OWNER)
        
        # Add observer
        chat.add_observer(self.notification_service)
        
        self.chats[chat_id] = chat
        
        # Update user membership
        self.users[creator_id].join_chat(chat_id, UserRole.OWNER)
        
        print(f"👥 Group chat created: {name}")
        return chat
    
    def join_group_chat(self, chat_id: str, user_id: str, invited_by: str = "") -> bool:
        """Join group chat."""
        chat = self.chats.get(chat_id)
        user = self.users.get(user_id)
        
        if not chat or not user or chat.chat_type != ChatType.GROUP:
            return False
        
        # Check if user can join
        if not chat.settings.get('allow_member_invites', True) and not invited_by:
            return False
        
        if chat.add_member(user_id, UserRole.MEMBER, invited_by):
            user.join_chat(chat_id)
            return True
        
        return False
    
    def leave_chat(self, chat_id: str, user_id: str) -> bool:
        """Leave chat."""
        chat = self.chats.get(chat_id)
        user = self.users.get(user_id)
        
        if not chat or not user:
            return False
        
        if chat.remove_member(user_id):
            user.leave_chat(chat_id)
            
            # Delete private chat if empty
            if chat.chat_type == ChatType.PRIVATE and len(chat.members) == 0:
                del self.chats[chat_id]
            
            return True
        
        return False
    
    def send_message(self, sender_id: str, chat_id: str, content: MessageContent,
                    message_type: MessageType = MessageType.TEXT, reply_to: str = "") -> Optional[Message]:
        """Send message to chat."""
        chat = self.chats.get(chat_id)
        sender = self.users.get(sender_id)
        
        if not chat or not sender or sender_id not in chat.members:
            return None
        
        # Check if sender is blocked by any recipient
        for member_id in chat.members:
            member = self.users.get(member_id)
            if member and member.is_blocked(sender_id):
                continue  # Skip blocked users
        
        # Create message
        message = Message(sender_id, content, message_type, chat_id, reply_to)
        
        # Encrypt message if needed
        if isinstance(self.encryption_strategy, EndToEndEncryption):
            self.encryption_strategy.set_current_user(sender_id)
        
        message.encrypt_message(self.encryption_strategy)
        
        # Add to chat
        if chat.add_message(message):
            self.total_messages += 1
            
            # Add delivery receipts for online members
            for member_id in chat.members:
                if member_id != sender_id:
                    member = self.users.get(member_id)
                    if member and member.status == UserStatus.ONLINE:
                        message.add_delivery_receipt(member_id, MessageStatus.DELIVERED)
            
            return message
        
        return None
    
    def mark_message_as_read(self, user_id: str, message_id: str) -> bool:
        """Mark message as read by user."""
        # Find message in all chats
        for chat in self.chats.values():
            if user_id in chat.members:
                message = chat.get_message_by_id(message_id)
                if message:
                    message.add_delivery_receipt(user_id, MessageStatus.READ)
                    return True
        
        return False
    
    def add_reaction(self, user_id: str, message_id: str, emoji: str) -> bool:
        """Add reaction to message."""
        # Find message and add reaction
        for chat in self.chats.values():
            if user_id in chat.members:
                message = chat.get_message_by_id(message_id)
                if message:
                    message.add_reaction(user_id, emoji)
                    
                    # Notify observers
                    chat.notify_observers("reaction_added", {
                        'message_id': message_id,
                        'user_id': user_id,
                        'emoji': emoji
                    })
                    return True
        
        return False
    
    def set_typing_status(self, user_id: str, chat_id: str, is_typing: bool) -> bool:
        """Set typing status for user in chat."""
        user = self.users.get(user_id)
        chat = self.chats.get(chat_id)
        
        if not user or not chat or user_id not in chat.members:
            return False
        
        user.set_typing(chat_id, is_typing)
        
        # Notify other members
        chat.notify_observers("typing_status", {
            'user_id': user_id,
            'is_typing': is_typing
        })
        
        return True
    
    def search_messages(self, user_id: str, query: str, chat_id: str = None) -> List[Dict[str, Any]]:
        """Search messages across chats."""
        results = []
        
        chats_to_search = [self.chats[chat_id]] if chat_id else self.chats.values()
        
        for chat in chats_to_search:
            if user_id in chat.members:
                messages = chat.search_messages(query, limit=10)
                for message in messages:
                    results.append({
                        'chat_id': chat.chat_id,
                        'chat_name': chat.name,
                        'message': message.get_message_info(self.encryption_strategy)
                    })
        
        return results
    
    def get_user_chats(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for user."""
        user = self.users.get(user_id)
        if not user:
            return []
        
        user_chats = []
        for chat_id in user.chat_memberships:
            chat = self.chats.get(chat_id)
            if chat:
                chat_info = chat.get_chat_info()
                
                # Add unread message count
                unread_count = 0
                for message in chat.messages:
                    if message.sender_id != user_id:
                        read_receipts = [r for r in message.delivery_receipts 
                                       if r.user_id == user_id and r.status == MessageStatus.READ]
                        if not read_receipts:
                            unread_count += 1
                
                chat_info['unread_count'] = unread_count
                chat_info['user_role'] = user.chat_memberships[chat_id].value
                
                user_chats.append(chat_info)
        
        # Sort by last activity
        user_chats.sort(key=lambda x: x['last_activity'], reverse=True)
        return user_chats
    
    def get_chat_messages(self, user_id: str, chat_id: str, count: int = 50, 
                         before_message_id: str = None) -> List[Dict[str, Any]]:
        """Get messages from chat."""
        chat = self.chats.get(chat_id)
        if not chat or user_id not in chat.members:
            return []
        
        messages = chat.get_recent_messages(count, before_message_id)
        
        # Decrypt messages for user
        if isinstance(self.encryption_strategy, EndToEndEncryption):
            self.encryption_strategy.set_current_user(user_id)
        
        return [message.get_message_info(self.encryption_strategy) for message in messages]
    
    def get_application_stats(self) -> Dict[str, Any]:
        """Get application statistics."""
        return {
            'app_name': self.app_name,
            'total_users': self.total_users,
            'active_users': self.active_users,
            'total_chats': len(self.chats),
            'total_messages': self.total_messages,
            'encryption_strategy': self.encryption_strategy.get_strategy_name(),
            'chat_types': {
                'private': len([c for c in self.chats.values() if c.chat_type == ChatType.PRIVATE]),
                'group': len([c for c in self.chats.values() if c.chat_type == ChatType.GROUP])
            }
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_chat_application():
    """Demonstrate the chat application system."""
    print("=== CHAT APPLICATION DESIGN DEMONSTRATION ===\n")
    
    # Initialize chat application
    app = ChatApplication("ChatApp Pro")
    
    print("1. USER REGISTRATION AND AUTHENTICATION:")
    
    # Register users
    users_data = [
        ("alice", "alice@example.com", "password123"),
        ("bob", "bob@example.com", "password456"),
        ("charlie", "charlie@example.com", "password789"),
        ("diana", "diana@example.com", "password000"),
        ("eve", "eve@example.com", "password111")
    ]
    
    user_sessions = {}
    
    for username, email, password in users_data:
        user = app.register_user(username, email, password)
        if user:
            # Authenticate user
            session_token = app.authenticate_user(username, password)
            if session_token:
                user_sessions[username] = session_token
                print(f"   ✓ {username} registered and authenticated")
    
    print()
    
    # Test encryption strategies
    print("2. ENCRYPTION STRATEGY TESTING:")
    
    strategies = [
        NoEncryption(),
        SimpleEncryption(),
        EndToEndEncryption({user_id: f"key_{user_id}" for user_id in app.users.keys()})
    ]
    
    for strategy in strategies:
        app.set_encryption_strategy(strategy)
        print(f"   ✓ Testing {strategy.get_strategy_name()}")
    
    # Use simple encryption for demo
    app.set_encryption_strategy(SimpleEncryption())
    
    print()
    
    # Create chats
    print("3. CHAT CREATION:")
    
    # Get user IDs
    alice = app.get_user_from_session(user_sessions["alice"])
    bob = app.get_user_from_session(user_sessions["bob"])
    charlie = app.get_user_from_session(user_sessions["charlie"])
    diana = app.get_user_from_session(user_sessions["diana"])
    eve = app.get_user_from_session(user_sessions["eve"])
    
    # Create private chat
    private_chat = app.create_private_chat(alice.user_id, bob.user_id)
    print(f"   ✓ Private chat created between Alice and Bob")
    
    # Create group chat
    group_chat = app.create_group_chat(alice.user_id, "Team Chat", "Work discussion group")
    print(f"   ✓ Group chat 'Team Chat' created by Alice")
    
    # Add members to group
    app.join_group_chat(group_chat.chat_id, charlie.user_id, alice.user_id)
    app.join_group_chat(group_chat.chat_id, diana.user_id, alice.user_id)
    print(f"   ✓ Charlie and Diana joined the group")
    
    print()
    
    # Send messages
    print("4. MESSAGE EXCHANGE:")
    
    # Private messages
    message1 = app.send_message(
        alice.user_id, 
        private_chat.chat_id,
        MessageContent(text="Hey Bob! How are you?")
    )
    
    message2 = app.send_message(
        bob.user_id,
        private_chat.chat_id,
        MessageContent(text="Hi Alice! I'm doing great, thanks for asking!")
    )
    
    print(f"   ✓ Private messages exchanged between Alice and Bob")
    
    # Group messages
    message3 = app.send_message(
        alice.user_id,
        group_chat.chat_id,
        MessageContent(text="Welcome everyone to our team chat!")
    )
    
    message4 = app.send_message(
        charlie.user_id,
        group_chat.chat_id,
        MessageContent(text="Thanks Alice! Excited to be here.")
    )
    
    message5 = app.send_message(
        diana.user_id,
        group_chat.chat_id,
        MessageContent(text="Hello team! 👋")
    )
    
    print(f"   ✓ Group messages sent in Team Chat")
    
    # File sharing simulation
    file_message = app.send_message(
        bob.user_id,
        group_chat.chat_id,
        MessageContent(
            text="Here's the project document",
            file_url="https://example.com/document.pdf",
            file_name="project_spec.pdf",
            file_size=1024000
        ),
        MessageType.FILE
    )
    
    print(f"   ✓ File shared by Bob")
    
    print()
    
    # Test message features
    print("5. MESSAGE FEATURES TESTING:")
    
    # Mark messages as read
    if message1:
        app.mark_message_as_read(bob.user_id, message1.message_id)
        print(f"   ✓ Bob marked Alice's message as read")
    
    # Add reactions
    if message3:
        app.add_reaction(charlie.user_id, message3.message_id, "👍")
        app.add_reaction(diana.user_id, message3.message_id, "❤️")
        print(f"   ✓ Reactions added to Alice's welcome message")
    
    # Edit message
    if message4:
        group_chat.edit_message(
            message4.message_id,
            MessageContent(text="Thanks Alice! Really excited to be part of this team."),
            charlie.user_id
        )
        print(f"   ✓ Charlie edited his message")
    
    # Pin message
    if message3:
        group_chat.pin_message(message3.message_id, alice.user_id)
        print(f"   ✓ Alice pinned the welcome message")
    
    print()
    
    # Test typing indicators
    print("6. TYPING INDICATORS:")
    
    app.set_typing_status(eve.user_id, group_chat.chat_id, True)
    print(f"   ✓ Eve is typing in Team Chat")
    
    time.sleep(1)
    
    app.set_typing_status(eve.user_id, group_chat.chat_id, False)
    print(f"   ✓ Eve stopped typing")
    
    print()
    
    # Test search functionality
    print("7. MESSAGE SEARCH:")
    
    search_results = app.search_messages(alice.user_id, "excited")
    print(f"   ✓ Search for 'excited' returned {len(search_results)} results")
    
    for result in search_results:
        print(f"     - Found in {result['chat_name']}: {result['message']['content']['text'][:50]}...")
    
    print()
    
    # Test user management
    print("8. USER MANAGEMENT:")
    
    # Block user
    diana.block_user(eve.user_id)
    print(f"   ✓ Diana blocked Eve")
    
    # Try to add Eve to group (should work, but Diana won't see messages)
    app.join_group_chat(group_chat.chat_id, eve.user_id, alice.user_id)
    print(f"   ✓ Eve joined the group")
    
    # Change user role
    group_chat.change_member_role(charlie.user_id, UserRole.ADMIN, alice.user_id)
    print(f"   ✓ Alice promoted Charlie to admin")
    
    print()
    
    # Show chat information
    print("9. CHAT INFORMATION:")
    
    # Private chat info
    private_info = private_chat.get_chat_info()
    print(f"   Private Chat:")
    print(f"     Members: {private_info['member_count']}")
    print(f"     Messages: {private_info['message_count']}")
    print(f"     Last Activity: {private_info['last_activity']}")
    
    # Group chat info
    group_info = group_chat.get_chat_info()
    print(f"   Group Chat '{group_info['name']}':")
    print(f"     Members: {group_info['member_count']}")
    print(f"     Messages: {group_info['message_count']}")
    print(f"     Pinned Messages: {group_info['pinned_message_count']}")
    
    # Show member roles
    print(f"     Member Roles:")
    for user_id, role in group_info['members'].items():
        username = next((u.username for u in app.users.values() if u.user_id == user_id), "Unknown")
        print(f"       {username}: {role}")
    
    print()
    
    # Show user chats
    print("10. USER CHAT LISTS:")
    
    alice_chats = app.get_user_chats(alice.user_id)
    print(f"   Alice's Chats ({len(alice_chats)}):")
    for chat_info in alice_chats:
        chat_name = chat_info['name'] or f"Private with {chat_info['chat_id']}"
        print(f"     - {chat_name} ({chat_info['chat_type']}) - {chat_info['unread_count']} unread")
    
    print()
    
    # Show recent messages
    print("11. RECENT MESSAGES:")
    
    group_messages = app.get_chat_messages(alice.user_id, group_chat.chat_id, count=10)
    print(f"   Recent messages in Team Chat:")
    
    for msg in group_messages[-5:]:  # Show last 5
        sender_username = next((u.username for u in app.users.values() if u.user_id == msg['sender_id']), "System")
        timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M')
        content = msg['content']['text'][:50] + ("..." if len(msg['content']['text']) > 50 else "")
        
        reactions_str = ""
        if msg['reactions']:
            reactions_str = " " + " ".join(f"{emoji}({len(users)})" for emoji, users in msg['reactions'].items())
        
        edited_str = " (edited)" if msg['edited_at'] else ""
        pinned_str = " 📌" if msg['is_pinned'] else ""
        
        print(f"     [{timestamp}] {sender_username}: {content}{reactions_str}{edited_str}{pinned_str}")
    
    print()
    
    # Show application statistics
    print("12. APPLICATION STATISTICS:")
    
    stats = app.get_application_stats()
    print(f"   Application: {stats['app_name']}")
    print(f"   Total Users: {stats['total_users']}")
    print(f"   Active Users: {stats['active_users']}")
    print(f"   Total Chats: {stats['total_chats']}")
    print(f"     - Private: {stats['chat_types']['private']}")
    print(f"     - Group: {stats['chat_types']['group']}")
    print(f"   Total Messages: {stats['total_messages']}")
    print(f"   Encryption: {stats['encryption_strategy']}")
    
    print()
    
    # Test notifications
    print("13. NOTIFICATION SYSTEM:")
    
    notifications = app.notification_service.get_notifications_for_user(alice.user_id)
    print(f"   Recent notifications for Alice ({len(notifications)}):")
    
    for notification in notifications[-5:]:  # Show last 5
        timestamp = datetime.fromisoformat(notification['timestamp']).strftime('%H:%M:%S')
        print(f"     [{timestamp}] {notification['event_type']} in chat {notification['chat_id'][:8]}")
    
    print()
    
    # Cleanup - logout users
    print("14. USER LOGOUT:")
    
    for username, session_token in user_sessions.items():
        app.logout_user(session_token)
        print(f"   ✓ {username} logged out")
    
    final_stats = app.get_application_stats()
    print(f"   Active users after logout: {final_stats['active_users']}")
    
    print()
    print("=== CHAT APPLICATION DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_chat_application()
