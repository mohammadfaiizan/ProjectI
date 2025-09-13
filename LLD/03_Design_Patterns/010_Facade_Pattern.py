"""
FACADE PATTERN - Structural Design Pattern
==========================================

Problem Statement:
Implement the Facade pattern to provide a simplified interface to a complex
subsystem:
- Hide complexity of subsystem interactions
- Provide unified interface for multiple subsystems
- Reduce dependencies between clients and subsystems
- Create higher-level interfaces for ease of use
- Implement system integration layers

Learning Objectives:
- Understand when to use Facade pattern
- Design simplified interfaces for complex systems
- Reduce coupling between clients and subsystems
- Create integration layers and API gateways
- Handle complex system initialization and coordination
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import time
import json
from datetime import datetime
from enum import Enum


# ============================================================================
# COMPLEX SUBSYSTEM CLASSES
# ============================================================================

class DatabaseConnection:
    """Complex database subsystem."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.is_connected = False
        self.transaction_active = False
        self.connection_pool_size = 10
        self.active_connections = 0
    
    def connect(self) -> bool:
        """Establish database connection."""
        if self.active_connections >= self.connection_pool_size:
            print("DatabaseConnection: Connection pool exhausted")
            return False
        
        print(f"DatabaseConnection: Connecting to {self.connection_string}")
        self.is_connected = True
        self.active_connections += 1
        time.sleep(0.1)  # Simulate connection time
        print("DatabaseConnection: Connected successfully")
        return True
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.is_connected:
            print("DatabaseConnection: Disconnecting from database")
            self.is_connected = False
            self.active_connections = max(0, self.active_connections - 1)
    
    def begin_transaction(self) -> bool:
        """Begin database transaction."""
        if not self.is_connected:
            print("DatabaseConnection: Cannot start transaction - not connected")
            return False
        
        print("DatabaseConnection: Beginning transaction")
        self.transaction_active = True
        return True
    
    def commit_transaction(self) -> bool:
        """Commit database transaction."""
        if not self.transaction_active:
            print("DatabaseConnection: No active transaction to commit")
            return False
        
        print("DatabaseConnection: Committing transaction")
        self.transaction_active = False
        return True
    
    def rollback_transaction(self) -> bool:
        """Rollback database transaction."""
        if not self.transaction_active:
            print("DatabaseConnection: No active transaction to rollback")
            return False
        
        print("DatabaseConnection: Rolling back transaction")
        self.transaction_active = False
        return True
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute database query."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        print(f"DatabaseConnection: Executing query: {query[:50]}...")
        time.sleep(0.05)  # Simulate query execution
        
        # Simulate query results
        if "SELECT" in query.upper():
            return [{"id": 1, "name": "Sample Data"}]
        return []
    
    def execute_update(self, query: str) -> int:
        """Execute database update."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        print(f"DatabaseConnection: Executing update: {query[:50]}...")
        time.sleep(0.03)  # Simulate update execution
        return 1  # Simulate affected rows


class CacheManager:
    """Complex caching subsystem."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize cache system."""
        print("CacheManager: Initializing cache system")
        self.is_initialized = True
        print(f"CacheManager: Cache initialized with size {self.cache_size}")
        return True
    
    def shutdown(self) -> None:
        """Shutdown cache system."""
        if self.is_initialized:
            print("CacheManager: Shutting down cache system")
            self.cache.clear()
            self.is_initialized = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.is_initialized:
            print("CacheManager: Cache not initialized")
            return None
        
        if key in self.cache:
            self.cache_stats['hits'] += 1
            print(f"CacheManager: Cache hit for key '{key}'")
            return self.cache[key]
        else:
            self.cache_stats['misses'] += 1
            print(f"CacheManager: Cache miss for key '{key}'")
            return None
    
    def put(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Put value in cache."""
        if not self.is_initialized:
            print("CacheManager: Cache not initialized")
            return False
        
        if len(self.cache) >= self.cache_size:
            # Simple eviction - remove first item
            evicted_key = next(iter(self.cache))
            del self.cache[evicted_key]
            self.cache_stats['evictions'] += 1
            print(f"CacheManager: Evicted key '{evicted_key}'")
        
        self.cache[key] = {
            'value': value,
            'ttl': ttl,
            'created_at': time.time()
        }
        print(f"CacheManager: Cached key '{key}'")
        return True
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        if key in self.cache:
            del self.cache[key]
            print(f"CacheManager: Invalidated key '{key}'")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_ratio = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache_size,
            'hit_ratio': round(hit_ratio, 2),
            'stats': self.cache_stats.copy()
        }


class LoggingSystem:
    """Complex logging subsystem."""
    
    def __init__(self):
        self.log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.current_level = 'INFO'
        self.log_handlers = []
        self.log_formatters = {}
        self.is_configured = False
    
    def configure(self, level: str = 'INFO', handlers: List[str] = None) -> bool:
        """Configure logging system."""
        print("LoggingSystem: Configuring logging system")
        
        if level not in self.log_levels:
            print(f"LoggingSystem: Invalid log level '{level}'")
            return False
        
        self.current_level = level
        self.log_handlers = handlers or ['console', 'file']
        
        # Configure formatters
        self.log_formatters = {
            'console': '[{timestamp}] {level}: {message}',
            'file': '{timestamp} | {level} | {message}',
            'json': '{"timestamp": "{timestamp}", "level": "{level}", "message": "{message}"}'
        }
        
        self.is_configured = True
        print(f"LoggingSystem: Configured with level '{level}' and handlers {self.log_handlers}")
        return True
    
    def log(self, level: str, message: str, context: Dict[str, Any] = None) -> None:
        """Log message."""
        if not self.is_configured:
            print("LoggingSystem: Logging system not configured")
            return
        
        if self.log_levels.index(level) < self.log_levels.index(self.current_level):
            return  # Message level too low
        
        timestamp = datetime.now().isoformat()
        
        for handler in self.log_handlers:
            formatter = self.log_formatters.get(handler, self.log_formatters['console'])
            formatted_message = formatter.format(
                timestamp=timestamp,
                level=level,
                message=message
            )
            
            if handler == 'console':
                print(f"LoggingSystem[{handler}]: {formatted_message}")
            elif handler == 'file':
                print(f"LoggingSystem[{handler}]: Writing to log file: {formatted_message}")
            elif handler == 'json':
                print(f"LoggingSystem[{handler}]: {formatted_message}")
    
    def debug(self, message: str, context: Dict[str, Any] = None) -> None:
        self.log('DEBUG', message, context)
    
    def info(self, message: str, context: Dict[str, Any] = None) -> None:
        self.log('INFO', message, context)
    
    def warning(self, message: str, context: Dict[str, Any] = None) -> None:
        self.log('WARNING', message, context)
    
    def error(self, message: str, context: Dict[str, Any] = None) -> None:
        self.log('ERROR', message, context)
    
    def critical(self, message: str, context: Dict[str, Any] = None) -> None:
        self.log('CRITICAL', message, context)


class SecurityManager:
    """Complex security subsystem."""
    
    def __init__(self):
        self.is_initialized = False
        self.encryption_keys = {}
        self.access_tokens = {}
        self.security_policies = {}
        self.audit_log = []
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize security system."""
        print("SecurityManager: Initializing security system")
        
        # Load encryption keys
        self.encryption_keys = {
            'default': 'default_encryption_key_12345',
            'database': 'db_encryption_key_67890',
            'api': 'api_encryption_key_abcdef'
        }
        
        # Set security policies
        self.security_policies = {
            'password_min_length': config.get('password_min_length', 8),
            'session_timeout': config.get('session_timeout', 3600),
            'max_login_attempts': config.get('max_login_attempts', 3),
            'require_2fa': config.get('require_2fa', False)
        }
        
        self.is_initialized = True
        print("SecurityManager: Security system initialized")
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return access token."""
        if not self.is_initialized:
            print("SecurityManager: Security system not initialized")
            return None
        
        print(f"SecurityManager: Authenticating user '{username}'")
        
        # Simulate authentication logic
        if len(password) >= self.security_policies['password_min_length']:
            token = f"token_{username}_{int(time.time())}"
            self.access_tokens[token] = {
                'username': username,
                'created_at': time.time(),
                'expires_at': time.time() + self.security_policies['session_timeout']
            }
            
            self.audit_log.append({
                'action': 'user_authenticated',
                'username': username,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"SecurityManager: User '{username}' authenticated successfully")
            return token
        else:
            self.audit_log.append({
                'action': 'authentication_failed',
                'username': username,
                'reason': 'password_too_short',
                'timestamp': datetime.now().isoformat()
            })
            print(f"SecurityManager: Authentication failed for user '{username}'")
            return None
    
    def validate_token(self, token: str) -> bool:
        """Validate access token."""
        if not self.is_initialized:
            return False
        
        if token not in self.access_tokens:
            print(f"SecurityManager: Invalid token")
            return False
        
        token_info = self.access_tokens[token]
        if time.time() > token_info['expires_at']:
            print(f"SecurityManager: Token expired")
            del self.access_tokens[token]
            return False
        
        print(f"SecurityManager: Token validated for user '{token_info['username']}'")
        return True
    
    def encrypt_data(self, data: str, key_name: str = 'default') -> str:
        """Encrypt data using specified key."""
        if not self.is_initialized:
            return data
        
        key = self.encryption_keys.get(key_name, self.encryption_keys['default'])
        encrypted = f"ENCRYPTED[{data}]_KEY[{key_name}]"
        print(f"SecurityManager: Data encrypted using key '{key_name}'")
        return encrypted
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data."""
        if not self.is_initialized:
            return encrypted_data
        
        if encrypted_data.startswith("ENCRYPTED[") and "]_KEY[" in encrypted_data:
            end_data = encrypted_data.find("]_KEY[")
            data = encrypted_data[10:end_data]  # Remove "ENCRYPTED[" and find end
            print("SecurityManager: Data decrypted")
            return data
        
        return encrypted_data
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get security audit log."""
        return self.audit_log.copy()


class NotificationService:
    """Complex notification subsystem."""
    
    def __init__(self):
        self.email_config = {}
        self.sms_config = {}
        self.push_config = {}
        self.notification_templates = {}
        self.is_configured = False
        self.sent_notifications = []
    
    def configure(self, email_config: Dict[str, Any], sms_config: Dict[str, Any], 
                 push_config: Dict[str, Any]) -> bool:
        """Configure notification service."""
        print("NotificationService: Configuring notification service")
        
        self.email_config = email_config
        self.sms_config = sms_config
        self.push_config = push_config
        
        # Load notification templates
        self.notification_templates = {
            'welcome': {
                'subject': 'Welcome to our service!',
                'body': 'Thank you for joining us, {username}!'
            },
            'password_reset': {
                'subject': 'Password Reset Request',
                'body': 'Click here to reset your password: {reset_link}'
            },
            'order_confirmation': {
                'subject': 'Order Confirmation',
                'body': 'Your order #{order_id} has been confirmed.'
            }
        }
        
        self.is_configured = True
        print("NotificationService: Notification service configured")
        return True
    
    def send_email(self, recipient: str, template: str, variables: Dict[str, Any] = None) -> bool:
        """Send email notification."""
        if not self.is_configured:
            print("NotificationService: Service not configured")
            return False
        
        if template not in self.notification_templates:
            print(f"NotificationService: Template '{template}' not found")
            return False
        
        template_data = self.notification_templates[template]
        variables = variables or {}
        
        subject = template_data['subject'].format(**variables)
        body = template_data['body'].format(**variables)
        
        notification = {
            'type': 'email',
            'recipient': recipient,
            'subject': subject,
            'body': body,
            'sent_at': datetime.now().isoformat(),
            'status': 'sent'
        }
        
        self.sent_notifications.append(notification)
        print(f"NotificationService: Email sent to {recipient} - {subject}")
        return True
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification."""
        if not self.is_configured:
            print("NotificationService: Service not configured")
            return False
        
        notification = {
            'type': 'sms',
            'recipient': phone_number,
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'status': 'sent'
        }
        
        self.sent_notifications.append(notification)
        print(f"NotificationService: SMS sent to {phone_number}")
        return True
    
    def send_push_notification(self, device_id: str, title: str, message: str) -> bool:
        """Send push notification."""
        if not self.is_configured:
            print("NotificationService: Service not configured")
            return False
        
        notification = {
            'type': 'push',
            'recipient': device_id,
            'title': title,
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'status': 'sent'
        }
        
        self.sent_notifications.append(notification)
        print(f"NotificationService: Push notification sent to {device_id}")
        return True
    
    def get_notification_history(self) -> List[Dict[str, Any]]:
        """Get notification history."""
        return self.sent_notifications.copy()


# ============================================================================
# FACADE CLASSES
# ============================================================================

class ApplicationFacade:
    """Main application facade that coordinates all subsystems."""
    
    def __init__(self):
        # Initialize all subsystems
        self.database = DatabaseConnection("postgresql://localhost:5432/app_db")
        self.cache = CacheManager(cache_size=2000)
        self.logger = LoggingSystem()
        self.security = SecurityManager()
        self.notifications = NotificationService()
        
        self.is_initialized = False
    
    def initialize_application(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the entire application with all subsystems."""
        print("ApplicationFacade: Initializing application...")
        
        config = config or {}
        
        try:
            # Initialize logging first
            log_config = config.get('logging', {})
            if not self.logger.configure(
                level=log_config.get('level', 'INFO'),
                handlers=log_config.get('handlers', ['console', 'file'])
            ):
                return False
            
            self.logger.info("Application initialization started")
            
            # Initialize security
            security_config = config.get('security', {})
            if not self.security.initialize(security_config):
                self.logger.error("Failed to initialize security system")
                return False
            
            # Initialize cache
            if not self.cache.initialize():
                self.logger.error("Failed to initialize cache system")
                return False
            
            # Initialize database
            if not self.database.connect():
                self.logger.error("Failed to connect to database")
                return False
            
            # Initialize notifications
            notification_config = config.get('notifications', {})
            if not self.notifications.configure(
                email_config=notification_config.get('email', {}),
                sms_config=notification_config.get('sms', {}),
                push_config=notification_config.get('push', {})
            ):
                self.logger.error("Failed to configure notification service")
                return False
            
            self.is_initialized = True
            self.logger.info("Application initialized successfully")
            return True
            
        except Exception as e:
            if self.logger.is_configured:
                self.logger.error(f"Application initialization failed: {str(e)}")
            else:
                print(f"ApplicationFacade: Initialization failed: {str(e)}")
            return False
    
    def shutdown_application(self) -> None:
        """Shutdown the entire application gracefully."""
        if not self.is_initialized:
            return
        
        print("ApplicationFacade: Shutting down application...")
        
        try:
            self.logger.info("Application shutdown started")
            
            # Shutdown in reverse order
            self.database.disconnect()
            self.cache.shutdown()
            self.logger.info("Application shutdown completed")
            
            self.is_initialized = False
            
        except Exception as e:
            print(f"ApplicationFacade: Shutdown error: {str(e)}")
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Simplified user authentication."""
        if not self.is_initialized:
            return None
        
        self.logger.info(f"Authentication attempt for user: {username}")
        
        token = self.security.authenticate_user(username, password)
        if token:
            # Cache user session
            self.cache.put(f"session_{token}", {
                'username': username,
                'authenticated_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"User {username} authenticated successfully")
        else:
            self.logger.warning(f"Authentication failed for user: {username}")
        
        return token
    
    def get_user_data(self, token: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user data with authentication, caching, and logging."""
        if not self.is_initialized:
            return None
        
        # Validate token
        if not self.security.validate_token(token):
            self.logger.warning("Invalid token used for data access")
            return None
        
        # Check cache first
        cache_key = f"user_data_{user_id}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.logger.debug(f"User data retrieved from cache for user {user_id}")
            return cached_data['value']
        
        # Query database
        try:
            query = f"SELECT * FROM users WHERE id = {user_id}"
            results = self.database.execute_query(query)
            
            if results:
                user_data = results[0]
                # Cache the result
                self.cache.put(cache_key, user_data)
                self.logger.info(f"User data retrieved from database for user {user_id}")
                return user_data
            else:
                self.logger.warning(f"User {user_id} not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Database error retrieving user {user_id}: {str(e)}")
            return None
    
    def create_user(self, token: str, user_data: Dict[str, Any]) -> bool:
        """Create new user with full system integration."""
        if not self.is_initialized:
            return False
        
        # Validate token
        if not self.security.validate_token(token):
            self.logger.warning("Invalid token used for user creation")
            return False
        
        try:
            # Begin database transaction
            if not self.database.begin_transaction():
                return False
            
            # Encrypt sensitive data
            if 'password' in user_data:
                user_data['password'] = self.security.encrypt_data(
                    user_data['password'], 'database'
                )
            
            # Insert user into database
            query = f"INSERT INTO users (username, email, password) VALUES ('{user_data['username']}', '{user_data['email']}', '{user_data['password']}')"
            affected_rows = self.database.execute_update(query)
            
            if affected_rows > 0:
                # Commit transaction
                self.database.commit_transaction()
                
                # Invalidate related cache entries
                self.cache.invalidate(f"user_data_{user_data.get('id', 'unknown')}")
                
                # Send welcome notification
                self.notifications.send_email(
                    user_data['email'],
                    'welcome',
                    {'username': user_data['username']}
                )
                
                self.logger.info(f"User created successfully: {user_data['username']}")
                return True
            else:
                # Rollback transaction
                self.database.rollback_transaction()
                self.logger.error("Failed to create user - no rows affected")
                return False
                
        except Exception as e:
            # Rollback transaction on error
            self.database.rollback_transaction()
            self.logger.error(f"User creation failed: {str(e)}")
            return False
    
    def send_notification(self, token: str, recipient: str, notification_type: str, 
                         template: str, variables: Dict[str, Any] = None) -> bool:
        """Send notification with authentication and logging."""
        if not self.is_initialized:
            return False
        
        # Validate token
        if not self.security.validate_token(token):
            self.logger.warning("Invalid token used for notification")
            return False
        
        try:
            success = False
            
            if notification_type == 'email':
                success = self.notifications.send_email(recipient, template, variables)
            elif notification_type == 'sms':
                message = variables.get('message', 'Default SMS message')
                success = self.notifications.send_sms(recipient, message)
            elif notification_type == 'push':
                title = variables.get('title', 'Notification')
                message = variables.get('message', 'You have a new notification')
                success = self.notifications.send_push_notification(recipient, title, message)
            
            if success:
                self.logger.info(f"Notification sent: {notification_type} to {recipient}")
            else:
                self.logger.error(f"Failed to send notification: {notification_type} to {recipient}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Notification error: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'running',
            'database': {
                'connected': self.database.is_connected,
                'active_connections': self.database.active_connections,
                'transaction_active': self.database.transaction_active
            },
            'cache': self.cache.get_stats(),
            'security': {
                'initialized': self.security.is_initialized,
                'active_tokens': len(self.security.access_tokens),
                'audit_entries': len(self.security.audit_log)
            },
            'notifications': {
                'configured': self.notifications.is_configured,
                'sent_count': len(self.notifications.sent_notifications)
            },
            'logging': {
                'configured': self.logger.is_configured,
                'current_level': self.logger.current_level
            }
        }


class DatabaseFacade:
    """Simplified facade for database operations."""
    
    def __init__(self, connection_string: str):
        self.db = DatabaseConnection(connection_string)
        self.logger = LoggingSystem()
        self.security = SecurityManager()
        self.is_ready = False
    
    def initialize(self) -> bool:
        """Initialize database facade."""
        print("DatabaseFacade: Initializing database facade...")
        
        # Configure logging
        if not self.logger.configure('INFO', ['console']):
            return False
        
        # Initialize security
        if not self.security.initialize({}):
            return False
        
        # Connect to database
        if not self.db.connect():
            return False
        
        self.is_ready = True
        self.logger.info("Database facade initialized")
        return True
    
    def find_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find user by email address."""
        if not self.is_ready:
            return None
        
        try:
            query = f"SELECT * FROM users WHERE email = '{email}'"
            results = self.db.execute_query(query)
            
            if results:
                user_data = results[0]
                # Decrypt sensitive data
                if 'password' in user_data:
                    user_data['password'] = self.security.decrypt_data(user_data['password'])
                
                self.logger.info(f"User found by email: {email}")
                return user_data
            else:
                self.logger.info(f"No user found with email: {email}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding user by email: {str(e)}")
            return None
    
    def save_user(self, user_data: Dict[str, Any]) -> bool:
        """Save user data with transaction support."""
        if not self.is_ready:
            return False
        
        try:
            # Begin transaction
            if not self.db.begin_transaction():
                return False
            
            # Encrypt password if present
            if 'password' in user_data:
                user_data['password'] = self.security.encrypt_data(
                    user_data['password'], 'database'
                )
            
            # Execute insert/update
            if 'id' in user_data:
                # Update existing user
                query = f"UPDATE users SET username='{user_data['username']}', email='{user_data['email']}' WHERE id={user_data['id']}"
            else:
                # Insert new user
                query = f"INSERT INTO users (username, email, password) VALUES ('{user_data['username']}', '{user_data['email']}', '{user_data['password']}')"
            
            affected_rows = self.db.execute_update(query)
            
            if affected_rows > 0:
                self.db.commit_transaction()
                self.logger.info(f"User saved successfully: {user_data['username']}")
                return True
            else:
                self.db.rollback_transaction()
                self.logger.error("Failed to save user - no rows affected")
                return False
                
        except Exception as e:
            self.db.rollback_transaction()
            self.logger.error(f"Error saving user: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup database facade."""
        if self.is_ready:
            self.db.disconnect()
            self.is_ready = False
            print("DatabaseFacade: Cleaned up")


class NotificationFacade:
    """Simplified facade for notification operations."""
    
    def __init__(self):
        self.notification_service = NotificationService()
        self.logger = LoggingSystem()
        self.is_ready = False
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize notification facade."""
        print("NotificationFacade: Initializing notification facade...")
        
        # Configure logging
        if not self.logger.configure('INFO', ['console']):
            return False
        
        # Configure notification service
        config = config or {}
        if not self.notification_service.configure(
            email_config=config.get('email', {'smtp_server': 'localhost'}),
            sms_config=config.get('sms', {'provider': 'twilio'}),
            push_config=config.get('push', {'service': 'firebase'})
        ):
            return False
        
        self.is_ready = True
        self.logger.info("Notification facade initialized")
        return True
    
    def send_welcome_email(self, email: str, username: str) -> bool:
        """Send welcome email to new user."""
        if not self.is_ready:
            return False
        
        return self.notification_service.send_email(
            email, 'welcome', {'username': username}
        )
    
    def send_password_reset(self, email: str, reset_link: str) -> bool:
        """Send password reset email."""
        if not self.is_ready:
            return False
        
        return self.notification_service.send_email(
            email, 'password_reset', {'reset_link': reset_link}
        )
    
    def send_order_confirmation(self, email: str, order_id: str) -> bool:
        """Send order confirmation email."""
        if not self.is_ready:
            return False
        
        return self.notification_service.send_email(
            email, 'order_confirmation', {'order_id': order_id}
        )
    
    def send_urgent_sms(self, phone: str, message: str) -> bool:
        """Send urgent SMS notification."""
        if not self.is_ready:
            return False
        
        urgent_message = f"URGENT: {message}"
        return self.notification_service.send_sms(phone, urgent_message)
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        if not self.is_ready:
            return {}
        
        history = self.notification_service.get_notification_history()
        
        stats = {
            'total_sent': len(history),
            'by_type': {},
            'recent_count': 0
        }
        
        recent_threshold = time.time() - 3600  # Last hour
        
        for notification in history:
            # Count by type
            notif_type = notification['type']
            stats['by_type'][notif_type] = stats['by_type'].get(notif_type, 0) + 1
            
            # Count recent notifications
            sent_time = datetime.fromisoformat(notification['sent_at']).timestamp()
            if sent_time > recent_threshold:
                stats['recent_count'] += 1
        
        return stats


def demonstrate_facade_pattern():
    """
    Demonstrate Facade pattern implementations.
    """
    print("=== FACADE PATTERN DEMONSTRATION ===\n")
    
    # 1. Application Facade - Complete System
    print("1. APPLICATION FACADE - COMPLETE SYSTEM:")
    
    app_facade = ApplicationFacade()
    
    # Initialize application with configuration
    app_config = {
        'logging': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        },
        'security': {
            'password_min_length': 8,
            'session_timeout': 3600,
            'require_2fa': False
        },
        'notifications': {
            'email': {'smtp_server': 'smtp.example.com'},
            'sms': {'provider': 'twilio', 'api_key': 'test_key'},
            'push': {'service': 'firebase', 'server_key': 'test_key'}
        }
    }
    
    print("   Initializing complete application...")
    success = app_facade.initialize_application(app_config)
    print(f"   Application initialization: {'Success' if success else 'Failed'}")
    
    if success:
        # Get system status
        status = app_facade.get_system_status()
        print(f"   System status: {status['status']}")
        print(f"   Database connected: {status['database']['connected']}")
        print(f"   Cache hit ratio: {status['cache']['hit_ratio']}%")
        print()
        
        # 2. User Authentication through Facade
        print("2. USER AUTHENTICATION THROUGH FACADE:")
        
        # Authenticate user
        token = app_facade.authenticate_user("john_doe", "secure_password123")
        if token:
            print(f"   User authenticated successfully")
            print(f"   Token: {token[:20]}...")
            
            # Create new user
            print("\n   Creating new user through facade...")
            new_user_data = {
                'username': 'jane_smith',
                'email': 'jane@example.com',
                'password': 'another_secure_password'
            }
            
            user_created = app_facade.create_user(token, new_user_data)
            print(f"   User creation: {'Success' if user_created else 'Failed'}")
            
            # Get user data
            print("\n   Retrieving user data through facade...")
            user_data = app_facade.get_user_data(token, 1)
            if user_data:
                print(f"   Retrieved user: {user_data.get('name', 'Sample User')}")
            
            # Send notification
            print("\n   Sending notification through facade...")
            notification_sent = app_facade.send_notification(
                token, 'jane@example.com', 'email', 'welcome',
                {'username': 'jane_smith'}
            )
            print(f"   Notification sent: {'Success' if notification_sent else 'Failed'}")
        
        print()
        
        # 3. System Status and Monitoring
        print("3. SYSTEM STATUS AND MONITORING:")
        
        final_status = app_facade.get_system_status()
        print("   Final system status:")
        print(f"     Overall: {final_status['status']}")
        print(f"     Active tokens: {final_status['security']['active_tokens']}")
        print(f"     Notifications sent: {final_status['notifications']['sent_count']}")
        print(f"     Cache items: {final_status['cache']['cache_size']}")
        print(f"     Audit entries: {final_status['security']['audit_entries']}")
        
        # Shutdown application
        print("\n   Shutting down application...")
        app_facade.shutdown_application()
        print("   Application shutdown completed")
    
    print()
    
    # 4. Database Facade - Simplified Database Operations
    print("4. DATABASE FACADE - SIMPLIFIED OPERATIONS:")
    
    db_facade = DatabaseFacade("postgresql://localhost:5432/user_db")
    
    print("   Initializing database facade...")
    if db_facade.initialize():
        print("   Database facade initialized successfully")
        
        # Find user by email
        print("\n   Finding user by email...")
        user = db_facade.find_user_by_email("john@example.com")
        if user:
            print(f"   Found user: {user.get('name', 'John Doe')}")
        else:
            print("   User not found")
        
        # Save new user
        print("\n   Saving new user...")
        new_user = {
            'username': 'alice_wonder',
            'email': 'alice@example.com',
            'password': 'wonderland123'
        }
        
        saved = db_facade.save_user(new_user)
        print(f"   User saved: {'Success' if saved else 'Failed'}")
        
        # Cleanup
        db_facade.cleanup()
        print("   Database facade cleaned up")
    
    print()
    
    # 5. Notification Facade - Simplified Notifications
    print("5. NOTIFICATION FACADE - SIMPLIFIED NOTIFICATIONS:")
    
    notif_facade = NotificationFacade()
    
    notif_config = {
        'email': {'smtp_server': 'smtp.gmail.com', 'port': 587},
        'sms': {'provider': 'twilio', 'account_sid': 'test_sid'},
        'push': {'service': 'firebase', 'project_id': 'test_project'}
    }
    
    print("   Initializing notification facade...")
    if notif_facade.initialize(notif_config):
        print("   Notification facade initialized successfully")
        
        # Send various notifications
        print("\n   Sending welcome email...")
        welcome_sent = notif_facade.send_welcome_email("newuser@example.com", "NewUser")
        print(f"   Welcome email sent: {'Success' if welcome_sent else 'Failed'}")
        
        print("\n   Sending password reset...")
        reset_sent = notif_facade.send_password_reset(
            "user@example.com", 
            "https://example.com/reset?token=abc123"
        )
        print(f"   Password reset sent: {'Success' if reset_sent else 'Failed'}")
        
        print("\n   Sending order confirmation...")
        order_sent = notif_facade.send_order_confirmation("customer@example.com", "ORD-12345")
        print(f"   Order confirmation sent: {'Success' if order_sent else 'Failed'}")
        
        print("\n   Sending urgent SMS...")
        sms_sent = notif_facade.send_urgent_sms("+1234567890", "System maintenance in 5 minutes")
        print(f"   Urgent SMS sent: {'Success' if sms_sent else 'Failed'}")
        
        # Get notification statistics
        print("\n   Notification statistics:")
        stats = notif_facade.get_notification_stats()
        print(f"     Total sent: {stats['total_sent']}")
        print(f"     Recent notifications: {stats['recent_count']}")
        print(f"     By type: {stats['by_type']}")
    
    print()
    
    # 6. Facade Benefits Demonstration
    print("6. FACADE BENEFITS DEMONSTRATION:")
    
    print("   WITHOUT FACADE (Complex client code):")
    print("   ```")
    print("   # Client needs to know about all subsystems")
    print("   db = DatabaseConnection('connection_string')")
    print("   cache = CacheManager(1000)")
    print("   logger = LoggingSystem()")
    print("   security = SecurityManager()")
    print("   notifications = NotificationService()")
    print("   ")
    print("   # Complex initialization sequence")
    print("   logger.configure('INFO', ['console'])")
    print("   security.initialize({})")
    print("   cache.initialize()")
    print("   db.connect()")
    print("   notifications.configure({}, {}, {})")
    print("   ")
    print("   # Complex operations")
    print("   token = security.authenticate_user(username, password)")
    print("   if security.validate_token(token):")
    print("       db.begin_transaction()")
    print("       encrypted_data = security.encrypt_data(data)")
    print("       db.execute_update(query)")
    print("       db.commit_transaction()")
    print("       cache.invalidate(key)")
    print("       notifications.send_email(email, template, vars)")
    print("   ```")
    
    print("\n   WITH FACADE (Simple client code):")
    print("   ```")
    print("   # Simple initialization")
    print("   app = ApplicationFacade()")
    print("   app.initialize_application(config)")
    print("   ")
    print("   # Simple operations")
    print("   token = app.authenticate_user(username, password)")
    print("   app.create_user(token, user_data)")
    print("   app.send_notification(token, recipient, type, template)")
    print("   ```")
    
    print()
    
    # 7. Facade Pattern Benefits
    print("7. FACADE PATTERN BENEFITS:")
    print("   ✓ Simplified Interface: Hide complex subsystem interactions")
    print("   ✓ Reduced Coupling: Clients don't depend on subsystem classes")
    print("   ✓ Easier Testing: Mock the facade instead of multiple subsystems")
    print("   ✓ Centralized Control: Single point for system coordination")
    print("   ✓ Layered Architecture: Clear separation between layers")
    print("   ✓ Backward Compatibility: Facade can evolve while maintaining interface")
    print("   ✓ Error Handling: Centralized error handling and recovery")
    print("   ✓ Configuration Management: Single point for system configuration")
    print()
    
    print("=== FACADE PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_facade_pattern()
