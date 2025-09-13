"""
MEDIATOR PATTERN - Behavioral Design Pattern
============================================

Problem Statement:
Implement the Mediator pattern to define how a set of objects interact with
each other, promoting loose coupling by keeping objects from referring to
each other explicitly and letting you vary their interaction independently:
- Centralized communication between objects
- Loose coupling between interacting components
- Complex interaction logic encapsulation
- UI component coordination and event handling
- Workflow and business process orchestration

Learning Objectives:
- Understand Mediator vs Observer pattern differences
- Implement centralized communication hubs
- Design loose coupling between components
- Handle complex interaction scenarios
- Create reusable mediation patterns
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable, Union
import time
import json
from datetime import datetime
from enum import Enum
import threading
from collections import defaultdict


# ============================================================================
# MEDIATOR INTERFACE
# ============================================================================

class Mediator(ABC):
    """Abstract mediator interface."""
    
    @abstractmethod
    def notify(self, sender: 'Component', event: str, data: Any = None) -> None:
        """Handle notification from a component."""
        pass
    
    @abstractmethod
    def register_component(self, component: 'Component') -> None:
        """Register a component with the mediator."""
        pass
    
    @abstractmethod
    def unregister_component(self, component: 'Component') -> None:
        """Unregister a component from the mediator."""
        pass


class Component(ABC):
    """Abstract component that communicates through mediator."""
    
    def __init__(self, name: str, mediator: Mediator = None):
        self.name = name
        self._mediator = mediator
        self.event_count = 0
        self.last_activity = None
        
        if mediator:
            mediator.register_component(self)
    
    def set_mediator(self, mediator: Mediator) -> None:
        """Set the mediator for this component."""
        if self._mediator:
            self._mediator.unregister_component(self)
        
        self._mediator = mediator
        if mediator:
            mediator.register_component(self)
    
    def notify_mediator(self, event: str, data: Any = None) -> None:
        """Notify mediator of an event."""
        if self._mediator:
            self.event_count += 1
            self.last_activity = datetime.now()
            self._mediator.notify(self, event, data)
    
    @abstractmethod
    def handle_event(self, event: str, sender: 'Component', data: Any = None) -> None:
        """Handle event from mediator."""
        pass
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'event_count': self.event_count,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'has_mediator': self._mediator is not None
        }


# ============================================================================
# CHAT ROOM MEDIATOR
# ============================================================================

class ChatRoom(Mediator):
    """Chat room mediator for user communication."""
    
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.users: Set['User'] = set()
        self.message_history: List[Dict[str, Any]] = []
        self.banned_users: Set[str] = set()
        self.moderators: Set[str] = set()
        self.room_settings = {
            'max_message_length': 500,
            'allow_private_messages': True,
            'profanity_filter': True,
            'rate_limit_seconds': 1
        }
        self.user_last_message = {}  # user_name -> timestamp
        
    def register_component(self, component: 'Component') -> None:
        """Register user with chat room."""
        if isinstance(component, User):
            if component.username not in self.banned_users:
                self.users.add(component)
                self.broadcast_system_message(f"{component.username} joined the chat")
                print(f"User {component.username} joined chat room '{self.room_name}'")
            else:
                print(f"User {component.username} is banned from chat room '{self.room_name}'")
    
    def unregister_component(self, component: 'Component') -> None:
        """Unregister user from chat room."""
        if isinstance(component, User) and component in self.users:
            self.users.remove(component)
            self.broadcast_system_message(f"{component.username} left the chat")
            print(f"User {component.username} left chat room '{self.room_name}'")
    
    def notify(self, sender: 'Component', event: str, data: Any = None) -> None:
        """Handle notifications from users."""
        if not isinstance(sender, User):
            return
        
        if sender.username in self.banned_users:
            sender.handle_event('error', self, 'You are banned from this chat room')
            return
        
        if event == 'send_message':
            self.handle_message(sender, data)
        elif event == 'private_message':
            self.handle_private_message(sender, data)
        elif event == 'user_typing':
            self.handle_typing_notification(sender)
        elif event == 'moderate_user':
            self.handle_moderation(sender, data)
    
    def handle_message(self, sender: 'User', message_data: Dict[str, Any]) -> None:
        """Handle public message from user."""
        message = message_data.get('message', '')
        
        # Rate limiting
        current_time = time.time()
        last_message_time = self.user_last_message.get(sender.username, 0)
        
        if current_time - last_message_time < self.room_settings['rate_limit_seconds']:
            sender.handle_event('error', self, 'Please wait before sending another message')
            return
        
        # Message validation
        if len(message) > self.room_settings['max_message_length']:
            sender.handle_event('error', self, 
                              f'Message too long (max {self.room_settings["max_message_length"]} characters)')
            return
        
        # Profanity filter (simplified)
        if self.room_settings['profanity_filter']:
            profanity_words = ['spam', 'badword']  # Simplified list
            if any(word in message.lower() for word in profanity_words):
                sender.handle_event('error', self, 'Message contains inappropriate content')
                return
        
        # Create message record
        message_record = {
            'id': len(self.message_history) + 1,
            'sender': sender.username,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'type': 'public'
        }
        
        self.message_history.append(message_record)
        self.user_last_message[sender.username] = current_time
        
        # Broadcast to all users except sender
        for user in self.users:
            if user != sender:
                user.handle_event('message_received', self, message_record)
        
        # Confirm to sender
        sender.handle_event('message_sent', self, message_record)
    
    def handle_private_message(self, sender: 'User', message_data: Dict[str, Any]) -> None:
        """Handle private message between users."""
        if not self.room_settings['allow_private_messages']:
            sender.handle_event('error', self, 'Private messages are disabled')
            return
        
        recipient_name = message_data.get('recipient')
        message = message_data.get('message', '')
        
        # Find recipient
        recipient = None
        for user in self.users:
            if user.username == recipient_name:
                recipient = user
                break
        
        if not recipient:
            sender.handle_event('error', self, f'User {recipient_name} not found')
            return
        
        # Create private message record
        private_message = {
            'id': f"pm_{len(self.message_history) + 1}",
            'sender': sender.username,
            'recipient': recipient_name,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'type': 'private'
        }
        
        # Send to recipient and sender
        recipient.handle_event('private_message_received', self, private_message)
        sender.handle_event('private_message_sent', self, private_message)
    
    def handle_typing_notification(self, sender: 'User') -> None:
        """Handle typing notification."""
        typing_data = {
            'user': sender.username,
            'timestamp': datetime.now().isoformat()
        }
        
        # Notify other users
        for user in self.users:
            if user != sender:
                user.handle_event('user_typing', self, typing_data)
    
    def handle_moderation(self, sender: 'User', moderation_data: Dict[str, Any]) -> None:
        """Handle moderation actions."""
        if sender.username not in self.moderators:
            sender.handle_event('error', self, 'You do not have moderation privileges')
            return
        
        action = moderation_data.get('action')
        target_user = moderation_data.get('target_user')
        
        if action == 'ban':
            self.ban_user(target_user, sender.username)
        elif action == 'kick':
            self.kick_user(target_user, sender.username)
        elif action == 'mute':
            self.mute_user(target_user, sender.username)
    
    def broadcast_system_message(self, message: str) -> None:
        """Broadcast system message to all users."""
        system_message = {
            'id': f"sys_{len(self.message_history) + 1}",
            'sender': 'SYSTEM',
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'type': 'system'
        }
        
        self.message_history.append(system_message)
        
        for user in self.users:
            user.handle_event('system_message', self, system_message)
    
    def ban_user(self, username: str, moderator: str) -> None:
        """Ban user from chat room."""
        self.banned_users.add(username)
        
        # Remove user if currently in room
        user_to_remove = None
        for user in self.users:
            if user.username == username:
                user_to_remove = user
                break
        
        if user_to_remove:
            self.users.remove(user_to_remove)
            user_to_remove.handle_event('banned', self, f'Banned by {moderator}')
        
        self.broadcast_system_message(f"{username} was banned by {moderator}")
    
    def kick_user(self, username: str, moderator: str) -> None:
        """Kick user from chat room."""
        user_to_kick = None
        for user in self.users:
            if user.username == username:
                user_to_kick = user
                break
        
        if user_to_kick:
            self.users.remove(user_to_kick)
            user_to_kick.handle_event('kicked', self, f'Kicked by {moderator}')
            self.broadcast_system_message(f"{username} was kicked by {moderator}")
    
    def add_moderator(self, username: str) -> None:
        """Add user as moderator."""
        self.moderators.add(username)
        self.broadcast_system_message(f"{username} is now a moderator")
    
    def get_room_stats(self) -> Dict[str, Any]:
        """Get chat room statistics."""
        return {
            'room_name': self.room_name,
            'active_users': len(self.users),
            'total_messages': len(self.message_history),
            'banned_users': len(self.banned_users),
            'moderators': len(self.moderators),
            'settings': self.room_settings
        }


class User(Component):
    """User component in chat room."""
    
    def __init__(self, username: str, mediator: Mediator = None):
        super().__init__(username, mediator)
        self.username = username
        self.received_messages: List[Dict[str, Any]] = []
        self.sent_messages: List[Dict[str, Any]] = []
        self.is_typing = False
        self.status = 'online'
    
    def send_message(self, message: str) -> None:
        """Send public message to chat room."""
        self.notify_mediator('send_message', {'message': message})
    
    def send_private_message(self, recipient: str, message: str) -> None:
        """Send private message to specific user."""
        self.notify_mediator('private_message', {
            'recipient': recipient,
            'message': message
        })
    
    def start_typing(self) -> None:
        """Notify that user is typing."""
        if not self.is_typing:
            self.is_typing = True
            self.notify_mediator('user_typing')
    
    def stop_typing(self) -> None:
        """Stop typing notification."""
        self.is_typing = False
    
    def moderate_user(self, action: str, target_user: str) -> None:
        """Perform moderation action."""
        self.notify_mediator('moderate_user', {
            'action': action,
            'target_user': target_user
        })
    
    def handle_event(self, event: str, sender: Mediator, data: Any = None) -> None:
        """Handle events from mediator."""
        if event == 'message_received':
            self.received_messages.append(data)
            print(f"[{self.username}] Received: {data['sender']}: {data['message']}")
        
        elif event == 'message_sent':
            self.sent_messages.append(data)
            print(f"[{self.username}] Sent message: {data['message']}")
        
        elif event == 'private_message_received':
            self.received_messages.append(data)
            print(f"[{self.username}] Private from {data['sender']}: {data['message']}")
        
        elif event == 'private_message_sent':
            self.sent_messages.append(data)
            print(f"[{self.username}] Private to {data['recipient']}: {data['message']}")
        
        elif event == 'system_message':
            print(f"[{self.username}] SYSTEM: {data['message']}")
        
        elif event == 'user_typing':
            print(f"[{self.username}] {data['user']} is typing...")
        
        elif event == 'error':
            print(f"[{self.username}] ERROR: {data}")
        
        elif event == 'banned':
            print(f"[{self.username}] You have been banned: {data}")
            self.status = 'banned'
        
        elif event == 'kicked':
            print(f"[{self.username}] You have been kicked: {data}")
            self.status = 'kicked'
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get user message statistics."""
        return {
            'username': self.username,
            'messages_sent': len(self.sent_messages),
            'messages_received': len(self.received_messages),
            'status': self.status,
            'is_typing': self.is_typing
        }


# ============================================================================
# DIALOG BOX MEDIATOR
# ============================================================================

class DialogMediator(Mediator):
    """Mediator for dialog box components."""
    
    def __init__(self, dialog_title: str):
        self.dialog_title = dialog_title
        self.components: Dict[str, 'UIComponent'] = {}
        self.form_data: Dict[str, Any] = {}
        self.validation_errors: Dict[str, str] = {}
        self.is_valid = True
        self.event_log: List[Dict[str, Any]] = []
    
    def register_component(self, component: 'Component') -> None:
        """Register UI component."""
        if isinstance(component, UIComponent):
            self.components[component.name] = component
            print(f"Registered UI component: {component.name}")
    
    def unregister_component(self, component: 'Component') -> None:
        """Unregister UI component."""
        if isinstance(component, UIComponent) and component.name in self.components:
            del self.components[component.name]
            print(f"Unregistered UI component: {component.name}")
    
    def notify(self, sender: 'Component', event: str, data: Any = None) -> None:
        """Handle UI component events."""
        if not isinstance(sender, UIComponent):
            return
        
        # Log event
        self.event_log.append({
            'timestamp': datetime.now().isoformat(),
            'component': sender.name,
            'event': event,
            'data': data
        })
        
        if event == 'value_changed':
            self.handle_value_change(sender, data)
        elif event == 'button_clicked':
            self.handle_button_click(sender, data)
        elif event == 'focus_gained':
            self.handle_focus_change(sender, True)
        elif event == 'focus_lost':
            self.handle_focus_change(sender, False)
        elif event == 'validation_requested':
            self.handle_validation(sender)
    
    def handle_value_change(self, sender: 'UIComponent', new_value: Any) -> None:
        """Handle value change in UI component."""
        # Update form data
        self.form_data[sender.name] = new_value
        
        # Clear validation error for this field
        if sender.name in self.validation_errors:
            del self.validation_errors[sender.name]
        
        # Handle specific component interactions
        if sender.name == 'country_dropdown':
            self.update_state_dropdown(new_value)
        elif sender.name == 'user_type_radio':
            self.update_form_visibility(new_value)
        elif sender.name == 'email_input':
            self.validate_email_format(new_value)
        elif sender.name == 'password_input':
            self.update_password_strength(new_value)
        elif sender.name == 'confirm_password_input':
            self.validate_password_match()
        
        # Update submit button state
        self.update_submit_button_state()
    
    def handle_button_click(self, sender: 'UIComponent', button_data: Dict[str, Any]) -> None:
        """Handle button click events."""
        button_type = button_data.get('type', 'button')
        
        if button_type == 'submit':
            self.handle_form_submission()
        elif button_type == 'cancel':
            self.handle_form_cancellation()
        elif button_type == 'reset':
            self.handle_form_reset()
        elif button_type == 'validate':
            self.validate_all_fields()
    
    def handle_focus_change(self, sender: 'UIComponent', has_focus: bool) -> None:
        """Handle focus change events."""
        if has_focus:
            # Clear any error highlighting when field gains focus
            sender.handle_event('clear_error', self)
        else:
            # Validate field when it loses focus
            self.validate_field(sender.name)
    
    def update_state_dropdown(self, country: str) -> None:
        """Update state dropdown based on country selection."""
        state_dropdown = self.components.get('state_dropdown')
        if not state_dropdown:
            return
        
        # Simulated state data
        states_by_country = {
            'USA': ['California', 'New York', 'Texas', 'Florida'],
            'Canada': ['Ontario', 'Quebec', 'British Columbia', 'Alberta'],
            'UK': ['England', 'Scotland', 'Wales', 'Northern Ireland']
        }
        
        states = states_by_country.get(country, [])
        state_dropdown.handle_event('update_options', self, states)
        
        # Clear current state selection
        if 'state_dropdown' in self.form_data:
            del self.form_data['state_dropdown']
    
    def update_form_visibility(self, user_type: str) -> None:
        """Update form field visibility based on user type."""
        company_field = self.components.get('company_input')
        student_id_field = self.components.get('student_id_input')
        
        if user_type == 'business':
            if company_field:
                company_field.handle_event('show', self)
            if student_id_field:
                student_id_field.handle_event('hide', self)
        elif user_type == 'student':
            if company_field:
                company_field.handle_event('hide', self)
            if student_id_field:
                student_id_field.handle_event('show', self)
        else:  # individual
            if company_field:
                company_field.handle_event('hide', self)
            if student_id_field:
                student_id_field.handle_event('hide', self)
    
    def validate_email_format(self, email: str) -> None:
        """Validate email format."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        email_input = self.components.get('email_input')
        if not email_input:
            return
        
        if email and not re.match(email_pattern, email):
            self.validation_errors['email_input'] = 'Invalid email format'
            email_input.handle_event('show_error', self, 'Invalid email format')
        else:
            if 'email_input' in self.validation_errors:
                del self.validation_errors['email_input']
            email_input.handle_event('clear_error', self)
    
    def update_password_strength(self, password: str) -> None:
        """Update password strength indicator."""
        strength_indicator = self.components.get('password_strength')
        if not strength_indicator:
            return
        
        # Calculate password strength
        strength = 0
        if len(password) >= 8:
            strength += 1
        if any(c.isupper() for c in password):
            strength += 1
        if any(c.islower() for c in password):
            strength += 1
        if any(c.isdigit() for c in password):
            strength += 1
        if any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            strength += 1
        
        strength_levels = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong']
        strength_text = strength_levels[min(strength, 4)]
        
        strength_indicator.handle_event('update_strength', self, {
            'level': strength,
            'text': strength_text
        })
    
    def validate_password_match(self) -> None:
        """Validate password confirmation."""
        password = self.form_data.get('password_input', '')
        confirm_password = self.form_data.get('confirm_password_input', '')
        
        confirm_input = self.components.get('confirm_password_input')
        if not confirm_input:
            return
        
        if confirm_password and password != confirm_password:
            self.validation_errors['confirm_password_input'] = 'Passwords do not match'
            confirm_input.handle_event('show_error', self, 'Passwords do not match')
        else:
            if 'confirm_password_input' in self.validation_errors:
                del self.validation_errors['confirm_password_input']
            confirm_input.handle_event('clear_error', self)
    
    def validate_field(self, field_name: str) -> bool:
        """Validate individual field."""
        component = self.components.get(field_name)
        if not component:
            return True
        
        value = self.form_data.get(field_name, '')
        
        # Required field validation
        if hasattr(component, 'required') and component.required:
            if not value or (isinstance(value, str) and not value.strip()):
                error_msg = f'{field_name.replace("_", " ").title()} is required'
                self.validation_errors[field_name] = error_msg
                component.handle_event('show_error', self, error_msg)
                return False
        
        return True
    
    def validate_all_fields(self) -> bool:
        """Validate all form fields."""
        self.validation_errors.clear()
        
        for field_name in self.components:
            self.validate_field(field_name)
        
        self.is_valid = len(self.validation_errors) == 0
        
        # Update form validation status
        for component in self.components.values():
            component.handle_event('validation_complete', self, {
                'is_valid': self.is_valid,
                'errors': self.validation_errors
            })
        
        return self.is_valid
    
    def update_submit_button_state(self) -> None:
        """Update submit button enabled/disabled state."""
        submit_button = self.components.get('submit_button')
        if not submit_button:
            return
        
        # Check if required fields are filled
        required_fields = ['email_input', 'password_input', 'confirm_password_input']
        all_required_filled = all(
            self.form_data.get(field) for field in required_fields
        )
        
        has_errors = len(self.validation_errors) > 0
        
        if all_required_filled and not has_errors:
            submit_button.handle_event('enable', self)
        else:
            submit_button.handle_event('disable', self)
    
    def handle_form_submission(self) -> None:
        """Handle form submission."""
        if self.validate_all_fields():
            print(f"Form '{self.dialog_title}' submitted successfully!")
            print(f"Form data: {json.dumps(self.form_data, indent=2)}")
            
            # Notify all components of successful submission
            for component in self.components.values():
                component.handle_event('form_submitted', self, self.form_data)
        else:
            print(f"Form validation failed. Errors: {self.validation_errors}")
    
    def handle_form_cancellation(self) -> None:
        """Handle form cancellation."""
        print(f"Form '{self.dialog_title}' cancelled")
        
        for component in self.components.values():
            component.handle_event('form_cancelled', self)
    
    def handle_form_reset(self) -> None:
        """Handle form reset."""
        self.form_data.clear()
        self.validation_errors.clear()
        
        print(f"Form '{self.dialog_title}' reset")
        
        for component in self.components.values():
            component.handle_event('form_reset', self)
    
    def get_dialog_state(self) -> Dict[str, Any]:
        """Get current dialog state."""
        return {
            'title': self.dialog_title,
            'components': list(self.components.keys()),
            'form_data': self.form_data.copy(),
            'validation_errors': self.validation_errors.copy(),
            'is_valid': self.is_valid,
            'events_processed': len(self.event_log)
        }


class UIComponent(Component):
    """Base UI component class."""
    
    def __init__(self, name: str, component_type: str, mediator: Mediator = None, required: bool = False):
        super().__init__(name, mediator)
        self.component_type = component_type
        self.value = None
        self.is_visible = True
        self.is_enabled = True
        self.has_error = False
        self.error_message = ""
        self.required = required
    
    def set_value(self, value: Any) -> None:
        """Set component value and notify mediator."""
        old_value = self.value
        self.value = value
        
        if old_value != value:
            self.notify_mediator('value_changed', value)
    
    def click(self, button_data: Dict[str, Any] = None) -> None:
        """Simulate component click."""
        if self.is_enabled:
            self.notify_mediator('button_clicked', button_data or {})
    
    def focus(self) -> None:
        """Simulate component gaining focus."""
        self.notify_mediator('focus_gained')
    
    def blur(self) -> None:
        """Simulate component losing focus."""
        self.notify_mediator('focus_lost')
    
    def handle_event(self, event: str, sender: Mediator, data: Any = None) -> None:
        """Handle events from mediator."""
        if event == 'show':
            self.is_visible = True
            print(f"[{self.name}] Component shown")
        
        elif event == 'hide':
            self.is_visible = False
            print(f"[{self.name}] Component hidden")
        
        elif event == 'enable':
            self.is_enabled = True
            print(f"[{self.name}] Component enabled")
        
        elif event == 'disable':
            self.is_enabled = False
            print(f"[{self.name}] Component disabled")
        
        elif event == 'show_error':
            self.has_error = True
            self.error_message = str(data) if data else ""
            print(f"[{self.name}] Error: {self.error_message}")
        
        elif event == 'clear_error':
            self.has_error = False
            self.error_message = ""
            print(f"[{self.name}] Error cleared")
        
        elif event == 'update_options':
            if hasattr(self, 'options'):
                self.options = data or []
                print(f"[{self.name}] Options updated: {self.options}")
        
        elif event == 'update_strength':
            if self.component_type == 'password_strength':
                strength_data = data or {}
                print(f"[{self.name}] Password strength: {strength_data.get('text', 'Unknown')}")
        
        elif event == 'form_submitted':
            print(f"[{self.name}] Form submitted")
        
        elif event == 'form_cancelled':
            print(f"[{self.name}] Form cancelled")
        
        elif event == 'form_reset':
            self.value = None
            self.has_error = False
            self.error_message = ""
            print(f"[{self.name}] Component reset")


# ============================================================================
# WORKFLOW MEDIATOR
# ============================================================================

class WorkflowMediator(Mediator):
    """Mediator for workflow orchestration."""
    
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.steps: Dict[str, 'WorkflowStep'] = {}
        self.workflow_data: Dict[str, Any] = {}
        self.current_step = None
        self.completed_steps: List[str] = []
        self.workflow_status = 'initialized'
        self.execution_log: List[Dict[str, Any]] = []
    
    def register_component(self, component: 'Component') -> None:
        """Register workflow step."""
        if isinstance(component, WorkflowStep):
            self.steps[component.name] = component
            print(f"Registered workflow step: {component.name}")
    
    def unregister_component(self, component: 'Component') -> None:
        """Unregister workflow step."""
        if isinstance(component, WorkflowStep) and component.name in self.steps:
            del self.steps[component.name]
            print(f"Unregistered workflow step: {component.name}")
    
    def notify(self, sender: 'Component', event: str, data: Any = None) -> None:
        """Handle workflow step notifications."""
        if not isinstance(sender, WorkflowStep):
            return
        
        # Log event
        self.execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': sender.name,
            'event': event,
            'data': data
        })
        
        if event == 'step_completed':
            self.handle_step_completion(sender, data)
        elif event == 'step_failed':
            self.handle_step_failure(sender, data)
        elif event == 'data_updated':
            self.handle_data_update(sender, data)
        elif event == 'step_skipped':
            self.handle_step_skip(sender, data)
    
    def start_workflow(self, initial_data: Dict[str, Any] = None) -> None:
        """Start workflow execution."""
        self.workflow_data.update(initial_data or {})
        self.workflow_status = 'running'
        self.completed_steps = []
        
        print(f"Starting workflow: {self.workflow_name}")
        
        # Find and execute first step
        first_step = self.find_next_step()
        if first_step:
            self.execute_step(first_step)
        else:
            print("No steps found in workflow")
            self.workflow_status = 'completed'
    
    def execute_step(self, step_name: str) -> None:
        """Execute specific workflow step."""
        if step_name not in self.steps:
            print(f"Step '{step_name}' not found")
            return
        
        step = self.steps[step_name]
        self.current_step = step_name
        
        print(f"Executing step: {step_name}")
        
        # Notify step to execute
        step.handle_event('execute', self, self.workflow_data)
    
    def handle_step_completion(self, sender: 'WorkflowStep', result_data: Any) -> None:
        """Handle step completion."""
        step_name = sender.name
        
        # Update workflow data with step results
        if isinstance(result_data, dict):
            self.workflow_data.update(result_data)
        
        # Mark step as completed
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
        
        print(f"Step '{step_name}' completed successfully")
        
        # Find and execute next step
        next_step = self.find_next_step()
        if next_step:
            self.execute_step(next_step)
        else:
            # Workflow completed
            self.workflow_status = 'completed'
            self.current_step = None
            print(f"Workflow '{self.workflow_name}' completed successfully")
            
            # Notify all steps of workflow completion
            for step in self.steps.values():
                step.handle_event('workflow_completed', self, self.workflow_data)
    
    def handle_step_failure(self, sender: 'WorkflowStep', error_data: Any) -> None:
        """Handle step failure."""
        step_name = sender.name
        error_message = str(error_data) if error_data else "Unknown error"
        
        print(f"Step '{step_name}' failed: {error_message}")
        
        # Check if step can be retried
        if hasattr(sender, 'retry_count') and sender.retry_count > 0:
            sender.retry_count -= 1
            print(f"Retrying step '{step_name}' ({sender.retry_count} retries left)")
            self.execute_step(step_name)
        else:
            # Workflow failed
            self.workflow_status = 'failed'
            self.current_step = None
            
            # Notify all steps of workflow failure
            for step in self.steps.values():
                step.handle_event('workflow_failed', self, {
                    'failed_step': step_name,
                    'error': error_message
                })
    
    def handle_data_update(self, sender: 'WorkflowStep', data: Dict[str, Any]) -> None:
        """Handle workflow data update."""
        if isinstance(data, dict):
            self.workflow_data.update(data)
            print(f"Workflow data updated by step '{sender.name}'")
    
    def handle_step_skip(self, sender: 'WorkflowStep', reason: str) -> None:
        """Handle step skip."""
        step_name = sender.name
        print(f"Step '{step_name}' skipped: {reason}")
        
        # Mark as completed and continue
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
        
        # Find next step
        next_step = self.find_next_step()
        if next_step:
            self.execute_step(next_step)
        else:
            self.workflow_status = 'completed'
    
    def find_next_step(self) -> Optional[str]:
        """Find next step to execute based on dependencies."""
        for step_name, step in self.steps.items():
            if step_name not in self.completed_steps:
                # Check if all dependencies are completed
                if hasattr(step, 'dependencies'):
                    if all(dep in self.completed_steps for dep in step.dependencies):
                        return step_name
                else:
                    return step_name
        
        return None
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            'workflow_name': self.workflow_name,
            'status': self.workflow_status,
            'current_step': self.current_step,
            'completed_steps': self.completed_steps.copy(),
            'total_steps': len(self.steps),
            'completion_percentage': (len(self.completed_steps) / len(self.steps)) * 100 if self.steps else 0,
            'workflow_data_keys': list(self.workflow_data.keys()),
            'execution_events': len(self.execution_log)
        }


class WorkflowStep(Component):
    """Workflow step component."""
    
    def __init__(self, name: str, mediator: Mediator = None, dependencies: List[str] = None):
        super().__init__(name, mediator)
        self.dependencies = dependencies or []
        self.retry_count = 3
        self.execution_time = 0
        self.step_data = {}
    
    def handle_event(self, event: str, sender: Mediator, data: Any = None) -> None:
        """Handle events from workflow mediator."""
        if event == 'execute':
            self.execute_step(data or {})
        elif event == 'workflow_completed':
            self.on_workflow_completed(data)
        elif event == 'workflow_failed':
            self.on_workflow_failed(data)
    
    def execute_step(self, workflow_data: Dict[str, Any]) -> None:
        """Execute this workflow step."""
        start_time = time.time()
        
        try:
            # Perform step-specific logic
            result = self.perform_step_logic(workflow_data)
            
            self.execution_time = time.time() - start_time
            
            # Notify mediator of completion
            self.notify_mediator('step_completed', result)
            
        except Exception as e:
            self.execution_time = time.time() - start_time
            self.notify_mediator('step_failed', str(e))
    
    @abstractmethod
    def perform_step_logic(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform step-specific logic - must be implemented by subclasses."""
        pass
    
    def update_workflow_data(self, data: Dict[str, Any]) -> None:
        """Update workflow data."""
        self.notify_mediator('data_updated', data)
    
    def skip_step(self, reason: str) -> None:
        """Skip this step."""
        self.notify_mediator('step_skipped', reason)
    
    def on_workflow_completed(self, workflow_data: Dict[str, Any]) -> None:
        """Called when workflow completes."""
        print(f"[{self.name}] Workflow completed")
    
    def on_workflow_failed(self, failure_data: Dict[str, Any]) -> None:
        """Called when workflow fails."""
        print(f"[{self.name}] Workflow failed: {failure_data.get('error', 'Unknown error')}")


def demonstrate_mediator_pattern():
    """
    Demonstrate Mediator pattern implementations.
    """
    print("=== MEDIATOR PATTERN DEMONSTRATION ===\n")
    
    # 1. Chat Room Mediator
    print("1. CHAT ROOM MEDIATOR:")
    
    # Create chat room
    chat_room = ChatRoom("General Chat")
    
    # Create users
    alice = User("alice", chat_room)
    bob = User("bob", chat_room)
    charlie = User("charlie", chat_room)
    
    # Add moderator
    chat_room.add_moderator("alice")
    
    print("\n   Chat room interactions:")
    print("   " + "=" * 40)
    
    # Simulate chat interactions
    alice.send_message("Hello everyone!")
    time.sleep(0.1)
    
    bob.send_message("Hi Alice! How are you?")
    time.sleep(0.1)
    
    charlie.send_message("Good morning!")
    time.sleep(0.1)
    
    # Private message
    alice.send_private_message("bob", "Can we discuss the project privately?")
    time.sleep(0.1)
    
    # Typing notification
    bob.start_typing()
    time.sleep(0.1)
    bob.stop_typing()
    
    bob.send_message("Sure, let's discuss it later")
    time.sleep(0.1)
    
    # Moderation action
    alice.moderate_user("kick", "charlie")
    time.sleep(0.1)
    
    # Show chat room statistics
    stats = chat_room.get_room_stats()
    print(f"\n   Chat Room Statistics:")
    print(f"     Active users: {stats['active_users']}")
    print(f"     Total messages: {stats['total_messages']}")
    print(f"     Moderators: {stats['moderators']}")
    
    # Show user statistics
    for user in [alice, bob]:
        user_stats = user.get_message_stats()
        print(f"     {user_stats['username']}: {user_stats['messages_sent']} sent, "
              f"{user_stats['messages_received']} received")
    
    print()
    
    # 2. Dialog Box Mediator
    print("2. DIALOG BOX MEDIATOR:")
    
    # Create dialog mediator
    dialog = DialogMediator("User Registration")
    
    # Create UI components
    email_input = UIComponent("email_input", "text_input", dialog, required=True)
    password_input = UIComponent("password_input", "password_input", dialog, required=True)
    confirm_password_input = UIComponent("confirm_password_input", "password_input", dialog, required=True)
    country_dropdown = UIComponent("country_dropdown", "dropdown", dialog)
    state_dropdown = UIComponent("state_dropdown", "dropdown", dialog)
    user_type_radio = UIComponent("user_type_radio", "radio_group", dialog)
    company_input = UIComponent("company_input", "text_input", dialog)
    student_id_input = UIComponent("student_id_input", "text_input", dialog)
    password_strength = UIComponent("password_strength", "password_strength", dialog)
    submit_button = UIComponent("submit_button", "button", dialog)
    
    print("\n   Dialog box interactions:")
    print("   " + "=" * 40)
    
    # Simulate user interactions
    email_input.set_value("alice@example.com")
    password_input.set_value("MySecurePass123!")
    confirm_password_input.set_value("MySecurePass123!")
    
    # Country selection triggers state dropdown update
    country_dropdown.set_value("USA")
    state_dropdown.set_value("California")
    
    # User type selection affects form visibility
    user_type_radio.set_value("business")
    company_input.set_value("Acme Corp")
    
    # Test validation
    email_input.set_value("invalid-email")  # Should trigger validation error
    email_input.set_value("alice@example.com")  # Should clear error
    
    # Test password mismatch
    confirm_password_input.set_value("different-password")  # Should show error
    confirm_password_input.set_value("MySecurePass123!")  # Should clear error
    
    # Submit form
    submit_button.click({"type": "submit"})
    
    # Show dialog state
    dialog_state = dialog.get_dialog_state()
    print(f"\n   Dialog State:")
    print(f"     Components: {len(dialog_state['components'])}")
    print(f"     Form valid: {dialog_state['is_valid']}")
    print(f"     Events processed: {dialog_state['events_processed']}")
    print(f"     Form data keys: {list(dialog_state['form_data'].keys())}")
    
    print()
    
    # 3. Workflow Mediator
    print("3. WORKFLOW MEDIATOR:")
    
    # Create workflow steps
    class DataValidationStep(WorkflowStep):
        def perform_step_logic(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
            print(f"[{self.name}] Validating input data...")
            
            # Simulate validation
            input_data = workflow_data.get('input_data', {})
            if not input_data:
                raise Exception("No input data provided")
            
            return {'validation_status': 'passed', 'validated_records': len(input_data)}
    
    class DataProcessingStep(WorkflowStep):
        def perform_step_logic(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
            print(f"[{self.name}] Processing data...")
            
            validated_records = workflow_data.get('validated_records', 0)
            
            # Simulate processing
            time.sleep(0.1)
            
            processed_records = validated_records * 0.95  # 95% success rate
            
            return {
                'processing_status': 'completed',
                'processed_records': int(processed_records),
                'processing_time': 0.1
            }
    
    class DataStorageStep(WorkflowStep):
        def perform_step_logic(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
            print(f"[{self.name}] Storing processed data...")
            
            processed_records = workflow_data.get('processed_records', 0)
            
            # Simulate storage
            time.sleep(0.05)
            
            return {
                'storage_status': 'completed',
                'stored_records': processed_records,
                'storage_location': 'database_cluster_1'
            }
    
    class NotificationStep(WorkflowStep):
        def perform_step_logic(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
            print(f"[{self.name}] Sending notifications...")
            
            stored_records = workflow_data.get('stored_records', 0)
            
            return {
                'notification_status': 'sent',
                'notifications_sent': 3,  # Admin, user, monitoring system
                'summary': f"Successfully processed {stored_records} records"
            }
    
    # Create workflow mediator
    workflow = WorkflowMediator("Data Processing Workflow")
    
    # Create and register workflow steps
    validation_step = DataValidationStep("data_validation", workflow)
    processing_step = DataProcessingStep("data_processing", workflow, dependencies=["data_validation"])
    storage_step = DataStorageStep("data_storage", workflow, dependencies=["data_processing"])
    notification_step = NotificationStep("notification", workflow, dependencies=["data_storage"])
    
    print("\n   Workflow execution:")
    print("   " + "=" * 40)
    
    # Start workflow with initial data
    initial_data = {
        'input_data': {'records': list(range(100))},  # 100 sample records
        'workflow_id': 'WF_001',
        'started_by': 'system'
    }
    
    workflow.start_workflow(initial_data)
    
    # Show workflow status
    workflow_status = workflow.get_workflow_status()
    print(f"\n   Workflow Status:")
    print(f"     Status: {workflow_status['status']}")
    print(f"     Completion: {workflow_status['completion_percentage']:.1f}%")
    print(f"     Completed steps: {workflow_status['completed_steps']}")
    print(f"     Total events: {workflow_status['execution_events']}")
    
    print()
    
    # 4. Complex Dialog with Dependencies
    print("4. COMPLEX DIALOG WITH DEPENDENCIES:")
    
    # Create complex form dialog
    complex_dialog = DialogMediator("Complex Registration Form")
    
    # Create interdependent components
    components = {
        'account_type': UIComponent("account_type", "radio", complex_dialog),
        'email': UIComponent("email", "text", complex_dialog, required=True),
        'password': UIComponent("password", "password", complex_dialog, required=True),
        'confirm_password': UIComponent("confirm_password", "password", complex_dialog, required=True),
        'company_name': UIComponent("company_name", "text", complex_dialog),
        'tax_id': UIComponent("tax_id", "text", complex_dialog),
        'personal_name': UIComponent("personal_name", "text", complex_dialog),
        'birth_date': UIComponent("birth_date", "date", complex_dialog),
        'terms_checkbox': UIComponent("terms_checkbox", "checkbox", complex_dialog, required=True),
        'submit_btn': UIComponent("submit_btn", "button", complex_dialog)
    }
    
    print("\n   Complex form interactions:")
    print("   " + "=" * 30)
    
    # Simulate complex interactions
    components['account_type'].set_value('business')
    components['email'].set_value('business@example.com')
    components['password'].set_value('SecurePass123!')
    components['confirm_password'].set_value('SecurePass123!')
    components['company_name'].set_value('Tech Solutions Inc.')
    components['tax_id'].set_value('12-3456789')
    components['terms_checkbox'].set_value(True)
    
    # Switch to personal account
    print("\n   Switching to personal account:")
    components['account_type'].set_value('personal')
    components['personal_name'].set_value('John Doe')
    components['birth_date'].set_value('1990-01-01')
    
    # Submit form
    components['submit_btn'].click({'type': 'submit'})
    
    final_state = complex_dialog.get_dialog_state()
    print(f"\n   Final form state:")
    print(f"     Valid: {final_state['is_valid']}")
    print(f"     Errors: {len(final_state['validation_errors'])}")
    
    print()
    
    # 5. Mediator Pattern Benefits
    print("5. MEDIATOR PATTERN BENEFITS:")
    print("   ✓ Loose Coupling: Components don't reference each other directly")
    print("   ✓ Centralized Control: Complex interactions are managed in one place")
    print("   ✓ Reusability: Components can be reused with different mediators")
    print("   ✓ Maintainability: Interaction logic is centralized and easier to modify")
    print("   ✓ Extensibility: New components can be added without changing existing ones")
    print("   ✓ Single Responsibility: Each component focuses on its specific functionality")
    print("   ✓ Testability: Mediator and components can be tested independently")
    print("   ✓ Flexibility: Interaction patterns can be changed without affecting components")
    print("   ✓ Consistency: Ensures consistent behavior across component interactions")
    print("   ✓ Monitoring: Centralized event handling enables better monitoring and logging")
    print()
    
    print("=== MEDIATOR PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_mediator_pattern()
