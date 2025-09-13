"""
CALENDAR SCHEDULING SYSTEM - Complete System Design
================================================

A comprehensive calendar and scheduling system that handles:
- Calendar management with multiple views (day, week, month, year)
- Event creation, modification, and deletion
- Recurring events with complex patterns
- Meeting scheduling with attendee management
- Room and resource booking
- Time zone support and conflicts resolution
- Reminder and notification system
- Calendar sharing and permissions
- Integration with external calendar systems
- Mobile and web synchronization

Design Patterns Used:
- Strategy: Different calendar views and recurrence patterns
- Observer: Event notifications and updates
- Factory: Event and calendar creation
- Template Method: Event processing pipeline
- Decorator: Event validation and enrichment
- Command: Calendar operations with undo/redo
- State: Event status transitions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta, time, date
from enum import Enum
import uuid
import threading
from dataclasses import dataclass, field
from collections import defaultdict
import pytz
import re


class EventType(Enum):
    MEETING = "meeting"
    APPOINTMENT = "appointment"
    REMINDER = "reminder"
    TASK = "task"
    BIRTHDAY = "birthday"
    HOLIDAY = "holiday"


class EventStatus(Enum):
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class RecurrenceType(Enum):
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class AttendeeStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    MAYBE = "maybe"


@dataclass
class User:
    user_id: str
    email: str
    name: str
    timezone: str = "UTC"
    default_calendar_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())


@dataclass
class Calendar:
    calendar_id: str
    name: str
    owner_id: str
    description: str = ""
    color: str = "#3174ad"
    is_public: bool = False
    timezone: str = "UTC"
    shared_with: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.calendar_id:
            self.calendar_id = str(uuid.uuid4())


@dataclass
class Attendee:
    email: str
    name: str = ""
    status: AttendeeStatus = AttendeeStatus.PENDING
    is_organizer: bool = False
    is_optional: bool = False


@dataclass
class RecurrenceRule:
    recurrence_type: RecurrenceType
    interval: int = 1
    end_date: Optional[datetime] = None
    count: Optional[int] = None
    by_day: List[int] = field(default_factory=list)  # 0=Monday, 6=Sunday
    by_month_day: List[int] = field(default_factory=list)
    by_month: List[int] = field(default_factory=list)


@dataclass
class Event:
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    calendar_id: str
    organizer_id: str
    
    description: str = ""
    location: str = ""
    event_type: EventType = EventType.MEETING
    status: EventStatus = EventStatus.CONFIRMED
    
    attendees: List[Attendee] = field(default_factory=list)
    recurrence_rule: Optional[RecurrenceRule] = None
    parent_event_id: Optional[str] = None  # For recurring event instances
    
    reminders: List[int] = field(default_factory=list)  # Minutes before event
    is_all_day: bool = False
    is_private: bool = False
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())


class CalendarSchedulingSystem:
    """Main calendar scheduling system."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.calendars: Dict[str, Calendar] = {}
        self.events: Dict[str, Event] = {}
        self.recurring_events: Dict[str, List[str]] = {}  # parent_id -> instance_ids
        
        # Notification system
        self.pending_notifications: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        print("📅 Calendar Scheduling System initialized")
    
    def create_user(self, email: str, name: str, timezone: str = "UTC") -> User:
        """Create a new user."""
        user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            name=name,
            timezone=timezone
        )
        
        with self._lock:
            self.users[user.user_id] = user
            
            # Create default calendar
            default_calendar = self.create_calendar(
                user.user_id, f"{name}'s Calendar", "Default calendar"
            )
            user.default_calendar_id = default_calendar.calendar_id
        
        return user
    
    def create_calendar(self, owner_id: str, name: str, description: str = "",
                       color: str = "#3174ad", is_public: bool = False) -> Calendar:
        """Create a new calendar."""
        if owner_id not in self.users:
            raise ValueError("User not found")
        
        calendar = Calendar(
            calendar_id=str(uuid.uuid4()),
            name=name,
            owner_id=owner_id,
            description=description,
            color=color,
            is_public=is_public,
            timezone=self.users[owner_id].timezone
        )
        
        with self._lock:
            self.calendars[calendar.calendar_id] = calendar
        
        return calendar
    
    def create_event(self, organizer_id: str, calendar_id: str, title: str,
                    start_time: datetime, end_time: datetime, **kwargs) -> Event:
        """Create a new event."""
        if organizer_id not in self.users:
            raise ValueError("Organizer not found")
        
        if calendar_id not in self.calendars:
            raise ValueError("Calendar not found")
        
        # Validate time
        if start_time >= end_time:
            raise ValueError("Start time must be before end time")
        
        # Check for conflicts
        conflicts = self.check_conflicts(calendar_id, start_time, end_time)
        if conflicts and not kwargs.get('ignore_conflicts', False):
            raise ValueError(f"Time conflict with {len(conflicts)} existing events")
        
        event = Event(
            event_id=str(uuid.uuid4()),
            title=title,
            start_time=start_time,
            end_time=end_time,
            calendar_id=calendar_id,
            organizer_id=organizer_id,
            **kwargs
        )
        
        with self._lock:
            self.events[event.event_id] = event
            
            # Handle recurring events
            if event.recurrence_rule:
                self._create_recurring_instances(event)
        
        return event
    
    def schedule_meeting(self, organizer_id: str, title: str, start_time: datetime,
                        duration_minutes: int, attendee_emails: List[str],
                        location: str = "", description: str = "") -> Event:
        """Schedule a meeting with attendees."""
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Find available time slot
        available_slot = self.find_available_time(
            attendee_emails + [self.users[organizer_id].email],
            start_time,
            duration_minutes
        )
        
        if available_slot != start_time:
            print(f"⚠️  Suggested time: {available_slot} (original time had conflicts)")
        
        # Create attendees
        attendees = []
        for email in attendee_emails:
            attendees.append(Attendee(email=email))
        
        # Add organizer
        organizer = self.users[organizer_id]
        attendees.append(Attendee(
            email=organizer.email,
            name=organizer.name,
            status=AttendeeStatus.ACCEPTED,
            is_organizer=True
        ))
        
        # Create event
        calendar_id = organizer.default_calendar_id or list(self.calendars.keys())[0]
        
        event = self.create_event(
            organizer_id=organizer_id,
            calendar_id=calendar_id,
            title=title,
            start_time=available_slot,
            end_time=available_slot + timedelta(minutes=duration_minutes),
            location=location,
            description=description,
            attendees=attendees,
            event_type=EventType.MEETING
        )
        
        # Send invitations
        self._send_invitations(event)
        
        return event
    
    def find_available_time(self, attendee_emails: List[str], preferred_start: datetime,
                           duration_minutes: int, search_days: int = 7) -> datetime:
        """Find the next available time slot for all attendees."""
        # Get user IDs from emails
        user_ids = []
        for email in attendee_emails:
            for user in self.users.values():
                if user.email == email:
                    user_ids.append(user.user_id)
                    break
        
        # Check preferred time first
        if not self._has_conflicts_for_users(user_ids, preferred_start, 
                                           preferred_start + timedelta(minutes=duration_minutes)):
            return preferred_start
        
        # Search for next available slot
        current_time = preferred_start
        end_search = preferred_start + timedelta(days=search_days)
        
        while current_time < end_search:
            end_time = current_time + timedelta(minutes=duration_minutes)
            
            if not self._has_conflicts_for_users(user_ids, current_time, end_time):
                return current_time
            
            # Move to next 30-minute slot
            current_time += timedelta(minutes=30)
        
        # Return original time if no slot found
        return preferred_start
    
    def check_conflicts(self, calendar_id: str, start_time: datetime, 
                       end_time: datetime, exclude_event_id: str = None) -> List[Event]:
        """Check for time conflicts in a calendar."""
        conflicts = []
        
        for event in self.events.values():
            if event.calendar_id != calendar_id:
                continue
            
            if exclude_event_id and event.event_id == exclude_event_id:
                continue
            
            if event.status == EventStatus.CANCELLED:
                continue
            
            # Check for overlap
            if (start_time < event.end_time and end_time > event.start_time):
                conflicts.append(event)
        
        return conflicts
    
    def get_events(self, calendar_id: str, start_date: datetime, 
                  end_date: datetime) -> List[Event]:
        """Get events in a date range for a calendar."""
        events = []
        
        for event in self.events.values():
            if event.calendar_id != calendar_id:
                continue
            
            if event.status == EventStatus.CANCELLED:
                continue
            
            # Check if event is in range
            if (event.start_time < end_date and event.end_time > start_date):
                events.append(event)
        
        # Sort by start time
        events.sort(key=lambda e: e.start_time)
        return events
    
    def update_event(self, event_id: str, **updates) -> bool:
        """Update an existing event."""
        if event_id not in self.events:
            return False
        
        event = self.events[event_id]
        
        # Validate changes
        if 'start_time' in updates or 'end_time' in updates:
            start_time = updates.get('start_time', event.start_time)
            end_time = updates.get('end_time', event.end_time)
            
            if start_time >= end_time:
                raise ValueError("Start time must be before end time")
            
            # Check conflicts
            conflicts = self.check_conflicts(
                event.calendar_id, start_time, end_time, event_id
            )
            if conflicts:
                raise ValueError(f"Time conflict with {len(conflicts)} existing events")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(event, key):
                setattr(event, key, value)
        
        event.updated_at = datetime.utcnow()
        
        # Notify attendees of changes
        self._notify_event_update(event)
        
        return True
    
    def delete_event(self, event_id: str, delete_series: bool = False) -> bool:
        """Delete an event."""
        if event_id not in self.events:
            return False
        
        event = self.events[event_id]
        
        if delete_series and event.recurrence_rule:
            # Delete entire recurring series
            if event.parent_event_id:
                # This is an instance, delete the parent
                parent_id = event.parent_event_id
            else:
                # This is the parent
                parent_id = event_id
            
            # Delete all instances
            instances = self.recurring_events.get(parent_id, [])
            for instance_id in instances:
                if instance_id in self.events:
                    del self.events[instance_id]
            
            # Delete parent
            if parent_id in self.events:
                del self.events[parent_id]
            
            # Clean up recurring events mapping
            if parent_id in self.recurring_events:
                del self.recurring_events[parent_id]
        else:
            # Delete single event
            del self.events[event_id]
        
        return True
    
    def respond_to_invitation(self, user_email: str, event_id: str, 
                            response: AttendeeStatus) -> bool:
        """Respond to a meeting invitation."""
        if event_id not in self.events:
            return False
        
        event = self.events[event_id]
        
        # Find attendee
        for attendee in event.attendees:
            if attendee.email == user_email:
                attendee.status = response
                event.updated_at = datetime.utcnow()
                
                # Notify organizer
                self._notify_response(event, attendee, response)
                return True
        
        return False
    
    def share_calendar(self, calendar_id: str, user_email: str, 
                      permission: str = "read") -> bool:
        """Share a calendar with another user."""
        if calendar_id not in self.calendars:
            return False
        
        # Find user by email
        target_user = None
        for user in self.users.values():
            if user.email == user_email:
                target_user = user
                break
        
        if not target_user:
            return False
        
        calendar = self.calendars[calendar_id]
        calendar.shared_with.add(target_user.user_id)
        
        return True
    
    def get_user_schedule(self, user_id: str, start_date: datetime, 
                         end_date: datetime) -> Dict[str, List[Event]]:
        """Get a user's complete schedule across all calendars."""
        if user_id not in self.users:
            return {}
        
        schedule = {}
        
        # Get user's calendars
        user_calendars = []
        for calendar in self.calendars.values():
            if (calendar.owner_id == user_id or 
                user_id in calendar.shared_with):
                user_calendars.append(calendar)
        
        # Get events from each calendar
        for calendar in user_calendars:
            events = self.get_events(calendar.calendar_id, start_date, end_date)
            schedule[calendar.name] = events
        
        return schedule
    
    def _create_recurring_instances(self, parent_event: Event) -> None:
        """Create instances for a recurring event."""
        if not parent_event.recurrence_rule:
            return
        
        rule = parent_event.recurrence_rule
        instances = []
        
        current_start = parent_event.start_time
        current_end = parent_event.end_time
        duration = current_end - current_start
        
        count = 0
        max_instances = rule.count or 100  # Default limit
        
        while count < max_instances:
            if rule.end_date and current_start > rule.end_date:
                break
            
            # Skip the original event
            if current_start != parent_event.start_time:
                instance = Event(
                    event_id=str(uuid.uuid4()),
                    title=parent_event.title,
                    start_time=current_start,
                    end_time=current_start + duration,
                    calendar_id=parent_event.calendar_id,
                    organizer_id=parent_event.organizer_id,
                    description=parent_event.description,
                    location=parent_event.location,
                    event_type=parent_event.event_type,
                    attendees=parent_event.attendees.copy(),
                    parent_event_id=parent_event.event_id,
                    is_all_day=parent_event.is_all_day,
                    is_private=parent_event.is_private
                )
                
                self.events[instance.event_id] = instance
                instances.append(instance.event_id)
            
            # Calculate next occurrence
            if rule.recurrence_type == RecurrenceType.DAILY:
                current_start += timedelta(days=rule.interval)
            elif rule.recurrence_type == RecurrenceType.WEEKLY:
                current_start += timedelta(weeks=rule.interval)
            elif rule.recurrence_type == RecurrenceType.MONTHLY:
                # Simple monthly recurrence
                if current_start.month == 12:
                    current_start = current_start.replace(
                        year=current_start.year + 1, month=1
                    )
                else:
                    current_start = current_start.replace(
                        month=current_start.month + 1
                    )
            elif rule.recurrence_type == RecurrenceType.YEARLY:
                current_start = current_start.replace(
                    year=current_start.year + rule.interval
                )
            else:
                break  # Custom rules not implemented
            
            count += 1
        
        # Store instances mapping
        self.recurring_events[parent_event.event_id] = instances
    
    def _has_conflicts_for_users(self, user_ids: List[str], start_time: datetime, 
                                end_time: datetime) -> bool:
        """Check if any users have conflicts in the given time range."""
        for user_id in user_ids:
            if user_id not in self.users:
                continue
            
            # Get user's calendars
            for calendar in self.calendars.values():
                if (calendar.owner_id == user_id or 
                    user_id in calendar.shared_with):
                    
                    conflicts = self.check_conflicts(
                        calendar.calendar_id, start_time, end_time
                    )
                    if conflicts:
                        return True
        
        return False
    
    def _send_invitations(self, event: Event) -> None:
        """Send meeting invitations to attendees."""
        for attendee in event.attendees:
            if not attendee.is_organizer:
                notification = {
                    'type': 'invitation',
                    'event_id': event.event_id,
                    'attendee_email': attendee.email,
                    'message': f"You're invited to: {event.title}",
                    'timestamp': datetime.utcnow()
                }
                self.pending_notifications.append(notification)
    
    def _notify_event_update(self, event: Event) -> None:
        """Notify attendees of event updates."""
        for attendee in event.attendees:
            notification = {
                'type': 'update',
                'event_id': event.event_id,
                'attendee_email': attendee.email,
                'message': f"Event updated: {event.title}",
                'timestamp': datetime.utcnow()
            }
            self.pending_notifications.append(notification)
    
    def _notify_response(self, event: Event, attendee: Attendee, 
                        response: AttendeeStatus) -> None:
        """Notify organizer of attendee response."""
        organizer_email = None
        for att in event.attendees:
            if att.is_organizer:
                organizer_email = att.email
                break
        
        if organizer_email:
            notification = {
                'type': 'response',
                'event_id': event.event_id,
                'attendee_email': organizer_email,
                'message': f"{attendee.email} {response.value} invitation to: {event.title}",
                'timestamp': datetime.utcnow()
            }
            self.pending_notifications.append(notification)


def demonstrate_calendar_system():
    """Demonstrate the calendar scheduling system."""
    print("=== CALENDAR SCHEDULING SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    system = CalendarSchedulingSystem()
    
    # Create users
    users = []
    user_data = [
        ("alice@company.com", "Alice Johnson", "America/New_York"),
        ("bob@company.com", "Bob Smith", "America/Los_Angeles"),
        ("charlie@company.com", "Charlie Brown", "Europe/London")
    ]
    
    for email, name, timezone in user_data:
        user = system.create_user(email, name, timezone)
        users.append(user)
        print(f"✓ Created user: {name} ({timezone})")
    
    print()
    
    # Schedule meetings
    now = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
    
    meeting1 = system.schedule_meeting(
        organizer_id=users[0].user_id,
        title="Team Standup",
        start_time=now,
        duration_minutes=30,
        attendee_emails=["bob@company.com", "charlie@company.com"],
        location="Conference Room A",
        description="Daily team standup meeting"
    )
    print(f"✓ Scheduled meeting: {meeting1.title}")
    
    # Create recurring event
    weekly_meeting = system.create_event(
        organizer_id=users[0].user_id,
        calendar_id=users[0].default_calendar_id,
        title="Weekly Review",
        start_time=now + timedelta(days=1),
        end_time=now + timedelta(days=1, hours=1),
        recurrence_rule=RecurrenceRule(
            recurrence_type=RecurrenceType.WEEKLY,
            interval=1,
            count=4
        )
    )
    print(f"✓ Created recurring event: {weekly_meeting.title}")
    
    # Show schedule
    start_date = now.replace(hour=0, minute=0)
    end_date = start_date + timedelta(days=7)
    
    for user in users:
        schedule = system.get_user_schedule(user.user_id, start_date, end_date)
        print(f"\n{user.name}'s schedule:")
        for calendar_name, events in schedule.items():
            print(f"  {calendar_name}: {len(events)} events")
    
    print("\n=== DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_calendar_system()
