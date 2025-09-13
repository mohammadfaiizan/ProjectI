"""
BRIDGE PATTERN - Structural Design Pattern
==========================================

Problem Statement:
Implement the Bridge pattern to separate abstraction from implementation:
- Decouple abstraction and implementation hierarchies
- Support multiple implementations for the same abstraction
- Enable runtime switching between implementations
- Cross-platform compatibility layers
- Plugin architecture with bridge pattern

Learning Objectives:
- Understand Bridge vs Adapter pattern differences
- Design separate abstraction and implementation hierarchies
- Implement runtime implementation switching
- Handle cross-platform compatibility
- Create flexible plugin architectures
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import time
import json
from datetime import datetime
from enum import Enum


# ============================================================================
# IMPLEMENTATION INTERFACES (Implementor)
# ============================================================================

class DrawingAPI(ABC):
    """Implementation interface for drawing operations."""
    
    @abstractmethod
    def draw_line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw a line from (x1,y1) to (x2,y2)."""
        pass
    
    @abstractmethod
    def draw_circle(self, x: float, y: float, radius: float) -> None:
        """Draw a circle at (x,y) with given radius."""
        pass
    
    @abstractmethod
    def draw_rectangle(self, x: float, y: float, width: float, height: float) -> None:
        """Draw a rectangle at (x,y) with given dimensions."""
        pass
    
    @abstractmethod
    def set_color(self, color: str) -> None:
        """Set drawing color."""
        pass
    
    @abstractmethod
    def set_line_width(self, width: float) -> None:
        """Set line width."""
        pass
    
    @abstractmethod
    def get_canvas_info(self) -> Dict[str, Any]:
        """Get canvas information."""
        pass


class NotificationSender(ABC):
    """Implementation interface for sending notifications."""
    
    @abstractmethod
    def send_message(self, recipient: str, subject: str, message: str) -> bool:
        """Send a message to recipient."""
        pass
    
    @abstractmethod
    def get_delivery_status(self, message_id: str) -> str:
        """Get delivery status of a message."""
        pass
    
    @abstractmethod
    def get_sender_info(self) -> Dict[str, Any]:
        """Get information about the sender implementation."""
        pass


class DatabaseDriver(ABC):
    """Implementation interface for database operations."""
    
    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        """Connect to database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from database."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        pass
    
    @abstractmethod
    def execute_update(self, query: str) -> int:
        """Execute an update query and return affected rows."""
        pass
    
    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a database transaction."""
        pass
    
    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit current transaction."""
        pass
    
    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback current transaction."""
        pass
    
    @abstractmethod
    def get_driver_info(self) -> Dict[str, Any]:
        """Get driver information."""
        pass


# ============================================================================
# CONCRETE IMPLEMENTATIONS (ConcreteImplementor)
# ============================================================================

class SVGDrawingAPI(DrawingAPI):
    """SVG implementation of drawing API."""
    
    def __init__(self):
        self.svg_elements = []
        self.current_color = "black"
        self.current_line_width = 1.0
        self.canvas_width = 800
        self.canvas_height = 600
        print("SVGDrawingAPI: Initialized SVG drawing implementation")
    
    def draw_line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw SVG line."""
        svg_line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" ' \
                  f'stroke="{self.current_color}" stroke-width="{self.current_line_width}"/>'
        self.svg_elements.append(svg_line)
        print(f"SVGDrawingAPI: Drew line from ({x1},{y1}) to ({x2},{y2})")
    
    def draw_circle(self, x: float, y: float, radius: float) -> None:
        """Draw SVG circle."""
        svg_circle = f'<circle cx="{x}" cy="{y}" r="{radius}" ' \
                    f'stroke="{self.current_color}" stroke-width="{self.current_line_width}" fill="none"/>'
        self.svg_elements.append(svg_circle)
        print(f"SVGDrawingAPI: Drew circle at ({x},{y}) with radius {radius}")
    
    def draw_rectangle(self, x: float, y: float, width: float, height: float) -> None:
        """Draw SVG rectangle."""
        svg_rect = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" ' \
                  f'stroke="{self.current_color}" stroke-width="{self.current_line_width}" fill="none"/>'
        self.svg_elements.append(svg_rect)
        print(f"SVGDrawingAPI: Drew rectangle at ({x},{y}) size {width}x{height}")
    
    def set_color(self, color: str) -> None:
        """Set SVG color."""
        self.current_color = color
        print(f"SVGDrawingAPI: Set color to {color}")
    
    def set_line_width(self, width: float) -> None:
        """Set SVG line width."""
        self.current_line_width = width
        print(f"SVGDrawingAPI: Set line width to {width}")
    
    def get_canvas_info(self) -> Dict[str, Any]:
        """Get SVG canvas information."""
        return {
            'type': 'SVG',
            'width': self.canvas_width,
            'height': self.canvas_height,
            'elements_count': len(self.svg_elements),
            'current_color': self.current_color,
            'current_line_width': self.current_line_width
        }
    
    def get_svg_content(self) -> str:
        """Get complete SVG content."""
        svg_header = f'<svg width="{self.canvas_width}" height="{self.canvas_height}" xmlns="http://www.w3.org/2000/svg">'
        svg_footer = '</svg>'
        return svg_header + '\n' + '\n'.join(self.svg_elements) + '\n' + svg_footer


class CanvasDrawingAPI(DrawingAPI):
    """HTML5 Canvas implementation of drawing API."""
    
    def __init__(self):
        self.canvas_commands = []
        self.current_color = "black"
        self.current_line_width = 1.0
        self.canvas_width = 800
        self.canvas_height = 600
        print("CanvasDrawingAPI: Initialized Canvas drawing implementation")
    
    def draw_line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw Canvas line."""
        commands = [
            "ctx.beginPath();",
            f"ctx.moveTo({x1}, {y1});",
            f"ctx.lineTo({x2}, {y2});",
            "ctx.stroke();"
        ]
        self.canvas_commands.extend(commands)
        print(f"CanvasDrawingAPI: Drew line from ({x1},{y1}) to ({x2},{y2})")
    
    def draw_circle(self, x: float, y: float, radius: float) -> None:
        """Draw Canvas circle."""
        commands = [
            "ctx.beginPath();",
            f"ctx.arc({x}, {y}, {radius}, 0, 2 * Math.PI);",
            "ctx.stroke();"
        ]
        self.canvas_commands.extend(commands)
        print(f"CanvasDrawingAPI: Drew circle at ({x},{y}) with radius {radius}")
    
    def draw_rectangle(self, x: float, y: float, width: float, height: float) -> None:
        """Draw Canvas rectangle."""
        commands = [
            f"ctx.strokeRect({x}, {y}, {width}, {height});"
        ]
        self.canvas_commands.extend(commands)
        print(f"CanvasDrawingAPI: Drew rectangle at ({x},{y}) size {width}x{height}")
    
    def set_color(self, color: str) -> None:
        """Set Canvas color."""
        self.current_color = color
        self.canvas_commands.append(f'ctx.strokeStyle = "{color}";')
        print(f"CanvasDrawingAPI: Set color to {color}")
    
    def set_line_width(self, width: float) -> None:
        """Set Canvas line width."""
        self.current_line_width = width
        self.canvas_commands.append(f"ctx.lineWidth = {width};")
        print(f"CanvasDrawingAPI: Set line width to {width}")
    
    def get_canvas_info(self) -> Dict[str, Any]:
        """Get Canvas information."""
        return {
            'type': 'HTML5_Canvas',
            'width': self.canvas_width,
            'height': self.canvas_height,
            'commands_count': len(self.canvas_commands),
            'current_color': self.current_color,
            'current_line_width': self.current_line_width
        }
    
    def get_canvas_script(self) -> str:
        """Get complete Canvas JavaScript."""
        script_header = f'''
        const canvas = document.createElement('canvas');
        canvas.width = {self.canvas_width};
        canvas.height = {self.canvas_height};
        const ctx = canvas.getContext('2d');
        '''
        return script_header + '\n' + '\n'.join(self.canvas_commands)


class EmailNotificationSender(NotificationSender):
    """Email implementation of notification sender."""
    
    def __init__(self, smtp_server: str, port: int = 587):
        self.smtp_server = smtp_server
        self.port = port
        self.sent_messages = {}
        self.message_counter = 1
        print(f"EmailNotificationSender: Initialized with {smtp_server}:{port}")
    
    def send_message(self, recipient: str, subject: str, message: str) -> bool:
        """Send email message."""
        message_id = f"EMAIL_{self.message_counter:06d}"
        self.message_counter += 1
        
        # Simulate email sending
        email_data = {
            'id': message_id,
            'recipient': recipient,
            'subject': subject,
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'status': 'sent',
            'delivery_attempts': 1
        }
        
        self.sent_messages[message_id] = email_data
        print(f"EmailNotificationSender: Sent email to {recipient} - Subject: {subject}")
        return True
    
    def get_delivery_status(self, message_id: str) -> str:
        """Get email delivery status."""
        if message_id in self.sent_messages:
            return self.sent_messages[message_id]['status']
        return 'not_found'
    
    def get_sender_info(self) -> Dict[str, Any]:
        """Get email sender information."""
        return {
            'type': 'Email',
            'smtp_server': self.smtp_server,
            'port': self.port,
            'messages_sent': len(self.sent_messages),
            'success_rate': 100.0  # Simulated
        }


class SMSNotificationSender(NotificationSender):
    """SMS implementation of notification sender."""
    
    def __init__(self, api_key: str, service_provider: str):
        self.api_key = api_key
        self.service_provider = service_provider
        self.sent_messages = {}
        self.message_counter = 1
        print(f"SMSNotificationSender: Initialized with {service_provider}")
    
    def send_message(self, recipient: str, subject: str, message: str) -> bool:
        """Send SMS message."""
        message_id = f"SMS_{self.message_counter:06d}"
        self.message_counter += 1
        
        # SMS doesn't use subject, combine with message
        full_message = f"{subject}: {message}" if subject else message
        
        # Simulate SMS sending
        sms_data = {
            'id': message_id,
            'recipient': recipient,
            'message': full_message,
            'sent_at': datetime.now().isoformat(),
            'status': 'delivered',
            'character_count': len(full_message)
        }
        
        self.sent_messages[message_id] = sms_data
        print(f"SMSNotificationSender: Sent SMS to {recipient} - {len(full_message)} chars")
        return True
    
    def get_delivery_status(self, message_id: str) -> str:
        """Get SMS delivery status."""
        if message_id in self.sent_messages:
            return self.sent_messages[message_id]['status']
        return 'not_found'
    
    def get_sender_info(self) -> Dict[str, Any]:
        """Get SMS sender information."""
        return {
            'type': 'SMS',
            'service_provider': self.service_provider,
            'messages_sent': len(self.sent_messages),
            'total_characters': sum(msg['character_count'] for msg in self.sent_messages.values())
        }


class MySQLDatabaseDriver(DatabaseDriver):
    """MySQL implementation of database driver."""
    
    def __init__(self):
        self.connection_string = ""
        self.is_connected = False
        self.in_transaction = False
        self.query_log = []
        print("MySQLDatabaseDriver: Initialized MySQL driver")
    
    def connect(self, connection_string: str) -> bool:
        """Connect to MySQL database."""
        self.connection_string = connection_string
        self.is_connected = True
        print(f"MySQLDatabaseDriver: Connected to {connection_string}")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from MySQL."""
        self.is_connected = False
        print("MySQLDatabaseDriver: Disconnected from MySQL")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute MySQL SELECT query."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        self.query_log.append({
            'query': query,
            'type': 'SELECT',
            'timestamp': datetime.now().isoformat()
        })
        
        # Simulate query results
        if 'users' in query.lower():
            results = [
                {'id': 1, 'name': 'John Doe', 'email': 'john@example.com'},
                {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com'}
            ]
        else:
            results = [{'result': 'simulated_data'}]
        
        print(f"MySQLDatabaseDriver: Executed SELECT query, returned {len(results)} rows")
        return results
    
    def execute_update(self, query: str) -> int:
        """Execute MySQL UPDATE/INSERT/DELETE query."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        query_type = query.strip().split()[0].upper()
        self.query_log.append({
            'query': query,
            'type': query_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Simulate affected rows
        affected_rows = 1 if query_type in ['INSERT', 'UPDATE', 'DELETE'] else 0
        print(f"MySQLDatabaseDriver: Executed {query_type} query, affected {affected_rows} rows")
        return affected_rows
    
    def begin_transaction(self) -> None:
        """Begin MySQL transaction."""
        self.in_transaction = True
        print("MySQLDatabaseDriver: Transaction started")
    
    def commit_transaction(self) -> None:
        """Commit MySQL transaction."""
        if self.in_transaction:
            self.in_transaction = False
            print("MySQLDatabaseDriver: Transaction committed")
    
    def rollback_transaction(self) -> None:
        """Rollback MySQL transaction."""
        if self.in_transaction:
            self.in_transaction = False
            print("MySQLDatabaseDriver: Transaction rolled back")
    
    def get_driver_info(self) -> Dict[str, Any]:
        """Get MySQL driver information."""
        return {
            'type': 'MySQL',
            'version': '8.0.x',
            'is_connected': self.is_connected,
            'in_transaction': self.in_transaction,
            'queries_executed': len(self.query_log)
        }


class PostgreSQLDatabaseDriver(DatabaseDriver):
    """PostgreSQL implementation of database driver."""
    
    def __init__(self):
        self.connection_string = ""
        self.is_connected = False
        self.in_transaction = False
        self.query_log = []
        print("PostgreSQLDatabaseDriver: Initialized PostgreSQL driver")
    
    def connect(self, connection_string: str) -> bool:
        """Connect to PostgreSQL database."""
        self.connection_string = connection_string
        self.is_connected = True
        print(f"PostgreSQLDatabaseDriver: Connected to {connection_string}")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        self.is_connected = False
        print("PostgreSQLDatabaseDriver: Disconnected from PostgreSQL")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute PostgreSQL SELECT query."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        self.query_log.append({
            'query': query,
            'type': 'SELECT',
            'timestamp': datetime.now().isoformat(),
            'execution_plan': 'simulated_plan'
        })
        
        # Simulate query results with PostgreSQL-specific features
        if 'users' in query.lower():
            results = [
                {'id': 1, 'name': 'John Doe', 'email': 'john@example.com', 'created_at': '2024-01-01'},
                {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com', 'created_at': '2024-01-02'}
            ]
        else:
            results = [{'result': 'postgresql_data', 'uuid': 'abc-123-def'}]
        
        print(f"PostgreSQLDatabaseDriver: Executed SELECT query, returned {len(results)} rows")
        return results
    
    def execute_update(self, query: str) -> int:
        """Execute PostgreSQL UPDATE/INSERT/DELETE query."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        query_type = query.strip().split()[0].upper()
        self.query_log.append({
            'query': query,
            'type': query_type,
            'timestamp': datetime.now().isoformat(),
            'execution_plan': 'simulated_plan'
        })
        
        # Simulate affected rows
        affected_rows = 1 if query_type in ['INSERT', 'UPDATE', 'DELETE'] else 0
        print(f"PostgreSQLDatabaseDriver: Executed {query_type} query, affected {affected_rows} rows")
        return affected_rows
    
    def begin_transaction(self) -> None:
        """Begin PostgreSQL transaction."""
        self.in_transaction = True
        print("PostgreSQLDatabaseDriver: Transaction started with ACID compliance")
    
    def commit_transaction(self) -> None:
        """Commit PostgreSQL transaction."""
        if self.in_transaction:
            self.in_transaction = False
            print("PostgreSQLDatabaseDriver: Transaction committed with full ACID compliance")
    
    def rollback_transaction(self) -> None:
        """Rollback PostgreSQL transaction."""
        if self.in_transaction:
            self.in_transaction = False
            print("PostgreSQLDatabaseDriver: Transaction rolled back with full recovery")
    
    def get_driver_info(self) -> Dict[str, Any]:
        """Get PostgreSQL driver information."""
        return {
            'type': 'PostgreSQL',
            'version': '14.x',
            'is_connected': self.is_connected,
            'in_transaction': self.in_transaction,
            'queries_executed': len(self.query_log),
            'features': ['ACID', 'JSON', 'Arrays', 'Full-text search']
        }


# ============================================================================
# ABSTRACTIONS (Abstraction)
# ============================================================================

class Shape(ABC):
    """Abstract shape class (Abstraction)."""
    
    def __init__(self, drawing_api: DrawingAPI):
        self.drawing_api = drawing_api
        self.x = 0.0
        self.y = 0.0
    
    def set_position(self, x: float, y: float) -> None:
        """Set shape position."""
        self.x = x
        self.y = y
    
    @abstractmethod
    def draw(self) -> None:
        """Draw the shape."""
        pass
    
    @abstractmethod
    def get_area(self) -> float:
        """Calculate shape area."""
        pass
    
    def set_style(self, color: str, line_width: float) -> None:
        """Set shape style."""
        self.drawing_api.set_color(color)
        self.drawing_api.set_line_width(line_width)


class Notification(ABC):
    """Abstract notification class (Abstraction)."""
    
    def __init__(self, sender: NotificationSender):
        self.sender = sender
        self.default_priority = "normal"
    
    @abstractmethod
    def send(self, recipient: str, content: str) -> bool:
        """Send notification."""
        pass
    
    def get_sender_type(self) -> str:
        """Get sender implementation type."""
        info = self.sender.get_sender_info()
        return info.get('type', 'unknown')
    
    def check_delivery_status(self, message_id: str) -> str:
        """Check delivery status."""
        return self.sender.get_delivery_status(message_id)


class Database(ABC):
    """Abstract database class (Abstraction)."""
    
    def __init__(self, driver: DatabaseDriver):
        self.driver = driver
        self.connection_pool_size = 10
    
    @abstractmethod
    def initialize(self, connection_string: str) -> bool:
        """Initialize database connection."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup database resources."""
        pass
    
    def get_driver_type(self) -> str:
        """Get driver implementation type."""
        info = self.driver.get_driver_info()
        return info.get('type', 'unknown')


# ============================================================================
# REFINED ABSTRACTIONS (RefinedAbstraction)
# ============================================================================

class Circle(Shape):
    """Circle shape (RefinedAbstraction)."""
    
    def __init__(self, drawing_api: DrawingAPI, radius: float):
        super().__init__(drawing_api)
        self.radius = radius
    
    def draw(self) -> None:
        """Draw circle using the drawing API."""
        self.drawing_api.draw_circle(self.x, self.y, self.radius)
    
    def get_area(self) -> float:
        """Calculate circle area."""
        return 3.14159 * self.radius * self.radius
    
    def set_radius(self, radius: float) -> None:
        """Set circle radius."""
        self.radius = radius


class Rectangle(Shape):
    """Rectangle shape (RefinedAbstraction)."""
    
    def __init__(self, drawing_api: DrawingAPI, width: float, height: float):
        super().__init__(drawing_api)
        self.width = width
        self.height = height
    
    def draw(self) -> None:
        """Draw rectangle using the drawing API."""
        self.drawing_api.draw_rectangle(self.x, self.y, self.width, self.height)
    
    def get_area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height
    
    def set_dimensions(self, width: float, height: float) -> None:
        """Set rectangle dimensions."""
        self.width = width
        self.height = height


class Line(Shape):
    """Line shape (RefinedAbstraction)."""
    
    def __init__(self, drawing_api: DrawingAPI, end_x: float, end_y: float):
        super().__init__(drawing_api)
        self.end_x = end_x
        self.end_y = end_y
    
    def draw(self) -> None:
        """Draw line using the drawing API."""
        self.drawing_api.draw_line(self.x, self.y, self.end_x, self.end_y)
    
    def get_area(self) -> float:
        """Lines have no area."""
        return 0.0
    
    def get_length(self) -> float:
        """Calculate line length."""
        dx = self.end_x - self.x
        dy = self.end_y - self.y
        return (dx * dx + dy * dy) ** 0.5
    
    def set_end_point(self, end_x: float, end_y: float) -> None:
        """Set line end point."""
        self.end_x = end_x
        self.end_y = end_y


class AlertNotification(Notification):
    """Alert notification (RefinedAbstraction)."""
    
    def __init__(self, sender: NotificationSender):
        super().__init__(sender)
        self.alert_level = "warning"
    
    def send(self, recipient: str, content: str) -> bool:
        """Send alert notification."""
        subject = f"ALERT [{self.alert_level.upper()}]"
        formatted_content = f"⚠️ ALERT: {content}\n\nThis is an automated alert notification."
        
        return self.sender.send_message(recipient, subject, formatted_content)
    
    def set_alert_level(self, level: str) -> None:
        """Set alert level."""
        self.alert_level = level


class MarketingNotification(Notification):
    """Marketing notification (RefinedAbstraction)."""
    
    def __init__(self, sender: NotificationSender):
        super().__init__(sender)
        self.campaign_id = ""
    
    def send(self, recipient: str, content: str) -> bool:
        """Send marketing notification."""
        subject = "Special Offer - Don't Miss Out!"
        formatted_content = f"🎉 {content}\n\nCampaign ID: {self.campaign_id}\n\nUnsubscribe: [link]"
        
        return self.sender.send_message(recipient, subject, formatted_content)
    
    def set_campaign(self, campaign_id: str) -> None:
        """Set marketing campaign ID."""
        self.campaign_id = campaign_id


class UserDatabase(Database):
    """User database (RefinedAbstraction)."""
    
    def __init__(self, driver: DatabaseDriver):
        super().__init__(driver)
        self.table_name = "users"
    
    def initialize(self, connection_string: str) -> bool:
        """Initialize user database."""
        success = self.driver.connect(connection_string)
        if success:
            # Create users table if it doesn't exist
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.driver.execute_update(create_table_query)
        return success
    
    def cleanup(self) -> None:
        """Cleanup user database."""
        self.driver.disconnect()
    
    def create_user(self, name: str, email: str) -> int:
        """Create a new user."""
        query = f"INSERT INTO {self.table_name} (name, email) VALUES ('{name}', '{email}')"
        return self.driver.execute_update(query)
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        query = f"SELECT * FROM {self.table_name} WHERE id = {user_id}"
        results = self.driver.execute_query(query)
        return results[0] if results else None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        query = f"SELECT * FROM {self.table_name}"
        return self.driver.execute_query(query)
    
    def update_user(self, user_id: int, name: str = None, email: str = None) -> int:
        """Update user information."""
        updates = []
        if name:
            updates.append(f"name = '{name}'")
        if email:
            updates.append(f"email = '{email}'")
        
        if updates:
            query = f"UPDATE {self.table_name} SET {', '.join(updates)} WHERE id = {user_id}"
            return self.driver.execute_update(query)
        return 0
    
    def delete_user(self, user_id: int) -> int:
        """Delete user."""
        query = f"DELETE FROM {self.table_name} WHERE id = {user_id}"
        return self.driver.execute_update(query)


class ProductDatabase(Database):
    """Product database (RefinedAbstraction)."""
    
    def __init__(self, driver: DatabaseDriver):
        super().__init__(driver)
        self.table_name = "products"
    
    def initialize(self, connection_string: str) -> bool:
        """Initialize product database."""
        success = self.driver.connect(connection_string)
        if success:
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                category VARCHAR(100),
                stock_quantity INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.driver.execute_update(create_table_query)
        return success
    
    def cleanup(self) -> None:
        """Cleanup product database."""
        self.driver.disconnect()
    
    def add_product(self, name: str, price: float, category: str, stock: int = 0) -> int:
        """Add a new product."""
        query = f"INSERT INTO {self.table_name} (name, price, category, stock_quantity) VALUES ('{name}', {price}, '{category}', {stock})"
        return self.driver.execute_update(query)
    
    def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get products by category."""
        query = f"SELECT * FROM {self.table_name} WHERE category = '{category}'"
        return self.driver.execute_query(query)
    
    def update_stock(self, product_id: int, quantity: int) -> int:
        """Update product stock."""
        query = f"UPDATE {self.table_name} SET stock_quantity = {quantity} WHERE id = {product_id}"
        return self.driver.execute_update(query)


# ============================================================================
# BRIDGE PATTERN MANAGER
# ============================================================================

class BridgePatternManager:
    """Manager for demonstrating bridge pattern flexibility."""
    
    def __init__(self):
        self.drawing_apis = {}
        self.notification_senders = {}
        self.database_drivers = {}
    
    def register_drawing_api(self, name: str, api: DrawingAPI) -> None:
        """Register a drawing API implementation."""
        self.drawing_apis[name] = api
        print(f"BridgePatternManager: Registered drawing API '{name}'")
    
    def register_notification_sender(self, name: str, sender: NotificationSender) -> None:
        """Register a notification sender implementation."""
        self.notification_senders[name] = sender
        print(f"BridgePatternManager: Registered notification sender '{name}'")
    
    def register_database_driver(self, name: str, driver: DatabaseDriver) -> None:
        """Register a database driver implementation."""
        self.database_drivers[name] = driver
        print(f"BridgePatternManager: Registered database driver '{name}'")
    
    def create_shape(self, shape_type: str, api_name: str, **kwargs) -> Optional[Shape]:
        """Create a shape with specified drawing API."""
        api = self.drawing_apis.get(api_name)
        if not api:
            print(f"Drawing API '{api_name}' not found")
            return None
        
        if shape_type == "circle":
            return Circle(api, kwargs.get('radius', 50))
        elif shape_type == "rectangle":
            return Rectangle(api, kwargs.get('width', 100), kwargs.get('height', 80))
        elif shape_type == "line":
            return Line(api, kwargs.get('end_x', 100), kwargs.get('end_y', 100))
        else:
            print(f"Unknown shape type: {shape_type}")
            return None
    
    def create_notification(self, notification_type: str, sender_name: str) -> Optional[Notification]:
        """Create a notification with specified sender."""
        sender = self.notification_senders.get(sender_name)
        if not sender:
            print(f"Notification sender '{sender_name}' not found")
            return None
        
        if notification_type == "alert":
            return AlertNotification(sender)
        elif notification_type == "marketing":
            return MarketingNotification(sender)
        else:
            print(f"Unknown notification type: {notification_type}")
            return None
    
    def create_database(self, database_type: str, driver_name: str) -> Optional[Database]:
        """Create a database with specified driver."""
        driver = self.database_drivers.get(driver_name)
        if not driver:
            print(f"Database driver '{driver_name}' not found")
            return None
        
        if database_type == "user":
            return UserDatabase(driver)
        elif database_type == "product":
            return ProductDatabase(driver)
        else:
            print(f"Unknown database type: {database_type}")
            return None
    
    def get_available_implementations(self) -> Dict[str, List[str]]:
        """Get all available implementations."""
        return {
            'drawing_apis': list(self.drawing_apis.keys()),
            'notification_senders': list(self.notification_senders.keys()),
            'database_drivers': list(self.database_drivers.keys())
        }


def demonstrate_bridge_pattern():
    """
    Demonstrate Bridge pattern implementations.
    """
    print("=== BRIDGE PATTERN DEMONSTRATION ===\n")
    
    # 1. Basic Bridge Pattern - Drawing System
    print("1. BASIC BRIDGE PATTERN - DRAWING SYSTEM:")
    
    # Create different drawing API implementations
    svg_api = SVGDrawingAPI()
    canvas_api = CanvasDrawingAPI()
    
    # Create shapes with different implementations
    svg_circle = Circle(svg_api, 50)
    svg_circle.set_position(100, 100)
    svg_circle.set_style("red", 2.0)
    
    canvas_rectangle = Rectangle(canvas_api, 150, 100)
    canvas_rectangle.set_position(200, 150)
    canvas_rectangle.set_style("blue", 3.0)
    
    # Draw shapes
    print("   Drawing shapes with different APIs:")
    svg_circle.draw()
    canvas_rectangle.draw()
    
    # Show API information
    print(f"\n   SVG API info: {svg_api.get_canvas_info()}")
    print(f"   Canvas API info: {canvas_api.get_canvas_info()}")
    
    # Calculate areas
    print(f"   Circle area: {svg_circle.get_area():.2f}")
    print(f"   Rectangle area: {canvas_rectangle.get_area():.2f}")
    print()
    
    # 2. Runtime Implementation Switching
    print("2. RUNTIME IMPLEMENTATION SWITCHING:")
    
    # Create a shape and switch its implementation
    line = Line(svg_api, 300, 200)
    line.set_position(50, 50)
    line.set_style("green", 1.5)
    
    print("   Drawing line with SVG API:")
    line.draw()
    print(f"   Line length: {line.get_length():.2f}")
    
    # Switch to Canvas API
    line.drawing_api = canvas_api
    line.set_style("purple", 2.5)  # Re-apply style for new API
    
    print("\n   Drawing same line with Canvas API:")
    line.draw()
    print()
    
    # 3. Notification System Bridge
    print("3. NOTIFICATION SYSTEM BRIDGE:")
    
    # Create different notification senders
    email_sender = EmailNotificationSender("smtp.example.com", 587)
    sms_sender = SMSNotificationSender("api_key_123", "TwilioSMS")
    
    # Create notifications with different senders
    email_alert = AlertNotification(email_sender)
    email_alert.set_alert_level("critical")
    
    sms_marketing = MarketingNotification(sms_sender)
    sms_marketing.set_campaign("SUMMER2024")
    
    # Send notifications
    print("   Sending notifications through different channels:")
    
    email_success = email_alert.send("admin@example.com", "Server CPU usage exceeded 90%")
    print(f"   Email alert sent: {email_success}")
    print(f"   Email sender type: {email_alert.get_sender_type()}")
    
    sms_success = sms_marketing.send("+1234567890", "Get 50% off summer collection!")
    print(f"   SMS marketing sent: {sms_success}")
    print(f"   SMS sender type: {sms_marketing.get_sender_type()}")
    
    # Show sender information
    print(f"\n   Email sender info: {email_sender.get_sender_info()}")
    print(f"   SMS sender info: {sms_sender.get_sender_info()}")
    print()
    
    # 4. Database System Bridge
    print("4. DATABASE SYSTEM BRIDGE:")
    
    # Create different database drivers
    mysql_driver = MySQLDatabaseDriver()
    postgres_driver = PostgreSQLDatabaseDriver()
    
    # Create databases with different drivers
    mysql_users = UserDatabase(mysql_driver)
    postgres_products = ProductDatabase(postgres_driver)
    
    # Initialize databases
    print("   Initializing databases with different drivers:")
    mysql_users.initialize("mysql://localhost:3306/userdb")
    postgres_products.initialize("postgresql://localhost:5432/productdb")
    
    print(f"   MySQL users DB driver: {mysql_users.get_driver_type()}")
    print(f"   PostgreSQL products DB driver: {postgres_products.get_driver_type()}")
    
    # Perform database operations
    print("\n   Performing database operations:")
    
    # User operations with MySQL
    mysql_users.create_user("Alice Johnson", "alice@example.com")
    mysql_users.create_user("Bob Wilson", "bob@example.com")
    users = mysql_users.get_all_users()
    print(f"   MySQL users retrieved: {len(users)}")
    
    # Product operations with PostgreSQL
    postgres_products.add_product("Laptop", 999.99, "Electronics", 50)
    postgres_products.add_product("Mouse", 29.99, "Electronics", 100)
    electronics = postgres_products.get_products_by_category("Electronics")
    print(f"   PostgreSQL electronics retrieved: {len(electronics)}")
    
    # Show driver information
    print(f"\n   MySQL driver info: {mysql_driver.get_driver_info()}")
    print(f"   PostgreSQL driver info: {postgres_driver.get_driver_info()}")
    
    # Cleanup
    mysql_users.cleanup()
    postgres_products.cleanup()
    print()
    
    # 5. Bridge Pattern Manager
    print("5. BRIDGE PATTERN MANAGER:")
    
    manager = BridgePatternManager()
    
    # Register implementations
    manager.register_drawing_api("svg", svg_api)
    manager.register_drawing_api("canvas", canvas_api)
    manager.register_notification_sender("email", email_sender)
    manager.register_notification_sender("sms", sms_sender)
    manager.register_database_driver("mysql", mysql_driver)
    manager.register_database_driver("postgresql", postgres_driver)
    
    # Show available implementations
    implementations = manager.get_available_implementations()
    print(f"\n   Available implementations:")
    for category, impl_list in implementations.items():
        print(f"     {category}: {impl_list}")
    
    # Create objects through manager
    print("\n   Creating objects through manager:")
    
    # Create shapes with different APIs
    svg_rect = manager.create_shape("rectangle", "svg", width=200, height=150)
    canvas_circle = manager.create_shape("circle", "canvas", radius=75)
    
    if svg_rect and canvas_circle:
        svg_rect.set_position(50, 50)
        svg_rect.set_style("orange", 2.0)
        svg_rect.draw()
        
        canvas_circle.set_position(300, 200)
        canvas_circle.set_style("cyan", 1.5)
        canvas_circle.draw()
    
    # Create notifications with different senders
    email_marketing = manager.create_notification("marketing", "email")
    sms_alert = manager.create_notification("alert", "sms")
    
    if email_marketing and sms_alert:
        email_marketing.set_campaign("WINTER2024")
        email_marketing.send("customer@example.com", "Winter sale starts now!")
        
        sms_alert.set_alert_level("high")
        sms_alert.send("+1987654321", "System maintenance in 10 minutes")
    
    print()
    
    # 6. Cross-Platform Compatibility
    print("6. CROSS-PLATFORM COMPATIBILITY:")
    
    # Demonstrate how bridge pattern enables cross-platform compatibility
    platforms = [
        ("Web (SVG)", svg_api),
        ("Web (Canvas)", canvas_api)
    ]
    
    # Create the same drawing on different platforms
    for platform_name, api in platforms:
        print(f"\n   Drawing on {platform_name}:")
        
        # Create a simple drawing
        circle = Circle(api, 40)
        circle.set_position(100, 100)
        circle.set_style("red", 2.0)
        
        rect = Rectangle(api, 80, 60)
        rect.set_position(150, 80)
        rect.set_style("blue", 1.5)
        
        line = Line(api, 200, 150)
        line.set_position(50, 50)
        line.set_style("green", 1.0)
        
        # Draw all shapes
        circle.draw()
        rect.draw()
        line.draw()
        
        # Show platform-specific output
        if isinstance(api, SVGDrawingAPI):
            svg_content = api.get_svg_content()
            print(f"     SVG output: {len(svg_content)} characters")
        elif isinstance(api, CanvasDrawingAPI):
            canvas_script = api.get_canvas_script()
            print(f"     Canvas script: {len(canvas_script)} characters")
    
    print()
    
    # 7. Bridge vs Adapter Comparison
    print("7. BRIDGE VS ADAPTER COMPARISON:")
    print("   BRIDGE PATTERN:")
    print("   ✓ Separates abstraction from implementation")
    print("   ✓ Both hierarchies can evolve independently")
    print("   ✓ Implementation can be switched at runtime")
    print("   ✓ Designed upfront for flexibility")
    print("   ✓ One-to-many relationship (abstraction to implementations)")
    print()
    print("   ADAPTER PATTERN:")
    print("   ✓ Makes incompatible interfaces work together")
    print("   ✓ Usually applied to existing code")
    print("   ✓ Focuses on interface compatibility")
    print("   ✓ Often used for legacy system integration")
    print("   ✓ One-to-one relationship (adaptee to target)")
    print()
    
    # 8. Bridge Pattern Benefits
    print("8. BRIDGE PATTERN BENEFITS:")
    print("   ✓ Separation of Concerns: Abstraction and implementation separated")
    print("   ✓ Runtime Flexibility: Implementation can be changed at runtime")
    print("   ✓ Platform Independence: Same abstraction works across platforms")
    print("   ✓ Extensibility: New implementations can be added easily")
    print("   ✓ Maintainability: Changes in implementation don't affect abstraction")
    print("   ✓ Testability: Implementations can be mocked for testing")
    print("   ✓ Code Reuse: Same abstraction works with multiple implementations")
    print("   ✓ Plugin Architecture: Supports pluggable implementations")
    print()
    
    print("=== BRIDGE PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_bridge_pattern()
