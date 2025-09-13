"""
LOGGER FRAMEWORK DESIGN - Complete System Design
===============================================

Problem Statement:
Design a comprehensive logging framework that handles:
- Multiple log levels (DEBUG, INFO, WARN, ERROR, FATAL)
- Different output destinations (console, file, database, network)
- Log formatting and structured logging (JSON, XML, custom formats)
- Log rotation and archival policies
- Asynchronous logging for high-performance scenarios
- Hierarchical loggers with inheritance
- Log filtering and conditional logging
- Performance monitoring and metrics collection
- Distributed logging across multiple services
- Log aggregation and centralized logging

Requirements:
- Support multiple log levels with configurable thresholds
- Implement various appenders (console, file, database, remote)
- Provide flexible log formatting options
- Handle log rotation based on size, time, or custom criteria
- Support asynchronous logging to avoid blocking operations
- Implement hierarchical logger structure
- Provide filtering mechanisms for selective logging
- Include performance metrics and monitoring
- Support distributed tracing and correlation IDs
- Handle logging configuration via files or programmatically

Design Patterns Used:
- Factory: Logger and appender creation
- Strategy: Different formatting and output strategies
- Observer: Log event propagation
- Decorator: Log formatting and filtering
- Chain of Responsibility: Log level filtering
- Singleton: Logger manager
- Template Method: Log processing pipeline
- Command: Log operations with batching
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import json
import xml.etree.ElementTree as ET
import os
import gzip
import sqlite3
import socket
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import concurrent.futures


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class LogLevel(Enum):
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50
    
    def __ge__(self, other):
        return self.value >= other.value
    
    def __gt__(self, other):
        return self.value > other.value
    
    def __le__(self, other):
        return self.value <= other.value
    
    def __lt__(self, other):
        return self.value < other.value


class RotationPolicy(Enum):
    SIZE = "size"
    TIME = "time"
    COUNT = "count"
    NEVER = "never"


@dataclass
class LogRecord:
    """Log record containing all logging information."""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    thread_id: int
    process_id: int
    correlation_id: Optional[str] = None
    exception: Optional[Exception] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    source_function: Optional[str] = None
    
    def __post_init__(self):
        if self.correlation_id is None:
            self.correlation_id = str(uuid.uuid4())


@dataclass
class LoggerConfig:
    """Logger configuration."""
    name: str
    level: LogLevel = LogLevel.INFO
    propagate: bool = True
    appenders: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    
    
@dataclass
class AppenderConfig:
    """Appender configuration."""
    name: str
    type: str
    level: LogLevel = LogLevel.INFO
    formatter: str = "default"
    filters: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# LOG FORMATTERS
# ============================================================================

class LogFormatter(ABC):
    """Abstract log formatter."""
    
    @abstractmethod
    def format(self, record: LogRecord) -> str:
        """Format log record into string."""
        pass


class SimpleFormatter(LogFormatter):
    """Simple text formatter."""
    
    def __init__(self, pattern: str = "{timestamp} [{level}] {logger}: {message}"):
        self.pattern = pattern
    
    def format(self, record: LogRecord) -> str:
        """Format record using simple pattern."""
        return self.pattern.format(
            timestamp=record.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            level=record.level.name,
            logger=record.logger_name,
            message=record.message,
            thread=record.thread_id,
            process=record.process_id,
            correlation_id=record.correlation_id or "",
            **record.extra_fields
        )


class DetailedFormatter(LogFormatter):
    """Detailed formatter with more information."""
    
    def format(self, record: LogRecord) -> str:
        """Format record with detailed information."""
        formatted = f"{record.timestamp.isoformat()} "
        formatted += f"[{record.level.name:5}] "
        formatted += f"{record.logger_name} "
        formatted += f"(PID:{record.process_id}, TID:{record.thread_id}) "
        
        if record.correlation_id:
            formatted += f"[{record.correlation_id[:8]}] "
        
        formatted += f"- {record.message}"
        
        if record.source_file:
            formatted += f" ({record.source_file}:{record.source_line})"
        
        if record.extra_fields:
            extras = ", ".join(f"{k}={v}" for k, v in record.extra_fields.items())
            formatted += f" [{extras}]"
        
        if record.exception:
            formatted += f"\nException: {record.exception}"
            formatted += f"\n{''.join(traceback.format_tb(record.exception.__traceback__))}"
        
        return formatted


class JSONFormatter(LogFormatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: LogRecord) -> str:
        """Format record as JSON."""
        log_dict = {
            "timestamp": record.timestamp.isoformat(),
            "level": record.level.name,
            "logger": record.logger_name,
            "message": record.message,
            "thread_id": record.thread_id,
            "process_id": record.process_id,
            "correlation_id": record.correlation_id
        }
        
        # Add source information if available
        if record.source_file:
            log_dict["source"] = {
                "file": record.source_file,
                "line": record.source_line,
                "function": record.source_function
            }
        
        # Add exception information
        if record.exception:
            log_dict["exception"] = {
                "type": type(record.exception).__name__,
                "message": str(record.exception),
                "traceback": traceback.format_exception(
                    type(record.exception),
                    record.exception,
                    record.exception.__traceback__
                )
            }
        
        # Add extra fields
        if record.extra_fields:
            log_dict["extra"] = record.extra_fields
        
        return json.dumps(log_dict)


class XMLFormatter(LogFormatter):
    """XML formatter for structured logging."""
    
    def format(self, record: LogRecord) -> str:
        """Format record as XML."""
        root = ET.Element("logEntry")
        
        # Basic fields
        ET.SubElement(root, "timestamp").text = record.timestamp.isoformat()
        ET.SubElement(root, "level").text = record.level.name
        ET.SubElement(root, "logger").text = record.logger_name
        ET.SubElement(root, "message").text = record.message
        ET.SubElement(root, "threadId").text = str(record.thread_id)
        ET.SubElement(root, "processId").text = str(record.process_id)
        
        if record.correlation_id:
            ET.SubElement(root, "correlationId").text = record.correlation_id
        
        # Source information
        if record.source_file:
            source = ET.SubElement(root, "source")
            ET.SubElement(source, "file").text = record.source_file
            ET.SubElement(source, "line").text = str(record.source_line)
            ET.SubElement(source, "function").text = record.source_function
        
        # Exception information
        if record.exception:
            exception = ET.SubElement(root, "exception")
            ET.SubElement(exception, "type").text = type(record.exception).__name__
            ET.SubElement(exception, "message").text = str(record.exception)
        
        # Extra fields
        if record.extra_fields:
            extra = ET.SubElement(root, "extra")
            for key, value in record.extra_fields.items():
                field = ET.SubElement(extra, "field")
                field.set("name", key)
                field.text = str(value)
        
        return ET.tostring(root, encoding='unicode')


# ============================================================================
# LOG FILTERS
# ============================================================================

class LogFilter(ABC):
    """Abstract log filter."""
    
    @abstractmethod
    def filter(self, record: LogRecord) -> bool:
        """Return True if record should be logged."""
        pass


class LevelFilter(LogFilter):
    """Filter based on log level."""
    
    def __init__(self, min_level: LogLevel, max_level: LogLevel = LogLevel.FATAL):
        self.min_level = min_level
        self.max_level = max_level
    
    def filter(self, record: LogRecord) -> bool:
        """Filter by level range."""
        return self.min_level <= record.level <= self.max_level


class ThreadFilter(LogFilter):
    """Filter based on thread ID."""
    
    def __init__(self, thread_ids: List[int]):
        self.thread_ids = set(thread_ids)
    
    def filter(self, record: LogRecord) -> bool:
        """Filter by thread ID."""
        return record.thread_id in self.thread_ids


class RegexFilter(LogFilter):
    """Filter based on message regex pattern."""
    
    def __init__(self, pattern: str, match_required: bool = True):
        import re
        self.pattern = re.compile(pattern)
        self.match_required = match_required
    
    def filter(self, record: LogRecord) -> bool:
        """Filter by regex pattern."""
        matches = bool(self.pattern.search(record.message))
        return matches if self.match_required else not matches


class RateLimitFilter(LogFilter):
    """Rate limiting filter."""
    
    def __init__(self, max_messages: int, time_window: timedelta):
        self.max_messages = max_messages
        self.time_window = time_window
        self.message_times: List[datetime] = []
        self._lock = threading.Lock()
    
    def filter(self, record: LogRecord) -> bool:
        """Filter by rate limit."""
        with self._lock:
            now = record.timestamp
            
            # Remove old messages outside time window
            cutoff = now - self.time_window
            self.message_times = [t for t in self.message_times if t > cutoff]
            
            # Check if under limit
            if len(self.message_times) < self.max_messages:
                self.message_times.append(now)
                return True
            
            return False


# ============================================================================
# LOG APPENDERS
# ============================================================================

class LogAppender(ABC):
    """Abstract log appender."""
    
    def __init__(self, name: str, formatter: LogFormatter, filters: List[LogFilter] = None):
        self.name = name
        self.formatter = formatter
        self.filters = filters or []
        self._lock = threading.Lock()
    
    def append(self, record: LogRecord) -> None:
        """Append log record if it passes filters."""
        # Apply filters
        for filter_obj in self.filters:
            if not filter_obj.filter(record):
                return
        
        # Format and write
        formatted_message = self.formatter.format(record)
        self._write(formatted_message, record)
    
    @abstractmethod
    def _write(self, formatted_message: str, record: LogRecord) -> None:
        """Write formatted message to destination."""
        pass
    
    def close(self) -> None:
        """Close appender and cleanup resources."""
        pass


class ConsoleAppender(LogAppender):
    """Console output appender."""
    
    def __init__(self, name: str, formatter: LogFormatter, use_stderr: bool = False):
        super().__init__(name, formatter)
        self.use_stderr = use_stderr
    
    def _write(self, formatted_message: str, record: LogRecord) -> None:
        """Write to console."""
        import sys
        
        output = sys.stderr if self.use_stderr else sys.stdout
        
        with self._lock:
            print(formatted_message, file=output)
            output.flush()


class FileAppender(LogAppender):
    """File output appender with rotation."""
    
    def __init__(self, name: str, formatter: LogFormatter, 
                 filename: str, encoding: str = "utf-8",
                 rotation_policy: RotationPolicy = RotationPolicy.NEVER,
                 max_size: int = 10 * 1024 * 1024,  # 10MB
                 max_files: int = 5,
                 compress_rotated: bool = True):
        super().__init__(name, formatter)
        self.filename = Path(filename)
        self.encoding = encoding
        self.rotation_policy = rotation_policy
        self.max_size = max_size
        self.max_files = max_files
        self.compress_rotated = compress_rotated
        
        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file
        self.file_handle = None
        self._open_file()
    
    def _open_file(self) -> None:
        """Open log file for writing."""
        if self.file_handle:
            self.file_handle.close()
        
        self.file_handle = open(self.filename, 'a', encoding=self.encoding)
    
    def _write(self, formatted_message: str, record: LogRecord) -> None:
        """Write to file with rotation check."""
        with self._lock:
            if self._should_rotate():
                self._rotate_file()
            
            self.file_handle.write(formatted_message + '\n')
            self.file_handle.flush()
    
    def _should_rotate(self) -> bool:
        """Check if file should be rotated."""
        if self.rotation_policy == RotationPolicy.NEVER:
            return False
        
        if self.rotation_policy == RotationPolicy.SIZE:
            return self.filename.stat().st_size >= self.max_size
        
        # Add time-based rotation logic here
        return False
    
    def _rotate_file(self) -> None:
        """Rotate log file."""
        self.file_handle.close()
        
        # Move existing files
        for i in range(self.max_files - 1, 0, -1):
            old_file = self.filename.with_suffix(f".{i}")
            new_file = self.filename.with_suffix(f".{i + 1}")
            
            if old_file.exists():
                if i == self.max_files - 1:
                    old_file.unlink()  # Delete oldest
                else:
                    old_file.rename(new_file)
        
        # Move current file
        rotated_file = self.filename.with_suffix(".1")
        self.filename.rename(rotated_file)
        
        # Compress if requested
        if self.compress_rotated:
            self._compress_file(rotated_file)
        
        # Open new file
        self._open_file()
    
    def _compress_file(self, file_path: Path) -> None:
        """Compress rotated file."""
        compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        file_path.unlink()  # Remove uncompressed file
    
    def close(self) -> None:
        """Close file handle."""
        if self.file_handle:
            self.file_handle.close()


class DatabaseAppender(LogAppender):
    """Database appender for structured log storage."""
    
    def __init__(self, name: str, formatter: LogFormatter, 
                 db_path: str, table_name: str = "logs"):
        super().__init__(name, formatter)
        self.db_path = db_path
        self.table_name = table_name
        self.connection = None
        
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database and create table."""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create logs table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            logger_name TEXT NOT NULL,
            message TEXT NOT NULL,
            thread_id INTEGER,
            process_id INTEGER,
            correlation_id TEXT,
            source_file TEXT,
            source_line INTEGER,
            source_function TEXT,
            exception_type TEXT,
            exception_message TEXT,
            extra_fields TEXT
        )
        """
        
        self.connection.execute(create_table_sql)
        self.connection.commit()
    
    def _write(self, formatted_message: str, record: LogRecord) -> None:
        """Write to database."""
        with self._lock:
            insert_sql = f"""
            INSERT INTO {self.table_name} 
            (timestamp, level, logger_name, message, thread_id, process_id, 
             correlation_id, source_file, source_line, source_function,
             exception_type, exception_message, extra_fields)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                record.timestamp.isoformat(),
                record.level.name,
                record.logger_name,
                record.message,
                record.thread_id,
                record.process_id,
                record.correlation_id,
                record.source_file,
                record.source_line,
                record.source_function,
                type(record.exception).__name__ if record.exception else None,
                str(record.exception) if record.exception else None,
                json.dumps(record.extra_fields) if record.extra_fields else None
            )
            
            self.connection.execute(insert_sql, values)
            self.connection.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()


class NetworkAppender(LogAppender):
    """Network appender for remote logging."""
    
    def __init__(self, name: str, formatter: LogFormatter,
                 host: str, port: int, protocol: str = "tcp"):
        super().__init__(name, formatter)
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.socket = None
        
        self._connect()
    
    def _connect(self) -> None:
        """Connect to remote log server."""
        try:
            if self.protocol == "tcp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
            elif self.protocol == "udp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            print(f"Failed to connect to log server: {e}")
    
    def _write(self, formatted_message: str, record: LogRecord) -> None:
        """Send to remote server."""
        if not self.socket:
            return
        
        try:
            message_bytes = formatted_message.encode('utf-8')
            
            if self.protocol == "tcp":
                self.socket.send(message_bytes + b'\n')
            elif self.protocol == "udp":
                self.socket.sendto(message_bytes, (self.host, self.port))
                
        except Exception as e:
            print(f"Failed to send log message: {e}")
            # Try to reconnect
            self._connect()
    
    def close(self) -> None:
        """Close network connection."""
        if self.socket:
            self.socket.close()


# ============================================================================
# ASYNCHRONOUS LOGGING
# ============================================================================

class AsyncLogHandler:
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, appender: LogAppender, queue_size: int = 10000):
        self.appender = appender
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.running = True
        
        self.worker_thread.start()
    
    def handle_log(self, record: LogRecord) -> None:
        """Add log record to queue for async processing."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Queue is full, drop the message or handle overflow
            print("Log queue is full, dropping message")
    
    def _process_logs(self) -> None:
        """Process logs from queue."""
        while self.running:
            try:
                record = self.log_queue.get(timeout=1.0)
                self.appender.append(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing log: {e}")
    
    def close(self) -> None:
        """Close async handler."""
        self.running = False
        self.worker_thread.join(timeout=5.0)
        self.appender.close()


# ============================================================================
# LOGGER IMPLEMENTATION
# ============================================================================

class Logger:
    """Logger implementation."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self.appenders: List[LogAppender] = []
        self.async_handlers: List[AsyncLogHandler] = []
        self.filters: List[LogFilter] = []
        self.propagate = True
        self.parent: Optional['Logger'] = None
        self.children: List['Logger'] = []
        
        # Performance tracking
        self.log_count = 0
        self.last_log_time = None
        
    def add_appender(self, appender: LogAppender, async_mode: bool = False) -> None:
        """Add appender to logger."""
        if async_mode:
            async_handler = AsyncLogHandler(appender)
            self.async_handlers.append(async_handler)
        else:
            self.appenders.append(appender)
    
    def add_filter(self, filter_obj: LogFilter) -> None:
        """Add filter to logger."""
        self.filters.append(filter_obj)
    
    def set_level(self, level: LogLevel) -> None:
        """Set logger level."""
        self.level = level
    
    def is_enabled_for(self, level: LogLevel) -> bool:
        """Check if logger is enabled for given level."""
        return level >= self.level
    
    def _log(self, level: LogLevel, message: str, exception: Exception = None, 
            extra: Dict[str, Any] = None, **kwargs) -> None:
        """Internal logging method."""
        if not self.is_enabled_for(level):
            return
        
        # Get caller information
        import inspect
        frame = inspect.currentframe().f_back.f_back
        source_file = frame.f_code.co_filename
        source_line = frame.f_lineno
        source_function = frame.f_code.co_name
        
        # Create log record
        record = LogRecord(
            timestamp=datetime.now(),
            level=level,
            logger_name=self.name,
            message=message,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            exception=exception,
            extra_fields=extra or {},
            source_file=source_file,
            source_line=source_line,
            source_function=source_function
        )
        
        # Apply filters
        for filter_obj in self.filters:
            if not filter_obj.filter(record):
                return
        
        # Update performance tracking
        self.log_count += 1
        self.last_log_time = record.timestamp
        
        # Send to appenders
        self._handle_record(record)
    
    def _handle_record(self, record: LogRecord) -> None:
        """Handle log record by sending to appenders."""
        # Send to synchronous appenders
        for appender in self.appenders:
            try:
                appender.append(record)
            except Exception as e:
                print(f"Error in appender {appender.name}: {e}")
        
        # Send to asynchronous handlers
        for handler in self.async_handlers:
            try:
                handler.handle_log(record)
            except Exception as e:
                print(f"Error in async handler: {e}")
        
        # Propagate to parent if enabled
        if self.propagate and self.parent:
            self.parent._handle_record(record)
    
    def trace(self, message: str, **kwargs) -> None:
        """Log trace message."""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warn(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARN, message, **kwargs)
    
    def error(self, message: str, exception: Exception = None, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, exception=exception, **kwargs)
    
    def fatal(self, message: str, exception: Exception = None, **kwargs) -> None:
        """Log fatal message."""
        self._log(LogLevel.FATAL, message, exception=exception, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with current exception info."""
        import sys
        self._log(LogLevel.ERROR, message, exception=sys.exc_info()[1], **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            'name': self.name,
            'level': self.level.name,
            'log_count': self.log_count,
            'last_log_time': self.last_log_time.isoformat() if self.last_log_time else None,
            'appender_count': len(self.appenders),
            'async_handler_count': len(self.async_handlers),
            'filter_count': len(self.filters),
            'child_count': len(self.children)
        }


# ============================================================================
# LOGGER MANAGER
# ============================================================================

class LoggerManager:
    """Central logger management system."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.loggers: Dict[str, Logger] = {}
            self.formatters: Dict[str, LogFormatter] = {}
            self.appenders: Dict[str, LogAppender] = {}
            self.filters: Dict[str, LogFilter] = {}
            
            # Initialize default components
            self._initialize_defaults()
            
            self.initialized = True
            print("📝 Logger Manager initialized")
    
    def _initialize_defaults(self) -> None:
        """Initialize default formatters and appenders."""
        # Default formatters
        self.formatters["simple"] = SimpleFormatter()
        self.formatters["detailed"] = DetailedFormatter()
        self.formatters["json"] = JSONFormatter()
        self.formatters["xml"] = XMLFormatter()
        
        # Default console appender
        console_appender = ConsoleAppender("console", self.formatters["simple"])
        self.appenders["console"] = console_appender
    
    def get_logger(self, name: str) -> Logger:
        """Get or create logger by name."""
        if name not in self.loggers:
            self.loggers[name] = Logger(name)
            
            # Set up parent-child relationships
            self._setup_hierarchy(name)
        
        return self.loggers[name]
    
    def _setup_hierarchy(self, logger_name: str) -> None:
        """Set up logger hierarchy based on name."""
        logger = self.loggers[logger_name]
        
        # Find parent logger
        parts = logger_name.split('.')
        for i in range(len(parts) - 1, 0, -1):
            parent_name = '.'.join(parts[:i])
            if parent_name in self.loggers:
                parent_logger = self.loggers[parent_name]
                logger.parent = parent_logger
                parent_logger.children.append(logger)
                break
    
    def add_formatter(self, name: str, formatter: LogFormatter) -> None:
        """Add formatter to manager."""
        self.formatters[name] = formatter
    
    def add_appender(self, name: str, appender: LogAppender) -> None:
        """Add appender to manager."""
        self.appenders[name] = appender
    
    def add_filter(self, name: str, filter_obj: LogFilter) -> None:
        """Add filter to manager."""
        self.filters[name] = filter_obj
    
    def configure_logger(self, config: LoggerConfig) -> Logger:
        """Configure logger from configuration."""
        logger = self.get_logger(config.name)
        logger.set_level(config.level)
        logger.propagate = config.propagate
        
        # Add appenders
        for appender_name in config.appenders:
            if appender_name in self.appenders:
                logger.add_appender(self.appenders[appender_name])
        
        # Add filters
        for filter_name in config.filters:
            if filter_name in self.filters:
                logger.add_filter(self.filters[filter_name])
        
        return logger
    
    def configure_from_dict(self, config: Dict[str, Any]) -> None:
        """Configure logging from dictionary."""
        # Configure formatters
        for name, formatter_config in config.get('formatters', {}).items():
            formatter_type = formatter_config.get('type', 'simple')
            
            if formatter_type == 'simple':
                pattern = formatter_config.get('pattern', "{timestamp} [{level}] {logger}: {message}")
                self.add_formatter(name, SimpleFormatter(pattern))
            elif formatter_type == 'detailed':
                self.add_formatter(name, DetailedFormatter())
            elif formatter_type == 'json':
                self.add_formatter(name, JSONFormatter())
            elif formatter_type == 'xml':
                self.add_formatter(name, XMLFormatter())
        
        # Configure appenders
        for name, appender_config in config.get('appenders', {}).items():
            appender_type = appender_config.get('type')
            formatter_name = appender_config.get('formatter', 'simple')
            formatter = self.formatters.get(formatter_name, self.formatters['simple'])
            
            if appender_type == 'console':
                use_stderr = appender_config.get('use_stderr', False)
                appender = ConsoleAppender(name, formatter, use_stderr)
            elif appender_type == 'file':
                filename = appender_config.get('filename', 'app.log')
                max_size = appender_config.get('max_size', 10 * 1024 * 1024)
                max_files = appender_config.get('max_files', 5)
                appender = FileAppender(name, formatter, filename, max_size=max_size, max_files=max_files)
            elif appender_type == 'database':
                db_path = appender_config.get('db_path', 'logs.db')
                appender = DatabaseAppender(name, formatter, db_path)
            elif appender_type == 'network':
                host = appender_config.get('host', 'localhost')
                port = appender_config.get('port', 514)
                appender = NetworkAppender(name, formatter, host, port)
            else:
                continue
            
            self.add_appender(name, appender)
        
        # Configure loggers
        for name, logger_config in config.get('loggers', {}).items():
            level_name = logger_config.get('level', 'INFO')
            level = LogLevel[level_name]
            
            appender_names = logger_config.get('appenders', [])
            filter_names = logger_config.get('filters', [])
            propagate = logger_config.get('propagate', True)
            
            config_obj = LoggerConfig(name, level, propagate, appender_names, filter_names)
            self.configure_logger(config_obj)
    
    def get_all_loggers(self) -> Dict[str, Logger]:
        """Get all registered loggers."""
        return self.loggers.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging system statistics."""
        logger_stats = {}
        for name, logger in self.loggers.items():
            logger_stats[name] = logger.get_stats()
        
        return {
            'logger_count': len(self.loggers),
            'formatter_count': len(self.formatters),
            'appender_count': len(self.appenders),
            'filter_count': len(self.filters),
            'loggers': logger_stats
        }
    
    def shutdown(self) -> None:
        """Shutdown logging system."""
        # Close all appenders
        for appender in self.appenders.values():
            appender.close()
        
        # Close async handlers
        for logger in self.loggers.values():
            for handler in logger.async_handlers:
                handler.close()


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class LoggingPerformanceMonitor:
    """Monitor logging system performance."""
    
    def __init__(self):
        self.metrics = {
            'total_logs': 0,
            'logs_per_second': 0,
            'average_latency_ms': 0,
            'error_count': 0,
            'queue_depth': 0
        }
        
        self.start_time = time.time()
        self.last_measurement = time.time()
        self.log_times: List[float] = []
        self._lock = threading.Lock()
    
    def record_log_event(self, processing_time: float) -> None:
        """Record log processing event."""
        with self._lock:
            self.metrics['total_logs'] += 1
            self.log_times.append(processing_time)
            
            # Keep only last 1000 measurements
            if len(self.log_times) > 1000:
                self.log_times = self.log_times[-1000:]
            
            # Update metrics every 10 seconds
            now = time.time()
            if now - self.last_measurement >= 10:
                self._update_metrics()
                self.last_measurement = now
    
    def _update_metrics(self) -> None:
        """Update performance metrics."""
        if not self.log_times:
            return
        
        # Calculate logs per second
        elapsed = time.time() - self.start_time
        self.metrics['logs_per_second'] = self.metrics['total_logs'] / elapsed
        
        # Calculate average latency
        self.metrics['average_latency_ms'] = sum(self.log_times) / len(self.log_times) * 1000
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        with self._lock:
            return self.metrics.copy()


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_logger_framework():
    """Demonstrate the logger framework."""
    print("=== LOGGER FRAMEWORK DEMONSTRATION ===\n")
    
    # Get logger manager
    print("1. LOGGER MANAGER SETUP:")
    
    manager = LoggerManager()
    print("   ✓ Logger manager initialized with default components")
    print()
    
    # Create different formatters
    print("2. FORMATTER CREATION:")
    
    # Custom simple formatter
    custom_formatter = SimpleFormatter("[{timestamp}] {level} - {logger}: {message}")
    manager.add_formatter("custom", custom_formatter)
    print("   ✓ Added custom simple formatter")
    
    # Performance monitoring
    perf_monitor = LoggingPerformanceMonitor()
    
    print()
    
    # Create different appenders
    print("3. APPENDER CREATION:")
    
    # File appender with rotation
    file_appender = FileAppender(
        "main_file", 
        manager.formatters["detailed"],
        "logs/application.log",
        rotation_policy=RotationPolicy.SIZE,
        max_size=1024,  # Small size for demo
        max_files=3
    )
    manager.add_appender("main_file", file_appender)
    print("   ✓ Created file appender with rotation")
    
    # Database appender
    db_appender = DatabaseAppender(
        "database",
        manager.formatters["json"],
        "logs/application.db"
    )
    manager.add_appender("database", db_appender)
    print("   ✓ Created database appender")
    
    # JSON file appender
    json_file_appender = FileAppender(
        "json_file",
        manager.formatters["json"],
        "logs/application.json"
    )
    manager.add_appender("json_file", json_file_appender)
    print("   ✓ Created JSON file appender")
    
    print()
    
    # Create filters
    print("4. FILTER CREATION:")
    
    # Level filter
    error_filter = LevelFilter(LogLevel.ERROR)
    manager.add_filter("errors_only", error_filter)
    print("   ✓ Created error-level filter")
    
    # Rate limit filter
    rate_filter = RateLimitFilter(5, timedelta(seconds=10))
    manager.add_filter("rate_limit", rate_filter)
    print("   ✓ Created rate limiting filter")
    
    print()
    
    # Configure loggers
    print("5. LOGGER CONFIGURATION:")
    
    # Application logger
    app_logger = manager.get_logger("app")
    app_logger.set_level(LogLevel.DEBUG)
    app_logger.add_appender(manager.appenders["console"])
    app_logger.add_appender(manager.appenders["main_file"])
    app_logger.add_appender(manager.appenders["json_file"], async_mode=True)
    print("   ✓ Configured application logger")
    
    # Database logger
    db_logger = manager.get_logger("app.database")
    db_logger.add_appender(manager.appenders["database"])
    db_logger.add_filter(manager.filters["errors_only"])
    print("   ✓ Configured database logger (errors only)")
    
    # Service logger with rate limiting
    service_logger = manager.get_logger("app.service")
    service_logger.add_appender(manager.appenders["console"])
    service_logger.add_filter(manager.filters["rate_limit"])
    print("   ✓ Configured service logger with rate limiting")
    
    print()
    
    # Test basic logging
    print("6. BASIC LOGGING TEST:")
    
    app_logger.info("Application started successfully")
    app_logger.debug("Debug information for troubleshooting")
    app_logger.warn("This is a warning message")
    
    # Test with extra fields
    app_logger.info("User login", extra={"user_id": 12345, "ip": "192.168.1.100"})
    
    print("   ✓ Logged messages at different levels")
    print()
    
    # Test error logging with exceptions
    print("7. EXCEPTION LOGGING TEST:")
    
    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError as e:
        app_logger.error("Division by zero error", exception=e)
        app_logger.exception("Exception occurred during calculation")
        db_logger.error("Database connection failed", exception=e)
    
    print("   ✓ Logged exceptions with stack traces")
    print()
    
    # Test hierarchical logging
    print("8. HIERARCHICAL LOGGING TEST:")
    
    # Child loggers inherit from parent
    auth_logger = manager.get_logger("app.auth")
    payment_logger = manager.get_logger("app.payment")
    
    auth_logger.info("User authentication successful")
    payment_logger.info("Payment processed successfully")
    
    print("   ✓ Tested hierarchical logger structure")
    print()
    
    # Test rate limiting
    print("9. RATE LIMITING TEST:")
    
    print("   Sending 10 messages to rate-limited logger (limit: 5 per 10 seconds):")
    for i in range(10):
        service_logger.info(f"Service message {i + 1}")
    
    print("   ✓ Demonstrated rate limiting (some messages dropped)")
    print()
    
    # Test configuration from dictionary
    print("10. CONFIGURATION FROM DICT:")
    
    config = {
        "formatters": {
            "console_format": {
                "type": "simple",
                "pattern": "{timestamp} | {level:5} | {logger:20} | {message}"
            }
        },
        "appenders": {
            "new_console": {
                "type": "console",
                "formatter": "console_format",
                "use_stderr": True
            },
            "error_file": {
                "type": "file",
                "formatter": "detailed",
                "filename": "logs/errors.log",
                "max_size": 2048,
                "max_files": 2
            }
        },
        "loggers": {
            "config_test": {
                "level": "WARN",
                "appenders": ["new_console", "error_file"],
                "propagate": False
            }
        }
    }
    
    manager.configure_from_dict(config)
    
    config_logger = manager.get_logger("config_test")
    config_logger.warn("This is a warning from configured logger")
    config_logger.error("This is an error from configured logger")
    config_logger.info("This info message should not appear (level is WARN)")
    
    print("   ✓ Configured logger from dictionary")
    print()
    
    # Performance test
    print("11. PERFORMANCE TEST:")
    
    perf_logger = manager.get_logger("performance")
    perf_logger.add_appender(manager.appenders["console"], async_mode=True)
    
    start_time = time.time()
    message_count = 1000
    
    for i in range(message_count):
        start = time.time()
        perf_logger.info(f"Performance test message {i}")
        end = time.time()
        perf_monitor.record_log_event(end - start)
    
    end_time = time.time()
    
    print(f"   Logged {message_count} messages in {end_time - start_time:.3f} seconds")
    print(f"   Rate: {message_count / (end_time - start_time):.0f} messages/second")
    
    # Wait a bit for async processing
    time.sleep(1)
    
    print()
    
    # Show performance metrics
    print("12. PERFORMANCE METRICS:")
    
    metrics = perf_monitor.get_metrics()
    print(f"   Total logs: {metrics['total_logs']}")
    print(f"   Logs per second: {metrics['logs_per_second']:.2f}")
    print(f"   Average latency: {metrics['average_latency_ms']:.3f} ms")
    
    print()
    
    # Show logging statistics
    print("13. LOGGING STATISTICS:")
    
    stats = manager.get_stats()
    print(f"   Total loggers: {stats['logger_count']}")
    print(f"   Total formatters: {stats['formatter_count']}")
    print(f"   Total appenders: {stats['appender_count']}")
    print(f"   Total filters: {stats['filter_count']}")
    
    print("\n   Logger details:")
    for name, logger_stats in stats['loggers'].items():
        print(f"     {name}: {logger_stats['log_count']} logs, level={logger_stats['level']}")
    
    print()
    
    # Test structured logging
    print("14. STRUCTURED LOGGING TEST:")
    
    structured_logger = manager.get_logger("structured")
    structured_logger.add_appender(manager.appenders["json_file"])
    
    # Log structured data
    user_event = {
        "event_type": "user_action",
        "action": "login",
        "user_id": 12345,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0...",
            "session_id": "abc123def456"
        }
    }
    
    structured_logger.info("User login event", extra=user_event)
    
    purchase_event = {
        "event_type": "transaction",
        "action": "purchase",
        "user_id": 12345,
        "product_id": "PROD-789",
        "amount": 99.99,
        "currency": "USD"
    }
    
    structured_logger.info("Purchase completed", extra=purchase_event)
    
    print("   ✓ Logged structured events")
    print()
    
    # Test log file rotation
    print("15. LOG ROTATION TEST:")
    
    rotation_logger = manager.get_logger("rotation_test")
    rotation_appender = FileAppender(
        "rotation_test",
        manager.formatters["simple"],
        "logs/rotation_test.log",
        rotation_policy=RotationPolicy.SIZE,
        max_size=100,  # Very small for quick rotation
        max_files=3
    )
    rotation_logger.add_appender(rotation_appender)
    
    # Generate enough logs to trigger rotation
    for i in range(20):
        rotation_logger.info(f"Rotation test message {i} - This is a longer message to fill up the file faster")
    
    print("   ✓ Generated logs to test rotation (check logs/ directory)")
    print()
    
    # Show final system state
    print("16. FINAL SYSTEM STATE:")
    
    final_stats = manager.get_stats()
    print(f"   Active loggers: {final_stats['logger_count']}")
    print(f"   Total log events: {sum(stats['log_count'] for stats in final_stats['loggers'].values())}")
    
    # Check log files created
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*"))
        print(f"   Log files created: {len(log_files)}")
        for log_file in log_files:
            size = log_file.stat().st_size
            print(f"     {log_file.name}: {size} bytes")
    
    print()
    
    # Cleanup
    print("17. CLEANUP:")
    
    manager.shutdown()
    print("   ✓ Logger system shutdown complete")
    
    print()
    print("=== LOGGER FRAMEWORK DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_logger_framework()
