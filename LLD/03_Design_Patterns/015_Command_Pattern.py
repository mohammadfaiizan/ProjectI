"""
COMMAND PATTERN - Behavioral Design Pattern
===========================================

Problem Statement:
Implement the Command pattern to encapsulate requests as objects, allowing you
to parameterize clients with different requests, queue operations, and support undo:
- Encapsulate requests as command objects
- Decouple invoker from receiver
- Support undo/redo operations
- Queue and log commands for execution
- Macro commands and composite operations

Learning Objectives:
- Understand Command vs Strategy pattern differences
- Implement command encapsulation and execution
- Design undo/redo functionality
- Create command queues and batch operations
- Handle macro commands and command composition
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union, Deque
import time
import json
from datetime import datetime
from enum import Enum
from collections import deque
import threading
import copy


# ============================================================================
# COMMAND INTERFACE
# ============================================================================

class Command(ABC):
    """Abstract command interface."""
    
    @abstractmethod
    def execute(self) -> Any:
        """Execute the command."""
        pass
    
    @abstractmethod
    def undo(self) -> Any:
        """Undo the command."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get command description."""
        pass
    
    def can_undo(self) -> bool:
        """Check if command can be undone."""
        return True
    
    def get_execution_info(self) -> Dict[str, Any]:
        """Get command execution information."""
        return {
            'command_type': self.__class__.__name__,
            'description': self.get_description(),
            'can_undo': self.can_undo(),
            'timestamp': datetime.now().isoformat()
        }


class Receiver(ABC):
    """Abstract receiver interface."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the receiver."""
        pass


# ============================================================================
# TEXT EDITOR COMMANDS
# ============================================================================

class TextEditor(Receiver):
    """Text editor that receives commands."""
    
    def __init__(self):
        self.content = ""
        self.cursor_position = 0
        self.selection_start = 0
        self.selection_end = 0
        self.clipboard = ""
        self.history: List[Dict[str, Any]] = []
        
    def insert_text(self, text: str, position: int = None) -> None:
        """Insert text at specified position."""
        if position is None:
            position = self.cursor_position
        
        self.content = self.content[:position] + text + self.content[position:]
        self.cursor_position = position + len(text)
        self._record_state("insert_text")
    
    def delete_text(self, start: int, end: int) -> str:
        """Delete text between start and end positions."""
        deleted_text = self.content[start:end]
        self.content = self.content[:start] + self.content[end:]
        self.cursor_position = start
        self._record_state("delete_text")
        return deleted_text
    
    def replace_text(self, start: int, end: int, new_text: str) -> str:
        """Replace text between start and end with new text."""
        old_text = self.content[start:end]
        self.content = self.content[:start] + new_text + self.content[end:]
        self.cursor_position = start + len(new_text)
        self._record_state("replace_text")
        return old_text
    
    def set_selection(self, start: int, end: int) -> None:
        """Set text selection."""
        self.selection_start = start
        self.selection_end = end
        self.cursor_position = end
    
    def copy_selection(self) -> str:
        """Copy selected text to clipboard."""
        if self.selection_start != self.selection_end:
            self.clipboard = self.content[self.selection_start:self.selection_end]
        return self.clipboard
    
    def get_selection(self) -> str:
        """Get currently selected text."""
        return self.content[self.selection_start:self.selection_end]
    
    def set_cursor_position(self, position: int) -> None:
        """Set cursor position."""
        self.cursor_position = max(0, min(position, len(self.content)))
    
    def _record_state(self, operation: str) -> None:
        """Record current state for history."""
        state = {
            'operation': operation,
            'content': self.content,
            'cursor_position': self.cursor_position,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(state)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current editor state."""
        return {
            'content': self.content,
            'cursor_position': self.cursor_position,
            'selection_start': self.selection_start,
            'selection_end': self.selection_end,
            'clipboard': self.clipboard,
            'content_length': len(self.content)
        }
    
    def display(self) -> None:
        """Display editor content with cursor."""
        print(f"Content: '{self.content}'")
        print(f"Cursor at position: {self.cursor_position}")
        if self.selection_start != self.selection_end:
            print(f"Selection: [{self.selection_start}:{self.selection_end}] = '{self.get_selection()}'")


class InsertTextCommand(Command):
    """Command to insert text."""
    
    def __init__(self, editor: TextEditor, text: str, position: int = None):
        self.editor = editor
        self.text = text
        self.position = position if position is not None else editor.cursor_position
        self.original_position = None
        
    def execute(self) -> Any:
        """Execute text insertion."""
        self.original_position = self.editor.cursor_position
        self.editor.insert_text(self.text, self.position)
        return f"Inserted '{self.text}' at position {self.position}"
    
    def undo(self) -> Any:
        """Undo text insertion."""
        start = self.position
        end = self.position + len(self.text)
        self.editor.delete_text(start, end)
        self.editor.set_cursor_position(self.original_position)
        return f"Undid insertion of '{self.text}'"
    
    def get_description(self) -> str:
        """Get command description."""
        return f"Insert '{self.text}' at position {self.position}"


class DeleteTextCommand(Command):
    """Command to delete text."""
    
    def __init__(self, editor: TextEditor, start: int, end: int):
        self.editor = editor
        self.start = start
        self.end = end
        self.deleted_text = ""
        self.original_position = None
        
    def execute(self) -> Any:
        """Execute text deletion."""
        self.original_position = self.editor.cursor_position
        self.deleted_text = self.editor.delete_text(self.start, self.end)
        return f"Deleted '{self.deleted_text}' from position {self.start}"
    
    def undo(self) -> Any:
        """Undo text deletion."""
        self.editor.insert_text(self.deleted_text, self.start)
        self.editor.set_cursor_position(self.original_position)
        return f"Undid deletion of '{self.deleted_text}'"
    
    def get_description(self) -> str:
        """Get command description."""
        return f"Delete text from {self.start} to {self.end}"


class ReplaceTextCommand(Command):
    """Command to replace text."""
    
    def __init__(self, editor: TextEditor, start: int, end: int, new_text: str):
        self.editor = editor
        self.start = start
        self.end = end
        self.new_text = new_text
        self.old_text = ""
        self.original_position = None
        
    def execute(self) -> Any:
        """Execute text replacement."""
        self.original_position = self.editor.cursor_position
        self.old_text = self.editor.replace_text(self.start, self.end, self.new_text)
        return f"Replaced '{self.old_text}' with '{self.new_text}'"
    
    def undo(self) -> Any:
        """Undo text replacement."""
        new_end = self.start + len(self.new_text)
        self.editor.replace_text(self.start, new_end, self.old_text)
        self.editor.set_cursor_position(self.original_position)
        return f"Undid replacement: restored '{self.old_text}'"
    
    def get_description(self) -> str:
        """Get command description."""
        return f"Replace text [{self.start}:{self.end}] with '{self.new_text}'"


class CopyCommand(Command):
    """Command to copy text."""
    
    def __init__(self, editor: TextEditor):
        self.editor = editor
        self.copied_text = ""
        
    def execute(self) -> Any:
        """Execute copy operation."""
        self.copied_text = self.editor.copy_selection()
        return f"Copied '{self.copied_text}' to clipboard"
    
    def undo(self) -> Any:
        """Copy operations typically cannot be undone."""
        return "Copy operation cannot be undone"
    
    def can_undo(self) -> bool:
        """Copy operations cannot be undone."""
        return False
    
    def get_description(self) -> str:
        """Get command description."""
        return "Copy selected text to clipboard"


class PasteCommand(Command):
    """Command to paste text."""
    
    def __init__(self, editor: TextEditor, position: int = None):
        self.editor = editor
        self.position = position
        self.pasted_text = ""
        self.original_position = None
        
    def execute(self) -> Any:
        """Execute paste operation."""
        self.original_position = self.editor.cursor_position
        self.pasted_text = self.editor.clipboard
        
        if self.position is None:
            self.position = self.editor.cursor_position
            
        self.editor.insert_text(self.pasted_text, self.position)
        return f"Pasted '{self.pasted_text}' at position {self.position}"
    
    def undo(self) -> Any:
        """Undo paste operation."""
        start = self.position
        end = self.position + len(self.pasted_text)
        self.editor.delete_text(start, end)
        self.editor.set_cursor_position(self.original_position)
        return f"Undid paste of '{self.pasted_text}'"
    
    def get_description(self) -> str:
        """Get command description."""
        return f"Paste clipboard content at position {self.position}"


# ============================================================================
# CALCULATOR COMMANDS
# ============================================================================

class Calculator(Receiver):
    """Calculator that receives arithmetic commands."""
    
    def __init__(self):
        self.result = 0.0
        self.memory = 0.0
        self.history: List[Dict[str, Any]] = []
        
    def add(self, value: float) -> float:
        """Add value to result."""
        old_result = self.result
        self.result += value
        self._record_operation("add", value, old_result, self.result)
        return self.result
    
    def subtract(self, value: float) -> float:
        """Subtract value from result."""
        old_result = self.result
        self.result -= value
        self._record_operation("subtract", value, old_result, self.result)
        return self.result
    
    def multiply(self, value: float) -> float:
        """Multiply result by value."""
        old_result = self.result
        self.result *= value
        self._record_operation("multiply", value, old_result, self.result)
        return self.result
    
    def divide(self, value: float) -> float:
        """Divide result by value."""
        if value == 0:
            raise ValueError("Division by zero")
        
        old_result = self.result
        self.result /= value
        self._record_operation("divide", value, old_result, self.result)
        return self.result
    
    def set_value(self, value: float) -> float:
        """Set result to specific value."""
        old_result = self.result
        self.result = value
        self._record_operation("set", value, old_result, self.result)
        return self.result
    
    def clear(self) -> float:
        """Clear result to zero."""
        old_result = self.result
        self.result = 0.0
        self._record_operation("clear", 0, old_result, self.result)
        return self.result
    
    def store_memory(self) -> None:
        """Store current result in memory."""
        self.memory = self.result
        self._record_operation("store_memory", self.result, self.memory, self.result)
    
    def recall_memory(self) -> float:
        """Recall value from memory."""
        old_result = self.result
        self.result = self.memory
        self._record_operation("recall_memory", self.memory, old_result, self.result)
        return self.result
    
    def _record_operation(self, operation: str, operand: float, old_result: float, new_result: float) -> None:
        """Record operation in history."""
        record = {
            'operation': operation,
            'operand': operand,
            'old_result': old_result,
            'new_result': new_result,
            'timestamp': datetime.now().isoformat()
        }
        self.history.append(record)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current calculator state."""
        return {
            'result': self.result,
            'memory': self.memory,
            'operations_count': len(self.history)
        }
    
    def display(self) -> None:
        """Display calculator state."""
        print(f"Result: {self.result}")
        print(f"Memory: {self.memory}")


class ArithmeticCommand(Command):
    """Base class for arithmetic commands."""
    
    def __init__(self, calculator: Calculator, value: float):
        self.calculator = calculator
        self.value = value
        self.previous_result = None
        
    def execute(self) -> Any:
        """Execute arithmetic operation."""
        self.previous_result = self.calculator.result
        return self._perform_operation()
    
    def undo(self) -> Any:
        """Undo arithmetic operation."""
        self.calculator.set_value(self.previous_result)
        return f"Undid {self.get_description()}, restored result to {self.previous_result}"
    
    @abstractmethod
    def _perform_operation(self) -> Any:
        """Perform the specific arithmetic operation."""
        pass


class AddCommand(ArithmeticCommand):
    """Command to add a value."""
    
    def _perform_operation(self) -> Any:
        result = self.calculator.add(self.value)
        return f"Added {self.value}, result: {result}"
    
    def get_description(self) -> str:
        return f"Add {self.value}"


class SubtractCommand(ArithmeticCommand):
    """Command to subtract a value."""
    
    def _perform_operation(self) -> Any:
        result = self.calculator.subtract(self.value)
        return f"Subtracted {self.value}, result: {result}"
    
    def get_description(self) -> str:
        return f"Subtract {self.value}"


class MultiplyCommand(ArithmeticCommand):
    """Command to multiply by a value."""
    
    def _perform_operation(self) -> Any:
        result = self.calculator.multiply(self.value)
        return f"Multiplied by {self.value}, result: {result}"
    
    def get_description(self) -> str:
        return f"Multiply by {self.value}"


class DivideCommand(ArithmeticCommand):
    """Command to divide by a value."""
    
    def _perform_operation(self) -> Any:
        result = self.calculator.divide(self.value)
        return f"Divided by {self.value}, result: {result}"
    
    def get_description(self) -> str:
        return f"Divide by {self.value}"


# ============================================================================
# MACRO COMMANDS
# ============================================================================

class MacroCommand(Command):
    """Command that executes multiple commands."""
    
    def __init__(self, commands: List[Command], description: str = "Macro Command"):
        self.commands = commands
        self.description = description
        self.executed_commands: List[Command] = []
        
    def execute(self) -> Any:
        """Execute all commands in sequence."""
        results = []
        self.executed_commands = []
        
        for command in self.commands:
            try:
                result = command.execute()
                results.append(result)
                self.executed_commands.append(command)
            except Exception as e:
                # If any command fails, undo all executed commands
                self.undo()
                raise e
        
        return f"Executed macro with {len(results)} commands"
    
    def undo(self) -> Any:
        """Undo all executed commands in reverse order."""
        undone_count = 0
        
        # Undo in reverse order
        for command in reversed(self.executed_commands):
            if command.can_undo():
                try:
                    command.undo()
                    undone_count += 1
                except Exception as e:
                    print(f"Error undoing command {command.get_description()}: {e}")
        
        self.executed_commands = []
        return f"Undid {undone_count} commands from macro"
    
    def can_undo(self) -> bool:
        """Macro can be undone if any of its commands can be undone."""
        return any(cmd.can_undo() for cmd in self.executed_commands)
    
    def get_description(self) -> str:
        """Get macro description."""
        return self.description
    
    def add_command(self, command: Command) -> None:
        """Add command to macro."""
        self.commands.append(command)
    
    def get_command_descriptions(self) -> List[str]:
        """Get descriptions of all commands in macro."""
        return [cmd.get_description() for cmd in self.commands]


# ============================================================================
# COMMAND INVOKER
# ============================================================================

class CommandInvoker:
    """Invoker that executes commands and manages undo/redo."""
    
    def __init__(self, max_history: int = 100):
        self.command_history: Deque[Command] = deque(maxlen=max_history)
        self.undo_stack: Deque[Command] = deque(maxlen=max_history)
        self.redo_stack: Deque[Command] = deque(maxlen=max_history)
        self.execution_count = 0
        
    def execute_command(self, command: Command) -> Any:
        """Execute a command and add it to history."""
        try:
            result = command.execute()
            
            # Add to history and undo stack
            self.command_history.append(command)
            if command.can_undo():
                self.undo_stack.append(command)
            
            # Clear redo stack when new command is executed
            self.redo_stack.clear()
            
            self.execution_count += 1
            
            print(f"Executed: {command.get_description()}")
            return result
            
        except Exception as e:
            print(f"Error executing command {command.get_description()}: {e}")
            raise e
    
    def undo(self) -> Optional[str]:
        """Undo the last command."""
        if not self.undo_stack:
            return "Nothing to undo"
        
        command = self.undo_stack.pop()
        
        try:
            result = command.undo()
            self.redo_stack.append(command)
            print(f"Undid: {command.get_description()}")
            return result
        except Exception as e:
            # If undo fails, put command back on undo stack
            self.undo_stack.append(command)
            print(f"Error undoing command {command.get_description()}: {e}")
            return f"Undo failed: {e}"
    
    def redo(self) -> Optional[str]:
        """Redo the last undone command."""
        if not self.redo_stack:
            return "Nothing to redo"
        
        command = self.redo_stack.pop()
        
        try:
            result = command.execute()
            self.undo_stack.append(command)
            print(f"Redid: {command.get_description()}")
            return result
        except Exception as e:
            # If redo fails, put command back on redo stack
            self.redo_stack.append(command)
            print(f"Error redoing command {command.get_description()}: {e}")
            return f"Redo failed: {e}"
    
    def get_history(self) -> List[str]:
        """Get command execution history."""
        return [cmd.get_description() for cmd in self.command_history]
    
    def get_undo_stack(self) -> List[str]:
        """Get undo stack descriptions."""
        return [cmd.get_description() for cmd in self.undo_stack]
    
    def get_redo_stack(self) -> List[str]:
        """Get redo stack descriptions."""
        return [cmd.get_description() for cmd in self.redo_stack]
    
    def clear_history(self) -> None:
        """Clear all command history."""
        self.command_history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.execution_count = 0
        print("Command history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get invoker statistics."""
        return {
            'total_executions': self.execution_count,
            'history_size': len(self.command_history),
            'undo_available': len(self.undo_stack),
            'redo_available': len(self.redo_stack),
            'can_undo': len(self.undo_stack) > 0,
            'can_redo': len(self.redo_stack) > 0
        }


# ============================================================================
# COMMAND QUEUE AND BATCH PROCESSING
# ============================================================================

class CommandQueue:
    """Queue for batch command execution."""
    
    def __init__(self, invoker: CommandInvoker):
        self.invoker = invoker
        self.queue: Deque[Command] = deque()
        self.is_processing = False
        self.processed_count = 0
        
    def add_command(self, command: Command) -> None:
        """Add command to queue."""
        self.queue.append(command)
        print(f"Added to queue: {command.get_description()}")
    
    def add_commands(self, commands: List[Command]) -> None:
        """Add multiple commands to queue."""
        for command in commands:
            self.add_command(command)
    
    def execute_all(self) -> List[Any]:
        """Execute all commands in queue."""
        if self.is_processing:
            raise RuntimeError("Queue is already being processed")
        
        self.is_processing = True
        results = []
        
        try:
            while self.queue:
                command = self.queue.popleft()
                result = self.invoker.execute_command(command)
                results.append(result)
                self.processed_count += 1
                
                # Small delay to simulate processing time
                time.sleep(0.01)
            
            print(f"Processed {len(results)} commands from queue")
            return results
            
        finally:
            self.is_processing = False
    
    def execute_batch(self, batch_size: int) -> List[Any]:
        """Execute a batch of commands from queue."""
        if self.is_processing:
            raise RuntimeError("Queue is already being processed")
        
        self.is_processing = True
        results = []
        
        try:
            for _ in range(min(batch_size, len(self.queue))):
                if not self.queue:
                    break
                    
                command = self.queue.popleft()
                result = self.invoker.execute_command(command)
                results.append(result)
                self.processed_count += 1
            
            print(f"Processed batch of {len(results)} commands")
            return results
            
        finally:
            self.is_processing = False
    
    def clear_queue(self) -> int:
        """Clear all commands from queue."""
        count = len(self.queue)
        self.queue.clear()
        print(f"Cleared {count} commands from queue")
        return count
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get queue information."""
        return {
            'queue_size': len(self.queue),
            'is_processing': self.is_processing,
            'total_processed': self.processed_count,
            'pending_commands': [cmd.get_description() for cmd in list(self.queue)[:5]]  # Show first 5
        }


# ============================================================================
# COMMAND LOGGING AND PERSISTENCE
# ============================================================================

class CommandLogger:
    """Logger for command execution and persistence."""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.log_entries: List[Dict[str, Any]] = []
        
    def log_command(self, command: Command, operation: str, result: Any = None, error: str = None) -> None:
        """Log command execution."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,  # 'execute', 'undo', 'redo'
            'command_type': command.__class__.__name__,
            'description': command.get_description(),
            'result': str(result) if result is not None else None,
            'error': error,
            'execution_info': command.get_execution_info()
        }
        
        self.log_entries.append(log_entry)
        
        if self.log_file:
            self._write_to_file(log_entry)
    
    def _write_to_file(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get log summary statistics."""
        if not self.log_entries:
            return {'total_entries': 0}
        
        operations = {}
        command_types = {}
        errors = 0
        
        for entry in self.log_entries:
            # Count operations
            op = entry['operation']
            operations[op] = operations.get(op, 0) + 1
            
            # Count command types
            cmd_type = entry['command_type']
            command_types[cmd_type] = command_types.get(cmd_type, 0) + 1
            
            # Count errors
            if entry['error']:
                errors += 1
        
        return {
            'total_entries': len(self.log_entries),
            'operations': operations,
            'command_types': command_types,
            'errors': errors,
            'success_rate': ((len(self.log_entries) - errors) / len(self.log_entries)) * 100
        }
    
    def get_recent_logs(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        return self.log_entries[-count:] if self.log_entries else []


class LoggingCommandInvoker(CommandInvoker):
    """Command invoker with logging capabilities."""
    
    def __init__(self, max_history: int = 100, logger: CommandLogger = None):
        super().__init__(max_history)
        self.logger = logger or CommandLogger()
    
    def execute_command(self, command: Command) -> Any:
        """Execute command with logging."""
        try:
            result = super().execute_command(command)
            self.logger.log_command(command, 'execute', result)
            return result
        except Exception as e:
            self.logger.log_command(command, 'execute', error=str(e))
            raise e
    
    def undo(self) -> Optional[str]:
        """Undo command with logging."""
        if not self.undo_stack:
            return "Nothing to undo"
        
        command = self.undo_stack[-1]  # Peek at command before undoing
        
        try:
            result = super().undo()
            self.logger.log_command(command, 'undo', result)
            return result
        except Exception as e:
            self.logger.log_command(command, 'undo', error=str(e))
            return f"Undo failed: {e}"
    
    def redo(self) -> Optional[str]:
        """Redo command with logging."""
        if not self.redo_stack:
            return "Nothing to redo"
        
        command = self.redo_stack[-1]  # Peek at command before redoing
        
        try:
            result = super().redo()
            self.logger.log_command(command, 'redo', result)
            return result
        except Exception as e:
            self.logger.log_command(command, 'redo', error=str(e))
            return f"Redo failed: {e}"


def demonstrate_command_pattern():
    """
    Demonstrate Command pattern implementations.
    """
    print("=== COMMAND PATTERN DEMONSTRATION ===\n")
    
    # 1. Text Editor Commands
    print("1. TEXT EDITOR COMMANDS:")
    
    # Create text editor and invoker
    editor = TextEditor()
    invoker = CommandInvoker()
    
    # Create and execute commands
    commands = [
        InsertTextCommand(editor, "Hello", 0),
        InsertTextCommand(editor, " World", 5),
        InsertTextCommand(editor, "!", 11),
        ReplaceTextCommand(editor, 6, 11, "Python"),
        InsertTextCommand(editor, " Programming", 12)
    ]
    
    print("   Executing text editor commands:")
    for command in commands:
        invoker.execute_command(command)
        editor.display()
        print()
    
    # Test undo operations
    print("   Testing undo operations:")
    for i in range(3):
        result = invoker.undo()
        print(f"   {result}")
        editor.display()
        print()
    
    # Test redo operations
    print("   Testing redo operations:")
    for i in range(2):
        result = invoker.redo()
        print(f"   {result}")
        editor.display()
        print()
    
    # Show command history
    print("   Command History:")
    history = invoker.get_history()
    for i, cmd in enumerate(history, 1):
        print(f"     {i}. {cmd}")
    
    print()
    
    # 2. Calculator Commands
    print("2. CALCULATOR COMMANDS:")
    
    # Create calculator and new invoker
    calculator = Calculator()
    calc_invoker = CommandInvoker()
    
    # Create arithmetic commands
    calc_commands = [
        AddCommand(calculator, 10),
        MultiplyCommand(calculator, 5),
        SubtractCommand(calculator, 15),
        DivideCommand(calculator, 7),
        AddCommand(calculator, 3)
    ]
    
    print("   Executing calculator commands:")
    for command in calc_commands:
        result = calc_invoker.execute_command(command)
        print(f"   {result}")
        calculator.display()
        print()
    
    # Test calculator undo
    print("   Testing calculator undo:")
    for i in range(3):
        result = calc_invoker.undo()
        print(f"   {result}")
        calculator.display()
        print()
    
    print()
    
    # 3. Macro Commands
    print("3. MACRO COMMANDS:")
    
    # Create new editor for macro demo
    macro_editor = TextEditor()
    macro_invoker = CommandInvoker()
    
    # Create macro command for "Hello World" insertion
    hello_world_macro = MacroCommand([
        InsertTextCommand(macro_editor, "Hello", 0),
        InsertTextCommand(macro_editor, " ", 5),
        InsertTextCommand(macro_editor, "World", 6),
        InsertTextCommand(macro_editor, "!", 11)
    ], "Insert Hello World")
    
    print("   Executing macro command:")
    print(f"   Macro contains: {hello_world_macro.get_command_descriptions()}")
    
    result = macro_invoker.execute_command(hello_world_macro)
    print(f"   {result}")
    macro_editor.display()
    print()
    
    # Create formatting macro
    formatting_macro = MacroCommand([
        ReplaceTextCommand(macro_editor, 0, 5, "Hi"),
        InsertTextCommand(macro_editor, " there", 8),
        InsertTextCommand(macro_editor, " How are you?", 14)
    ], "Format Greeting")
    
    result = macro_invoker.execute_command(formatting_macro)
    print(f"   {result}")
    macro_editor.display()
    print()
    
    # Test macro undo
    print("   Testing macro undo:")
    result = macro_invoker.undo()
    print(f"   {result}")
    macro_editor.display()
    print()
    
    result = macro_invoker.undo()
    print(f"   {result}")
    macro_editor.display()
    print()
    
    # 4. Command Queue and Batch Processing
    print("4. COMMAND QUEUE AND BATCH PROCESSING:")
    
    # Create new calculator for queue demo
    queue_calculator = Calculator()
    queue_invoker = CommandInvoker()
    command_queue = CommandQueue(queue_invoker)
    
    # Add commands to queue
    batch_commands = [
        AddCommand(queue_calculator, 5),
        MultiplyCommand(queue_calculator, 3),
        SubtractCommand(queue_calculator, 2),
        DivideCommand(queue_calculator, 4),
        AddCommand(queue_calculator, 10),
        MultiplyCommand(queue_calculator, 2)
    ]
    
    print("   Adding commands to queue:")
    command_queue.add_commands(batch_commands)
    
    queue_info = command_queue.get_queue_info()
    print(f"   Queue size: {queue_info['queue_size']}")
    print(f"   Pending commands: {queue_info['pending_commands']}")
    print()
    
    # Execute batch of commands
    print("   Executing batch of 3 commands:")
    results = command_queue.execute_batch(3)
    queue_calculator.display()
    print()
    
    # Execute remaining commands
    print("   Executing remaining commands:")
    results = command_queue.execute_all()
    queue_calculator.display()
    print()
    
    # Show final queue info
    final_queue_info = command_queue.get_queue_info()
    print(f"   Final queue info: {final_queue_info}")
    
    print()
    
    # 5. Command Logging
    print("5. COMMAND LOGGING:")
    
    # Create logging invoker
    logger = CommandLogger()
    logging_invoker = LoggingCommandInvoker(logger=logger)
    
    # Create new calculator for logging demo
    log_calculator = Calculator()
    
    # Execute commands with logging
    log_commands = [
        AddCommand(log_calculator, 100),
        MultiplyCommand(log_calculator, 0.5),
        DivideCommand(log_calculator, 10),
        SubtractCommand(log_calculator, 5)
    ]
    
    print("   Executing commands with logging:")
    for command in log_commands:
        logging_invoker.execute_command(command)
    
    # Test undo with logging
    print("\n   Testing undo with logging:")
    logging_invoker.undo()
    logging_invoker.undo()
    
    # Test redo with logging
    print("\n   Testing redo with logging:")
    logging_invoker.redo()
    
    # Show log summary
    log_summary = logger.get_log_summary()
    print(f"\n   Log Summary:")
    print(f"     Total entries: {log_summary['total_entries']}")
    print(f"     Operations: {log_summary['operations']}")
    print(f"     Command types: {log_summary['command_types']}")
    print(f"     Success rate: {log_summary['success_rate']:.1f}%")
    
    # Show recent logs
    print(f"\n   Recent log entries:")
    recent_logs = logger.get_recent_logs(3)
    for log_entry in recent_logs:
        print(f"     {log_entry['timestamp']}: {log_entry['operation']} - {log_entry['description']}")
    
    print()
    
    # 6. Thread-Safe Command Execution
    print("6. THREAD-SAFE COMMAND EXECUTION:")
    
    class ThreadSafeInvoker(CommandInvoker):
        """Thread-safe command invoker."""
        
        def __init__(self, max_history: int = 100):
            super().__init__(max_history)
            self._lock = threading.Lock()
        
        def execute_command(self, command: Command) -> Any:
            with self._lock:
                return super().execute_command(command)
        
        def undo(self) -> Optional[str]:
            with self._lock:
                return super().undo()
        
        def redo(self) -> Optional[str]:
            with self._lock:
                return super().redo()
    
    # Create thread-safe invoker
    thread_safe_invoker = ThreadSafeInvoker()
    thread_calculator = Calculator()
    
    # Function for concurrent command execution
    def execute_commands_concurrently(invoker, calculator, commands, thread_id):
        for i, command in enumerate(commands):
            try:
                result = invoker.execute_command(command)
                print(f"   Thread {thread_id}: {result}")
                time.sleep(0.01)  # Small delay
            except Exception as e:
                print(f"   Thread {thread_id} error: {e}")
    
    # Create commands for different threads
    thread1_commands = [
        AddCommand(thread_calculator, 10),
        MultiplyCommand(thread_calculator, 2)
    ]
    
    thread2_commands = [
        SubtractCommand(thread_calculator, 5),
        DivideCommand(thread_calculator, 3)
    ]
    
    # Execute commands concurrently
    print("   Executing commands concurrently:")
    
    thread1 = threading.Thread(
        target=execute_commands_concurrently,
        args=(thread_safe_invoker, thread_calculator, thread1_commands, 1)
    )
    
    thread2 = threading.Thread(
        target=execute_commands_concurrently,
        args=(thread_safe_invoker, thread_calculator, thread2_commands, 2)
    )
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    print(f"   Final calculator state:")
    thread_calculator.display()
    
    # Show invoker statistics
    stats = thread_safe_invoker.get_statistics()
    print(f"   Invoker statistics: {stats}")
    
    print()
    
    # 7. Command Pattern Benefits
    print("7. COMMAND PATTERN BENEFITS:")
    print("   ✓ Request Encapsulation: Requests are encapsulated as objects")
    print("   ✓ Decoupling: Invoker is decoupled from receiver")
    print("   ✓ Undo/Redo Support: Easy to implement undo and redo operations")
    print("   ✓ Macro Commands: Multiple commands can be combined")
    print("   ✓ Queuing: Commands can be queued for batch processing")
    print("   ✓ Logging: Command execution can be logged and audited")
    print("   ✓ Transactional Behavior: Commands can be rolled back on failure")
    print("   ✓ Remote Execution: Commands can be serialized and executed remotely")
    print("   ✓ Progress Tracking: Command execution progress can be monitored")
    print("   ✓ Parameterization: Clients can be parameterized with different requests")
    print()
    
    print("=== COMMAND PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_command_pattern()
