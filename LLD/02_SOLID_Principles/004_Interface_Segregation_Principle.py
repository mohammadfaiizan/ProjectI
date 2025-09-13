"""
INTERFACE SEGREGATION PRINCIPLE - ISP and Focused Interfaces
============================================================

Problem Statement:
Demonstrate the Interface Segregation Principle (ISP):
- Clients should not be forced to depend on interfaces they don't use
- Many specific interfaces are better than one general-purpose interface
- Avoiding fat interfaces and interface pollution
- Creating focused, cohesive interfaces
- Role-based interface design

Learning Objectives:
- Understand the Interface Segregation Principle
- Identify fat interfaces and interface pollution
- Design focused, role-based interfaces
- Avoid forcing clients to implement unused methods
- Create cohesive interface hierarchies
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
from enum import Enum


# VIOLATION EXAMPLE - Fat interface that violates ISP
class BadWorker(ABC):
    """
    BAD EXAMPLE: Fat interface that forces all workers to implement all methods.
    This violates ISP because not all workers can do all tasks.
    """
    
    @abstractmethod
    def work(self) -> str:
        """All workers can work."""
        pass
    
    @abstractmethod
    def eat(self) -> str:
        """All workers can eat."""
        pass
    
    @abstractmethod
    def sleep(self) -> str:
        """All workers can sleep."""
        pass
    
    @abstractmethod
    def program(self) -> str:
        """ISP VIOLATION: Not all workers can program!"""
        pass
    
    @abstractmethod
    def design(self) -> str:
        """ISP VIOLATION: Not all workers can design!"""
        pass
    
    @abstractmethod
    def manage_team(self) -> str:
        """ISP VIOLATION: Not all workers can manage teams!"""
        pass
    
    @abstractmethod
    def write_documentation(self) -> str:
        """ISP VIOLATION: Not all workers write documentation!"""
        pass


class BadDeveloper(BadWorker):
    """Developer forced to implement all methods, even irrelevant ones."""
    
    def work(self) -> str:
        return "Developer is coding"
    
    def eat(self) -> str:
        return "Developer is eating lunch"
    
    def sleep(self) -> str:
        return "Developer is sleeping"
    
    def program(self) -> str:
        return "Developer is programming"
    
    def design(self) -> str:
        return "Developer is designing software architecture"
    
    def manage_team(self) -> str:
        # ISP VIOLATION: Not all developers manage teams!
        raise NotImplementedError("This developer doesn't manage teams")
    
    def write_documentation(self) -> str:
        return "Developer is writing technical documentation"


class BadManager(BadWorker):
    """Manager forced to implement all methods, even irrelevant ones."""
    
    def work(self) -> str:
        return "Manager is working on strategy"
    
    def eat(self) -> str:
        return "Manager is eating in meeting room"
    
    def sleep(self) -> str:
        return "Manager is sleeping"
    
    def program(self) -> str:
        # ISP VIOLATION: Not all managers program!
        raise NotImplementedError("This manager doesn't program")
    
    def design(self) -> str:
        # ISP VIOLATION: Not all managers design!
        raise NotImplementedError("This manager doesn't design")
    
    def manage_team(self) -> str:
        return "Manager is managing the team"
    
    def write_documentation(self) -> str:
        return "Manager is writing project documentation"


# GOOD EXAMPLE - ISP-compliant design with focused interfaces

# 1. Basic worker interface - common to all workers
class Worker(ABC):
    """Basic worker interface with common behaviors."""
    
    def __init__(self, name: str, employee_id: str):
        self.name = name
        self.employee_id = employee_id
        self.start_date = datetime.now()
    
    @abstractmethod
    def work(self) -> str:
        """All workers can work."""
        pass
    
    def take_break(self) -> str:
        """All workers can take breaks."""
        return f"{self.name} is taking a break"
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get worker information."""
        return {
            'name': self.name,
            'employee_id': self.employee_id,
            'start_date': self.start_date.isoformat(),
            'type': self.__class__.__name__
        }


# 2. Specific interfaces for different capabilities
class Programmer(Protocol):
    """Interface for workers who can program."""
    
    def program(self) -> str:
        """Program software."""
        ...
    
    def debug_code(self) -> str:
        """Debug code issues."""
        ...
    
    def code_review(self) -> str:
        """Review code."""
        ...


class Designer(Protocol):
    """Interface for workers who can design."""
    
    def design(self) -> str:
        """Create designs."""
        ...
    
    def create_mockups(self) -> str:
        """Create design mockups."""
        ...
    
    def user_research(self) -> str:
        """Conduct user research."""
        ...


class TeamManager(Protocol):
    """Interface for workers who can manage teams."""
    
    def manage_team(self) -> str:
        """Manage team members."""
        ...
    
    def conduct_meetings(self) -> str:
        """Conduct team meetings."""
        ...
    
    def performance_review(self) -> str:
        """Conduct performance reviews."""
        ...


class DocumentWriter(Protocol):
    """Interface for workers who write documentation."""
    
    def write_documentation(self) -> str:
        """Write documentation."""
        ...
    
    def update_wiki(self) -> str:
        """Update project wiki."""
        ...


class TechnicalWriter(Protocol):
    """Interface for technical writing."""
    
    def write_technical_specs(self) -> str:
        """Write technical specifications."""
        ...
    
    def create_user_manuals(self) -> str:
        """Create user manuals."""
        ...


# 3. Concrete implementations using only relevant interfaces
class Developer(Worker):
    """Developer implementing only relevant interfaces."""
    
    def __init__(self, name: str, employee_id: str, programming_languages: List[str]):
        super().__init__(name, employee_id)
        self.programming_languages = programming_languages
        self.projects_completed = 0
    
    def work(self) -> str:
        """Developer's work."""
        return f"{self.name} is developing software"
    
    def program(self) -> str:
        """Programming implementation."""
        return f"{self.name} is programming in {', '.join(self.programming_languages)}"
    
    def debug_code(self) -> str:
        """Debug code implementation."""
        return f"{self.name} is debugging code issues"
    
    def code_review(self) -> str:
        """Code review implementation."""
        return f"{self.name} is reviewing code for quality"
    
    def write_documentation(self) -> str:
        """Technical documentation implementation."""
        return f"{self.name} is writing technical documentation"
    
    def update_wiki(self) -> str:
        """Wiki update implementation."""
        return f"{self.name} is updating the project wiki"


class UIDesigner(Worker):
    """UI Designer implementing only relevant interfaces."""
    
    def __init__(self, name: str, employee_id: str, design_tools: List[str]):
        super().__init__(name, employee_id)
        self.design_tools = design_tools
        self.designs_created = 0
    
    def work(self) -> str:
        """Designer's work."""
        return f"{self.name} is creating UI designs"
    
    def design(self) -> str:
        """Design implementation."""
        return f"{self.name} is designing user interfaces using {', '.join(self.design_tools)}"
    
    def create_mockups(self) -> str:
        """Mockup creation implementation."""
        return f"{self.name} is creating design mockups"
    
    def user_research(self) -> str:
        """User research implementation."""
        return f"{self.name} is conducting user research"


class ProjectManager(Worker):
    """Project Manager implementing only relevant interfaces."""
    
    def __init__(self, name: str, employee_id: str, team_size: int):
        super().__init__(name, employee_id)
        self.team_size = team_size
        self.projects_managed = 0
    
    def work(self) -> str:
        """Manager's work."""
        return f"{self.name} is managing projects and team of {self.team_size}"
    
    def manage_team(self) -> str:
        """Team management implementation."""
        return f"{self.name} is managing a team of {self.team_size} members"
    
    def conduct_meetings(self) -> str:
        """Meeting implementation."""
        return f"{self.name} is conducting team meetings"
    
    def performance_review(self) -> str:
        """Performance review implementation."""
        return f"{self.name} is conducting performance reviews"
    
    def write_documentation(self) -> str:
        """Project documentation implementation."""
        return f"{self.name} is writing project documentation"
    
    def update_wiki(self) -> str:
        """Wiki update implementation."""
        return f"{self.name} is updating project status on wiki"


class TechLead(Worker):
    """Tech Lead implementing multiple relevant interfaces."""
    
    def __init__(self, name: str, employee_id: str, technologies: List[str]):
        super().__init__(name, employee_id)
        self.technologies = technologies
        self.team_members = []
    
    def work(self) -> str:
        """Tech lead's work."""
        return f"{self.name} is leading technical development"
    
    # Programmer interface
    def program(self) -> str:
        return f"{self.name} is programming complex features"
    
    def debug_code(self) -> str:
        return f"{self.name} is debugging critical issues"
    
    def code_review(self) -> str:
        return f"{self.name} is reviewing team's code"
    
    # Designer interface (architectural design)
    def design(self) -> str:
        return f"{self.name} is designing system architecture"
    
    def create_mockups(self) -> str:
        return f"{self.name} is creating technical diagrams"
    
    def user_research(self) -> str:
        return f"{self.name} is researching technical requirements"
    
    # Team Manager interface
    def manage_team(self) -> str:
        return f"{self.name} is managing technical team"
    
    def conduct_meetings(self) -> str:
        return f"{self.name} is conducting technical meetings"
    
    def performance_review(self) -> str:
        return f"{self.name} is reviewing technical performance"
    
    # Documentation interface
    def write_documentation(self) -> str:
        return f"{self.name} is writing technical specifications"
    
    def update_wiki(self) -> str:
        return f"{self.name} is updating technical wiki"
    
    def write_technical_specs(self) -> str:
        return f"{self.name} is writing detailed technical specifications"
    
    def create_user_manuals(self) -> str:
        return f"{self.name} is creating technical user manuals"


# DEVICE INTERFACE EXAMPLE - Another ISP demonstration

# Fat interface violation
class BadDevice(ABC):
    """BAD EXAMPLE: Fat interface forcing all devices to implement all methods."""
    
    @abstractmethod
    def turn_on(self) -> bool:
        pass
    
    @abstractmethod
    def turn_off(self) -> bool:
        pass
    
    @abstractmethod
    def print_document(self, document: str) -> bool:
        """ISP VIOLATION: Not all devices can print!"""
        pass
    
    @abstractmethod
    def scan_document(self) -> str:
        """ISP VIOLATION: Not all devices can scan!"""
        pass
    
    @abstractmethod
    def send_fax(self, number: str, document: str) -> bool:
        """ISP VIOLATION: Not all devices can fax!"""
        pass
    
    @abstractmethod
    def copy_document(self, copies: int) -> bool:
        """ISP VIOLATION: Not all devices can copy!"""
        pass


# ISP-compliant device interfaces
class Device(ABC):
    """Basic device interface."""
    
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.is_on = False
    
    @abstractmethod
    def turn_on(self) -> bool:
        """Turn device on."""
        pass
    
    @abstractmethod
    def turn_off(self) -> bool:
        """Turn device off."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get device status."""
        return {
            'name': self.name,
            'model': self.model,
            'is_on': self.is_on
        }


class Printer(Protocol):
    """Interface for devices that can print."""
    
    def print_document(self, document: str) -> bool:
        """Print a document."""
        ...
    
    def get_print_queue(self) -> List[str]:
        """Get current print queue."""
        ...


class Scanner(Protocol):
    """Interface for devices that can scan."""
    
    def scan_document(self) -> str:
        """Scan a document."""
        ...
    
    def set_scan_quality(self, quality: str) -> None:
        """Set scanning quality."""
        ...


class FaxMachine(Protocol):
    """Interface for devices that can fax."""
    
    def send_fax(self, number: str, document: str) -> bool:
        """Send a fax."""
        ...
    
    def receive_fax(self) -> Optional[str]:
        """Receive a fax."""
        ...


class Copier(Protocol):
    """Interface for devices that can copy."""
    
    def copy_document(self, copies: int) -> bool:
        """Copy a document."""
        ...
    
    def set_copy_settings(self, color: bool, double_sided: bool) -> None:
        """Set copy settings."""
        ...


# Concrete device implementations
class SimplePrinter(Device):
    """Simple printer that only prints."""
    
    def __init__(self, name: str, model: str):
        super().__init__(name, model)
        self.print_queue = []
    
    def turn_on(self) -> bool:
        self.is_on = True
        return True
    
    def turn_off(self) -> bool:
        self.is_on = False
        return True
    
    def print_document(self, document: str) -> bool:
        if self.is_on:
            print(f"{self.name} printing: {document}")
            return True
        return False
    
    def get_print_queue(self) -> List[str]:
        return self.print_queue.copy()


class DocumentScanner(Device):
    """Scanner that only scans."""
    
    def __init__(self, name: str, model: str):
        super().__init__(name, model)
        self.scan_quality = "medium"
    
    def turn_on(self) -> bool:
        self.is_on = True
        return True
    
    def turn_off(self) -> bool:
        self.is_on = False
        return True
    
    def scan_document(self) -> str:
        if self.is_on:
            return f"{self.name} scanned document at {self.scan_quality} quality"
        return "Scanner is off"
    
    def set_scan_quality(self, quality: str) -> None:
        self.scan_quality = quality


class MultiFunctionPrinter(Device):
    """Multi-function printer implementing multiple interfaces."""
    
    def __init__(self, name: str, model: str):
        super().__init__(name, model)
        self.print_queue = []
        self.scan_quality = "high"
        self.fax_number = None
    
    def turn_on(self) -> bool:
        self.is_on = True
        return True
    
    def turn_off(self) -> bool:
        self.is_on = False
        return True
    
    # Printer interface
    def print_document(self, document: str) -> bool:
        if self.is_on:
            print(f"{self.name} printing: {document}")
            return True
        return False
    
    def get_print_queue(self) -> List[str]:
        return self.print_queue.copy()
    
    # Scanner interface
    def scan_document(self) -> str:
        if self.is_on:
            return f"{self.name} scanned document at {self.scan_quality} quality"
        return "Device is off"
    
    def set_scan_quality(self, quality: str) -> None:
        self.scan_quality = quality
    
    # Copier interface
    def copy_document(self, copies: int) -> bool:
        if self.is_on:
            print(f"{self.name} making {copies} copies")
            return True
        return False
    
    def set_copy_settings(self, color: bool, double_sided: bool) -> None:
        settings = f"Color: {color}, Double-sided: {double_sided}"
        print(f"{self.name} copy settings: {settings}")
    
    # Fax interface
    def send_fax(self, number: str, document: str) -> bool:
        if self.is_on:
            print(f"{self.name} sending fax to {number}: {document}")
            return True
        return False
    
    def receive_fax(self) -> Optional[str]:
        if self.is_on:
            return f"Fax received by {self.name}"
        return None


# Functions that work with specific interfaces
def print_documents(printer: Printer, documents: List[str]) -> None:
    """Function that works with any printer."""
    for doc in documents:
        printer.print_document(doc)


def scan_batch(scanner: Scanner, count: int) -> List[str]:
    """Function that works with any scanner."""
    results = []
    for i in range(count):
        result = scanner.scan_document()
        results.append(result)
    return results


def manage_programmers(programmers: List[Programmer]) -> None:
    """Function that works with any programmer."""
    for programmer in programmers:
        print(f"  - {programmer.program()}")
        print(f"  - {programmer.debug_code()}")
        print(f"  - {programmer.code_review()}")


def manage_team_leaders(managers: List[TeamManager]) -> None:
    """Function that works with any team manager."""
    for manager in managers:
        print(f"  - {manager.manage_team()}")
        print(f"  - {manager.conduct_meetings()}")
        print(f"  - {manager.performance_review()}")


def demonstrate_interface_segregation_principle():
    """
    Demonstrate Interface Segregation Principle with practical examples.
    """
    print("=== INTERFACE SEGREGATION PRINCIPLE DEMONSTRATION ===\n")
    
    # 1. Show ISP violation problem
    print("1. ISP VIOLATION PROBLEM:")
    print("   BadWorker interface forces all workers to implement all methods,")
    print("   even those they don't need or can't perform.")
    print("   This leads to:")
    print("   - Empty implementations")
    print("   - NotImplementedError exceptions")
    print("   - Tight coupling")
    print("   - Difficult maintenance")
    print()
    
    # 2. ISP-compliant worker system
    print("2. ISP-COMPLIANT WORKER SYSTEM:")
    
    # Create workers with only relevant capabilities
    workers = [
        Developer("Alice", "DEV001", ["Python", "JavaScript"]),
        UIDesigner("Bob", "DES001", ["Figma", "Sketch"]),
        ProjectManager("Carol", "PM001", 8),
        TechLead("David", "TL001", ["Python", "AWS", "Docker"])
    ]
    
    print("   Created workers with focused responsibilities:")
    for worker in workers:
        print(f"     {worker.name}: {worker.__class__.__name__}")
    
    print()
    
    # 3. Use workers based on their specific capabilities
    print("3. USING WORKERS BY THEIR CAPABILITIES:")
    
    # Find programmers
    programmers = [w for w in workers if isinstance(w, Programmer)]
    if programmers:
        print("   Programming tasks:")
        manage_programmers(programmers)
    
    # Find team managers
    managers = [w for w in workers if isinstance(w, TeamManager)]
    if managers:
        print("\n   Management tasks:")
        manage_team_leaders(managers)
    
    # Find designers
    designers = [w for w in workers if isinstance(w, Designer)]
    if designers:
        print("\n   Design tasks:")
        for designer in designers:
            print(f"     - {designer.design()}")
            print(f"     - {designer.create_mockups()}")
            print(f"     - {designer.user_research()}")
    
    print()
    
    # 4. Device interface example
    print("4. DEVICE INTERFACE EXAMPLE:")
    
    # Create devices with specific capabilities
    devices = [
        SimplePrinter("HP Printer", "LaserJet Pro"),
        DocumentScanner("Canon Scanner", "CanoScan"),
        MultiFunctionPrinter("Epson MFP", "WorkForce Pro")
    ]
    
    print("   Created devices:")
    for device in devices:
        device.turn_on()
        print(f"     {device.name} ({device.__class__.__name__}): {'On' if device.is_on else 'Off'}")
    
    # Use devices based on their capabilities
    print("\n   Using devices by capability:")
    
    # Print documents
    printers = [d for d in devices if isinstance(d, Printer)]
    if printers:
        print("     Printing documents:")
        documents = ["Report.pdf", "Invoice.doc"]
        for printer in printers:
            print_documents(printer, documents)
    
    # Scan documents
    scanners = [d for d in devices if isinstance(d, Scanner)]
    if scanners:
        print("\n     Scanning documents:")
        for scanner in scanners:
            scanned = scan_batch(scanner, 2)
            for result in scanned:
                print(f"       {result}")
    
    # Copy documents (only multi-function devices)
    copiers = [d for d in devices if isinstance(d, Copier)]
    if copiers:
        print("\n     Copying documents:")
        for copier in copiers:
            copier.copy_document(3)
            copier.set_copy_settings(True, False)
    
    print()
    
    # 5. Show flexibility of ISP design
    print("5. FLEXIBILITY OF ISP DESIGN:")
    
    # Add new worker type that implements multiple interfaces
    class FullStackDeveloper(Worker):
        """Full-stack developer with multiple capabilities."""
        
        def __init__(self, name: str, employee_id: str):
            super().__init__(name, employee_id)
        
        def work(self) -> str:
            return f"{self.name} is doing full-stack development"
        
        # Programmer interface
        def program(self) -> str:
            return f"{self.name} is programming both frontend and backend"
        
        def debug_code(self) -> str:
            return f"{self.name} is debugging full-stack issues"
        
        def code_review(self) -> str:
            return f"{self.name} is reviewing full-stack code"
        
        # Designer interface
        def design(self) -> str:
            return f"{self.name} is designing system architecture and UI"
        
        def create_mockups(self) -> str:
            return f"{self.name} is creating technical and UI mockups"
        
        def user_research(self) -> str:
            return f"{self.name} is researching user and technical requirements"
        
        # Documentation interface
        def write_documentation(self) -> str:
            return f"{self.name} is writing comprehensive documentation"
        
        def update_wiki(self) -> str:
            return f"{self.name} is updating project wiki"
    
    fullstack_dev = FullStackDeveloper("Eve", "FS001")
    print(f"   Added new worker type: {fullstack_dev.name} ({fullstack_dev.__class__.__name__})")
    
    # Full-stack developer can work in multiple roles
    print("   Full-stack developer capabilities:")
    print(f"     Programming: {fullstack_dev.program()}")
    print(f"     Design: {fullstack_dev.design()}")
    print(f"     Documentation: {fullstack_dev.write_documentation()}")
    
    print()
    
    # 6. ISP Benefits
    print("6. ISP BENEFITS:")
    print("   ✓ Clients depend only on interfaces they actually use")
    print("   ✓ Interfaces are focused and cohesive")
    print("   ✓ Changes to one interface don't affect unrelated clients")
    print("   ✓ Easier to implement and test")
    print("   ✓ Better separation of concerns")
    print("   ✓ More flexible and maintainable code")
    print("   ✓ Supports role-based design")
    print()
    
    print("   ISP Guidelines:")
    print("   • Keep interfaces small and focused")
    print("   • Group related methods together")
    print("   • Avoid fat interfaces with many unrelated methods")
    print("   • Use composition of interfaces when needed")
    print("   • Design interfaces from the client's perspective")
    print("   • Prefer many specific interfaces over one general interface")
    print()
    
    print("=== INTERFACE SEGREGATION PRINCIPLE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_interface_segregation_principle()
