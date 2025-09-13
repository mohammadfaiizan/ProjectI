"""
OBJECT RELATIONSHIPS - Association, Aggregation, Composition
============================================================

Problem Statement:
Demonstrate different types of object relationships:
- Association (uses-a relationship)
- Aggregation (has-a relationship, weak ownership)
- Composition (part-of relationship, strong ownership)
- Dependency relationships
- Multiplicity in relationships

Learning Objectives:
- Understand different object relationship types
- Implement association, aggregation, and composition
- Choose appropriate relationship patterns
- Design flexible object interactions
- Handle object lifecycle in relationships
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from abc import ABC, abstractmethod
import weakref


# Association Example - "Uses-a" relationship
class Printer:
    """Printer class for association example."""
    
    def __init__(self, printer_id: str, model: str, location: str):
        self.printer_id = printer_id
        self.model = model
        self.location = location
        self.is_available = True
        self.paper_count = 500
        self.toner_level = 100
    
    def print_document(self, document: 'Document') -> bool:
        """Print a document."""
        if not self.is_available:
            print(f"Printer {self.printer_id} is not available")
            return False
        
        if self.paper_count < document.pages:
            print(f"Not enough paper. Need {document.pages}, have {self.paper_count}")
            return False
        
        if self.toner_level < document.pages * 2:
            print(f"Low toner level. Need {document.pages * 2}, have {self.toner_level}")
            return False
        
        # Simulate printing
        self.paper_count -= document.pages
        self.toner_level -= document.pages * 2
        print(f"Printed '{document.title}' ({document.pages} pages) on {self.model}")
        return True
    
    def refill_paper(self, sheets: int) -> None:
        """Refill paper."""
        self.paper_count += sheets
        print(f"Added {sheets} sheets. Total: {self.paper_count}")
    
    def replace_toner(self) -> None:
        """Replace toner cartridge."""
        self.toner_level = 100
        print(f"Toner replaced on {self.model}")
    
    def __str__(self) -> str:
        return f"Printer({self.printer_id}, {self.model}, {self.location})"


class Document:
    """Document class for association example."""
    
    def __init__(self, title: str, content: str, author: str):
        self.title = title
        self.content = content
        self.author = author
        self.pages = max(1, len(content) // 500)  # Estimate pages
        self.created_at = datetime.now()
    
    def print_on(self, printer: Printer) -> bool:
        """
        Association: Document uses Printer.
        Document knows about printer but doesn't own it.
        """
        return printer.print_document(self)
    
    def __str__(self) -> str:
        return f"Document('{self.title}', {self.pages} pages, by {self.author})"


# Aggregation Example - "Has-a" relationship with weak ownership
class Student:
    """Student class for aggregation example."""
    
    def __init__(self, student_id: str, name: str, email: str):
        self.student_id = student_id
        self.name = name
        self.email = email
        self.enrolled_courses: Set['Course'] = set()
        self.gpa = 0.0
    
    def enroll_in_course(self, course: 'Course') -> bool:
        """Enroll in a course."""
        if course.add_student(self):
            self.enrolled_courses.add(course)
            return True
        return False
    
    def drop_course(self, course: 'Course') -> bool:
        """Drop a course."""
        if course in self.enrolled_courses:
            course.remove_student(self)
            self.enrolled_courses.remove(course)
            return True
        return False
    
    def get_enrolled_courses(self) -> List[str]:
        """Get list of enrolled course names."""
        return [course.course_name for course in self.enrolled_courses]
    
    def __str__(self) -> str:
        return f"Student({self.student_id}, {self.name})"


class Course:
    """
    Course class demonstrating aggregation.
    Course has students, but students can exist independently.
    """
    
    def __init__(self, course_id: str, course_name: str, instructor: str, max_capacity: int = 30):
        self.course_id = course_id
        self.course_name = course_name
        self.instructor = instructor
        self.max_capacity = max_capacity
        self.enrolled_students: Set[Student] = set()  # Aggregation - weak ownership
        self.schedule = {}
    
    def add_student(self, student: Student) -> bool:
        """Add student to course (aggregation)."""
        if len(self.enrolled_students) >= self.max_capacity:
            print(f"Course {self.course_name} is at capacity")
            return False
        
        self.enrolled_students.add(student)
        print(f"Student {student.name} enrolled in {self.course_name}")
        return True
    
    def remove_student(self, student: Student) -> bool:
        """Remove student from course."""
        if student in self.enrolled_students:
            self.enrolled_students.remove(student)
            print(f"Student {student.name} dropped from {self.course_name}")
            return True
        return False
    
    def get_enrollment_count(self) -> int:
        """Get current enrollment count."""
        return len(self.enrolled_students)
    
    def get_student_list(self) -> List[str]:
        """Get list of enrolled student names."""
        return [student.name for student in self.enrolled_students]
    
    def __str__(self) -> str:
        return f"Course({self.course_id}, {self.course_name}, {len(self.enrolled_students)}/{self.max_capacity})"


# Composition Example - "Part-of" relationship with strong ownership
class Engine:
    """Engine class for composition example."""
    
    def __init__(self, engine_type: str, horsepower: int, fuel_type: str):
        self.engine_type = engine_type
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.is_running = False
        self.temperature = 20  # Celsius
        self.mileage = 0
    
    def start(self) -> bool:
        """Start the engine."""
        if not self.is_running:
            self.is_running = True
            self.temperature = 90
            print(f"{self.engine_type} engine started")
            return True
        return False
    
    def stop(self) -> bool:
        """Stop the engine."""
        if self.is_running:
            self.is_running = False
            self.temperature = 20
            print(f"{self.engine_type} engine stopped")
            return True
        return False
    
    def run(self, distance: float) -> None:
        """Run engine for distance."""
        if self.is_running:
            self.mileage += distance
            print(f"Engine ran {distance} miles. Total: {self.mileage}")
    
    def __str__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"Engine({self.engine_type}, {self.horsepower}HP, {status})"


class Transmission:
    """Transmission class for composition example."""
    
    def __init__(self, transmission_type: str, gears: int):
        self.transmission_type = transmission_type
        self.gears = gears
        self.current_gear = 0  # 0 = park/neutral
    
    def shift_up(self) -> bool:
        """Shift to higher gear."""
        if self.current_gear < self.gears:
            self.current_gear += 1
            print(f"Shifted up to gear {self.current_gear}")
            return True
        return False
    
    def shift_down(self) -> bool:
        """Shift to lower gear."""
        if self.current_gear > 0:
            self.current_gear -= 1
            print(f"Shifted down to gear {self.current_gear}")
            return True
        return False
    
    def __str__(self) -> str:
        return f"Transmission({self.transmission_type}, gear {self.current_gear}/{self.gears})"


class Car:
    """
    Car class demonstrating composition.
    Car owns its engine and transmission - they cannot exist without the car.
    """
    
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year
        
        # Composition - Car creates and owns these components
        self.engine = Engine("V6", 300, "gasoline")  # Strong ownership
        self.transmission = Transmission("automatic", 8)  # Strong ownership
        
        self.mileage = 0.0
        self.fuel_level = 50.0
    
    def start(self) -> bool:
        """Start the car."""
        return self.engine.start()
    
    def stop(self) -> bool:
        """Stop the car."""
        return self.engine.stop()
    
    def drive(self, distance: float) -> bool:
        """Drive the car."""
        if not self.engine.is_running:
            print("Cannot drive - engine not running")
            return False
        
        if self.fuel_level < distance * 0.1:
            print("Not enough fuel")
            return False
        
        # Use composed objects
        self.engine.run(distance)
        if distance > 10:  # Shift gears for longer distances
            self.transmission.shift_up()
        
        self.mileage += distance
        self.fuel_level -= distance * 0.1
        print(f"Drove {distance} miles. Total mileage: {self.mileage}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get car status including composed objects."""
        return {
            'car': f"{self.year} {self.make} {self.model}",
            'mileage': self.mileage,
            'fuel_level': self.fuel_level,
            'engine': str(self.engine),
            'transmission': str(self.transmission)
        }
    
    def __str__(self) -> str:
        return f"Car({self.year} {self.make} {self.model})"
    
    def __del__(self):
        """When car is destroyed, its composed parts are also destroyed."""
        print(f"Car {self.make} {self.model} destroyed (engine and transmission destroyed too)")


# Dependency Example
class EmailService:
    """Email service for dependency example."""
    
    def __init__(self, smtp_server: str, port: int):
        self.smtp_server = smtp_server
        self.port = port
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to email server."""
        self.is_connected = True
        print(f"Connected to {self.smtp_server}:{self.port}")
        return True
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email."""
        if not self.is_connected:
            self.connect()
        
        print(f"Email sent to {to}: {subject}")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from email server."""
        self.is_connected = False
        print(f"Disconnected from {self.smtp_server}")


class NotificationService:
    """
    Notification service demonstrating dependency.
    Depends on EmailService but doesn't own it.
    """
    
    def __init__(self):
        self.notification_history: List[Dict[str, Any]] = []
    
    def send_notification(self, user_email: str, message: str, email_service: EmailService) -> bool:
        """
        Send notification using email service (dependency).
        EmailService is passed as parameter - loose coupling.
        """
        subject = "Notification"
        success = email_service.send_email(user_email, subject, message)
        
        # Log notification
        self.notification_history.append({
            'timestamp': datetime.now(),
            'recipient': user_email,
            'message': message,
            'success': success
        })
        
        return success
    
    def get_notification_count(self) -> int:
        """Get total notification count."""
        return len(self.notification_history)


# Complex relationship example
class Department:
    """Department class showing multiple relationship types."""
    
    def __init__(self, dept_id: str, name: str, budget: float):
        self.dept_id = dept_id
        self.name = name
        self.budget = budget
        
        # Aggregation - department has employees, but employees can exist without department
        self.employees: Set['Employee'] = set()
        
        # Composition - department owns projects, projects cannot exist without department
        self.projects: List['Project'] = []
        
        # Association - department uses resources
        self.used_resources: Set['Resource'] = set()
    
    def hire_employee(self, employee: 'Employee') -> bool:
        """Hire employee (aggregation)."""
        self.employees.add(employee)
        employee.department = self  # Back reference
        print(f"Hired {employee.name} in {self.name}")
        return True
    
    def fire_employee(self, employee: 'Employee') -> bool:
        """Fire employee."""
        if employee in self.employees:
            self.employees.remove(employee)
            employee.department = None
            print(f"Fired {employee.name} from {self.name}")
            return True
        return False
    
    def create_project(self, project_name: str, budget: float) -> 'Project':
        """Create project (composition)."""
        project = Project(f"PROJ_{len(self.projects)+1}", project_name, budget, self)
        self.projects.append(project)
        print(f"Created project '{project_name}' in {self.name}")
        return project
    
    def close_project(self, project: 'Project') -> bool:
        """Close project."""
        if project in self.projects:
            self.projects.remove(project)
            print(f"Closed project '{project.name}' in {self.name}")
            return True
        return False
    
    def use_resource(self, resource: 'Resource') -> bool:
        """Use resource (association)."""
        if resource.is_available:
            self.used_resources.add(resource)
            resource.allocate_to_department(self)
            return True
        return False
    
    def release_resource(self, resource: 'Resource') -> bool:
        """Release resource."""
        if resource in self.used_resources:
            self.used_resources.remove(resource)
            resource.deallocate()
            return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get department summary."""
        return {
            'department': self.name,
            'employees': len(self.employees),
            'projects': len(self.projects),
            'resources': len(self.used_resources),
            'budget': self.budget
        }
    
    def __str__(self) -> str:
        return f"Department({self.dept_id}, {self.name})"


class Employee:
    """Employee class for complex relationships."""
    
    def __init__(self, emp_id: str, name: str, position: str):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department: Optional[Department] = None  # Back reference to department
        self.assigned_projects: Set['Project'] = set()
    
    def assign_to_project(self, project: 'Project') -> bool:
        """Assign employee to project."""
        self.assigned_projects.add(project)
        project.add_team_member(self)
        return True
    
    def remove_from_project(self, project: 'Project') -> bool:
        """Remove employee from project."""
        if project in self.assigned_projects:
            self.assigned_projects.remove(project)
            project.remove_team_member(self)
            return True
        return False
    
    def __str__(self) -> str:
        dept_name = self.department.name if self.department else "No Department"
        return f"Employee({self.emp_id}, {self.name}, {dept_name})"


class Project:
    """Project class owned by department (composition)."""
    
    def __init__(self, project_id: str, name: str, budget: float, department: Department):
        self.project_id = project_id
        self.name = name
        self.budget = budget
        self.department = department  # Strong reference to owning department
        self.team_members: Set[Employee] = set()
        self.status = "active"
    
    def add_team_member(self, employee: Employee) -> bool:
        """Add team member to project."""
        self.team_members.add(employee)
        print(f"Added {employee.name} to project {self.name}")
        return True
    
    def remove_team_member(self, employee: Employee) -> bool:
        """Remove team member from project."""
        if employee in self.team_members:
            self.team_members.remove(employee)
            print(f"Removed {employee.name} from project {self.name}")
            return True
        return False
    
    def __str__(self) -> str:
        return f"Project({self.project_id}, {self.name}, {len(self.team_members)} members)"


class Resource:
    """Resource class for association example."""
    
    def __init__(self, resource_id: str, name: str, resource_type: str):
        self.resource_id = resource_id
        self.name = name
        self.resource_type = resource_type
        self.is_available = True
        self.allocated_to: Optional[Department] = None
    
    def allocate_to_department(self, department: Department) -> bool:
        """Allocate resource to department."""
        if self.is_available:
            self.is_available = False
            self.allocated_to = department
            print(f"Resource {self.name} allocated to {department.name}")
            return True
        return False
    
    def deallocate(self) -> bool:
        """Deallocate resource."""
        if not self.is_available:
            self.is_available = True
            dept_name = self.allocated_to.name if self.allocated_to else "Unknown"
            self.allocated_to = None
            print(f"Resource {self.name} deallocated from {dept_name}")
            return True
        return False
    
    def __str__(self) -> str:
        status = "available" if self.is_available else "allocated"
        return f"Resource({self.resource_id}, {self.name}, {status})"


def demonstrate_object_relationships():
    """
    Demonstrate different types of object relationships.
    """
    print("=== OBJECT RELATIONSHIPS DEMONSTRATION ===\n")
    
    # 1. Association - "Uses-a" relationship
    print("1. ASSOCIATION - Document uses Printer:")
    
    printer1 = Printer("P001", "HP LaserJet", "Office A")
    printer2 = Printer("P002", "Canon Inkjet", "Office B")
    
    doc1 = Document("Report Q1", "This is a quarterly report with financial data and analysis." * 20, "Alice")
    doc2 = Document("Presentation", "Marketing presentation slides." * 10, "Bob")
    
    print(f"Created: {printer1}")
    print(f"Created: {printer2}")
    print(f"Created: {doc1}")
    print(f"Created: {doc2}")
    
    # Documents use printers (association)
    doc1.print_on(printer1)
    doc2.print_on(printer2)
    doc1.print_on(printer2)  # Same document can use different printers
    
    print(f"Printer1 status: Paper={printer1.paper_count}, Toner={printer1.toner_level}")
    print()
    
    # 2. Aggregation - "Has-a" relationship with weak ownership
    print("2. AGGREGATION - Course has Students (weak ownership):")
    
    # Create students
    student1 = Student("S001", "Charlie", "charlie@example.com")
    student2 = Student("S002", "Diana", "diana@example.com")
    student3 = Student("S003", "Eve", "eve@example.com")
    
    # Create courses
    course1 = Course("CS101", "Introduction to Programming", "Dr. Smith", 2)
    course2 = Course("MATH201", "Calculus I", "Dr. Johnson", 3)
    
    print(f"Created students: {student1.name}, {student2.name}, {student3.name}")
    print(f"Created courses: {course1.course_name}, {course2.course_name}")
    
    # Students enroll in courses (aggregation)
    student1.enroll_in_course(course1)
    student2.enroll_in_course(course1)
    student1.enroll_in_course(course2)
    student3.enroll_in_course(course1)  # Should fail - course at capacity
    
    print(f"Course1 students: {course1.get_student_list()}")
    print(f"Course2 students: {course2.get_student_list()}")
    print(f"Student1 courses: {student1.get_enrolled_courses()}")
    
    # Student can drop course and still exist
    student1.drop_course(course1)
    print(f"After student1 drops course1: {course1.get_student_list()}")
    print()
    
    # 3. Composition - "Part-of" relationship with strong ownership
    print("3. COMPOSITION - Car owns Engine and Transmission:")
    
    car = Car("Toyota", "Camry", 2023)
    print(f"Created: {car}")
    print(f"Car components: {car.engine}, {car.transmission}")
    
    # Use composed objects through car interface
    car.start()
    car.drive(25.0)
    car.drive(50.0)
    car.stop()
    
    print(f"Car status: {car.get_status()}")
    
    # When car is destroyed, its components are destroyed too
    del car  # This will trigger destructor
    print()
    
    # 4. Dependency - Service depends on another service
    print("4. DEPENDENCY - NotificationService depends on EmailService:")
    
    email_service = EmailService("smtp.gmail.com", 587)
    notification_service = NotificationService()
    
    # NotificationService depends on EmailService (passed as parameter)
    notification_service.send_notification("user1@example.com", "Welcome!", email_service)
    notification_service.send_notification("user2@example.com", "Update available", email_service)
    
    print(f"Total notifications sent: {notification_service.get_notification_count()}")
    email_service.disconnect()
    print()
    
    # 5. Complex relationships in one system
    print("5. COMPLEX RELATIONSHIPS - Department System:")
    
    # Create department
    it_dept = Department("IT001", "Information Technology", 500000.0)
    
    # Create employees (aggregation)
    emp1 = Employee("E001", "Frank", "Developer")
    emp2 = Employee("E002", "Grace", "Designer")
    emp3 = Employee("E003", "Henry", "Manager")
    
    # Hire employees
    it_dept.hire_employee(emp1)
    it_dept.hire_employee(emp2)
    it_dept.hire_employee(emp3)
    
    # Create projects (composition)
    project1 = it_dept.create_project("Website Redesign", 50000.0)
    project2 = it_dept.create_project("Mobile App", 75000.0)
    
    # Assign employees to projects
    emp1.assign_to_project(project1)
    emp2.assign_to_project(project1)
    emp1.assign_to_project(project2)
    emp3.assign_to_project(project2)
    
    # Create and use resources (association)
    server1 = Resource("R001", "Web Server", "Hardware")
    server2 = Resource("R002", "Database Server", "Hardware")
    software1 = Resource("R003", "Design Software License", "Software")
    
    it_dept.use_resource(server1)
    it_dept.use_resource(server2)
    it_dept.use_resource(software1)
    
    # Show department summary
    summary = it_dept.get_summary()
    print(f"Department summary: {summary}")
    
    # Show project details
    for project in it_dept.projects:
        print(f"  {project} in {project.department.name}")
    
    # Show employee assignments
    for employee in it_dept.employees:
        projects = [p.name for p in employee.assigned_projects]
        print(f"  {employee.name}: {projects}")
    
    # Show resource allocation
    for resource in it_dept.used_resources:
        print(f"  {resource}")
    
    print()
    
    # 6. Relationship Types Summary
    print("6. RELATIONSHIP TYPES SUMMARY:")
    print("ASSOCIATION (uses-a):")
    print("  - Loose coupling, temporary relationship")
    print("  - Objects can exist independently")
    print("  - Example: Document uses Printer")
    
    print("\nAGGREGATION (has-a, weak ownership):")
    print("  - 'Has-a' relationship with weak ownership")
    print("  - Child can exist without parent")
    print("  - Example: Course has Students")
    
    print("\nCOMPOSITION (part-of, strong ownership):")
    print("  - 'Part-of' relationship with strong ownership")
    print("  - Child cannot exist without parent")
    print("  - Example: Car owns Engine")
    
    print("\nDEPENDENCY:")
    print("  - One class depends on another to function")
    print("  - Usually passed as parameter or injected")
    print("  - Example: NotificationService depends on EmailService")
    
    print()
    
    print("=== OBJECT RELATIONSHIPS DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_object_relationships()
