"""
CLASSES AND OBJECTS - Basic Class Design and Instantiation
==========================================================

Problem Statement:
Demonstrate fundamental concepts of classes and objects including:
- Class definition and structure
- Object instantiation and initialization
- Instance variables and methods
- Class variables and methods
- Object lifecycle management

Learning Objectives:
- Understand the difference between class and object
- Learn proper class design principles
- Master object creation and initialization
- Implement class and instance members
"""

from typing import List, Optional
from datetime import datetime


class Student:
    """
    Student class demonstrating basic class structure and object concepts.
    
    Class Variables:
    - total_students: Tracks total number of students created
    - school_name: Shared across all student instances
    """
    
    # Class variables (shared by all instances)
    total_students = 0
    school_name = "Tech University"
    
    def __init__(self, student_id: str, name: str, age: int, email: str):
        """
        Constructor to initialize student object.
        
        Args:
            student_id: Unique identifier for student
            name: Student's full name
            age: Student's age
            email: Student's email address
        """
        # Instance variables (unique to each object)
        self.student_id = student_id
        self.name = name
        self.age = age
        self.email = email
        self.enrollment_date = datetime.now()
        self.courses: List[str] = []
        self.gpa: float = 0.0
        
        # Increment class variable when new student is created
        Student.total_students += 1
    
    def enroll_course(self, course_name: str) -> bool:
        """
        Instance method to enroll student in a course.
        
        Args:
            course_name: Name of the course to enroll in
            
        Returns:
            bool: True if enrollment successful, False otherwise
        """
        if course_name not in self.courses:
            self.courses.append(course_name)
            print(f"{self.name} enrolled in {course_name}")
            return True
        else:
            print(f"{self.name} is already enrolled in {course_name}")
            return False
    
    def drop_course(self, course_name: str) -> bool:
        """
        Instance method to drop a course.
        
        Args:
            course_name: Name of the course to drop
            
        Returns:
            bool: True if drop successful, False otherwise
        """
        if course_name in self.courses:
            self.courses.remove(course_name)
            print(f"{self.name} dropped {course_name}")
            return True
        else:
            print(f"{self.name} is not enrolled in {course_name}")
            return False
    
    def update_gpa(self, new_gpa: float) -> None:
        """
        Update student's GPA.
        
        Args:
            new_gpa: New GPA value (0.0 to 4.0)
        """
        if 0.0 <= new_gpa <= 4.0:
            self.gpa = new_gpa
            print(f"{self.name}'s GPA updated to {new_gpa}")
        else:
            print("Invalid GPA. Must be between 0.0 and 4.0")
    
    def get_student_info(self) -> dict:
        """
        Get complete student information.
        
        Returns:
            dict: Dictionary containing all student information
        """
        return {
            'student_id': self.student_id,
            'name': self.name,
            'age': self.age,
            'email': self.email,
            'enrollment_date': self.enrollment_date.strftime("%Y-%m-%d"),
            'courses': self.courses,
            'gpa': self.gpa,
            'school': Student.school_name
        }
    
    @classmethod
    def get_total_students(cls) -> int:
        """
        Class method to get total number of students.
        
        Returns:
            int: Total number of students created
        """
        return cls.total_students
    
    @classmethod
    def change_school_name(cls, new_name: str) -> None:
        """
        Class method to change school name for all students.
        
        Args:
            new_name: New school name
        """
        cls.school_name = new_name
        print(f"School name changed to {new_name}")
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Static method to validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            bool: True if email is valid, False otherwise
        """
        return "@" in email and "." in email.split("@")[1]
    
    def __str__(self) -> str:
        """String representation of student object."""
        return f"Student(ID: {self.student_id}, Name: {self.name}, GPA: {self.gpa})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation of student object."""
        return f"Student('{self.student_id}', '{self.name}', {self.age}, '{self.email}')"
    
    def __eq__(self, other) -> bool:
        """Check equality based on student ID."""
        if isinstance(other, Student):
            return self.student_id == other.student_id
        return False
    
    def __del__(self):
        """Destructor called when object is garbage collected."""
        Student.total_students -= 1
        print(f"Student {self.name} object destroyed. Total students: {Student.total_students}")


class Course:
    """
    Course class to demonstrate object relationships.
    """
    
    def __init__(self, course_id: str, course_name: str, credits: int, instructor: str):
        self.course_id = course_id
        self.course_name = course_name
        self.credits = credits
        self.instructor = instructor
        self.enrolled_students: List[Student] = []
        self.max_capacity = 30
    
    def add_student(self, student: Student) -> bool:
        """
        Add a student to the course.
        
        Args:
            student: Student object to add
            
        Returns:
            bool: True if student added successfully
        """
        if len(self.enrolled_students) < self.max_capacity:
            if student not in self.enrolled_students:
                self.enrolled_students.append(student)
                student.enroll_course(self.course_name)
                return True
            else:
                print(f"{student.name} is already enrolled in {self.course_name}")
                return False
        else:
            print(f"Course {self.course_name} is at maximum capacity")
            return False
    
    def remove_student(self, student: Student) -> bool:
        """
        Remove a student from the course.
        
        Args:
            student: Student object to remove
            
        Returns:
            bool: True if student removed successfully
        """
        if student in self.enrolled_students:
            self.enrolled_students.remove(student)
            student.drop_course(self.course_name)
            return True
        else:
            print(f"{student.name} is not enrolled in {self.course_name}")
            return False
    
    def get_enrollment_count(self) -> int:
        """Get current enrollment count."""
        return len(self.enrolled_students)
    
    def get_course_info(self) -> dict:
        """Get complete course information."""
        return {
            'course_id': self.course_id,
            'course_name': self.course_name,
            'credits': self.credits,
            'instructor': self.instructor,
            'enrolled_students': len(self.enrolled_students),
            'max_capacity': self.max_capacity,
            'available_spots': self.max_capacity - len(self.enrolled_students)
        }
    
    def __str__(self) -> str:
        return f"Course({self.course_name}, Instructor: {self.instructor}, Enrolled: {len(self.enrolled_students)})"


def demonstrate_classes_and_objects():
    """
    Demonstrate classes and objects concepts with practical examples.
    """
    print("=== CLASSES AND OBJECTS DEMONSTRATION ===\n")
    
    # 1. Object Creation and Initialization
    print("1. Creating Student Objects:")
    student1 = Student("S001", "Alice Johnson", 20, "alice@email.com")
    student2 = Student("S002", "Bob Smith", 21, "bob@email.com")
    student3 = Student("S003", "Carol Davis", 19, "carol@email.com")
    
    print(f"Created students: {student1}, {student2}, {student3}")
    print(f"Total students created: {Student.get_total_students()}\n")
    
    # 2. Instance Methods
    print("2. Using Instance Methods:")
    student1.enroll_course("Computer Science 101")
    student1.enroll_course("Mathematics 201")
    student1.update_gpa(3.8)
    
    student2.enroll_course("Computer Science 101")
    student2.enroll_course("Physics 101")
    student2.update_gpa(3.6)
    
    print()
    
    # 3. Class Methods and Variables
    print("3. Class Methods and Variables:")
    print(f"School name: {Student.school_name}")
    Student.change_school_name("Advanced Tech University")
    print(f"Updated school name: {Student.school_name}")
    print(f"Total students: {Student.get_total_students()}\n")
    
    # 4. Static Methods
    print("4. Static Method Usage:")
    emails = ["valid@email.com", "invalid-email", "another@valid.edu"]
    for email in emails:
        is_valid = Student.validate_email(email)
        print(f"Email '{email}' is {'valid' if is_valid else 'invalid'}")
    print()
    
    # 5. Object Information
    print("5. Object Information:")
    for student in [student1, student2, student3]:
        info = student.get_student_info()
        print(f"Student Info: {info}")
    print()
    
    # 6. Object Relationships
    print("6. Object Relationships with Course:")
    cs_course = Course("CS101", "Computer Science 101", 3, "Dr. Wilson")
    math_course = Course("MATH201", "Mathematics 201", 4, "Dr. Brown")
    
    cs_course.add_student(student1)
    cs_course.add_student(student2)
    math_course.add_student(student1)
    
    print(f"CS Course Info: {cs_course.get_course_info()}")
    print(f"Math Course Info: {math_course.get_course_info()}")
    print()
    
    # 7. Object Comparison
    print("7. Object Comparison:")
    student4 = Student("S001", "Alice Johnson Clone", 20, "alice2@email.com")
    print(f"student1 == student4: {student1 == student4}")  # Same ID
    print(f"student1 == student2: {student1 == student2}")  # Different ID
    print()
    
    # 8. String Representations
    print("8. String Representations:")
    print(f"str(student1): {str(student1)}")
    print(f"repr(student1): {repr(student1)}")
    print()
    
    print("=== DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_classes_and_objects()
