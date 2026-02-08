"""
Sample input module for Research Assistant system.
Contains predefined research topics and expected sections for testing.
"""

from typing import Dict, List
from Main import Setup_Research_System, Run_Research


# Predefined research topics with detailed descriptions
RESEARCH_TOPICS = [
    {
        "topic": "Impact of AI agents on software development",
        "description": """
        This topic explores how AI agents are transforming software development practices,
        including code generation, automated testing, deployment automation, and developer
        productivity. The research should cover both current applications and future potential.
        """,
        "key_aspects": [
            "Code generation and autocomplete tools",
            "Automated testing and quality assurance",
            "CI/CD pipeline automation",
            "Developer productivity metrics",
            "Challenges and limitations",
            "Future trends and predictions"
        ]
    },
    {
        "topic": "Comparison of RAG vs fine-tuning for enterprise LLM applications",
        "description": """
        This topic compares Retrieval-Augmented Generation (RAG) and fine-tuning approaches
        for deploying large language models in enterprise settings. The research should analyze
        trade-offs in cost, performance, accuracy, and maintenance requirements.
        """,
        "key_aspects": [
            "RAG architecture and implementation",
            "Fine-tuning process and requirements",
            "Cost comparison and scalability",
            "Accuracy and performance metrics",
            "Maintenance and update strategies",
            "Use case recommendations"
        ]
    },
    {
        "topic": "Multi-agent systems in autonomous driving",
        "description": """
        This topic examines how multi-agent systems are used in autonomous vehicle technology,
        including coordination between agents, decision-making processes, safety considerations,
        and real-world deployment challenges.
        """,
        "key_aspects": [
            "Multi-agent architecture in vehicles",
            "Agent coordination and communication",
            "Perception and planning agents",
            "Safety and reliability mechanisms",
            "Real-world deployment examples",
            "Regulatory and ethical considerations"
        ]
    }
]


# Expected report sections for each topic
EXPECTED_SECTIONS = {
    "Impact of AI agents on software development": [
        "Executive Summary",
        "Introduction",
        "Current State of AI Agents in Development",
        "Code Generation and Automation Tools",
        "Testing and Quality Assurance",
        "Productivity and Efficiency Gains",
        "Challenges and Limitations",
        "Future Outlook",
        "Conclusion",
        "References"
    ],
    "Comparison of RAG vs fine-tuning for enterprise LLM applications": [
        "Executive Summary",
        "Introduction",
        "Methodology",
        "RAG Architecture Overview",
        "Fine-Tuning Approach",
        "Cost Analysis",
        "Performance Comparison",
        "Use Case Analysis",
        "Recommendations",
        "Conclusion",
        "References"
    ],
    "Multi-agent systems in autonomous driving": [
        "Executive Summary",
        "Introduction",
        "Multi-Agent System Architecture",
        "Agent Coordination Mechanisms",
        "Perception and Planning Agents",
        "Safety and Reliability",
        "Real-World Applications",
        "Challenges and Future Directions",
        "Conclusion",
        "References"
    ]
}


def Run_Samples():
    """
    Run research on all predefined sample topics and display results.
    """
    print("\n" + "="*70)
    print("Research Assistant - Sample Topics Execution")
    print("="*70)
    
    # Setup research system
    try:
        research_graph = Setup_Research_System()
        print("\nResearch system initialized successfully.\n")
    except Exception as e:
        print(f"Error initializing research system: {e}")
        print("Please check your API key configuration.")
        return
    
    # Process each topic
    for idx, topic_data in enumerate(RESEARCH_TOPICS, 1):
        topic = topic_data["topic"]
        description = topic_data["description"].strip()
        key_aspects = topic_data["key_aspects"]
        
        print("\n" + "="*70)
        print(f"Sample {idx}/{len(RESEARCH_TOPICS)}: {topic}")
        print("="*70)
        print(f"\nDescription: {description}")
        print(f"\nKey Aspects to Cover:")
        for aspect in key_aspects:
            print(f"  - {aspect}")
        
        # Get expected sections
        expected = EXPECTED_SECTIONS.get(topic, [])
        if expected:
            print(f"\nExpected Report Sections:")
            for section in expected:
                print(f"  - {section}")
        
        print("\n" + "-"*70)
        print("Executing Research...")
        print("-"*70 + "\n")
        
        try:
            # Run research
            report = Run_Research(topic, research_graph)
            
            # Display report
            print("\n" + "="*70)
            print(f"RESEARCH REPORT: {topic}")
            print("="*70)
            print(report)
            print("="*70)
            
            # Verify sections (basic check)
            if expected:
                print("\nSection Verification:")
                found_sections = []
                for section in expected:
                    if section.lower() in report.lower():
                        found_sections.append(section)
                        print(f"  ✓ Found: {section}")
                    else:
                        print(f"  ✗ Missing: {section}")
                
                coverage = len(found_sections) / len(expected) * 100
                print(f"\nSection Coverage: {coverage:.1f}% ({len(found_sections)}/{len(expected)} sections)")
            
        except Exception as e:
            print(f"\nError researching topic '{topic}': {e}")
            print("Skipping to next topic...\n")
            continue
        
        # Pause between topics (except for last one)
        if idx < len(RESEARCH_TOPICS):
            print("\n" + "-"*70)
            input("Press Enter to continue to next sample topic...")
            print("-"*70)
    
    print("\n" + "="*70)
    print("All sample topics completed!")
    print("="*70 + "\n")


def Run_Single_Sample(topic_index: int = 0):
    """
    Run research on a single sample topic by index.
    
    Args:
        topic_index: Index of topic in RESEARCH_TOPICS (0-based)
    """
    if topic_index < 0 or topic_index >= len(RESEARCH_TOPICS):
        print(f"Invalid topic index. Please use 0-{len(RESEARCH_TOPICS)-1}")
        return
    
    topic_data = RESEARCH_TOPICS[topic_index]
    topic = topic_data["topic"]
    
    print(f"\nRunning single sample: {topic}\n")
    
    try:
        research_graph = Setup_Research_System()
        report = Run_Research(topic, research_graph)
        
        print("\n" + "="*70)
        print(f"RESEARCH REPORT: {topic}")
        print("="*70)
        print(report)
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")


def List_Samples():
    """
    List all available sample topics with their descriptions.
    """
    print("\n" + "="*70)
    print("Available Sample Research Topics")
    print("="*70 + "\n")
    
    for idx, topic_data in enumerate(RESEARCH_TOPICS):
        topic = topic_data["topic"]
        description = topic_data["description"].strip()
        key_aspects = topic_data["key_aspects"]
        
        print(f"{idx + 1}. {topic}")
        print(f"   Description: {description[:100]}...")
        print(f"   Key Aspects: {len(key_aspects)} aspects to cover")
        print()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            List_Samples()
        elif command == "single" and len(sys.argv) > 2:
            try:
                index = int(sys.argv[2])
                Run_Single_Sample(index)
            except ValueError:
                print("Invalid index. Please provide a number.")
        else:
            print("Usage:")
            print("  python Sample_Input.py           - Run all samples")
            print("  python Sample_Input.py list      - List available topics")
            print("  python Sample_Input.py single N  - Run sample N (0-based index)")
    else:
        # Run all samples by default
        Run_Samples()
