#!/usr/bin/env python3
"""
State Machine Agents: Managing Complex Behavior Through States
=============================================================

WHAT IS THE PROBLEM?
==================
Complex systems need to behave differently in different situations, but managing all the possible behaviors and transitions becomes chaotic without structure.

Example: Traffic Light Without State Management
BAD APPROACH:
- Check current time
- Check traffic sensors  
- Check emergency signals
- Check pedestrian buttons
- Try to figure out what to do with all this information
- Make random decisions
- Result: Chaos, accidents, inefficiency

REAL WORLD EXAMPLE:
=================
How does a real traffic light system work?

STATES: Red, Yellow, Green, Emergency
CLEAR RULES FOR EACH STATE:

RED STATE:
- Cars must stop
- Pedestrians can cross
- Wait 30 seconds, then transition to GREEN
- If emergency vehicle detected, go to EMERGENCY

GREEN STATE:
- Cars can go
- Pedestrians must wait
- After 45 seconds, transition to YELLOW
- If emergency vehicle detected, go to EMERGENCY

YELLOW STATE:
- Cars should prepare to stop
- Wait 5 seconds, then transition to RED

EMERGENCY STATE:
- All directions stop except emergency route
- When emergency clears, return to previous state

THE ALGORITHM:
=============
1. DEFINE STATES: Identify all possible system states
2. DEFINE TRANSITIONS: Specify when to move between states
3. DEFINE BEHAVIORS: Define what to do in each state
4. EXECUTE: Run the state machine, responding to events
5. MONITOR: Track state changes and system behavior

PSEUDO CODE:
===========
class StateMachine:
    def __init__(self):
        self.current_state = initial_state
        self.states = {
            'state1': State1Handler(),
            'state2': State2Handler()
        }
        self.transitions = {
            ('state1', 'event_x'): 'state2',
            ('state2', 'event_y'): 'state1'
        }
    
    def handle_event(self, event):
        # Execute current state behavior
        result = self.states[self.current_state].handle(event)
        
        # Check for state transition
        transition_key = (self.current_state, event.type)
        if transition_key in self.transitions:
            old_state = self.current_state
            self.current_state = self.transitions[transition_key]
            self.on_state_change(old_state, self.current_state)
        
        return result

WHY IS THIS POWERFUL?
===================
- Makes complex behavior manageable and predictable
- Prevents impossible or dangerous state combinations
- Makes system behavior easy to understand and debug
- Enables systematic testing of all possible scenarios
- Scales to very complex systems with many states
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class EventType(Enum):
    USER_INPUT = "user_input"
    TIMER = "timer"
    EXTERNAL_SIGNAL = "external_signal"
    ERROR = "error"
    COMPLETION = "completion"
    INTERRUPTION = "interruption"

@dataclass
class Event:
    """Represents an event that can trigger state transitions"""
    event_type: EventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

@dataclass
class StateTransition:
    """Represents a transition between states"""
    from_state: str
    to_state: str
    trigger_event: EventType
    condition: Optional[Callable] = None
    action: Optional[Callable] = None

class StateHandler(ABC):
    """Abstract base class for state handlers"""
    
    @abstractmethod
    async def on_enter(self, context: Dict[str, Any]) -> None:
        """Called when entering this state"""
        pass
    
    @abstractmethod
    async def handle_event(self, event: Event, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an event while in this state"""
        pass
    
    @abstractmethod
    async def on_exit(self, context: Dict[str, Any]) -> None:
        """Called when leaving this state"""
        pass
    
    @abstractmethod
    def get_state_name(self) -> str:
        """Get the name of this state"""
        pass

class StateMachineAgent:
    """
    An agent that manages complex behavior through well-defined states
    
    EXAMPLE USAGE:
    =============
    # Create a customer service agent with different states
    agent = StateMachineAgent("customer_service")
    
    # Define states: greeting, understanding, solving, closing
    agent.add_state("greeting", GreetingHandler())
    agent.add_state("understanding", UnderstandingHandler())
    agent.add_state("solving", SolvingHandler())
    agent.add_state("closing", ClosingHandler())
    
    # Define transitions between states
    agent.add_transition("greeting", "understanding", EventType.USER_INPUT)
    agent.add_transition("understanding", "solving", EventType.COMPLETION)
    
    # Process customer interactions
    await agent.handle_event(Event(EventType.USER_INPUT, "I need help with my order"))
    """
    
    def __init__(self, agent_id: str, initial_state: str = "initial"):
        self.agent_id = agent_id
        self.current_state = initial_state
        self.states: Dict[str, StateHandler] = {}
        self.transitions: Dict[Tuple[str, EventType], str] = {}
        self.conditional_transitions: List[StateTransition] = []
        self.context: Dict[str, Any] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.event_queue: List[Event] = []
        
        # Metrics and monitoring
        self.state_durations: Dict[str, float] = {}
        self.transition_counts: Dict[Tuple[str, str], int] = {}
        self.current_state_start_time = time.time()
    
    def add_state(self, state_name: str, handler: StateHandler) -> None:
        """Add a state with its handler"""
        self.states[state_name] = handler
        print(f"Added state: {state_name}")
    
    def add_transition(self, from_state: str, to_state: str, trigger_event: EventType, 
                      condition: Optional[Callable] = None, action: Optional[Callable] = None) -> None:
        """Add a state transition rule"""
        if condition or action:
            # Conditional transition
            transition = StateTransition(from_state, to_state, trigger_event, condition, action)
            self.conditional_transitions.append(transition)
        else:
            # Simple transition
            self.transitions[(from_state, trigger_event)] = to_state
        
        print(f"Added transition: {from_state} -> {to_state} on {trigger_event.value}")
    
    async def start(self) -> None:
        """Start the state machine"""
        print(f"Starting state machine in state: {self.current_state}")
        
        if self.current_state in self.states:
            await self.states[self.current_state].on_enter(self.context)
        
        self.current_state_start_time = time.time()
    
    async def handle_event(self, event: Event) -> Dict[str, Any]:
        """
        Handle an incoming event, potentially triggering state transitions
        """
        print(f"\nHandling event: {event.event_type.value} in state: {self.current_state}")
        
        # Record event
        self.event_queue.append(event)
        
        # Handle event in current state
        if self.current_state in self.states:
            handler = self.states[self.current_state]
            result = await handler.handle_event(event, self.context)
        else:
            result = {"error": f"No handler for state {self.current_state}"}
        
        # Check for state transitions
        await self.check_transitions(event)
        
        return {
            "current_state": self.current_state,
            "event_handled": True,
            "result": result,
            "context": self.context.copy()
        }
    
    async def check_transitions(self, event: Event) -> None:
        """Check if event triggers a state transition"""
        
        # Check simple transitions first
        transition_key = (self.current_state, event.event_type)
        if transition_key in self.transitions:
            new_state = self.transitions[transition_key]
            await self.transition_to_state(new_state, event)
            return
        
        # Check conditional transitions
        for transition in self.conditional_transitions:
            if (transition.from_state == self.current_state and 
                transition.trigger_event == event.event_type):
                
                # Check condition if present
                if transition.condition is None or transition.condition(event, self.context):
                    # Execute action if present
                    if transition.action:
                        await transition.action(event, self.context)
                    
                    await self.transition_to_state(transition.to_state, event)
                    return
    
    async def transition_to_state(self, new_state: str, triggering_event: Event) -> None:
        """Transition from current state to new state"""
        if new_state not in self.states:
            print(f"Error: State {new_state} not found")
            return
        
        old_state = self.current_state
        
        # Record state duration
        duration = time.time() - self.current_state_start_time
        self.state_durations[old_state] = self.state_durations.get(old_state, 0) + duration
        
        # Exit current state
        if old_state in self.states:
            await self.states[old_state].on_exit(self.context)
        
        # Record transition
        self.state_history.append({
            "from_state": old_state,
            "to_state": new_state,
            "trigger_event": triggering_event.event_type.value,
            "timestamp": time.time(),
            "context_snapshot": self.context.copy()
        })
        
        # Update transition counts
        transition_pair = (old_state, new_state)
        self.transition_counts[transition_pair] = self.transition_counts.get(transition_pair, 0) + 1
        
        # Change state
        self.current_state = new_state
        self.current_state_start_time = time.time()
        
        # Enter new state
        await self.states[new_state].on_enter(self.context)
        
        print(f"Transitioned: {old_state} -> {new_state}")
    
    def get_current_state(self) -> str:
        """Get current state name"""
        return self.current_state
    
    def get_possible_transitions(self) -> List[str]:
        """Get list of possible next states from current state"""
        possible_states = []
        
        # From simple transitions
        for (from_state, event_type), to_state in self.transitions.items():
            if from_state == self.current_state:
                possible_states.append(to_state)
        
        # From conditional transitions
        for transition in self.conditional_transitions:
            if transition.from_state == self.current_state:
                possible_states.append(transition.to_state)
        
        return list(set(possible_states))
    
    def get_state_machine_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the state machine"""
        return {
            "agent_id": self.agent_id,
            "current_state": self.current_state,
            "total_states": len(self.states),
            "total_transitions": len(self.transitions) + len(self.conditional_transitions),
            "state_history": len(self.state_history),
            "events_processed": len(self.event_queue),
            "possible_next_states": self.get_possible_transitions(),
            "state_durations": self.state_durations.copy(),
            "most_common_transitions": sorted(self.transition_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:3]
        }

# EXAMPLE STATE HANDLERS
# =====================

class CustomerServiceGreetingState(StateHandler):
    """State for greeting customers"""
    
    async def on_enter(self, context: Dict[str, Any]) -> None:
        context["greeting_given"] = True
        context["customer_acknowledged"] = True
        print("  [GREETING] Welcome! How can I help you today?")
    
    async def handle_event(self, event: Event, context: Dict[str, Any]) -> Dict[str, Any]:
        if event.event_type == EventType.USER_INPUT:
            customer_message = event.data
            context["customer_request"] = customer_message
            print(f"  [GREETING] Customer said: {customer_message}")
            
            # Acknowledge and prepare to understand their need
            return {
                "response": "Thank you for contacting us. Let me understand your request better.",
                "action": "transition_to_understanding"
            }
        
        return {"response": "I'm here to help. Please tell me what you need."}
    
    async def on_exit(self, context: Dict[str, Any]) -> None:
        print("  [GREETING] Moving to understand customer needs...")
    
    def get_state_name(self) -> str:
        return "greeting"

class CustomerServiceUnderstandingState(StateHandler):
    """State for understanding customer problems"""
    
    async def on_enter(self, context: Dict[str, Any]) -> None:
        context["understanding_phase"] = True
        print("  [UNDERSTANDING] Analyzing customer request...")
    
    async def handle_event(self, event: Event, context: Dict[str, Any]) -> Dict[str, Any]:
        if event.event_type == EventType.USER_INPUT:
            details = event.data
            context["problem_details"] = details
            context["understanding_complete"] = True
            
            print(f"  [UNDERSTANDING] Problem details: {details}")
            
            # Analyze complexity to determine next action
            if "simple" in str(details).lower() or "quick" in str(details).lower():
                context["problem_complexity"] = "simple"
                return {"response": "I understand. This is a straightforward issue I can resolve quickly."}
            else:
                context["problem_complexity"] = "complex"
                return {"response": "I see. This requires some investigation. Let me work on this for you."}
        
        return {"response": "Could you provide more details about your issue?"}
    
    async def on_exit(self, context: Dict[str, Any]) -> None:
        print("  [UNDERSTANDING] Customer problem understood, moving to solution...")
    
    def get_state_name(self) -> str:
        return "understanding"

class CustomerServiceSolvingState(StateHandler):
    """State for solving customer problems"""
    
    async def on_enter(self, context: Dict[str, Any]) -> None:
        context["solving_phase"] = True
        complexity = context.get("problem_complexity", "unknown")
        print(f"  [SOLVING] Working on {complexity} problem...")
    
    async def handle_event(self, event: Event, context: Dict[str, Any]) -> Dict[str, Any]:
        if event.event_type == EventType.COMPLETION:
            # Solution found
            context["solution_provided"] = True
            context["customer_satisfied"] = True
            
            print("  [SOLVING] Solution implemented successfully!")
            return {
                "response": "Great! I've resolved your issue. Is there anything else I can help you with?",
                "solution_status": "completed"
            }
        
        elif event.event_type == EventType.USER_INPUT:
            feedback = event.data
            if "thanks" in str(feedback).lower() or "solved" in str(feedback).lower():
                context["customer_satisfied"] = True
                return {"response": "You're welcome! Glad I could help."}
            else:
                return {"response": "Let me try a different approach to resolve this."}
        
        return {"response": "Working on your request..."}
    
    async def on_exit(self, context: Dict[str, Any]) -> None:
        print("  [SOLVING] Solution phase complete, wrapping up...")
    
    def get_state_name(self) -> str:
        return "solving"

class CustomerServiceClosingState(StateHandler):
    """State for closing customer interactions"""
    
    async def on_enter(self, context: Dict[str, Any]) -> None:
        context["closing_phase"] = True
        print("  [CLOSING] Wrapping up customer interaction...")
    
    async def handle_event(self, event: Event, context: Dict[str, Any]) -> Dict[str, Any]:
        if event.event_type == EventType.USER_INPUT:
            message = event.data
            if "goodbye" in str(message).lower() or "bye" in str(message).lower():
                context["interaction_complete"] = True
                return {
                    "response": "Thank you for contacting us. Have a great day!",
                    "interaction_status": "completed"
                }
            else:
                # Customer has additional questions
                context["additional_request"] = message
                return {
                    "response": "Of course! Let me help you with that as well.",
                    "action": "transition_to_understanding"
                }
        
        return {"response": "Is there anything else I can help you with today?"}
    
    async def on_exit(self, context: Dict[str, Any]) -> None:
        if context.get("interaction_complete"):
            print("  [CLOSING] Customer interaction completed successfully")
        else:
            print("  [CLOSING] Handling additional customer request...")
    
    def get_state_name(self) -> str:
        return "closing"

# EXAMPLE DEMONSTRATIONS
# =====================

async def create_customer_service_agent() -> StateMachineAgent:
    """Create a customer service agent with state machine"""
    agent = StateMachineAgent("customer_service_bot", "greeting")
    
    # Add states
    agent.add_state("greeting", CustomerServiceGreetingState())
    agent.add_state("understanding", CustomerServiceUnderstandingState())
    agent.add_state("solving", CustomerServiceSolvingState())
    agent.add_state("closing", CustomerServiceClosingState())
    
    # Add transitions
    agent.add_transition("greeting", "understanding", EventType.USER_INPUT)
    
    # Conditional transition based on understanding completion
    def understanding_complete(event, context):
        return context.get("understanding_complete", False)
    
    agent.add_transition("understanding", "solving", EventType.COMPLETION, understanding_complete)
    
    # Transition to closing when solution is provided
    def solution_provided(event, context):
        return context.get("solution_provided", False)
    
    agent.add_transition("solving", "closing", EventType.COMPLETION, solution_provided)
    
    # Allow returning to understanding from closing for additional questions
    def has_additional_request(event, context):
        return context.get("additional_request") is not None
    
    agent.add_transition("closing", "understanding", EventType.USER_INPUT, has_additional_request)
    
    return agent

async def demo_customer_service_interaction():
    """Demo: Customer service conversation using state machine"""
    print("\nDEMO 1: CUSTOMER SERVICE STATE MACHINE")
    print("=" * 50)
    
    agent = await create_customer_service_agent()
    await agent.start()
    
    # Simulate customer interaction
    interactions = [
        ("I need help with my order", EventType.USER_INPUT),
        ("My order hasn't arrived yet", EventType.USER_INPUT),
        ("Solution found", EventType.COMPLETION),  # Simulating solution
        ("Thank you!", EventType.USER_INPUT),
        ("Actually, I have another question", EventType.USER_INPUT),
        ("How do I return an item?", EventType.USER_INPUT),
        ("Problem solved", EventType.COMPLETION),
        ("Goodbye", EventType.USER_INPUT)
    ]
    
    for message, event_type in interactions:
        print(f"\nCustomer/System: {message}")
        event = Event(event_type, message)
        result = await agent.handle_event(event)
        
        if "response" in result["result"]:
            print(f"Agent: {result['result']['response']}")
        
        # Brief delay for readability
        await asyncio.sleep(0.2)
    
    # Show final status
    status = agent.get_state_machine_status()
    print(f"\nFinal Status:")
    print(f"Current State: {status['current_state']}")
    print(f"Total State Changes: {status['state_history']}")
    print(f"Events Processed: {status['events_processed']}")

async def demo_traffic_light_system():
    """Demo: Traffic light system using state machine"""
    print("\nDEMO 2: TRAFFIC LIGHT STATE MACHINE")
    print("=" * 50)
    
    # Simple traffic light implementation
    class TrafficLightRedState(StateHandler):
        async def on_enter(self, context): 
            print("  TRAFFIC LIGHT: RED - Cars stop, pedestrians cross")
        async def handle_event(self, event, context): 
            if event.event_type == EventType.TIMER:
                return {"action": "change_to_green"}
            return {"status": "red_active"}
        async def on_exit(self, context): 
            print("  TRAFFIC LIGHT: Changing from RED...")
        def get_state_name(self): 
            return "red"
    
    class TrafficLightGreenState(StateHandler):
        async def on_enter(self, context): 
            print("  TRAFFIC LIGHT: GREEN - Cars go, pedestrians wait")
        async def handle_event(self, event, context): 
            if event.event_type == EventType.TIMER:
                return {"action": "change_to_yellow"}
            return {"status": "green_active"}
        async def on_exit(self, context): 
            print("  TRAFFIC LIGHT: Changing from GREEN...")
        def get_state_name(self): 
            return "green"
    
    class TrafficLightYellowState(StateHandler):
        async def on_enter(self, context): 
            print("  TRAFFIC LIGHT: YELLOW - Cars prepare to stop")
        async def handle_event(self, event, context): 
            if event.event_type == EventType.TIMER:
                return {"action": "change_to_red"}
            return {"status": "yellow_active"}
        async def on_exit(self, context): 
            print("  TRAFFIC LIGHT: Changing from YELLOW...")
        def get_state_name(self): 
            return "yellow"
    
    # Create traffic light agent
    traffic_light = StateMachineAgent("traffic_light", "red")
    traffic_light.add_state("red", TrafficLightRedState())
    traffic_light.add_state("green", TrafficLightGreenState())
    traffic_light.add_state("yellow", TrafficLightYellowState())
    
    # Add transitions
    traffic_light.add_transition("red", "green", EventType.TIMER)
    traffic_light.add_transition("green", "yellow", EventType.TIMER)
    traffic_light.add_transition("yellow", "red", EventType.TIMER)
    
    await traffic_light.start()
    
    # Simulate timer events
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        for phase in ["Timer: 30s (Red)", "Timer: 45s (Green)", "Timer: 5s (Yellow)"]:
            print(f"\nSystem: {phase}")
            event = Event(EventType.TIMER, phase)
            await traffic_light.handle_event(event)
            await asyncio.sleep(0.3)

async def main():
    """
    Demonstrate State Machine Agents managing complex behavior
    
    WHAT YOU'LL LEARN:
    ================
    1. How to model complex behavior using states and transitions
    2. How state machines prevent invalid or dangerous states
    3. How to handle events and trigger appropriate responses
    4. How to make system behavior predictable and debuggable
    5. How state machines scale to very complex systems
    
    REAL WORLD APPLICATIONS:
    =======================
    - Customer service chatbots with conversation flow
    - Game AI with different behavior modes
    - Workflow management systems
    - IoT device controllers (smart home, robots)
    - Financial transaction processing
    - User interface navigation and interaction
    """
    
    print("STATE MACHINE AGENTS DEMONSTRATION")
    print("This shows how to manage complex behavior through well-defined states!")
    
    await demo_customer_service_interaction()
    await demo_traffic_light_system()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ States make complex behavior manageable and predictable")
    print("✓ Transitions define clear rules for changing behavior")
    print("✓ Event handling enables responsive system behavior")
    print("✓ State machines prevent impossible or dangerous conditions")
    print("✓ Complex systems become easier to understand and debug")
    print("\nTRY IT YOURSELF:")
    print("- Model your own complex system using states")
    print("- Add error handling and recovery states")
    print("- Implement hierarchical state machines for more complexity")
    print("- Add state machine visualization and monitoring")

if __name__ == "__main__":
    asyncio.run(main())
