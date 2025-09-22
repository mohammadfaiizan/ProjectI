#!/usr/bin/env python3
"""
Peer-to-Peer Collaboration: Decentralized Agent Cooperation
===========================================================

WHAT IS THE PROBLEM?
==================
Having one central coordinator (master) creates a single point of failure and bottleneck. When agents can work together directly, they're more resilient and efficient.

Example: Group Project Without Central Coordinator
BAD APPROACH (All going through one person):
- Everyone must ask Sarah before making any decision
- Sarah becomes overwhelmed, slows down everything
- If Sarah is sick, entire project stops
- No one can work independently or innovate
- Sarah becomes the bottleneck for all communication

REAL WORLD EXAMPLE:
=================
How does Wikipedia actually work?

PEER-TO-PEER COLLABORATION:
- Anyone can edit any article directly
- Editors collaborate directly with each other
- No single person controls all decisions
- Changes are reviewed by peers, not central authority
- Knowledge sharing happens between contributors
- System continues working even if many people are offline

COLLABORATIVE WORKFLOW:
1. Editor A finds an article that needs improvement
2. Editor A makes changes and explains reasoning
3. Editor B sees the changes, adds more information
4. Editor C spots an error, discusses with A and B directly
5. All three collaborate to improve the article
6. Other editors can join the collaboration at any time
7. Result: High-quality article created through peer cooperation

THE ALGORITHM:
=============
1. DISCOVER: Find other agents who can help with the task
2. NEGOTIATE: Discuss who will do what parts
3. COLLABORATE: Work together, sharing information and resources
4. COORDINATE: Keep each other informed of progress
5. RESOLVE: Handle conflicts through peer discussion
6. COMBINE: Integrate everyone's contributions

PSEUDO CODE:
===========
class PeerAgent:
    def __init__(self, agent_id, capabilities):
        self.id = agent_id
        self.capabilities = capabilities
        self.peer_network = []
        self.active_collaborations = {}
    
    def collaborate_on_task(self, task):
        # Find peers who can help
        relevant_peers = self.discover_relevant_peers(task)
        
        # Propose collaboration
        collaboration = self.initiate_collaboration(task, relevant_peers)
        
        # Negotiate responsibilities
        responsibilities = self.negotiate_roles(collaboration)
        
        # Work together
        while not task_complete:
            my_progress = self.work_on_my_part(responsibilities.my_part)
            peer_updates = self.get_peer_updates(collaboration)
            
            # Share progress and coordinate
            self.share_progress(collaboration, my_progress)
            self.coordinate_with_peers(collaboration, peer_updates)
        
        # Combine results
        final_result = self.combine_peer_contributions(collaboration)
        return final_result

WHY IS THIS POWERFUL?
===================
- No single point of failure - system is resilient
- Scales better - no central bottleneck
- Enables innovation through direct peer interaction
- Agents can specialize and share knowledge
- More efficient - direct communication between relevant parties
- Self-organizing - adapts to changing conditions automatically
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

class CollaborationStatus(Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MessageType(Enum):
    COLLABORATION_PROPOSAL = "collaboration_proposal"
    COLLABORATION_RESPONSE = "collaboration_response"
    PROGRESS_UPDATE = "progress_update"
    RESOURCE_SHARE = "resource_share"
    COORDINATION_REQUEST = "coordination_request"
    CONFLICT_RESOLUTION = "conflict_resolution"
    COMPLETION_NOTIFICATION = "completion_notification"

@dataclass
class CollaborationProposal:
    """Proposal for agents to collaborate on a task"""
    id: str
    task_description: str
    required_capabilities: List[str]
    proposed_by: str
    estimated_duration: float
    priority: int = 1
    max_participants: int = 5
    deadline: Optional[float] = None

@dataclass
class Message:
    """Message between peer agents"""
    id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: Any
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False

@dataclass
class CollaborationRole:
    """Role assignment in a collaboration"""
    agent_id: str
    responsibilities: List[str]
    estimated_effort: float
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)

@dataclass
class Collaboration:
    """Active collaboration between agents"""
    id: str
    task_description: str
    participants: List[str]
    roles: Dict[str, CollaborationRole]
    status: CollaborationStatus
    created_at: float = field(default_factory=time.time)
    progress: Dict[str, float] = field(default_factory=dict)  # agent_id -> progress %
    shared_resources: Dict[str, Any] = field(default_factory=dict)
    communications: List[Message] = field(default_factory=list)

class PeerAgent:
    """
    An agent that collaborates with peers in a decentralized manner
    
    EXAMPLE USAGE:
    =============
    # Create peer agents with different capabilities
    researcher = PeerAgent("researcher_1", ["research", "analysis"])
    writer = PeerAgent("writer_1", ["writing", "editing"])
    designer = PeerAgent("designer_1", ["design", "visualization"])
    
    # Connect agents to each other
    network = PeerNetwork()
    network.add_agent(researcher)
    network.add_agent(writer)
    network.add_agent(designer)
    
    # Agents collaborate on tasks
    result = await researcher.propose_collaboration("Create research report", network)
    """
    
    def __init__(self, agent_id: str, capabilities: List[str], max_concurrent_collaborations: int = 3):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.max_concurrent_collaborations = max_concurrent_collaborations
        
        # Peer network and collaboration
        self.known_peers: Dict[str, Dict[str, Any]] = {}  # peer_id -> info
        self.active_collaborations: Dict[str, Collaboration] = {}
        self.collaboration_history: List[Collaboration] = []
        
        # Message handling
        self.message_queue: List[Message] = []
        self.message_handlers = {
            MessageType.COLLABORATION_PROPOSAL: self.handle_collaboration_proposal,
            MessageType.COLLABORATION_RESPONSE: self.handle_collaboration_response,
            MessageType.PROGRESS_UPDATE: self.handle_progress_update,
            MessageType.RESOURCE_SHARE: self.handle_resource_share,
            MessageType.COORDINATION_REQUEST: self.handle_coordination_request,
            MessageType.CONFLICT_RESOLUTION: self.handle_conflict_resolution,
            MessageType.COMPLETION_NOTIFICATION: self.handle_completion_notification
        }
        
        # Performance tracking
        self.collaboration_success_rate = 0.0
        self.average_contribution_rating = 0.0
        self.peer_ratings: Dict[str, float] = {}  # How other peers rate this agent
        
        # Work simulation
        self.current_workload = 0.0
        self.max_workload = 1.0
    
    def discover_peers(self, network: 'PeerNetwork') -> None:
        """Discover other agents in the network"""
        available_peers = network.get_all_agents()
        
        for peer in available_peers:
            if peer.agent_id != self.agent_id:
                self.known_peers[peer.agent_id] = {
                    "capabilities": peer.capabilities,
                    "success_rate": peer.collaboration_success_rate,
                    "average_rating": peer.average_contribution_rating,
                    "current_workload": peer.current_workload,
                    "last_seen": time.time()
                }
        
        print(f"{self.agent_id} discovered {len(self.known_peers)} peers")
    
    async def propose_collaboration(self, task_description: str, required_capabilities: List[str],
                                  network: 'PeerNetwork', max_participants: int = 3) -> Optional[Dict[str, Any]]:
        """
        Propose a collaboration to relevant peers
        
        Args:
            task_description: What needs to be done
            required_capabilities: What capabilities are needed
            network: The peer network to find collaborators
            max_participants: Maximum number of participants
            
        Returns:
            Results of the collaboration
        """
        print(f"\n{self.agent_id} PROPOSING COLLABORATION")
        print(f"Task: {task_description}")
        print(f"Required capabilities: {required_capabilities}")
        print("-" * 50)
        
        # Check if we can handle this alone
        if all(cap in self.capabilities for cap in required_capabilities):
            print(f"Agent can handle task alone, but seeking collaboration for better results")
        
        # Find suitable peers
        suitable_peers = self.find_suitable_peers(required_capabilities, max_participants - 1)
        
        if not suitable_peers:
            print(f"No suitable peers found, working alone")
            return await self.work_alone(task_description)
        
        # Create collaboration proposal
        proposal = CollaborationProposal(
            id=f"collab_{int(time.time())}_{self.agent_id}",
            task_description=task_description,
            required_capabilities=required_capabilities,
            proposed_by=self.agent_id,
            estimated_duration=self.estimate_task_duration(task_description),
            max_participants=max_participants
        )
        
        # Send proposals to suitable peers
        responses = await self.send_collaboration_proposals(proposal, suitable_peers, network)
        
        # Select participants based on responses
        participants = self.select_collaboration_participants(proposal, responses)
        
        if not participants:
            print(f"No peers accepted collaboration, working alone")
            return await self.work_alone(task_description)
        
        # Create and execute collaboration
        collaboration = await self.create_collaboration(proposal, participants, network)
        result = await self.execute_collaboration(collaboration, network)
        
        return result
    
    def find_suitable_peers(self, required_capabilities: List[str], max_peers: int) -> List[str]:
        """Find peers who have the required capabilities and availability"""
        
        suitable_peers = []
        
        for peer_id, peer_info in self.known_peers.items():
            peer_capabilities = peer_info["capabilities"]
            peer_workload = peer_info["current_workload"]
            peer_rating = peer_info["average_rating"]
            
            # Check if peer has any required capabilities
            has_relevant_capabilities = any(cap in peer_capabilities for cap in required_capabilities)
            
            # Check availability (not overloaded)
            is_available = peer_workload < 0.8
            
            # Check reputation (decent rating)
            has_good_reputation = peer_rating >= 0.6
            
            if has_relevant_capabilities and is_available and has_good_reputation:
                # Calculate suitability score
                capability_match = len(set(required_capabilities) & set(peer_capabilities))
                availability_score = 1.0 - peer_workload
                reputation_score = peer_rating
                
                suitability_score = (capability_match * 0.5 + availability_score * 0.3 + reputation_score * 0.2)
                
                suitable_peers.append((peer_id, suitability_score))
        
        # Sort by suitability and return top candidates
        suitable_peers.sort(key=lambda x: x[1], reverse=True)
        return [peer_id for peer_id, score in suitable_peers[:max_peers]]
    
    async def send_collaboration_proposals(self, proposal: CollaborationProposal, 
                                         target_peers: List[str], network: 'PeerNetwork') -> Dict[str, Any]:
        """Send collaboration proposals to target peers and collect responses"""
        
        responses = {}
        
        for peer_id in target_peers:
            print(f"Sending collaboration proposal to {peer_id}")
            
            message = Message(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                sender_id=self.agent_id,
                recipient_id=peer_id,
                message_type=MessageType.COLLABORATION_PROPOSAL,
                content=proposal,
                requires_response=True
            )
            
            # Send message through network
            response = await network.send_message(message)
            if response:
                responses[peer_id] = response
        
        return responses
    
    def select_collaboration_participants(self, proposal: CollaborationProposal, 
                                        responses: Dict[str, Any]) -> List[str]:
        """Select the best participants based on their responses"""
        
        accepted_participants = []
        
        for peer_id, response in responses.items():
            if response.get("accepted", False):
                participant_info = {
                    "id": peer_id,
                    "proposed_role": response.get("proposed_role", []),
                    "estimated_effort": response.get("estimated_effort", 1.0),
                    "confidence": response.get("confidence", 0.5)
                }
                accepted_participants.append((peer_id, participant_info))
        
        # Sort by confidence and select best participants
        accepted_participants.sort(key=lambda x: x[1]["confidence"], reverse=True)
        selected_count = min(len(accepted_participants), proposal.max_participants - 1)
        
        selected = [self.agent_id]  # Include self
        for i in range(selected_count):
            selected.append(accepted_participants[i][0])
        
        print(f"Selected participants: {selected}")
        return selected
    
    async def create_collaboration(self, proposal: CollaborationProposal, 
                                 participants: List[str], network: 'PeerNetwork') -> Collaboration:
        """Create and initialize a new collaboration"""
        
        collaboration = Collaboration(
            id=proposal.id,
            task_description=proposal.task_description,
            participants=participants,
            roles={},
            status=CollaborationStatus.ACTIVE
        )
        
        # Assign roles to participants
        await self.assign_collaboration_roles(collaboration, proposal, network)
        
        # Add to active collaborations
        self.active_collaborations[collaboration.id] = collaboration
        
        # Notify all participants
        for participant_id in participants:
            if participant_id != self.agent_id:
                message = Message(
                    id=f"msg_{uuid.uuid4().hex[:8]}",
                    sender_id=self.agent_id,
                    recipient_id=participant_id,
                    message_type=MessageType.COLLABORATION_RESPONSE,
                    content={"collaboration": collaboration, "action": "collaboration_created"}
                )
                await network.send_message(message)
        
        print(f"Created collaboration {collaboration.id} with {len(participants)} participants")
        return collaboration
    
    async def assign_collaboration_roles(self, collaboration: Collaboration, 
                                       proposal: CollaborationProposal, network: 'PeerNetwork') -> None:
        """Assign specific roles and responsibilities to each participant"""
        
        # Break down the task based on required capabilities
        task_breakdown = self.break_down_task(proposal.task_description, proposal.required_capabilities)
        
        # Assign roles based on agent capabilities
        for participant_id in collaboration.participants:
            if participant_id == self.agent_id:
                participant_capabilities = self.capabilities
            else:
                participant_capabilities = self.known_peers[participant_id]["capabilities"]
            
            # Find best matching responsibilities for this participant
            assigned_responsibilities = []
            for subtask, required_cap in task_breakdown.items():
                if required_cap in participant_capabilities:
                    assigned_responsibilities.append(subtask)
            
            # Create role assignment
            role = CollaborationRole(
                agent_id=participant_id,
                responsibilities=assigned_responsibilities,
                estimated_effort=len(assigned_responsibilities) * 0.3,
                deliverables=[f"{subtask}_result" for subtask in assigned_responsibilities]
            )
            
            collaboration.roles[participant_id] = role
            collaboration.progress[participant_id] = 0.0
            
            print(f"Assigned role to {participant_id}: {assigned_responsibilities}")
    
    def break_down_task(self, task_description: str, required_capabilities: List[str]) -> Dict[str, str]:
        """Break down a task into subtasks matching required capabilities"""
        
        # Simulate task breakdown based on task description
        task_breakdown = {}
        
        if "research" in required_capabilities:
            task_breakdown["gather_information"] = "research"
            task_breakdown["analyze_data"] = "analysis"
        
        if "writing" in required_capabilities:
            task_breakdown["create_content"] = "writing"
            task_breakdown["edit_content"] = "editing"
        
        if "design" in required_capabilities:
            task_breakdown["create_visuals"] = "design"
            task_breakdown["layout_design"] = "visualization"
        
        if "development" in required_capabilities:
            task_breakdown["implement_solution"] = "development"
            task_breakdown["test_implementation"] = "testing"
        
        # Add generic tasks if no specific capabilities matched
        if not task_breakdown:
            task_breakdown["primary_task"] = "general_processing"
            task_breakdown["quality_review"] = "review"
        
        return task_breakdown
    
    async def execute_collaboration(self, collaboration: Collaboration, 
                                  network: 'PeerNetwork') -> Dict[str, Any]:
        """Execute the collaboration, coordinating work among participants"""
        
        print(f"\nEXECUTING COLLABORATION: {collaboration.task_description}")
        print("-" * 50)
        
        start_time = time.time()
        my_role = collaboration.roles[self.agent_id]
        
        # Start working on assigned responsibilities
        asyncio.create_task(self.work_on_collaboration_role(collaboration, my_role, network))
        
        # Monitor collaboration progress
        while collaboration.status == CollaborationStatus.ACTIVE:
            # Check overall progress
            total_progress = sum(collaboration.progress.values()) / len(collaboration.participants)
            
            if total_progress >= 0.95:  # 95% completion threshold
                print(f"Collaboration nearing completion ({total_progress:.1%})")
                break
            
            # Handle incoming messages
            await self.process_pending_messages(network)
            
            # Coordinate with peers
            await self.coordinate_with_peers(collaboration, network)
            
            # Brief pause
            await asyncio.sleep(0.2)
        
        # Finalize collaboration
        final_result = await self.finalize_collaboration(collaboration, network)
        execution_time = time.time() - start_time
        
        print(f"Collaboration completed in {execution_time:.2f} seconds")
        return final_result
    
    async def work_on_collaboration_role(self, collaboration: Collaboration, 
                                       role: CollaborationRole, network: 'PeerNetwork') -> None:
        """Work on assigned role in the collaboration"""
        
        for responsibility in role.responsibilities:
            print(f"{self.agent_id} working on: {responsibility}")
            
            # Simulate work progress
            work_time = random.uniform(0.5, 2.0)
            steps = 5
            
            for step in range(steps):
                await asyncio.sleep(work_time / steps)
                
                # Update progress
                step_progress = (step + 1) / steps
                overall_progress = step_progress / len(role.responsibilities)
                responsibility_index = role.responsibilities.index(responsibility)
                base_progress = responsibility_index / len(role.responsibilities)
                
                collaboration.progress[self.agent_id] = base_progress + overall_progress
                
                # Share progress update occasionally
                if step % 2 == 0:
                    await self.share_progress_update(collaboration, network)
        
        print(f"{self.agent_id} completed all assigned responsibilities")
        collaboration.progress[self.agent_id] = 1.0
        await self.share_progress_update(collaboration, network)
    
    async def share_progress_update(self, collaboration: Collaboration, network: 'PeerNetwork') -> None:
        """Share progress update with other participants"""
        
        my_progress = collaboration.progress[self.agent_id]
        
        for participant_id in collaboration.participants:
            if participant_id != self.agent_id:
                message = Message(
                    id=f"msg_{uuid.uuid4().hex[:8]}",
                    sender_id=self.agent_id,
                    recipient_id=participant_id,
                    message_type=MessageType.PROGRESS_UPDATE,
                    content={
                        "collaboration_id": collaboration.id,
                        "progress": my_progress,
                        "status": "working"
                    }
                )
                await network.send_message(message)
    
    async def coordinate_with_peers(self, collaboration: Collaboration, network: 'PeerNetwork') -> None:
        """Coordinate work with peer participants"""
        
        # Check if coordination is needed
        my_role = collaboration.roles[self.agent_id]
        
        # Look for dependencies
        if my_role.dependencies:
            for dependency in my_role.dependencies:
                # Check if dependency is ready
                dependency_ready = self.check_dependency_status(collaboration, dependency)
                if not dependency_ready:
                    # Request coordination
                    await self.request_coordination(collaboration, dependency, network)
        
        # Share resources if available
        await self.share_available_resources(collaboration, network)
    
    def check_dependency_status(self, collaboration: Collaboration, dependency: str) -> bool:
        """Check if a dependency is satisfied"""
        # Simplified dependency checking
        for participant_id, progress in collaboration.progress.items():
            if progress > 0.5:  # Assume dependency is ready if participant is > 50% done
                return True
        return False
    
    async def request_coordination(self, collaboration: Collaboration, 
                                 dependency: str, network: 'PeerNetwork') -> None:
        """Request coordination with peers for dependencies"""
        
        for participant_id in collaboration.participants:
            if participant_id != self.agent_id:
                message = Message(
                    id=f"msg_{uuid.uuid4().hex[:8]}",
                    sender_id=self.agent_id,
                    recipient_id=participant_id,
                    message_type=MessageType.COORDINATION_REQUEST,
                    content={
                        "collaboration_id": collaboration.id,
                        "dependency": dependency,
                        "request_type": "status_update"
                    }
                )
                await network.send_message(message)
    
    async def share_available_resources(self, collaboration: Collaboration, network: 'PeerNetwork') -> None:
        """Share any available resources with collaboration participants"""
        
        # Simulate sharing intermediate results or useful resources
        if random.random() < 0.3:  # 30% chance to share something
            resource_name = f"resource_{self.agent_id}_{int(time.time())}"
            resource_data = {
                "type": "intermediate_result",
                "data": f"Useful data from {self.agent_id}",
                "applicable_to": ["general_processing", "review"]
            }
            
            collaboration.shared_resources[resource_name] = resource_data
            
            # Notify peers about shared resource
            for participant_id in collaboration.participants:
                if participant_id != self.agent_id:
                    message = Message(
                        id=f"msg_{uuid.uuid4().hex[:8]}",
                        sender_id=self.agent_id,
                        recipient_id=participant_id,
                        message_type=MessageType.RESOURCE_SHARE,
                        content={
                            "collaboration_id": collaboration.id,
                            "resource_name": resource_name,
                            "resource_data": resource_data
                        }
                    )
                    await network.send_message(message)
    
    async def finalize_collaboration(self, collaboration: Collaboration, 
                                   network: 'PeerNetwork') -> Dict[str, Any]:
        """Finalize the collaboration and combine results"""
        
        collaboration.status = CollaborationStatus.COMPLETED
        
        # Collect all deliverables
        final_deliverables = {}
        
        for participant_id, role in collaboration.roles.items():
            participant_deliverables = {}
            for deliverable in role.deliverables:
                # Simulate deliverable content
                participant_deliverables[deliverable] = f"Completed {deliverable} by {participant_id}"
            
            final_deliverables[participant_id] = participant_deliverables
        
        # Combine shared resources
        combined_result = {
            "task_description": collaboration.task_description,
            "participants": collaboration.participants,
            "deliverables": final_deliverables,
            "shared_resources": collaboration.shared_resources,
            "collaboration_id": collaboration.id,
            "completion_time": time.time(),
            "total_messages": len(collaboration.communications),
            "success": True
        }
        
        # Notify all participants of completion
        for participant_id in collaboration.participants:
            if participant_id != self.agent_id:
                message = Message(
                    id=f"msg_{uuid.uuid4().hex[:8]}",
                    sender_id=self.agent_id,
                    recipient_id=participant_id,
                    message_type=MessageType.COMPLETION_NOTIFICATION,
                    content={
                        "collaboration_id": collaboration.id,
                        "final_result": combined_result
                    }
                )
                await network.send_message(message)
        
        # Move to history
        self.collaboration_history.append(collaboration)
        del self.active_collaborations[collaboration.id]
        
        return combined_result
    
    async def work_alone(self, task_description: str) -> Dict[str, Any]:
        """Work on task without collaboration"""
        print(f"{self.agent_id} working alone on: {task_description}")
        
        # Simulate solo work
        work_time = random.uniform(2.0, 4.0)
        await asyncio.sleep(work_time)
        
        return {
            "task_description": task_description,
            "completed_by": self.agent_id,
            "solo_work": True,
            "completion_time": work_time,
            "result": f"Solo completion of {task_description}"
        }
    
    def estimate_task_duration(self, task_description: str) -> float:
        """Estimate how long a task will take"""
        base_duration = 2.0
        complexity_factor = len(task_description.split()) / 10
        return base_duration + complexity_factor
    
    # MESSAGE HANDLERS
    # ===============
    
    async def handle_collaboration_proposal(self, message: Message, network: 'PeerNetwork') -> Dict[str, Any]:
        """Handle incoming collaboration proposal"""
        proposal = message.content
        
        # Decide whether to accept
        should_accept = self.evaluate_collaboration_proposal(proposal)
        
        if should_accept:
            proposed_role = self.propose_my_role(proposal)
            response = {
                "accepted": True,
                "agent_id": self.agent_id,
                "proposed_role": proposed_role,
                "estimated_effort": len(proposed_role) * 0.3,
                "confidence": 0.8
            }
            print(f"{self.agent_id} accepted collaboration proposal from {message.sender_id}")
        else:
            response = {
                "accepted": False,
                "agent_id": self.agent_id,
                "reason": "Current workload too high or capabilities don't match"
            }
            print(f"{self.agent_id} declined collaboration proposal from {message.sender_id}")
        
        return response
    
    def evaluate_collaboration_proposal(self, proposal: CollaborationProposal) -> bool:
        """Evaluate whether to accept a collaboration proposal"""
        
        # Check workload
        if self.current_workload > 0.7:
            return False
        
        # Check capability match
        my_relevant_capabilities = set(self.capabilities) & set(proposal.required_capabilities)
        if not my_relevant_capabilities:
            return False
        
        # Check if we're already in too many collaborations
        if len(self.active_collaborations) >= self.max_concurrent_collaborations:
            return False
        
        return True
    
    def propose_my_role(self, proposal: CollaborationProposal) -> List[str]:
        """Propose what role this agent can take in the collaboration"""
        possible_roles = []
        
        for capability in self.capabilities:
            if capability in proposal.required_capabilities:
                if capability == "research":
                    possible_roles.extend(["gather_information", "analyze_data"])
                elif capability == "writing":
                    possible_roles.extend(["create_content", "edit_content"])
                elif capability == "design":
                    possible_roles.extend(["create_visuals", "layout_design"])
                elif capability == "development":
                    possible_roles.extend(["implement_solution", "test_implementation"])
                else:
                    possible_roles.append(f"{capability}_task")
        
        return possible_roles
    
    async def handle_collaboration_response(self, message: Message, network: 'PeerNetwork') -> None:
        """Handle response about collaboration"""
        content = message.content
        if content.get("action") == "collaboration_created":
            collaboration = content["collaboration"]
            self.active_collaborations[collaboration.id] = collaboration
            print(f"{self.agent_id} joined collaboration {collaboration.id}")
    
    async def handle_progress_update(self, message: Message, network: 'PeerNetwork') -> None:
        """Handle progress updates from peers"""
        content = message.content
        collaboration_id = content["collaboration_id"]
        
        if collaboration_id in self.active_collaborations:
            collaboration = self.active_collaborations[collaboration_id]
            collaboration.progress[message.sender_id] = content["progress"]
            
            total_progress = sum(collaboration.progress.values()) / len(collaboration.participants)
            if total_progress % 0.2 < 0.1:  # Print every 20%
                print(f"Collaboration {collaboration_id} progress: {total_progress:.1%}")
    
    async def handle_resource_share(self, message: Message, network: 'PeerNetwork') -> None:
        """Handle shared resources from peers"""
        content = message.content
        collaboration_id = content["collaboration_id"]
        
        if collaboration_id in self.active_collaborations:
            collaboration = self.active_collaborations[collaboration_id]
            resource_name = content["resource_name"]
            resource_data = content["resource_data"]
            
            collaboration.shared_resources[resource_name] = resource_data
            print(f"{self.agent_id} received shared resource: {resource_name}")
    
    async def handle_coordination_request(self, message: Message, network: 'PeerNetwork') -> None:
        """Handle coordination requests from peers"""
        content = message.content
        collaboration_id = content["collaboration_id"]
        
        if collaboration_id in self.active_collaborations:
            collaboration = self.active_collaborations[collaboration_id]
            my_progress = collaboration.progress[self.agent_id]
            
            # Respond with current status
            response_message = Message(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.COORDINATION_REQUEST,
                content={
                    "collaboration_id": collaboration_id,
                    "my_progress": my_progress,
                    "status": "active"
                }
            )
            await network.send_message(response_message)
    
    async def handle_conflict_resolution(self, message: Message, network: 'PeerNetwork') -> None:
        """Handle conflict resolution requests"""
        # Simplified conflict resolution
        print(f"{self.agent_id} handling conflict resolution request")
    
    async def handle_completion_notification(self, message: Message, network: 'PeerNetwork') -> None:
        """Handle collaboration completion notifications"""
        content = message.content
        collaboration_id = content["collaboration_id"]
        
        if collaboration_id in self.active_collaborations:
            collaboration = self.active_collaborations[collaboration_id]
            collaboration.status = CollaborationStatus.COMPLETED
            
            # Move to history
            self.collaboration_history.append(collaboration)
            del self.active_collaborations[collaboration_id]
            
            print(f"{self.agent_id} collaboration {collaboration_id} completed")
    
    async def process_pending_messages(self, network: 'PeerNetwork') -> None:
        """Process any pending messages"""
        pending_messages = network.get_messages_for_agent(self.agent_id)
        
        for message in pending_messages:
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                await handler(message, network)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "current_workload": self.current_workload,
            "active_collaborations": len(self.active_collaborations),
            "known_peers": len(self.known_peers),
            "collaboration_history": len(self.collaboration_history),
            "success_rate": self.collaboration_success_rate,
            "average_rating": self.average_contribution_rating
        }

class PeerNetwork:
    """Network facilitating peer-to-peer agent communication"""
    
    def __init__(self):
        self.agents: Dict[str, PeerAgent] = {}
        self.message_queues: Dict[str, List[Message]] = {}
        self.network_stats = {
            "total_messages": 0,
            "active_collaborations": 0,
            "completed_collaborations": 0
        }
    
    def add_agent(self, agent: PeerAgent) -> None:
        """Add an agent to the network"""
        self.agents[agent.agent_id] = agent
        self.message_queues[agent.agent_id] = []
        
        # Let agent discover other peers
        agent.discover_peers(self)
        
        print(f"Added agent {agent.agent_id} to network")
    
    def get_all_agents(self) -> List[PeerAgent]:
        """Get all agents in the network"""
        return list(self.agents.values())
    
    async def send_message(self, message: Message) -> Optional[Any]:
        """Send a message between agents"""
        if message.recipient_id in self.message_queues:
            self.message_queues[message.recipient_id].append(message)
            self.network_stats["total_messages"] += 1
            
            # If message requires response, simulate response
            if message.requires_response and message.recipient_id in self.agents:
                recipient = self.agents[message.recipient_id]
                if message.message_type in recipient.message_handlers:
                    handler = recipient.message_handlers[message.message_type]
                    response = await handler(message, self)
                    return response
        
        return None
    
    def get_messages_for_agent(self, agent_id: str) -> List[Message]:
        """Get pending messages for an agent"""
        if agent_id in self.message_queues:
            messages = self.message_queues[agent_id].copy()
            self.message_queues[agent_id].clear()
            return messages
        return []
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status and statistics"""
        active_collabs = sum(len(agent.active_collaborations) for agent in self.agents.values())
        
        return {
            "total_agents": len(self.agents),
            "total_messages": self.network_stats["total_messages"],
            "active_collaborations": active_collabs,
            "agent_capabilities": {agent_id: agent.capabilities for agent_id, agent in self.agents.items()}
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_research_collaboration():
    """Demo: Research team collaborating on a report"""
    print("\nDEMO 1: RESEARCH TEAM COLLABORATION")
    print("=" * 50)
    
    # Create network
    network = PeerNetwork()
    
    # Create specialized agents
    researcher = PeerAgent("researcher_alice", ["research", "analysis"])
    writer = PeerAgent("writer_bob", ["writing", "editing"])
    designer = PeerAgent("designer_carol", ["design", "visualization"])
    
    # Add to network
    network.add_agent(researcher)
    network.add_agent(writer)
    network.add_agent(designer)
    
    # Researcher proposes collaboration
    result = await researcher.propose_collaboration(
        "Create comprehensive AI research report with visuals",
        ["research", "analysis", "writing", "design"],
        network,
        max_participants=3
    )
    
    print(f"\nCollaboration result:")
    print(f"- Participants: {result.get('participants', [])}")
    print(f"- Success: {result.get('success', False)}")
    print(f"- Deliverables: {len(result.get('deliverables', {}))}")

async def demo_development_team():
    """Demo: Development team working on features"""
    print("\nDEMO 2: DEVELOPMENT TEAM COLLABORATION")
    print("=" * 50)
    
    network = PeerNetwork()
    
    # Create development team
    frontend_dev = PeerAgent("frontend_dev", ["development", "ui_design"])
    backend_dev = PeerAgent("backend_dev", ["development", "database"])
    tester = PeerAgent("tester", ["testing", "quality_assurance"])
    devops = PeerAgent("devops", ["deployment", "infrastructure"])
    
    # Add to network
    for agent in [frontend_dev, backend_dev, tester, devops]:
        network.add_agent(agent)
    
    # Backend developer proposes collaboration
    result = await backend_dev.propose_collaboration(
        "Implement new user authentication system",
        ["development", "testing", "deployment"],
        network,
        max_participants=4
    )
    
    print(f"\nDevelopment collaboration:")
    print(f"- Team size: {len(result.get('participants', []))}")
    print(f"- Shared resources: {len(result.get('shared_resources', {}))}")
    print(f"- Messages exchanged: {result.get('total_messages', 0)}")

async def demo_content_creation_network():
    """Demo: Content creation network with multiple collaborations"""
    print("\nDEMO 3: CONTENT CREATION NETWORK")
    print("=" * 50)
    
    network = PeerNetwork()
    
    # Create diverse content creation team
    content_strategist = PeerAgent("strategist", ["strategy", "planning"])
    copywriter = PeerAgent("copywriter", ["writing", "marketing"])
    graphic_designer = PeerAgent("designer", ["design", "graphics"])
    video_editor = PeerAgent("video_editor", ["video", "editing"])
    social_media_manager = PeerAgent("social_manager", ["social_media", "marketing"])
    
    # Add all to network
    agents = [content_strategist, copywriter, graphic_designer, video_editor, social_media_manager]
    for agent in agents:
        network.add_agent(agent)
    
    # Multiple simultaneous collaborations
    collaborations = []
    
    # Collaboration 1: Blog post creation
    blog_collab = await content_strategist.propose_collaboration(
        "Create engaging blog post with graphics",
        ["writing", "design", "strategy"],
        network
    )
    collaborations.append(blog_collab)
    
    # Collaboration 2: Social media campaign
    social_collab = await social_media_manager.propose_collaboration(
        "Design social media campaign with video content",
        ["social_media", "video", "design"],
        network
    )
    collaborations.append(social_collab)
    
    # Show network statistics
    network_status = network.get_network_status()
    print(f"\nNetwork activity:")
    print(f"- Total agents: {network_status['total_agents']}")
    print(f"- Active collaborations: {network_status['active_collaborations']}")
    print(f"- Messages exchanged: {network_status['total_messages']}")
    
    successful_collabs = sum(1 for c in collaborations if c.get('success', False))
    print(f"- Successful collaborations: {successful_collabs}/{len(collaborations)}")

async def main():
    """
    Demonstrate Peer-to-Peer Collaboration between agents
    
    WHAT YOU'LL LEARN:
    ================
    1. How agents can collaborate directly without central coordination
    2. How to design decentralized communication and negotiation
    3. How to handle role assignment and responsibility sharing
    4. How to coordinate work and share resources among peers
    5. How peer collaboration scales and adapts dynamically
    
    REAL WORLD APPLICATIONS:
    =======================
    - Open source software development (GitHub collaboration)
    - Wikipedia and collaborative knowledge creation
    - Blockchain and cryptocurrency networks
    - Peer-to-peer file sharing and content distribution
    - Distributed research and scientific collaboration
    - Cross-functional team coordination in organizations
    """
    
    print("PEER-TO-PEER COLLABORATION DEMONSTRATION")
    print("This shows how agents can work together directly without central control!")
    
    await demo_research_collaboration()
    await demo_development_team()
    await demo_content_creation_network()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Peer-to-peer collaboration enables decentralized coordination")
    print("✓ Direct agent communication reduces bottlenecks and increases efficiency")
    print("✓ Role negotiation allows optimal task distribution")
    print("✓ Resource sharing enhances collective capabilities")
    print("✓ Resilient networks continue working despite individual agent failures")
    print("\nTRY IT YOURSELF:")
    print("- Add reputation systems for better peer selection")
    print("- Implement conflict resolution mechanisms")
    print("- Create specialized collaboration protocols for different domains")
    print("- Add dynamic network topology and agent discovery")

if __name__ == "__main__":
    asyncio.run(main())
