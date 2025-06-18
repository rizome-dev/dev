import json
import re
import time
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import anyio
from pydantic import BaseModel, Field

from rn4s import (
    HuggingFaceClient,
    OpenRouterClient,
    Message,
    AgentMode,
    ActionStep,
    AgentMemory,
    CodeExecutor,
    CodeExecutionResult,
    JudgeEvaluator,
    Agent,
    ToolCall
)

# Try importing rich for better formatting if available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Define the different roles in our two-agent architecture
class AgentRole(str, Enum):
    """Roles for agents in the coding architecture."""
    PROJECT_MANAGER = "project_manager"
    CODER = "coder"

class CodeReview(BaseModel):
    """Code review feedback from the project manager."""
    code: str = Field(..., description="The code being reviewed")
    feedback: str = Field(..., description="Feedback on the code")
    issues: List[str] = Field(default_factory=list, description="Specific issues identified")
    suggestions: List[str] = Field(default_factory=list, description="Suggested improvements")
    approved: bool = Field(default=False, description="Whether the code is approved")

class CodingTask(BaseModel):
    """A coding task to be implemented."""
    description: str = Field(..., description="Natural language description of the task")
    specifications: Optional[str] = Field(None, description="Technical specifications from the project manager")
    requirements: List[str] = Field(default_factory=list, description="List of requirements/constraints")
    implementation: Optional[str] = Field(None, description="The code implementation")
    current_review: Optional[CodeReview] = Field(None, description="Current code review")
    execution_result: Optional[str] = Field(None, description="Result of executing the code")
    completed: bool = Field(default=False, description="Whether the task is completed")
    iteration: int = Field(default=0, description="Current iteration count")

class CodingAgentTracer:
    """Tracer for capturing coding agent progress with detailed logs."""
    
    def __init__(self, verbose: bool = True, debug: bool = False):
        """Initialize the tracer with verbosity settings."""
        self.verbose = verbose
        self.debug = debug
        self.console = Console() if RICH_AVAILABLE else None
        self.start_time = time.time()
        self.step_times = {}
    
    def task_start(self, description: str):
        """Log the start of a coding task."""
        self.start_time = time.time()
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            title = "🤖 New Coding Task"
            self.console.print("\n")
            self.console.print(Panel(
                description, 
                title=title, 
                expand=False, 
                border_style="green"
            ))
        else:
            separator = "=" * 80
            print(f"\n{separator}")
            print(f"🤖 NEW CODING TASK")
            print(f"{separator}")
            print(description)
    
    def specifications_generated(self, specs: str):
        """Log the generated specifications."""
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            title = "📋 Technical Specifications"
            self.console.print("\n")
            self.console.print(Panel(
                Markdown(specs), 
                title=title, 
                expand=False, 
                border_style="blue"
            ))
        else:
            separator = "-" * 80
            print(f"\n{separator}")
            print("📋 TECHNICAL SPECIFICATIONS")
            print(f"{separator}")
            print(specs)
    
    def iteration_start(self, iteration: int, max_iterations: int):
        """Log the start of an iteration."""
        self.step_times[iteration] = time.time()
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold blue]🔄 Iteration {iteration}/{max_iterations}[/bold blue]")
            self.console.print("─" * min(100, self.console.width))
        else:
            separator = "-" * 80
            print(f"\n{separator}")
            print(f"🔄 ITERATION {iteration}/{max_iterations}")
            print(separator)
    
    def code_implementation(self, code: str):
        """Log the code implementation."""
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            self.console.print("\n[bold cyan]💻 Code Implementation:[/bold cyan]")
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print("\n💻 CODE IMPLEMENTATION:")
            print(f"```python")
            print(code)
            print(f"```")
    
    def execution_result(self, result: str, success: bool = True):
        """Log the execution result."""
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            if success:
                self.console.print("\n[bold green]✅ Execution Result:[/bold green]")
                self.console.print(Panel(result, expand=False, border_style="green"))
            else:
                self.console.print("\n[bold red]❌ Execution Error:[/bold red]")
                self.console.print(Panel(result, expand=False, border_style="red"))
        else:
            if success:
                print("\n✅ EXECUTION RESULT:")
            else:
                print("\n❌ EXECUTION ERROR:")
            print(result)
    
    def code_review(self, review: CodeReview):
        """Log the code review."""
        if not self.verbose:
            return
            
        # Create a formatted review summary
        if RICH_AVAILABLE:
            self.console.print("\n[bold magenta]🔍 Code Review:[/bold magenta]")
            
            # Create a review panel
            if review.approved:
                title = "✅ APPROVED"
                style = "green"
            else:
                title = "⚠️ NEEDS REVISION"
                style = "yellow"
                
            self.console.print(Panel(
                Markdown(review.feedback),
                title=title,
                expand=False,
                border_style=style
            ))
            
            # Show issues and suggestions if any
            if review.issues:
                self.console.print("[bold red]Issues:[/bold red]")
                for issue in review.issues:
                    self.console.print(f"• {issue}")
                    
            if review.suggestions:
                self.console.print("\n[bold yellow]Suggestions:[/bold yellow]")
                for suggestion in review.suggestions:
                    self.console.print(f"• {suggestion}")
        else:
            print("\n🔍 CODE REVIEW:")
            print("-" * 80)
            if review.approved:
                print("✅ APPROVED")
            else:
                print("⚠️ NEEDS REVISION")
            print("-" * 80)
            print(review.feedback)
            
            if review.issues:
                print("\nIssues:")
                for issue in review.issues:
                    print(f"• {issue}")
                    
            if review.suggestions:
                print("\nSuggestions:")
                for suggestion in review.suggestions:
                    print(f"• {suggestion}")
    
    def task_complete(self, code: str, iterations: int, approved: bool):
        """Log the completion of the task."""
        total_time = time.time() - self.start_time
        
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            if approved:
                title = "✅ Final Approved Solution"
                style = "green"
            else:
                title = "⚠️ Final Solution (Not Approved)"
                style = "yellow"
                
            self.console.print("\n")
            self.console.print(Panel(
                Syntax(code, "python", theme="monokai", line_numbers=True),
                title=title,
                expand=False,
                border_style=style
            ))
            
            # Show summary stats
            table = Table(title="Task Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Total Time", f"{total_time:.2f} seconds")
            table.add_row("Iterations", str(iterations))
            table.add_row("Status", "Approved ✅" if approved else "Not Approved ⚠️")
            
            self.console.print(table)
        else:
            print("\n" + "=" * 80)
            if approved:
                print("✅ FINAL APPROVED SOLUTION")
            else:
                print("⚠️ FINAL SOLUTION (NOT APPROVED)")
            print("=" * 80)
            print(f"```python")
            print(code)
            print(f"```")
            
            print("\nTASK SUMMARY:")
            print(f"• Total Time: {total_time:.2f} seconds")
            print(f"• Iterations: {iterations}")
            print(f"• Status: {'Approved ✅' if approved else 'Not Approved ⚠️'}")
    
    def debug_message(self, message: str):
        """Log a debug message."""
        if not self.debug:
            return
            
        if RICH_AVAILABLE:
            self.console.print(f"[dim italic]{message}[/dim italic]")
        else:
            print(f"DEBUG: {message}")

class CodingAgentMemory(BaseModel):
    """Memory for the coding agent system."""
    task: CodingTask = Field(..., description="The coding task")
    conversation: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    max_iterations: int = Field(default=5, description="Maximum number of iterations")
    
    def add_message(self, role: AgentRole, content: str, message_type: str = "text"):
        """Add a message to the conversation history."""
        self.conversation.append({
            "role": role,
            "content": content,
            "type": message_type
        })
    
    def get_conversation_for_agent(self, agent_role: AgentRole) -> str:
        """Get a formatted conversation history for the given agent role."""
        conversation = ""
        
        # The project manager sees everything
        if agent_role == AgentRole.PROJECT_MANAGER:
            for message in self.conversation:
                role_name = "Project Manager" if message["role"] == AgentRole.PROJECT_MANAGER else "Coder"
                if message["type"] == "code":
                    conversation += f"{role_name}: [CODE]\n```python\n{message['content']}\n```\n\n"
                elif message["type"] == "execution":
                    conversation += f"Execution Result:\n{message['content']}\n\n"
                else:
                    conversation += f"{role_name}: {message['content']}\n\n"
        
        # The coder only sees its own exchanges with the project manager
        else:
            # Include the initial task and specifications
            if len(self.conversation) > 0:
                conversation += f"Project Manager: {self.task.description}\n\n"
            if self.task.specifications:
                conversation += f"Project Manager: Here are the specifications:\n{self.task.specifications}\n\n"
                
            # Add the most recent relevant exchanges, focusing on coder's code and PM's feedback
            relevant_exchanges = []
            for i, message in enumerate(self.conversation):
                if (message["role"] == AgentRole.PROJECT_MANAGER and 
                    i > 0 and self.conversation[i-1]["role"] == AgentRole.CODER):
                    # This is feedback to the coder
                    relevant_exchanges.append({
                        "role": "Project Manager",
                        "content": message["content"],
                        "type": message["type"]
                    })
                elif message["role"] == AgentRole.CODER and message["type"] == "code":
                    # This is code from the coder
                    relevant_exchanges.append({
                        "role": "Coder",
                        "content": message["content"],
                        "type": "code"
                    })
            
            # Add relevant exchanges to the conversation
            for exchange in relevant_exchanges[-4:]:
                if exchange["type"] == "code":
                    conversation += f"{exchange['role']}: [CODE]\n```python\n{exchange['content']}\n```\n\n"
                else:
                    conversation += f"{exchange['role']}: {exchange['content']}\n\n"
        
        return conversation
    
    def get_executable_code(self) -> Optional[str]:
        """Get the most recent code implementation from the conversation."""
        for message in reversed(self.conversation):
            if message["role"] == AgentRole.CODER and message["type"] == "code":
                return message["content"]
        return None

class CodingAgent(Agent):
    """
    A self-learning agent architecture for translating natural language into code implementations.
    Uses a project manager (QWQ-32b) to delegate and review, and a coder (deepcoder-14b) to implement.
    Inherits from Agent to leverage core agent capabilities.
    """
    
    # Required inputs specific to CodingAgent
    project_manager_api_token: str = Field(..., description="API token for the project manager (SambaNova)")
    openrouter_api_key: str = Field(..., description="API key for OpenRouter")
    
    # Optional configuration specific to CodingAgent
    max_iterations: int = Field(5, description="Maximum number of iterations")
    max_tokens_pm: int = Field(2048, description="Maximum tokens for project manager responses")
    max_tokens_coder: int = Field(4096, description="Maximum tokens for coder responses")
    execute_code: bool = Field(True, description="Whether to execute code for verification")
    debug: bool = Field(False, description="Enable debug output")
    verbose: bool = Field(True, description="Enable detailed output")
    planning_interval: int = Field(0, description="How often to include explicit planning steps (0 to disable)")
    
    # Clients for the two-agent architecture
    project_manager_client: Optional[HuggingFaceClient] = None
    coder_client: Optional[OpenRouterClient] = None
    
    # Additional components
    coding_memory: Optional[CodingAgentMemory] = None
    tracer: Optional[CodingAgentTracer] = None
    
    def __init__(self, **data):
        # Initialize the base Agent first with the project manager client
        if "client" not in data:
            # Use the project manager as the primary client for base class
            if "project_manager_api_token" in data:
                pm_token = data["project_manager_api_token"]
                max_tokens = data.get("max_tokens_pm", 2048)
                # Create the client with exact parameters that were working before
                data["client"] = HuggingFaceClient(
                    model_id="Qwen/QwQ-32B",
                    api_token=pm_token,
                    provider="sambanova"
                )
        
        # Set execution mode
        data["mode"] = AgentMode.CODE_EXECUTION
        
        # Make sure code_executor is provided or will be created by base class
        if "code_executor" not in data:
            data["code_executor"] = CodeExecutor()
            
        # Set system prompt for code generation
        if "system_prompt" not in data:
            data["system_prompt"] = """You are an AI coding assistant that creates high-quality Python code from specifications.
Your goal is to implement solutions that are correct, efficient, and maintainable."""
        
        # Ensure debug flag is passed to base class
        debug_value = data.get("debug", False)
        verbose_value = data.get("verbose", True)
        
        # Initialize base agent
        data["verbose"] = verbose_value
        data["show_thinking"] = debug_value
        
        # Initialize the base class
        super().__init__(**data)
        
        # Initialize tracer for better logging
        self.tracer = CodingAgentTracer(verbose=self.verbose, debug=self.debug)
        
        # Initialize the specialized clients exactly as they were in the original code
        if not self.project_manager_client:
            self.project_manager_client = HuggingFaceClient(
                model_id="Qwen/QwQ-32B",
                api_token=self.project_manager_api_token,
                provider="sambanova" 
            )
            
        if not self.coder_client:
            self.coder_client = OpenRouterClient(
                model_id="agentica-org/deepcoder-14b-preview:free",
                api_key=self.openrouter_api_key,
                max_tokens=self.max_tokens_coder,
                site_name="Judge Coding Agent",
                site_url="https://github.com/rizome-dev/judge"
            )
    
    def run(self, task_description: str) -> Dict[str, Any]:
        """
        Run the coding agent synchronously.
        
        Args:
            task_description: Natural language description of the coding task
            
        Returns:
            Dictionary with the results
        """
        return anyio.run(self.generate_code, task_description)
    
    async def generate_code(self, task_description: str) -> Dict[str, Any]:
        """
        Generate code for the given task using the two-agent architecture.
        
        Args:
            task_description: Natural language description of the coding task
            
        Returns:
            Dictionary with the final code, execution result, and conversation history
        """
        # Initialize the coding task and specialized memory
        coding_task = CodingTask(description=task_description)
        self.coding_memory = CodingAgentMemory(task=coding_task, max_iterations=self.max_iterations)
        
        # Also set task in base agent's memory
        self.memory.task = task_description
        
        # Log task start
        self.tracer.task_start(task_description)
        self.tracer.debug_message(f"Initialized CodingAgent with task: {task_description}")
        
        # Step 1: Project Manager creates specifications
        specs = await self._generate_specifications(task_description)
        coding_task.specifications = specs
        
        # Log specifications
        self.tracer.specifications_generated(specs)
        
        iteration = 0
        final_code = None
        final_answer = None
        
        # The main iteration loop
        while iteration < self.max_iterations and not coding_task.completed:
            iteration += 1
            coding_task.iteration = iteration
            
            # Log iteration start
            self.tracer.iteration_start(iteration, self.max_iterations)
            
            # Planning step if enabled
            if self.planning_interval > 0 and iteration % self.planning_interval == 0:
                await self._run_planning_step(coding_task)
            
            # Step 2: Coder implements the code
            code = await self._implement_code(coding_task)
            
            # Add the code to the specialized memory
            self.coding_memory.add_message(AgentRole.CODER, code, "code")
            
            # Log the code implementation
            self.tracer.code_implementation(code)
            
            # Step 3: Execute the code if enabled
            execution_result = None
            execution_success = True
            if self.execute_code:
                execution_result, execution_success = await self._execute_code(code)
                if execution_result:
                    self.coding_memory.add_message(AgentRole.PROJECT_MANAGER, execution_result, "execution")
                    coding_task.execution_result = execution_result
                    
                    # Log execution result
                    self.tracer.execution_result(execution_result, execution_success)
            
            # Step 4: Project Manager reviews the code
            review = await self._review_code(code, execution_result)
            self.coding_memory.add_message(AgentRole.PROJECT_MANAGER, review.feedback, "text")
            
            # Log code review
            self.tracer.code_review(review)
            
            coding_task.current_review = review
            
            # Create an ActionStep to store in the base agent's memory for self-learning
            action_step = ActionStep(step_number=iteration)
            action_step.code = code
            action_step.thoughts = f"Iteration {iteration}: Code implementation"
            
            if execution_result:
                action_step.observation = f"Execution result:\n{execution_result}"
            
            # Check for final_answer tool call in response
            if action_step.action and action_step.action.tool_name == "final_answer":
                final_answer = action_step.action.tool_input.get("answer", "")
                self.tracer.debug_message(f"Received final answer from agent: {final_answer}")
                
            # Get validation from judge evaluator (leveraging base class functionality)
            if self.judge_evaluator:
                try:
                    evaluation = await self.judge_evaluator.evaluate_step(
                        task=task_description,
                        thoughts=f"Code iteration {iteration}",
                        code=code,
                        result=execution_result or "",
                        step_number=iteration
                    )
                    action_step.evaluation = evaluation
                except Exception as e:
                    self.tracer.debug_message(f"Error getting judge evaluation: {str(e)}")
            
            # Add to base agent's memory for learning
            self.memory.add_step(action_step)
            
            # Check if the code is approved
            if review.approved:
                coding_task.completed = True
                final_code = code
                coding_task.implementation = code
                break
            
            # If not approved, coder will make another attempt in the next iteration
        
        # If we ran out of iterations, use the last code version
        if not final_code and self.coding_memory:
            final_code = self.coding_memory.get_executable_code()
            coding_task.implementation = final_code
        
        # Check for final_answer in any of the memory steps if not already found
        if not final_answer:
            for step in self.memory.steps:
                if step.action and step.action.tool_name == "final_answer":
                    final_answer = step.action.tool_input.get("answer", "")
                    break
        
        # Log task completion
        self.tracer.task_complete(
            code=final_code or "No code generated", 
            iterations=iteration,
            approved=coding_task.completed
        )
        
        # Return the results
        result = {
            "code": final_code,
            "execution_result": coding_task.execution_result,
            "approved": coding_task.completed,
            "iterations": iteration,
            "specifications": coding_task.specifications,
            "conversation": self.coding_memory.conversation if self.coding_memory else []
        }
        
        # Include final_answer if available
        if final_answer:
            result["final_answer"] = final_answer
            
        return result
    
    async def _run_planning_step(self, task: CodingTask):
        """
        Run a planning step to refine the approach.
        
        Args:
            task: The current coding task
        """
        planning_prompt = """You are a technical project manager reviewing the current state of a coding task.
Based on the specifications and current implementation progress, provide:

1. A summary of the current state
2. Key challenges that need to be addressed
3. An approach for the next iteration

Format your response as follows:
## Current State
[Brief summary of what has been implemented so far]

## Key Challenges
- [Challenge 1]
- [Challenge 2]
...

## Next Iteration Approach
[Detailed approach for the next iteration]
"""

        # Get current conversation context
        conversation = self.coding_memory.get_conversation_for_agent(AgentRole.PROJECT_MANAGER)
        
        # Create message for planning
        messages = [
            {"role": "system", "content": planning_prompt},
            {"role": "user", "content": f"Task: {task.description}\n\nSpecifications:\n{task.specifications}\n\nCurrent progress:\n{conversation}"}
        ]
        
        # Get planning response
        planning = await self.project_manager_client.chat(messages)
        
        # Log planning response
        if RICH_AVAILABLE and self.tracer.verbose:
            self.tracer.console.print("\n[bold yellow]📝 Planning Step:[/bold yellow]")
            self.tracer.console.print(Panel(
                Markdown(planning),
                title="Planning Update",
                expand=False,
                border_style="yellow"
            ))
        elif self.tracer.verbose:
            print("\n📝 PLANNING STEP:")
            print("-" * 80)
            print(planning)
            print("-" * 80)
    
    async def _generate_specifications(self, task_description: str) -> str:
        """
        Have the project manager generate technical specifications for the task.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            Technical specifications for the coder
        """
        # Create a system prompt for the project manager with explicit formatting
        system_prompt = """You are a senior technical project manager with expertise in software design and architecture. 
Your job is to translate user requirements into clear, detailed technical specifications that a developer can implement.

Follow these guidelines when creating specifications:
1. Analyze the requirements thoroughly and identify the core functionality
2. Break down the task into clear, specific requirements
3. Identify potential edge cases or considerations
4. Specify any constraints or performance requirements
5. Be precise and technical in your language
6. Provide specific data structures and functions that should be implemented
7. Include example input/output when relevant

Format your specifications EXACTLY like this:
## Technical Specifications
[Detailed technical description of what needs to be implemented]

## Requirements
- [Requirement 1]
- [Requirement 2]
...

## Data Structures
```python
# Example data structures or type hints
class SampleClass:
    field1: str
    field2: int
```

## Functions
```python
def function_name(param1: type, param2: type) -> return_type:
    \"\"\"
    Description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    \"\"\"
    # Function should do something like this
    pass
```

## Edge Cases
- [Edge case 1 to handle]
- [Edge case 2 to handle]
...

Keep your specifications comprehensive but concise. Focus on WHAT needs to be built rather than HOW to build it."""
        
        # Create a message for the project manager - use the exact format from original working code
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create comprehensive technical specifications for the following task:\n\n{task_description}\n\nMake sure to include clear specifications with example code, functions, and data structures."}
        ]
        
        # Call the project manager model and log progress
        self.tracer.debug_message("Generating specifications with Project Manager...")
        specifications = await self.project_manager_client.chat(messages)
        
        # Add to specialized memory
        self.coding_memory.add_message(AgentRole.PROJECT_MANAGER, specifications, "text")
        
        return specifications
    
    async def _implement_code(self, task: CodingTask) -> str:
        """
        Have the coder implement code based on the specifications.
        
        Args:
            task: The coding task with specifications
            
        Returns:
            Implemented code
        """
        # Create a system prompt for the coder with more explicit instructions for code format
        system_prompt = """You are an expert Python developer specializing in writing clean, efficient, and maintainable code. 
Your task is to implement complete, runnable Python code based on the specifications provided by the project manager.

Guidelines for your implementation:
1. Read the specifications carefully and implement EXACTLY what is requested
2. Write clean, well-documented code with appropriate docstrings
3. Include imports for all required modules
4. Handle all edge cases mentioned in the specifications
5. Follow PEP 8 style guidelines
6. Implement ALL required functions and classes completely
7. Ensure your code can be executed without additional modifications

IMPORTANT: Your response MUST contain ONLY the Python code implementation wrapped in a code block.
Do not include any explanations, comments, or text outside of the code block.

Example format of your response:
```python
# imports
import os
from typing import List, Dict

# implementation
def function_name(param1, param2):
    \"\"\"Docstring with description.\"\"\"
    # Implementation
    return result

# additional code as needed
class SomeClass:
    def __init__(self):
        pass
        
    def some_method(self):
        pass

# example usage if helpful
if __name__ == "__main__":
    # Example execution
    result = function_name("example", 123)
    print(result)
```

Ensure your code is complete and directly executable."""
        
        # Get the conversation history for context
        conversation = self.coding_memory.get_conversation_for_agent(AgentRole.CODER)
        
        # Create a message for the coder
        if task.iteration == 1:
            # First iteration, provide full specifications
            user_content = f"Implement Python code for the following task:\n\n{task.description}\n\nSpecifications:\n{task.specifications}\n\nProvide a COMPLETE implementation as a single Python file that can be executed directly."
        else:
            # Subsequent iterations, focus on feedback
            user_content = f"Revise your implementation based on the feedback received. Make sure to address ALL the issues mentioned in the review.\n\nTask: {task.description}\n\nConversation history:\n{conversation}\n\nProvide your COMPLETE revised implementation."
            
            if task.current_review and task.current_review.issues:
                user_content += "\n\nPlease fix these specific issues:\n"
                for issue in task.current_review.issues:
                    user_content += f"- {issue}\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Call the coder model - let the client handle retries
        self.tracer.debug_message("Requesting code implementation from Coder...")
        implementation = await self.coder_client.chat(messages)
        
        # Extract code from markdown if present
        code_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        code_match = re.search(code_pattern, implementation, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = implementation.strip()
        
        return code
    
    async def _execute_code(self, code: str) -> Tuple[Optional[str], bool]:
        """
        Execute the code and return the result.
        Uses the base agent's code executor.
        
        Args:
            code: The code to execute
            
        Returns:
            Tuple of (execution_result, success_flag)
        """
        if not code or not self.code_executor:
            return "No code to execute or executor not available", False
            
        try:
            # Use the base agent's code executor
            self.tracer.debug_message("Executing code...")
            execution_result = await self.code_executor.execute_code(code)
            
            if execution_result.success:
                result_text = execution_result.stdout or ""
                if execution_result.result:
                    if result_text:
                        result_text += f"\nResult: {execution_result.result}"
                    else:
                        result_text = f"Result: {execution_result.result}"
                return result_text, True
            else:
                error_message = f"Error: {execution_result.error or execution_result.stderr}"
                return error_message, False
        except Exception as e:
            error_message = f"Execution error: {str(e)}"
            return error_message, False
    
    async def _review_code(self, code: str, execution_result: Optional[str] = None) -> CodeReview:
        """
        Have the project manager review the code.
        
        Args:
            code: The code to review
            execution_result: The result of executing the code
            
        Returns:
            A CodeReview object with feedback and approval status
        """
        # Create a system prompt for the project manager with more structured review format
        system_prompt = """You are a senior technical project manager reviewing code implementation. 
Your role is to provide constructive, detailed feedback and determine if the code meets all requirements.

Guidelines for reviewing code:
1. Verify that the code implements ALL required functionality in the specs
2. Check for proper error handling and edge cases
3. Evaluate code clarity, organization, and documentation
4. Identify specific bugs, issues, or inefficiencies
5. Be constructive and specific in your feedback
6. If the code works as expected, approve it

When providing your review, use the following format:
## Code Review

[Your general assessment of the code]

### Issues
- [Issue 1]
- [Issue 2]
...

### Suggestions
- [Suggestion 1]
- [Suggestion 2]
...

### Approval
[YES/NO]: Clearly state whether you approve this code or not.

If you approve the code, please provide a final answer using the final_answer tool. For example:

```json
{
  "name": "final_answer",
  "arguments": {
    "answer": "The implementation is complete and correct. The code [brief description of what the code does]."
  }
}
```

Your review should be thorough but concise. Focus on issues that need to be fixed for functional correctness first."""

        # Get the coding task
        task = self.coding_memory.task
        
        # Create a context-rich message for the project manager to review the code
        user_message = f"""Review the following Python code implementation:

Task: {task.description}

Technical Specifications:
{task.specifications}

Code Implementation:
```python
{code}
```
"""

        # Add execution result if available
        if execution_result:
            user_message += f"\nExecution Result:\n{execution_result}\n"
            
        user_message += "\nProvide a comprehensive code review. Is this implementation correct, complete, and does it meet all requirements? If the code is fully functional and meets all requirements, approve it. Otherwise, provide specific feedback for improvements."
        
        # Create messages for the review
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Call the project manager model
        self.tracer.debug_message("Requesting code review from Project Manager...")
        review_text = await self.project_manager_client.chat(messages)
        
        # Parse the review to extract approval status
        approval_pattern = r"(?i)approval\s*[\r\n]+\s*(?:decision)?:?\s*\[?(YES|NO)\]?"
        approval_match = re.search(approval_pattern, review_text)
        
        is_approved = False
        if approval_match:
            is_approved = approval_match.group(1).upper() == "YES"
        
        # Try to extract issues and suggestions using regex
        issues = []
        suggestions = []
        
        issues_section = re.search(r"(?:###|##)\s*Issues\s*[\r\n]+(.+?)(?=(?:###|##)|$)", review_text, re.DOTALL)
        if issues_section:
            issues_text = issues_section.group(1).strip()
            # Extract bullet points
            issues = [issue.strip().lstrip("-").strip() for issue in re.findall(r"[\r\n]+\s*-\s*(.+?)(?=[\r\n]+\s*-|$)", issues_text + "\n-", re.DOTALL)]
            issues = [issue for issue in issues if issue]  # Remove empty items
        
        suggestions_section = re.search(r"(?:###|##)\s*Suggestions\s*[\r\n]+(.+?)(?=(?:###|##)|$)", review_text, re.DOTALL)
        if suggestions_section:
            suggestions_text = suggestions_section.group(1).strip()
            # Extract bullet points
            suggestions = [suggestion.strip().lstrip("-").strip() for suggestion in re.findall(r"[\r\n]+\s*-\s*(.+?)(?=[\r\n]+\s*-|$)", suggestions_text + "\n-", re.DOTALL)]
            suggestions = [suggestion for suggestion in suggestions if suggestion]  # Remove empty items
        
        # Extract final_answer if present
        final_answer = None
        if is_approved:
            # Check for JSON tool call format
            tool_pattern = r"```(?:json)?\s*\{[\s\S]*?\"(?:name|action|tool)\"[\s\S]*?\}(?:\s*\n)?\s*```"
            tool_matches = re.finditer(tool_pattern, review_text)
            
            for match in tool_matches:
                tool_json_str = match.group(0).strip()
                # Remove markdown code block syntax
                tool_json_str = re.sub(r"```(?:json)?\s*", "", tool_json_str)
                tool_json_str = re.sub(r"\s*```", "", tool_json_str)
                
                try:
                    tool_data = json.loads(tool_json_str)
                    
                    # Check for final_answer tool
                    tool_name = None
                    tool_input = None
                    
                    if "name" in tool_data:
                        tool_name = tool_data["name"]
                        tool_input = tool_data.get("arguments", {})
                    elif "action" in tool_data:
                        tool_name = tool_data["action"]
                        tool_input = tool_data.get("action_input", {})
                    elif "tool" in tool_data:
                        tool_name = tool_data["tool"]
                        tool_input = tool_data.get("tool_input", {})
                    
                    if tool_name == "final_answer" and isinstance(tool_input, dict) and "answer" in tool_input:
                        final_answer = tool_input["answer"]
                        self.tracer.debug_message(f"Extracted final answer from review: {final_answer}")
                        break
                except Exception as e:
                    self.tracer.debug_message(f"Error parsing tool call JSON: {str(e)}")
        
        # Create the code review object
        review = CodeReview(
            code=code,
            feedback=review_text,
            issues=issues,
            suggestions=suggestions,
            approved=is_approved
        )
        
        # If final_answer was found, create an action step with the tool call
        if final_answer:
            # Create an action step to store the tool call
            action_step = ActionStep(step_number=0)  # Placeholder step number
            action_step.action = ToolCall(
                tool_name="final_answer",
                tool_input={"answer": final_answer}
            )
            # Add the step to memory
            self.memory.add_step(action_step)
        
        return review 

if __name__ == "__main__":
    import sys
    import os
    import json
    from datetime import datetime

    from rn4s import add_final_answer_to_agent

    # Check if a task was provided as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python -m main \"<task description>\"")
        print("Example: python -m main \"Build me a Go API\"")
        sys.exit(1)
    
    # Get the task from command-line argument
    task = sys.argv[1]

    # Login using e.g. `huggingface-cli login` to access this dataset
    #ds = load_dataset("Maxwell-Jia/AIME_2024")

    def run_coding_agent_with_final_answer(task: str):
        """Run the CodingAgent with the FinalAnswer tool."""
        # Get API tokens from environment variables
        hf_token = os.environ.get("HF_TOKEN")
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not hf_token:
            print("Error: HF_TOKEN environment variable is not set.")
            print("Please set it to your Hugging Face API token.")
            return
        
        if not openrouter_api_key:
            print("Error: OPENROUTER_API_KEY environment variable is not set.")
            print("Please set it to your OpenRouter API key.")
            return
        
        print("Initializing CodingAgent with FinalAnswer tool...")
        
        # Initialize the CodingAgent
        agent = CodingAgent(
            project_manager_api_token=hf_token,
            openrouter_api_key=openrouter_api_key,
            max_iterations=10,
            execute_code=True,
            debug=False
        )
        
        # Add the FinalAnswer tool to the agent
        agent = add_final_answer_to_agent(agent)
        
        print(f"\nTask: {task}")
        print("\nRunning agent...")
        
        # Run the agent on the task
        result = agent.run(task)
        
        print("\n" + "="*50)
        print("RESULT:")
        print("="*50)
        print(f"Code: {result['code']}")
        print(f"Final answer: {result.get('final_answer', 'No final answer provided')}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{timestamp}.py"
        
        with open(output_file, "w") as f:
            f.write(result['code'])
        
        print(f"\nOutput saved to: {output_file}")

    run_coding_agent_with_final_answer(task=task)
