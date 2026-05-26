


_detecting_en = """There are some tasks whose solvability is unknown. You need to determine, based on the task description in <task>, whether the agent defined in <Agent_role> can determine if the task can be completed under the current conditions within the <Env> environment using the tools provided in <provided_tools>. Please note that you need to carefully review the requirements in <task>, the permissions in <Agent_role>, the data availability in <Env>, and the capability descriptions of the tools in <provided_tools>. At the same time, you must not use any tools that are not provided, violate role restrictions, or assume any parameters that are neither given in <Env> nor retrievable through the provided tools.

Wrap your final answer with <answer> and </answer>.
If you believe the task can be solved under the current circumstances, output '<answer>solvable</answer>'.
If you believe the task cannot be solved, output '<answer>unsolvable</answer>'.
If you are not confident in either judgment, output '<answer>uncertain</answer>'.
Do not output anything else outside of the <answer> tags."""


_planning_en = """Plan your tool usage based on the task description in <task>, the environment description in <Env>, and the available tools provided in <provided_tools>. List the function calls (tool name and arguments) you need to use to complete the task in order, wrapping ALL calls in a single <answer> and </answer> block and placing each call on its own line, e.g.:
<answer>
Tool1(arg1="value")
Tool2(arg2="Step_1_Output")
Tooln(arg="Step_n-1_Output")
finish(answer="final answer")
</answer>
Crucial Rule for Parameters:
If a parameter is explicitly known from the task or environment, use the exact value.
If a parameter depends on the output of a previous step (which is not executed yet), use a symbolic placeholder like "Step_n_Output" to indicate the data flow.
If a step of the task does not have a suitable tool or parameter that can be solved in the list of tools provided by <provided_tools>, use UnsolvableQuery() at the last step e.g.:
<answer>
Tool1(id="123")
tool2(id="456")
UnsolvableQuery(reason='explanation of missing tool capability or parameter')
finish(answer="final answer")
</answer>
If there is a limit to the number of tools that can be used in <task>, call UnsolvableQuery() to terminate the task on the next step after the limit is reached.
In a wrapped answer, do not output any redundancy other than the function calls, and do not interpret the output in any way.
You cannot use tools or parameters that are not provided in <provided_tools> and <Env>, you can't do something that is not allowed in <Agent_role>.
Ensure that the function calls are clear and separated, with each call on its own line.
Output only ONE <answer>...</answer> block with your best answer. Do NOT use multiple <answer> blocks.
Complete the task as much as possible with the fewest tools."""




_diagnosing_en = "Divide the task and develop a task plan for tool usage based on the task description in <task> and the tools in <provided_tool>. Requirements:\n1. Each step of the task planning based on the tools provided in the list of available tools may consist of multiple steps (t>=1), with each step corresponding to a sub-objective and a tool usage. \n2. Sub-objectives need to be logically related to each other to ensure that the completion of sub-objectives advances the overall task progress. \n3. Tools that are not in <provided_tool> cannot be used \n4. Use <answer> and </answer> to wrap the entire answer \nThe format for task planning can be seen in the following example: \n<answer>Subgoal 1: [description] Planned tool: [tool name] \nSubgoal 2: [description ] Planned tool: [tool name]\n... \nSubgoal t: [description] Planned tool: [tool name]</answer>\nState the relationship between the subgoal and the requirement in the [description] section, provide the tool name in the [tool name] section, and don't provide any description or explanation in the [tool name] section. \nIf there is a limit to the number of tools that can be used in a <task>, the next step after the limit is reached calls UnsolvableQuery to terminate the task. For example, to limit the number of tools to t or less: \n<answer>Subgoal 1: [description] Planned tool: [tool name]\n...\nSubgoal t: [description] Planned tool: [tool name]\nSubgoal t+1: [Task requires number of tools within t] Planned tool: UnsovlableQuery</answer>\n\nNow, let's start scheduling the current task, using <answer> to mark the start of the task and </answer> mark the end of the task."









_react_system_prompt_en = """You are an AI agent that can use tools to complete tasks. Use the ReAct (Reasoning + Acting) paradigm to solve the task step by step.

## Instructions:
1. At each step, first think about your approach (Thought), then call a tool (Action), then read the system-provided Observation.
2. Continue this Thought → Action → Observation loop until the task is completed or you determine it cannot be completed.
3. When the task is done, you MUST call `finish(answer='your_final_answer')` to terminate.
4. If a subtask cannot be completed due to missing tools, call `UnsolvableQuery(reason='explanation')`, then continue with remaining steps.
5. If the ENTIRE task is unsolvable, call `UnsolvableQuery(reason='...')` followed by `finish(answer='Task cannot be completed: reason')`.
6. Maximum steps allowed: 10. If you haven't finished by step 10, call `finish()` with partial results.
7. Complete the task as much as possible with the fewest tools.

## Output Format:
Each step you output must follow this EXACT format (only Thought and Action, nothing else):

Thought: [Your reasoning about what to do next]
Action: [tool_name(param1='value1', param2='value2')]

Then STOP and wait. The system will provide the Observation.

## Important Rules:
- Use exact parameter values from the task or previous Observations.
- Never hallucinate tool names or parameters not in <provided_tools>.
- Each Action must be exactly ONE tool call.
- Do NOT generate Observation yourself - STOP after Action and wait for the system to provide it.
- Do NOT repeat a tool call you have already made. If you already called a tool and got its Observation, use that result and move on to the next step.
- Finish MUST be called to end the task (even for unsolvable tasks).
"""



_react_few_shot_example = """
## Example Interaction:

<Agent_role>
You are a project management assistant with access to project tracking tools.
</Agent_role>

<Env>
Current project: PROJ-1050
User: example_user
</Env>

<provided_tools>
- get_project_status(project_id) – Returns the current status of a project
- get_assigned_tasks(user_id, project_id) – Returns tasks assigned to a user in a project
- update_task_status(task_id, new_status) – Updates the status of a task
- finish(answer) – Call this to end the task and provide the final answer
</provided_tools>

<task>
Check the status of project PROJ-1050 and find all tasks assigned to example_user.
</task>

---

Thought: I need to first check the overall project status, then find tasks assigned to example_user.
Action: get_project_status(project_id='PROJ-1050')

Observation: Status: In Progress, Deadline: 2024-03-15

Thought: Project is in progress with deadline March 15. Now I need to get tasks assigned to example_user.
Action: get_assigned_tasks(user_id='example_user', project_id='PROJ-1050')

Observation: [Task-101: Code Review (pending), Task-102: Documentation (in-progress)]

Thought: I have all the information needed. Project PROJ-1050 is in progress and example_user has 2 tasks assigned.
Action: finish(answer='Project PROJ-1050 is In Progress with deadline 2024-03-15. example_user has 2 assigned tasks: Task-101 (Code Review - pending) and Task-102 (Documentation - in-progress).')

---

Now solve the given task following the same format.
"""









_reflexion_system_prompt_en = """You are an AI agent that can use tools to complete tasks. Use the ReAct (Reasoning + Acting) paradigm to solve the task step by step.

## Instructions:
1. At each step, first think about your approach (Thought), then call a tool (Action), then read the system-provided Observation.
2. Continue this Thought → Action → Observation loop until the task is completed or you determine it cannot be completed.
3. When the task is done, you MUST call `finish(answer='your_final_answer')` to terminate.
4. If a subtask cannot be completed due to missing tools, call `UnsolvableQuery(reason='explanation')`, then continue with remaining steps.
5. If the ENTIRE task is unsolvable, call `UnsolvableQuery(reason='...')` followed by `finish(answer='Task cannot be completed: reason')`.
6. Maximum steps allowed: 10. If you haven't finished by step 10, call `finish()` with partial results.
7. Complete the task as much as possible with the fewest tools.

## Self-Improvement:
You have attempted this task before. The reflections below summarize what to preserve or improve. Use them to avoid repeating mistakes and maintain valid tool-use behavior.

{reflections}

## Output Format:
Each step you output must follow this EXACT format (only Thought and Action, nothing else):

Thought: [Your reasoning about what to do next]
Action: [tool_name(param1='value1', param2='value2')]

Then STOP and wait. The system will provide the Observation.

## Important Rules:
- Use exact parameter values from the task or previous Observations.
- Never hallucinate tool names or parameters not in <provided_tools>.
- Each Action must be exactly ONE tool call.
- Do NOT generate Observation yourself - STOP after Action and wait for the system to provide it.
- Do NOT repeat a tool call you have already made. If you already called a tool and got its Observation, use that result and move on to the next step.
- Finish MUST be called to end the task (even for unsolvable tasks).
"""



_reflexion_self_reflect_prompt_en = """You are an advanced reasoning agent that can improve based on self-reflection. You will be given a previous reasoning trial in which an AI agent attempted to solve a task using tools. The trial may have succeeded or failed.

Analyze the trial carefully:
1. Identify which specific tool calls should be preserved or changed.
2. Identify any wrong tool names, wrong parameters, wrong order, or hallucinated tool calls.
3. Identify any missed necessary tool calls.
4. Identify whether the reasoning should be preserved or corrected.

In a few sentences, provide a concise, actionable plan for the second execution round. Be specific — reference exact tool names and parameter values where possible.

## Previous Trial:

<Agent_role>
{agent_role}
</Agent_role>

<Env>
{env}
</Env>

<provided_tools>
{tool_list}
</provided_tools>

<task>
{task}
</task>

### Execution Trace:
{scratchpad}

### Result:
Finished: {finished}
Final Answer: {final_answer}

Reflection:"""









_pte_planner_prompt_en = """You are a task planner. Given a task, an agent role, an environment, and a list of available tools, produce a complete execution plan.

## Instructions:
1. Analyze the task and determine the exact sequence of tool calls needed.
2. For each step, specify the tool name and its arguments.
3. If a parameter's value is explicitly known from the task or environment, use the exact value.
4. If a parameter depends on the output of a previous step (which has not been executed yet), use the placeholder "Step_N_Output" to indicate the data flow (N is the step number whose output is needed).
5. End the plan with finish(answer=...) to provide the final answer. Use "Step_N_Output" if the answer depends on a previous step's result.
6. If a step cannot be completed due to missing tools, use UnsolvableQuery(reason='...') for that step, then continue planning the remaining steps.
7. If the ENTIRE task is unsolvable, include UnsolvableQuery(reason='...') followed by finish(answer='Task cannot be completed: reason').
8. Complete the task as much as possible with the fewest tools.

## Output Format:
First, explain your reasoning in a <reasoning> block: analyze the task requirements, which tools are needed and why, and the logical ordering of steps.
Then, wrap your plan in <plan> and </plan> tags. Each step on its own line:
<reasoning>
Explain your analysis of the task, why you chose these tools and this ordering, and any considerations about data flow between steps.
</reasoning>
<plan>
Step 1: tool_name(param1="value1", param2="value2")
Step 2: tool_name(param1="Step_1_Output")
...
Step N: finish(answer="Step_N-1_Output")
</plan>

You MUST output both <reasoning>...</reasoning> and <plan>...</plan> blocks."""



_pte_executor_prompt_en = """You are a task executor. You are given a plan and must execute the current step by replacing any placeholders with actual values from previous observations.

## Complete Plan:
{plan}

## Execution History:
{execution_history}

## Current Step to Execute:
{current_step}

## Instructions:
- First, reason about the current step: what does it need to accomplish, what previous observations provide relevant data, and how to resolve any placeholders.
- Then, replace any placeholder like "Step_N_Output" with the actual value from that step's Observation.
- If the planned tool call already has concrete values (no placeholders), output it as-is.

## Output Format:
Thought: <your reasoning about this step, how you resolve placeholders, and why you chose these parameter values>
Action:"""

