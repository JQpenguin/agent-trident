import re

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.evaluation_func import (
    _evaluate_detecting, _evaluate_planning, _evaluate_diagnosing,
    _evaluate_planning_analysis, _evaluate_diagnosing_analysis,
    _evaluate_planning_v2, _evaluate_planning_analysis_v2,
    _evaluate_react, _evaluate_reflexion, _evaluate_plan_execute,
)

from utils.extract_func import (
    _extract_subgoal_plantool_del_lastfinish, remove_finish_if_last,
    extract_tool_name_from_call, extract_params_from_call,
    extract_all_answer_blocks, normalize_param_value,
)





def _extract_action_with_balanced_parens(text):



    action_match = re.search(r'Action:\s*', text)

    if not action_match:

        return ""



    start_pos = action_match.end()

    remaining = text[start_pos:]





    func_match = re.match(r'(\S+?)\(', remaining)

    if not func_match:

        return ""



    func_name = func_match.group(1)

    paren_start = func_match.end() - 1





    depth = 0

    in_string = False

    string_char = None

    escape_next = False



    for i, char in enumerate(remaining[paren_start:]):

        if escape_next:

            escape_next = False

            continue



        if char == '\\':

            escape_next = True

            continue



        if char in ('"', "'") and not in_string:

            in_string = True

            string_char = char

        elif char == string_char and in_string:

            in_string = False

            string_char = None

        elif not in_string:

            if char == '(':

                depth += 1

            elif char == ')':

                depth -= 1

                if depth == 0:



                    return remaining[:paren_start + i + 1]





    simple_match = re.match(r'(\S+\([^)]*\))', remaining)

    return simple_match.group(1) if simple_match else ""





def convert_tools_list_to_str(tools_list):

    if isinstance(tools_list, str):

        return tools_list

    return "\n".join([f"{i+1}. {tool}" for i, tool in enumerate(tools_list)])





def _process_task_infer_react(task_item, generation_func, args, system_prompt_override=None):

    from utils.generation_prompt import _react_system_prompt_en, _react_few_shot_example



    task_query = task_item["task"]

    tool_list_str = convert_tools_list_to_str(task_item["tools"])

    agent_role = task_item.get("Agent_role", "")

    env = task_item.get("Env", "")

    simulation_script = task_item.get("simulation_script", {})





    system_prompt = system_prompt_override if system_prompt_override is not None else _react_system_prompt_en

    few_shot = _react_few_shot_example



    initial_prompt = f"""{system_prompt}

{few_shot}

<Agent_role>
{agent_role}
</Agent_role>

<Env>
{env}
</Env>

<provided_tools>
{tool_list_str}
</provided_tools>

<task>
{task_query}
</task>

Now begin solving the task. Remember to call finish() when done.
"""





    thoughts = []

    actions = []

    observations = []

    conversation_history = initial_prompt

    finished = False

    final_answer = ""

    max_steps = 10

    golden_tools = task_item.get("planning_tools", [])

    golden_cursor = 0



    for step_num in range(1, max_steps + 1):



        response = generation_func(conversation_history)





        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)



        action_str = _extract_action_with_balanced_parens(response)



        thought = thought_match.group(1).strip() if thought_match else ""



        thoughts.append(thought)

        actions.append(action_str)





        tool_name = extract_tool_name_from_call(action_str)

        params = extract_params_from_call(action_str)





        if tool_name.lower() == "finish":

            finished = True

            final_answer = params.get("answer", params.get("result", ""))

            observations.append(f"Task completed with answer: {final_answer}")

            break





        observation, golden_cursor = _simulate_tool_execution(

            tool_name, params, simulation_script, step_num,

            golden_tools=golden_tools, golden_cursor=golden_cursor

        )

        observations.append(observation)





        conversation_history += f"\n{response}\nObservation: {observation}\n\nStep {step_num + 1}:\n"



    return {

        "thoughts": thoughts,

        "actions": actions,

        "observations": observations,

        "total_steps": len(actions),

        "finished": finished,

        "final_answer": final_answer

    }





def _params_match(pred_params, golden_params):

    mismatched_keys = []

    for key, golden_val in golden_params.items():

        golden_val_clean = normalize_param_value(golden_val)



        if re.match(r'Step_\d+_Output', golden_val_clean):

            continue

        pred_val = pred_params.get(key, None)

        if pred_val is None:

            mismatched_keys.append(key)

            continue

        pred_val_clean = normalize_param_value(pred_val)

        if pred_val_clean != golden_val_clean:

            mismatched_keys.append(key)

    return (len(mismatched_keys) == 0, mismatched_keys)





def _simulate_tool_execution(tool_name, params, simulation_script, step_num,

                              golden_tools=None, golden_cursor=0):



    if not golden_tools:

        if tool_name == "UnsolvableQuery":

            reason = params.get("reason", "Unknown reason")

            return f"Query marked as unsolvable: {reason}", golden_cursor

        if tool_name in simulation_script:

            tool_sim = simulation_script[tool_name]

            if "return_value" in tool_sim:

                return str(tool_sim["return_value"]), golden_cursor

            elif "observation" in tool_sim:

                return str(tool_sim["observation"]), golden_cursor

        return f"Tool {tool_name} executed successfully", golden_cursor





    golden_no_finish = []

    for t in golden_tools:

        g_name = extract_tool_name_from_call(t)

        if g_name.lower() != "finish":

            golden_no_finish.append(t)





    def _try_return_from_script(t_name, t_params):

        if t_name == "UnsolvableQuery":

            reason = t_params.get("reason", "Unknown reason")

            return f"Query marked as unsolvable: {reason}"

        if t_name in simulation_script:

            tool_sim = simulation_script[t_name]

            if "return_value" in tool_sim:

                return str(tool_sim["return_value"])

            elif "observation" in tool_sim:

                return str(tool_sim["observation"])



        return "[No response received]"





    if golden_cursor >= len(golden_no_finish):

        return _try_return_from_script(tool_name, params), golden_cursor





    matched_j = None

    for j in range(golden_cursor, len(golden_no_finish)):

        g_name = extract_tool_name_from_call(golden_no_finish[j])

        if g_name == tool_name:

            matched_j = j

            break





    if matched_j is None:

        return _try_return_from_script(tool_name, params), golden_cursor





    golden_params = extract_params_from_call(golden_no_finish[matched_j])





    if tool_name == "UnsolvableQuery":

        new_cursor = matched_j + 1

        return _try_return_from_script(tool_name, params), new_cursor



    match_ok, mismatched_keys = _params_match(params, golden_params)

    if match_ok:



        new_cursor = matched_j + 1

        return _try_return_from_script(tool_name, params), new_cursor

    else:



        hint = ", ".join(f"'{k}'" for k in mismatched_keys)

        return (f"Tool '{tool_name}' execution failed. "

                f"The following parameter(s) have incorrect values: {hint}. "

                f"Please check and retry."), golden_cursor





def _process_task_infer_reflexion(task_item, generation_func, args, preloaded_react_result=None):

    from utils.generation_prompt import (

        _reflexion_system_prompt_en, _react_few_shot_example,

        _reflexion_self_reflect_prompt_en

    )



    max_trials = getattr(args, 'max_reflexion_trials', 2)



    task_query = task_item["task"]

    tool_list_str = convert_tools_list_to_str(task_item["tools"])

    agent_role = task_item.get("Agent_role", "")

    env = task_item.get("Env", "")

    golden_tools = task_item.get("planning_tools", [])



    all_trials = []

    reflections = []



    for trial_num in range(max_trials):



        if trial_num == 0 and preloaded_react_result is not None:



            trial_result = preloaded_react_result

        elif trial_num == 0 or not reflections:



            trial_result = _process_task_infer_react(task_item, generation_func, args)

        else:



            reflections_str = _format_reflections(reflections)

            reflexion_prompt = _reflexion_system_prompt_en.replace("{reflections}", reflections_str)

            trial_result = _process_task_infer_react(

                task_item, generation_func, args,

                system_prompt_override=reflexion_prompt

            )



        all_trials.append(trial_result)





        if trial_num < max_trials - 1:

            thoughts = trial_result["thoughts"]

            actions = trial_result["actions"]

            observations = trial_result["observations"]

            finished = trial_result["finished"]

            final_answer = trial_result["final_answer"]



            scratchpad = _build_scratchpad(thoughts, actions, observations)

            reflect_prompt = _reflexion_self_reflect_prompt_en.format(

                agent_role=agent_role,

                env=env,

                tool_list=tool_list_str,

                task=task_query,

                scratchpad=scratchpad,

                finished=str(finished),

                final_answer=final_answer

            )



            reflection = generation_func(reflect_prompt)



            reflection = reflection.strip()

            if reflection.startswith("Reflection:"):

                reflection = reflection[len("Reflection:"):].strip()

            reflections.append(reflection)





    final_trial = all_trials[-1]

    first_trial_success = _quick_evaluate_trial(all_trials[0], golden_tools) if all_trials else False

    final_trial_success = _quick_evaluate_trial(final_trial, golden_tools)



    return {

        "trials": all_trials,

        "reflections": reflections,

        "num_trials": len(all_trials),

        "final_result": final_trial,

        "improved": (not first_trial_success and final_trial_success),



        "thoughts": final_trial["thoughts"],

        "actions": final_trial["actions"],

        "observations": final_trial["observations"],

        "total_steps": final_trial["total_steps"],

        "finished": final_trial["finished"],

        "final_answer": final_trial["final_answer"]

    }





def _quick_evaluate_trial(trial_result, golden_tools):

    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call



    if not trial_result.get("finished", False):

        return False



    if not golden_tools:

        return True



    pred_actions = trial_result.get("actions", [])

    pred_tool_names = [extract_tool_name_from_call(a) for a in pred_actions]

    pred_tool_names_no_finish = []

    pred_actions_no_finish = []

    for i, t in enumerate(pred_tool_names):

        if t.lower() != "finish":

            pred_tool_names_no_finish.append(t)

            pred_actions_no_finish.append(pred_actions[i])



    golden_tool_names = [extract_tool_name_from_call(t) for t in golden_tools]

    golden_no_finish = []

    golden_names_no_finish = []

    for i, t in enumerate(golden_tool_names):

        if t.lower() != "finish":

            golden_names_no_finish.append(t)

            golden_no_finish.append(golden_tools[i])



    if len(pred_tool_names_no_finish) < len(golden_names_no_finish):

        return False





    golden_idx = 0

    for i, pred_name in enumerate(pred_tool_names_no_finish):

        if golden_idx >= len(golden_names_no_finish):

            break

        if pred_name == golden_names_no_finish[golden_idx]:

            golden_params = extract_params_from_call(golden_no_finish[golden_idx])

            pred_params = extract_params_from_call(pred_actions_no_finish[i])

            if _params_match(pred_params, golden_params)[0]:

                golden_idx += 1



    return golden_idx >= len(golden_names_no_finish)





def _build_scratchpad(thoughts, actions, observations):

    lines = []

    for i in range(len(actions)):

        if i < len(thoughts) and thoughts[i]:

            lines.append(f"Thought {i+1}: {thoughts[i]}")

        lines.append(f"Action {i+1}: {actions[i]}")

        if i < len(observations):

            lines.append(f"Observation {i+1}: {observations[i]}")

    return "\n".join(lines)





def _format_reflections(reflections, max_reflections=3):

    recent = reflections[-max_reflections:]

    lines = []

    for i, ref in enumerate(recent, 1):

        lines.append(f"Reflection {i}: {ref}")

    return "\n".join(lines)





def _process_task_infer_plan_execute(task_item, generation_func, args):

    from utils.generation_prompt import _pte_planner_prompt_en, _pte_executor_prompt_en



    task_query = task_item["task"]

    tool_list_str = convert_tools_list_to_str(task_item["tools"])

    agent_role = task_item.get("Agent_role", "")

    env_desc = task_item.get("Env", "")

    simulation_script = task_item.get("simulation_script", {})





    plan_prompt = f"""{_pte_planner_prompt_en}

<Agent_role>
{agent_role}
</Agent_role>

<Env>
{env_desc}
</Env>

<provided_tools>
{tool_list_str}
</provided_tools>

<task>
{task_query}
</task>
"""



    plan_response = generation_func(plan_prompt)

    planned_steps = _parse_plan_steps(plan_response)

    plan_reasoning = _extract_plan_reasoning(plan_response)





    if not planned_steps:

        return {

            "plan": [],

            "plan_reasoning": plan_reasoning,

            "actions": [],

            "observations": [],

            "thoughts": [plan_reasoning] if plan_reasoning else [],

            "total_steps": 0,

            "finished": False,

            "final_answer": ""

        }





    actions = []

    observations = []

    thoughts = []

    env_context = {}

    finished = False

    final_answer = ""

    golden_tools = task_item.get("planning_tools", [])

    golden_cursor = 0



    for step_i, planned_call in enumerate(planned_steps):



        execution_history = _build_execution_history(actions, observations)





        plan_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(planned_steps)])

        current_step_text = f"Step {step_i+1}: {planned_call}"



        executor_prompt = _pte_executor_prompt_en.format(

            plan=plan_text,

            execution_history=execution_history if execution_history else "(No steps executed yet)",

            current_step=current_step_text

        )





        executor_response = generation_func(executor_prompt)





        thought_str = _extract_thought_from_executor(executor_response)



        if step_i == 0 and plan_reasoning:

            thought_str = f"[Plan Reasoning] {plan_reasoning} [Step Reasoning] {thought_str}" if thought_str else f"[Plan Reasoning] {plan_reasoning}"

        thoughts.append(thought_str)



        action_str = _extract_action_from_executor(executor_response, planned_call, env_context)





        tool_name = extract_tool_name_from_call(action_str)

        params = extract_params_from_call(action_str)



        actions.append(action_str)





        if tool_name.lower() == "finish":

            finished = True

            final_answer = params.get("answer", params.get("result", ""))

            observations.append(f"Task completed with answer: {final_answer}")

            break





        observation, golden_cursor = _simulate_tool_execution(

            tool_name, params, simulation_script, step_i + 1,

            golden_tools=golden_tools, golden_cursor=golden_cursor

        )

        observations.append(observation)





        env_context[f"Step_{step_i+1}_Output"] = observation



    return {

        "plan": planned_steps,

        "plan_reasoning": plan_reasoning,

        "actions": actions,

        "observations": observations,

        "thoughts": thoughts,

        "total_steps": len(actions),

        "finished": finished,

        "final_answer": final_answer

    }





def _parse_plan_steps(plan_response):



    plan_match = re.search(r'<plan>(.*?)</plan>', plan_response, re.DOTALL)

    if plan_match:

        plan_text = plan_match.group(1).strip()

    else:



        plan_text = plan_response.strip()



    steps = []

    for line in plan_text.split('\n'):

        line = line.strip()

        if not line:

            continue



        step_match = re.match(r'(?:Step\s*\d+\s*[:：]\s*)(.*)', line)

        if step_match:

            steps.append(step_match.group(1).strip())

        elif re.match(r'\w+\(', line):



            steps.append(line.strip())



    return steps





def _extract_plan_reasoning(plan_response):

    match = re.search(r'<reasoning>(.*?)</reasoning>', plan_response, re.DOTALL)

    if match:

        return match.group(1).strip()

    return ""





def _extract_thought_from_executor(executor_response):



    match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', executor_response, re.DOTALL)

    if match:

        thought = match.group(1).strip()

        if thought:

            return thought

    return ""





def _build_execution_history(actions, observations):

    if not actions:

        return ""

    lines = []

    for i in range(len(actions)):

        lines.append(f"Step {i+1}: Action: {actions[i]}")

        if i < len(observations):

            lines.append(f"  Observation: {observations[i]}")

    return "\n".join(lines)





def _extract_action_from_executor(executor_response, planned_call, env_context):



    action_str = _extract_action_with_balanced_parens(

        "Action: " + executor_response if "Action:" not in executor_response else executor_response

    )



    result = action_str if action_str and re.match(r'\w+\(', action_str) else planned_call

    for placeholder, value in env_context.items():

        result = result.replace(f'"{placeholder}"', f'"{value}"')

        result = result.replace(f"'{placeholder}'", f"'{value}'")

        result = result.replace(placeholder, str(value))

    return result





def convert_hf_data(items):

    tasks_dict = {}

    for item in items:

        subtask = item["subtask"]

        if subtask not in tasks_dict:

            tasks_dict[subtask] = []

        tasks_dict[subtask].append(item)

    return tasks_dict



def _get_token_snapshot(generation_func):

    try:

        usage = generation_func.__self__.total_usage

        return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]}

    except (AttributeError, KeyError):

        return None



def _calc_token_delta(before, after):

    if before is None or after is None:

        return {}

    return {

        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],

        "completion_tokens": after["completion_tokens"] - before["completion_tokens"]

    }



def _process_task_infer(prompt_type, task_item, generation_func, args, preloaded_react_result=None):

    task_query = task_item["task"]

    tool_list_str = convert_tools_list_to_str(task_item["tools"])





    agent_role = task_item.get("Agent_role", "")

    env = task_item.get("Env", "")

    solvability = task_item.get("Solvability", "Solvable")





    token_before = _get_token_snapshot(generation_func)



    if prompt_type == "level_1":



        prompt = f"{args.detecting_prompt}\n\n<Agent_role>\n{agent_role}\n</Agent_role>\n<Env>\n{env}\n</Env>\n<task>\n{task_query}\n</task>\n<provided_tools>\n{tool_list_str}\n</provided_tools>"

        response = generation_func(prompt)





        golden_solvability = "solvable" if solvability == "Solvable" else "unsolvable"



        result = {"response": response, "golden": golden_solvability}

        token_after = _get_token_snapshot(generation_func)

        result.update(_calc_token_delta(token_before, token_after))

        return result



    elif prompt_type == "level_2":



        agent_role = task_item.get("Agent_role", "")

        env = task_item.get("Env", "")

        solvability = task_item.get("Solvability", "Solvable")



        prompt = f"{args.planning_prompt}\n\n<Agent_role>\n{agent_role}\n</Agent_role>\n<Env>\n{env}\n</Env>\n<task>\n{task_query}\n</task>\n<provided_tools>\n{tool_list_str}\n</provided_tools>"

        raw_response = generation_func(prompt)



        golden_solvability = "solvable" if solvability == "Solvable" else "unsolvable"





        parsed_tools = extract_all_answer_blocks(raw_response, args.answer_pattern)



        result = {

            "response": parsed_tools,

            "raw_response": raw_response,

            "golden": golden_solvability,

            "planning_tools": task_item.get("planning_tools", [])

        }

        token_after = _get_token_snapshot(generation_func)

        result.update(_calc_token_delta(token_before, token_after))

        return result



    elif prompt_type == "level_3":



        solvability = task_item.get("Solvability", "Solvable")

        golden_solvability = "solvable" if solvability == "Solvable" else "unsolvable"



        agent_strategy = getattr(args, 'agent_strategy', 'react')



        if agent_strategy == "reflexion":



            react_result = _process_task_infer_reflexion(task_item, generation_func, args, preloaded_react_result=preloaded_react_result)

        elif agent_strategy == "plan_execute":



            react_result = _process_task_infer_plan_execute(task_item, generation_func, args)

        else:



            react_result = _process_task_infer_react(task_item, generation_func, args)



        return {

            "response": react_result,

            "golden": golden_solvability,

            "planning_tools": task_item.get("planning_tools", []),

            **_calc_token_delta(token_before, _get_token_snapshot(generation_func))

        }



def _process_task_eval(prompt_type, task_item, task, args,

                    result_task):

    answer_pattern = args.answer_pattern

    tool_list_str = convert_tools_list_to_str(task_item["tools"])





    provided_tool_dict = {"UnsolvableQuery": "Used to determine that the Query or Subgoal can not be completed"}





    if isinstance(task_item["tools"], list):

        for tool_str in task_item["tools"]:



            tool_match = re.match(r'([^\(]+)', tool_str)

            if tool_match:

                tool_name = tool_match.group(1).strip()

                provided_tool_dict[tool_name] = tool_str

    else:



        pattern = re.compile(r'\d+\.\s*([^\:：]+)[:：]\s*(.*)')

        matches = pattern.findall(tool_list_str)

        for match in matches:

            tool_name, tool_desc = match

            provided_tool_dict[tool_name] = tool_desc





    solvability = task_item.get("Solvability", "Solvable")

    golden_solvability = "solvable" if solvability == "Solvable" else "unsolvable"



    if prompt_type == "level_1":



        response = result_task.get("response", "")

        golden = result_task.get("golden", golden_solvability)





        match = re.search(answer_pattern, response)

        pred = match.group(1).strip() if match else ""





        eval_result = _evaluate_detecting(pred, golden)





        result = {

            "metric": eval_result["metric"],

            "golden": golden,

            "pred": pred,

            "is_confident": eval_result["is_confident"],

            "is_correct": eval_result["is_correct"],

            "is_uncertain": eval_result["is_uncertain"],

            "pred_category": eval_result["pred_category"],

        }



        if "prompt_tokens" in result_task:

            result["prompt_tokens"] = result_task["prompt_tokens"]

            result["completion_tokens"] = result_task.get("completion_tokens", 0)

        return result



    elif prompt_type == "level_2":



        response = result_task.get("response", "")

        golden = result_task.get("golden", golden_solvability)





        simulation_script = task_item.get("simulation_script", {})





        provided_tool_list = list(task_item["tools"]) if isinstance(task_item["tools"], list) else []

        provided_tool_list.append("UnsolvableQuery")









        if isinstance(response, list):

            pred_tools = response

        else:

            pred_tools = extract_all_answer_blocks(response, answer_pattern)



        if pred_tools:





            scores = _evaluate_planning_v2(

                remove_finish_if_last(pred_tools),

                remove_finish_if_last(task_item["planning_tools"]),

                provided_tool_list,

                args.calculate_type,

                simulation_script,

                emb_model=getattr(args, 'emb_model', None)

            )



            scores_analysis, condition = _evaluate_planning_analysis_v2(

                remove_finish_if_last(pred_tools),

                remove_finish_if_last(task_item["planning_tools"]),

                provided_tool_list,

                args.calculate_type,

                simulation_script,

                emb_model=getattr(args, 'emb_model', None)

            )

        else:



            scores = {"tool_score_soft": 0.0, "param_score": 0.0,

                       "task_success_soft": 0, "planning_precision": 0.0,

                       "param_details": []}

            scores_analysis, condition = {"tool_score_soft": 0.0, "param_score": 0.0,

                                           "task_success_soft": 0, "planning_precision": 0.0}, "no_answer"





        result = {

            "tool_score_soft": scores["tool_score_soft"],

            "param_score": scores["param_score"],

            "task_success_soft": scores["task_success_soft"],

            "planning_precision": scores["planning_precision"],

            "condition": condition,

            "param_details": scores.get("param_details", []),

            "golden": golden,

            "pred_tool_count": len(remove_finish_if_last(pred_tools)) if pred_tools else 0,

            "golden_tool_count": len(remove_finish_if_last(task_item["planning_tools"])),

        }



        if "prompt_tokens" in result_task:

            result["prompt_tokens"] = result_task["prompt_tokens"]

            result["completion_tokens"] = result_task.get("completion_tokens", 0)

        return result



    elif prompt_type == "level_3":



        react_response = result_task.get("response", {})

        golden = result_task.get("golden", golden_solvability)

        planning_tools = result_task.get("planning_tools", task_item.get("planning_tools", []))

        simulation_script = task_item.get("simulation_script", {})

        task_description = task_item.get("task", "")





        provided_tool_list = list(task_item["tools"]) if isinstance(task_item["tools"], list) else []

        provided_tool_list.append("UnsolvableQuery")

        provided_tool_list.append("finish")



        agent_strategy = getattr(args, 'agent_strategy', 'react')



        if agent_strategy == "reflexion" and "trials" in react_response:



            metrics = _evaluate_reflexion(

                reflexion_response=react_response,

                golden_tools=planning_tools,

                provided_tools=provided_tool_list,

                simulation_script=simulation_script,

                task_description=task_description,

                calculate_type=args.calculate_type,

                emb_model=args.emb_model

            )

        elif agent_strategy == "plan_execute" and "plan" in react_response:



            metrics = _evaluate_plan_execute(

                pte_response=react_response,

                golden_tools=planning_tools,

                provided_tools=provided_tool_list,

                simulation_script=simulation_script,

                task_description=task_description,

                calculate_type=args.calculate_type,

                emb_model=args.emb_model

            )

        else:



            metrics = _evaluate_react(

                react_response=react_response,

                golden_tools=planning_tools,

                provided_tools=provided_tool_list,

                simulation_script=simulation_script,

                task_description=task_description,

                calculate_type=args.calculate_type,

                emb_model=args.emb_model

            )

        metrics["golden"] = golden



        if "prompt_tokens" in result_task:

            metrics["prompt_tokens"] = result_task["prompt_tokens"]

            metrics["completion_tokens"] = result_task.get("completion_tokens", 0)



        return metrics



def process_task(level, task_item, generation_func, task, args, result=None, preloaded_react_result=None):

    if args.mode in ["infer", "recover"]:

        return _process_task_infer(level, task_item, generation_func, args, preloaded_react_result=preloaded_react_result)

    elif args.mode == "eval":

        return _process_task_eval(level, task_item, task, args, result)

    else:

        raise Exception("Invalid mode")



def process_all_tasks_infer(items, args, generation_func, results=None):

    results = results or {}





    react_results = None

    if getattr(args, 'react_results_path', None) and getattr(args, 'agent_strategy', 'react') == 'reflexion':

        from utils.load_save import load_results

        react_results = load_results(args.react_results_path)

        print("Loaded configured ReAct results for Reflexion reuse.")



    for task, task_items in items.items():

        print(f"task: {task}, count: {len(task_items)}")

        results[task] = {}

        for task_item in tqdm(task_items, desc=f"Processing {task} items with {args.model_name_save}"):

            levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]

            task_item_index = task_items.index(task_item)

            for level in tqdm(levels, desc="Processing levels", leave=False):

                results[task] = results.get(task, {})

                result_list = results[task].get(level, [None] * len(task_items))

                result = result_list[task_item_index] if task_item_index < len(result_list) else None





                preloaded_react_result = None

                if level == "level_3" and react_results is not None:

                    try:

                        react_level3 = react_results.get(task, {}).get("level_3", [])

                        if task_item_index < len(react_level3):

                            preloaded_react_result = react_level3[task_item_index].get("response", None)

                    except (IndexError, AttributeError, TypeError):

                        pass



                level_result = process_task(level, task_item, generation_func, task, args, result, preloaded_react_result=preloaded_react_result)

                results[task][level] = results[task].get(level, []) + [level_result]



    return results





def _process_single_task_item(task, task_item, task_item_index, levels, generation_func, args, preloaded_react_result=None):

    item_results = {}

    for level in levels:

        pr = preloaded_react_result if level == "level_3" else None

        level_result = process_task(level, task_item, generation_func, task, args, None, preloaded_react_result=pr)

        item_results[level] = level_result

    return task, task_item_index, item_results





def process_all_tasks_infer_parallel(items, args, generation_func, results=None):

    results = results or {}

    num_workers = getattr(args, 'num_workers', 4)





    react_results = None

    if getattr(args, 'react_results_path', None) and getattr(args, 'agent_strategy', 'react') == 'reflexion':

        from utils.load_save import load_results

        react_results = load_results(args.react_results_path)

        print("Loaded configured ReAct results for Reflexion reuse.")





    all_tasks = []

    for task, task_items in items.items():

        print(f"task: {task}, count: {len(task_items)}")

        results[task] = {}

        levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]



        for task_item_index, task_item in enumerate(task_items):



            preloaded_react_result = None

            if react_results is not None:

                try:

                    react_level3 = react_results.get(task, {}).get("level_3", [])

                    if task_item_index < len(react_level3):

                        preloaded_react_result = react_level3[task_item_index].get("response", None)

                except (IndexError, AttributeError, TypeError):

                    pass

            all_tasks.append((task, task_item, task_item_index, levels, preloaded_react_result))



    print(f"\ntotal tasks: {len(all_tasks)}, parallel workers: {num_workers}")





    for task, task_items in items.items():

        levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]

        for level in levels:

            results[task][level] = [None] * len(task_items)





    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = {

            executor.submit(

                _process_single_task_item,

                task, task_item, task_item_index, levels, generation_func, args,

                preloaded_react_result=preloaded_react_result

            ): (task, task_item_index)

            for task, task_item, task_item_index, levels, preloaded_react_result in all_tasks

        }





        with tqdm(total=len(futures), desc="Parallel inference") as pbar:

            for future in as_completed(futures):

                try:

                    task, task_item_index, item_results = future.result()



                    for level, level_result in item_results.items():

                        results[task][level][task_item_index] = level_result

                    pbar.update(1)

                except Exception as e:

                    task, task_item_index = futures[future]

                    print(f"\nError processing task={task}, index={task_item_index}: {type(e).__name__}")

                    pbar.update(1)



    return results





def process_all_tasks_recover(items, args, generation_func, results):

    for task, task_items in items.items():

        print(f"task: {task}, count: {len(task_items)}")

        results[task] = results.get(task, {})

        for task_item in tqdm(task_items, desc=f"Processing {task} items with {args.model_name_save}"):

            levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]

            for level in tqdm(levels, desc="Processing levels", leave=False):

                result_list = results[task].get(level, [None] * len(task_items))

                task_item_index = task_items.index(task_item)

                if task_item_index < len(result_list):

                    result = result_list[task_item_index]

                else:

                    result = None





                if isinstance(result, dict):

                    needs_update = any(

                        (isinstance(v, str) and v == "") or

                        (isinstance(v, list) and len(v) == 0)

                        for v in result.values()

                    )

                else:

                    needs_update = result == ""



                if needs_update:

                    print(f"Updating result for: {task}, item_id: {task_item_index}, level: {level}")

                    level_result = process_task(level, task_item, generation_func, task, args, result)

                    result_list[task_item_index] = level_result

                    results[task][level] = result_list

                else:

                    print(f"Skipping for task: {task}, item_id: {task_item_index}, level: {level}")



    return results



def process_all_tasks_eval(items, args, results):

    metrics = {}



    for task, task_items in items.items():

        print(f"task: {task}, count: {len(task_items)}")

        metrics[task] = {}

        for task_item in tqdm(task_items, desc=f"Processing {task} items with {args.model_name_save}"):

            levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]

            for level in tqdm(levels, desc="Processing levels", leave=False):

                metrics[task] = metrics.get(task, {})

                result_list = results[task].get(level, [])

                task_item_index = task_items.index(task_item)

                if task_item_index < len(result_list):

                    result = result_list[task_item_index]

                else:

                    print(f"Result not found for task={task}, index={task_item_index}")

                    raise Exception("Result not found in eval mode")



                level_result = process_task(level, task_item, None, task, args, result)





                if level_result is not None and "domain" in task_item:

                    level_result["domain"] = task_item["domain"]



                metrics[task][level] = metrics[task].get(level, []) + [level_result]



    return metrics

