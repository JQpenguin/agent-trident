import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def check_if_value_from_previous_step(value, current_step_index, simulation_script, golden_planning_tools):
    from utils.extract_func import extract_tool_name_from_call
    if not simulation_script or current_step_index == 0:
        return False, None
    for i in range(current_step_index):
        prev_tool_call = golden_planning_tools[i]
        prev_tool_name = extract_tool_name_from_call(prev_tool_call)
        if prev_tool_name in simulation_script:
            prev_return = simulation_script[prev_tool_name].get("return_value", "")
            if str(value).strip() == str(prev_return).strip():
                return True, i + 1
    return False, None


def _calculate_step_utilization(pred_tool_names, golden_tool_names):
    from collections import Counter
    if not pred_tool_names:
        return 0.0
    golden_remaining = Counter(golden_tool_names)
    useful = 0
    for tool in pred_tool_names:
        if golden_remaining.get(tool, 0) > 0:
            useful += 1
            golden_remaining[tool] -= 1
    return useful / len(pred_tool_names)


def _calculate_process_precision(pred_tool_calls, golden_tool_calls, simulation_script=None, emb_model=None):
    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call
    if not pred_tool_calls:
        return 0.0
    if not golden_tool_calls:
        return 0.0
    valid_count = 0
    golden_cursor = 0
    simulation_script = simulation_script or {}
    golden_names = [extract_tool_name_from_call(call) for call in golden_tool_calls]
    for pred_call in pred_tool_calls:
        pred_name = extract_tool_name_from_call(pred_call)
        if not pred_name:
            continue
        matched_index = None
        for idx in range(golden_cursor, len(golden_tool_calls)):
            if golden_names[idx] == pred_name:
                matched_index = idx
                break
        if matched_index is None:
            continue
        pred_params = extract_params_from_call(pred_call)
        golden_params = extract_params_from_call(golden_tool_calls[matched_index])
        param_result = compare_params(
            pred_params, golden_params, matched_index, simulation_script, golden_tool_calls,
            tool_name=pred_name, emb_model=emb_model
        )
        if param_result["score"] >= 1.0:
            valid_count += 1
            golden_cursor = matched_index + 1
    return valid_count / len(pred_tool_calls)


def compare_params(pred_params, golden_params, step_index, simulation_script, golden_planning_tools, tool_name=None, emb_model=None):
    from utils.extract_func import is_step_output_placeholder, extract_step_number_from_placeholder
    if not golden_params:
        return {"score": 1.0, "missing": [], "wrong_value": [], "details": {}}
    if tool_name == "UnsolvableQuery":
        return {"score": 1.0, "missing": [], "wrong_value": [], "details": {
            k: {"status": "correct", "type": "unsolvable_query_bypass"} for k in golden_params
        }}
    result = {"missing": [], "wrong_value": [], "details": {}}
    correct_count = 0
    for param_name, golden_value in golden_params.items():
        if param_name not in pred_params:
            result["missing"].append(param_name)
            result["details"][param_name] = {"status": "missing", "expected": golden_value}
            continue
        pred_value = pred_params[param_name]
        if is_step_output_placeholder(pred_value):
            is_from_prev, expected_step = check_if_value_from_previous_step(
                golden_value, step_index, simulation_script, golden_planning_tools
            )
            actual_step = extract_step_number_from_placeholder(pred_value)
            if is_from_prev and actual_step == expected_step:
                correct_count += 1
                result["details"][param_name] = {
                    "status": "correct",
                    "type": "placeholder_match",
                    "expected_step": expected_step,
                    "actual_step": actual_step
                }
            elif is_from_prev:
                result["wrong_value"].append(param_name)
                result["details"][param_name] = {
                    "status": "wrong",
                    "expected": golden_value,
                    "got": pred_value,
                    "reason": "wrong_step_number",
                    "expected_step": expected_step,
                    "actual_step": actual_step
                }
            else:
                result["wrong_value"].append(param_name)
                result["details"][param_name] = {
                    "status": "wrong",
                    "expected": golden_value,
                    "got": pred_value,
                    "reason": "unexpected_placeholder"
                }
        else:
            from utils.extract_func import normalize_param_value
            if normalize_param_value(pred_value) == normalize_param_value(golden_value):
                correct_count += 1
                result["details"][param_name] = {"status": "correct", "type": "exact_match"}
            elif tool_name == "UnsolvableQuery" and emb_model is not None:
                from sklearn.metrics.pairwise import cosine_similarity
                pred_emb = emb_model.encode(str(pred_value).strip())
                golden_emb = emb_model.encode(str(golden_value).strip())
                similarity = cosine_similarity([pred_emb], [golden_emb])[0][0]
                if similarity >= 0.75:
                    correct_count += 1
                    result["details"][param_name] = {
                        "status": "correct",
                        "type": "semantic_match",
                        "similarity": float(similarity)
                    }
                else:
                    result["wrong_value"].append(param_name)
                    result["details"][param_name] = {
                        "status": "wrong",
                        "expected": golden_value,
                        "got": pred_value,
                        "reason": "semantic_mismatch",
                        "similarity": float(similarity)
                    }
            else:
                is_from_prev, expected_step = check_if_value_from_previous_step(
                    golden_value, step_index, simulation_script, golden_planning_tools
                )
                if is_from_prev:
                    result["wrong_value"].append(param_name)
                    result["details"][param_name] = {
                        "status": "wrong",
                        "expected": f"Step_{expected_step}_Output (or '{golden_value}')",
                        "got": pred_value,
                        "reason": "should_use_placeholder"
                    }
                else:
                    result["wrong_value"].append(param_name)
                    result["details"][param_name] = {
                        "status": "wrong",
                        "expected": golden_value,
                        "got": pred_value,
                        "reason": "value_mismatch"
                    }
    result["score"] = correct_count / len(golden_params) if golden_params else 1.0
    return result


def calculate_progress_rate(pred_planning_tool_list, golden_planning_tool_list, calculate_type):
    if not golden_planning_tool_list:
        return 1.0 if not pred_planning_tool_list else 0.0
    if calculate_type == "hard":
        match_number = 0
        for i in range(len(golden_planning_tool_list)):
            if i < len(pred_planning_tool_list):
                if pred_planning_tool_list[i] == golden_planning_tool_list[i]:
                    match_number += 1
                else:
                    break
            else:
                break
        progress_rate = match_number / len(golden_planning_tool_list)
        return progress_rate
    elif calculate_type == "soft":
        match_count = 0
        g_index, p_index = 0, 0
        while g_index < len(golden_planning_tool_list) and p_index < len(pred_planning_tool_list):
            if golden_planning_tool_list[g_index] == pred_planning_tool_list[p_index]:
                g_index += 1
                match_count += 1
            p_index += 1
        progress_rate = match_count / len(golden_planning_tool_list)
        return progress_rate
    else:
        raise Exception("calculate_type must be hard or soft")


def calculate_scorers(pred_subgoal_texts, pred_planning_tool_list,
                    golden_planning_tool_list,
                    origin_provided_subgoal_dict, solvable_planning_tool_list,
                    emb_model, tools_embedding, task, args):
    scorers = []
    for i in range(len(golden_planning_tool_list)):
        if golden_planning_tool_list[i] == "UnsolvableQuery":
            if i < len(pred_planning_tool_list):
                if golden_planning_tool_list[i] == pred_planning_tool_list[i]:
                    task_emb = tools_embedding[task]
                    pattern = r"Subgoal\s*\d+[:：]\s*(.*)"
                    pred_subgoal_text_pure = re.sub(pattern, r"\1", pred_subgoal_texts[i])
                    if args.embedding_model == "minilm":
                        pred_embedding = emb_model.encode(pred_subgoal_text_pure)
                    elif args.embedding_model == "gemini":
                        pred_embedding = emb_model(pred_subgoal_text_pure)["embedding"]
                    else:
                        raise ValueError(f"Unsupported embedding_model: {args.embedding_model}")
                    similarities = cosine_similarity([pred_embedding], task_emb["embeddings"])[0]
                    best_match_index = np.argmax(similarities)
                    best_task_name = task_emb["name"][best_match_index]
                    if best_task_name == solvable_planning_tool_list[i]:
                        scorers.append(1.0)
                    else:
                        true_deleted_tool_desc = re.sub(pattern, r"\1", origin_provided_subgoal_dict[solvable_planning_tool_list[i]])
                        if args.embedding_model == "minilm":
                            true_deleted_tool_emb = emb_model.encode(true_deleted_tool_desc)
                        elif args.embedding_model == "gemini":
                            true_deleted_tool_emb = emb_model(true_deleted_tool_desc)["embedding"]
                        else:
                            raise ValueError(f"Unsupported embedding_model: {args.embedding_model}")
                        scorers.append(float(
                            cosine_similarity([pred_embedding], [true_deleted_tool_emb])[0][0]))
                else:
                    scorers.append(0.0)
            else:
                scorers.append(0.0)
    return scorers


def _evaluate_detecting(pred_solvability, golden_solvability):
    pred = pred_solvability.lower().strip()
    golden = golden_solvability.lower().strip()
    if pred == "" or pred not in ("solvable", "unsolvable", "uncertain"):
        return {
            "metric": 0, "is_confident": False, "is_correct": False,
            "is_uncertain": False, "pred_category": "no_answer"
        }
    elif pred == "uncertain":
        return {
            "metric": 0, "is_confident": False, "is_correct": False,
            "is_uncertain": True, "pred_category": "uncertain"
        }
    else:
        is_correct = (pred == golden)
        return {
            "metric": 1 if is_correct else 0, "is_confident": True,
            "is_correct": is_correct, "is_uncertain": False,
            "pred_category": "confident_right" if is_correct else "confident_wrong"
        }


def _evaluate_planning_v2(
        pred_planning_tool_list,
        golden_planning_tool_list,
        provided_tool_list,
        calculate_type="hard",
        simulation_script=None,
        emb_model=None
    ):
    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call, extract_tool_name_from_definition
    empty_result = {
        "tool_score_soft": 0.0, "param_score": 0.0,
        "task_success_soft": 0, "planning_precision": 0.0,
        "param_details": []
    }
    if not golden_planning_tool_list:
        if not pred_planning_tool_list:
            return {
                "tool_score_soft": 1.0, "param_score": 1.0,
                "task_success_soft": 1, "planning_precision": 1.0,
                "param_details": []
            }
        else:
            return empty_result
    pred_tool_names = [extract_tool_name_from_call(t) for t in pred_planning_tool_list]
    golden_tool_names = [extract_tool_name_from_call(t) for t in golden_planning_tool_list]
    provided_tool_names = []
    for t in provided_tool_list:
        name = extract_tool_name_from_definition(t) if isinstance(t, str) else t
        if name:
            provided_tool_names.append(name)
    if "UnsolvableQuery" not in provided_tool_names:
        provided_tool_names.append("UnsolvableQuery")
    if not all(tool in provided_tool_names for tool in pred_tool_names):
        return empty_result
    if "UnsolvableQuery" in golden_tool_names and "UnsolvableQuery" not in pred_tool_names:
        return empty_result
    if "UnsolvableQuery" not in golden_tool_names and "UnsolvableQuery" in pred_tool_names:
        return empty_result
    tool_score_hard = calculate_progress_rate(pred_tool_names, golden_tool_names, "hard")
    tool_score_soft = calculate_progress_rate(pred_tool_names, golden_tool_names, "soft")
    if tool_score_hard == 0.0 and tool_score_soft == 0.0:
        return empty_result
    param_scores = []
    param_details = []
    simulation_script = simulation_script or {}
    for i in range(min(len(pred_planning_tool_list), len(golden_planning_tool_list))):
        pred_name = extract_tool_name_from_call(pred_planning_tool_list[i])
        golden_name = extract_tool_name_from_call(golden_planning_tool_list[i])
        if pred_name == golden_name:
            pred_params = extract_params_from_call(pred_planning_tool_list[i])
            golden_params = extract_params_from_call(golden_planning_tool_list[i])
            param_result = compare_params(
                pred_params, golden_params, i, simulation_script, golden_planning_tool_list,
                tool_name=golden_name, emb_model=emb_model
            )
            param_scores.append(param_result["score"])
            param_details.append(param_result)
    param_score = sum(param_scores) / len(param_scores) if param_scores else 0.0
    task_success_soft = _calculate_task_success(
        pred_planning_tool_list, golden_planning_tool_list,
        provided_tool_names, simulation_script or {}, True,
        emb_model=emb_model, mode="soft"
    )
    planning_precision = _calculate_process_precision(
        pred_planning_tool_list, golden_planning_tool_list,
        simulation_script=simulation_script, emb_model=emb_model
    )
    return {
        "tool_score_soft": tool_score_soft,
        "param_score": param_score,
        "task_success_soft": task_success_soft,
        "planning_precision": planning_precision,
        "param_details": param_details
    }


def _evaluate_planning_analysis_v2(
        pred_planning_tool_list,
        golden_planning_tool_list,
        provided_tool_list,
        calculate_type="hard",
        simulation_script=None,
        emb_model=None
    ):
    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call, extract_tool_name_from_definition
    zero_scores = {"tool_score_soft": 0.0, "param_score": 0.0,
                   "task_success_soft": 0, "planning_precision": 0.0}
    full_scores = {"tool_score_soft": 1.0, "param_score": 1.0,
                   "task_success_soft": 1, "planning_precision": 1.0}
    if not golden_planning_tool_list:
        if not pred_planning_tool_list:
            return full_scores, "correct"
        else:
            return zero_scores, "wrong_tools"
    pred_tool_names = [extract_tool_name_from_call(t) for t in pred_planning_tool_list]
    golden_tool_names = [extract_tool_name_from_call(t) for t in golden_planning_tool_list]
    provided_tool_names = []
    for t in provided_tool_list:
        name = extract_tool_name_from_definition(t) if isinstance(t, str) else t
        if name:
            provided_tool_names.append(name)
    if "UnsolvableQuery" not in provided_tool_names:
        provided_tool_names.append("UnsolvableQuery")
    if "finish" not in provided_tool_names:
        provided_tool_names.append("finish")
    invalid_tools = [t for t in pred_tool_names if t not in provided_tool_names and t != ""]
    if invalid_tools:
        result = dict(zero_scores)
        result["invalid_tools"] = invalid_tools
        return result, "non_existent_tools"
    if "UnsolvableQuery" in golden_tool_names and "UnsolvableQuery" not in pred_tool_names:
        return zero_scores, "solvability_hallu"
    if "UnsolvableQuery" not in golden_tool_names and "UnsolvableQuery" in pred_tool_names:
        return zero_scores, "false_unsolvable"
    tool_score_hard = calculate_progress_rate(pred_tool_names, golden_tool_names, "hard")
    tool_score_soft = calculate_progress_rate(pred_tool_names, golden_tool_names, "soft")
    condition = None
    if tool_score_hard == 0.0 and tool_score_soft == 0.0:
        pred_set = set(pred_tool_names)
        golden_set = set(golden_tool_names)
        if golden_set - pred_set:
            condition = "missing_tools"
        else:
            condition = "incorrect_tools"
        return zero_scores, condition
    param_scores = []
    simulation_script = simulation_script or {}
    has_missing_params = False
    has_wrong_param_value = False
    for i in range(min(len(pred_planning_tool_list), len(golden_planning_tool_list))):
        pred_name = extract_tool_name_from_call(pred_planning_tool_list[i])
        golden_name = extract_tool_name_from_call(golden_planning_tool_list[i])
        if pred_name == golden_name:
            pred_params = extract_params_from_call(pred_planning_tool_list[i])
            golden_params = extract_params_from_call(golden_planning_tool_list[i])
            param_result = compare_params(
                pred_params, golden_params, i, simulation_script, golden_planning_tool_list,
                tool_name=golden_name, emb_model=emb_model
            )
            param_scores.append(param_result["score"])
            if param_result["missing"]:
                has_missing_params = True
            if param_result["wrong_value"]:
                has_wrong_param_value = True
    param_score = sum(param_scores) / len(param_scores) if param_scores else 0.0
    task_success_soft = _calculate_task_success(
        pred_planning_tool_list, golden_planning_tool_list,
        provided_tool_names, simulation_script, True,
        emb_model=emb_model, mode="soft"
    )
    if task_success_soft == 1:
        if len(pred_tool_names) > len(golden_tool_names):
            condition = "redundant_steps"
        else:
            condition = "correct"
    elif tool_score_soft == 1.0 and param_score < 1.0:
        if has_missing_params:
            condition = "missing_params"
        elif has_wrong_param_value:
            condition = "wrong_param_value"
        else:
            condition = "tool_correct_param_error"
    elif tool_score_soft < 1.0:
        pred_set = set(pred_tool_names)
        golden_set = set(golden_tool_names)
        if golden_set - pred_set:
            condition = "missing_tools"
        else:
            condition = "incorrect_tools"
    else:
        condition = "wrong_param_value"
    planning_precision = _calculate_process_precision(
        pred_planning_tool_list, golden_planning_tool_list,
        simulation_script=simulation_script, emb_model=emb_model
    )
    return {
        "tool_score_soft": tool_score_soft, "param_score": param_score,
        "task_success_soft": task_success_soft,
        "planning_precision": planning_precision,
    }, condition


def _evaluate_planning(
        pred_planning_tool_list,
        golden_planning_tool_list,
        provided_tool_list,
        calculate_type = "hard"
    ):
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        return 0.0
    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        return 0.0
    progress_rate = calculate_progress_rate(pred_planning_tool_list, golden_planning_tool_list, calculate_type)
    return progress_rate


def _evaluate_planning_analysis(
        pred_planning_tool_list,
        golden_planning_tool_list,
        provided_tool_list,
        calculate_type = "hard"
    ):
    condition = None
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        return 0.0, "non_existent_tools"
    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        return 0.0, "solvability_hallu"
    progress_rate = calculate_progress_rate(pred_planning_tool_list, golden_planning_tool_list, calculate_type)
    if condition is None:
        if not all(tool in golden_planning_tool_list for tool in pred_planning_tool_list):
            condition = "wrong_tools"
        else:
            if progress_rate == 1.0:
                condition = "correct"
            else:
                if golden_planning_tool_list.index("UnsolvableQuery") != pred_planning_tool_list.index("UnsolvableQuery"):
                    condition = "wrong_unsolvable_index"
                else:
                    condition = "wrong_reasoning"
    return progress_rate, condition


def _evaluate_diagnosing(
        pred_planning_tuple,
        golden_planning_tool_list,
        origin_provided_subgoal_dict, solvable_planning_tool_list,
        provided_tool_list,
        emb_model, tools_embedding, task, args, calculate_type = "hard"
    ):
    pred_subgoal_texts, pred_planning_tool_list = pred_planning_tuple
    progress_rate = None
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        progress_rate = 0.0
    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        unsolvable_count = golden_planning_tool_list.count("UnsolvableQuery")
        return 0.0, [0.0] * unsolvable_count
    if progress_rate is None:
        progress_rate = calculate_progress_rate(
            pred_planning_tool_list, golden_planning_tool_list, calculate_type)
    if "UnsolvableQuery" in golden_planning_tool_list:
        scorers = calculate_scorers(pred_subgoal_texts, pred_planning_tool_list,
                                    golden_planning_tool_list,
                                    origin_provided_subgoal_dict, solvable_planning_tool_list,
                                    emb_model, tools_embedding, task, args)
    else:
        scorers = ""
    return progress_rate, scorers


def _evaluate_diagnosing_analysis(
        pred_planning_tuple,
        golden_planning_tool_list,
        origin_provided_subgoal_dict, solvable_planning_tool_list,
        provided_tool_list,
        emb_model, tools_embedding, task, args, calculate_type = "hard"
    ):
    pred_subgoal_texts, pred_planning_tool_list = pred_planning_tuple
    progress_rate = None
    condition = None
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        progress_rate = 0.0
        condition = "non_existent_tools"
    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        unsolvable_count = golden_planning_tool_list.count("UnsolvableQuery")
        return 0.0, [0.0] * unsolvable_count, "solvability_hallu"
    if progress_rate is None:
        progress_rate = calculate_progress_rate(
            pred_planning_tool_list, golden_planning_tool_list, calculate_type)
    if "UnsolvableQuery" in golden_planning_tool_list:
        scorers = calculate_scorers(pred_subgoal_texts, pred_planning_tool_list,
                                    golden_planning_tool_list,
                                    origin_provided_subgoal_dict, solvable_planning_tool_list,
                                    emb_model, tools_embedding, task, args)
    else:
        scorers = ""
    if condition is None:
        if not all(tool in golden_planning_tool_list for tool in pred_planning_tool_list):
            condition = "wrong_tools"
        else:
            if progress_rate == 1.0:
                condition = "correct"
            else:
                if golden_planning_tool_list.index("UnsolvableQuery") != pred_planning_tool_list.index("UnsolvableQuery"):
                    condition = "wrong_unsolvable_index"
                else:
                    condition = "wrong_reasoning"
    return progress_rate, scorers, condition


def _evaluate_react(
        react_response,
        golden_tools,
        provided_tools,
        simulation_script,
        task_description,
        calculate_type="hard",
        emb_model=None
    ):
    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call, extract_tool_name_from_definition
    if not react_response or not isinstance(react_response, dict):
        return {
            "task_success_soft": 0,
            "tool_call_exact_soft": 0.0,
            "parameter_accuracy": 0.0,
            "step_efficiency_soft": 0.0,
            "condition": "no_response",
            "details": {}
        }
    thoughts = react_response.get("thoughts", [])
    actions = react_response.get("actions", [])
    observations = react_response.get("observations", [])
    total_steps = react_response.get("total_steps", 0)
    finished = react_response.get("finished", False)
    final_answer = react_response.get("final_answer", "")
    pred_tool_names = [extract_tool_name_from_call(a) for a in actions]
    pred_tool_calls_full = actions
    pred_tool_names_no_finish = [t for t in pred_tool_names if t.lower() != "finish"]
    pred_tool_calls_no_finish = [a for a in actions if extract_tool_name_from_call(a).lower() != "finish"]
    golden_tool_names = [extract_tool_name_from_call(t) for t in golden_tools]
    golden_tool_names_no_finish = [t for t in golden_tool_names if t.lower() != "finish"]
    golden_tool_calls_no_finish = [t for t in golden_tools if extract_tool_name_from_call(t).lower() != "finish"]
    provided_tool_names = []
    for t in provided_tools:
        name = extract_tool_name_from_definition(t) if isinstance(t, str) else t
        if name:
            provided_tool_names.append(name)
    if "UnsolvableQuery" not in provided_tool_names:
        provided_tool_names.append("UnsolvableQuery")
    if "finish" not in provided_tool_names:
        provided_tool_names.append("finish")
    task_success_hard = _calculate_task_success(
        pred_tool_calls_no_finish,
        golden_tool_calls_no_finish,
        provided_tool_names,
        simulation_script,
        finished,
        emb_model=emb_model,
        mode="hard"
    )
    task_success_soft = _calculate_task_success(
        pred_tool_calls_no_finish,
        golden_tool_calls_no_finish,
        provided_tool_names,
        simulation_script,
        finished,
        emb_model=emb_model,
        mode="soft"
    )
    tool_call_exact_soft = calculate_progress_rate(
        pred_tool_names_no_finish,
        golden_tool_names_no_finish,
        "soft"
    )
    parameter_accuracy, has_missing_params, has_wrong_param_value = _calculate_parameter_accuracy(
        pred_tool_calls_no_finish,
        golden_tool_calls_no_finish,
        simulation_script,
        emb_model=emb_model
    )
    golden_steps = len(golden_tool_calls_no_finish)
    actual_steps = len(pred_tool_calls_no_finish)
    if actual_steps > 0 and golden_steps > 0:
        step_efficiency_soft = min(1.0, golden_steps / actual_steps)
    else:
        step_efficiency_soft = 0.0
    agent_capability_score = tool_call_exact_soft * parameter_accuracy * step_efficiency_soft
    effective_step_utilization = _calculate_process_precision(
        pred_tool_calls_no_finish, golden_tool_calls_no_finish,
        simulation_script=simulation_script, emb_model=emb_model
    )
    condition = _determine_react_condition(
        pred_tool_names_no_finish,
        golden_tool_names_no_finish,
        provided_tool_names,
        task_success_hard,
        tool_call_exact_soft,
        parameter_accuracy,
        finished,
        has_missing_params,
        has_wrong_param_value
    )
    return {
        "task_success_soft": task_success_soft,
        "tool_call_exact_soft": tool_call_exact_soft,
        "parameter_accuracy": parameter_accuracy,
        "step_efficiency_soft": step_efficiency_soft,
        "agent_capability_score": agent_capability_score,
        "effective_step_utilization": effective_step_utilization,
        "condition": condition,
        "details": {
            "total_steps": total_steps,
            "golden_steps": golden_steps,
            "finished": finished,
            "pred_tools": pred_tool_names_no_finish,
            "golden_tools": golden_tool_names_no_finish
        }
    }


def _calculate_task_success(pred_tool_calls, golden_tool_calls, provided_tools, simulation_script, finished, emb_model=None, mode="hard"):
    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call
    if not golden_tool_calls:
        return 1
    pred_names = [extract_tool_name_from_call(c) for c in pred_tool_calls]
    golden_names = [extract_tool_name_from_call(c) for c in golden_tool_calls]
    if "UnsolvableQuery" not in golden_names and "UnsolvableQuery" in pred_names:
        return 0
    if "UnsolvableQuery" in golden_names and "UnsolvableQuery" not in pred_names:
        return 0
    if mode == "hard":
        if len(pred_tool_calls) < len(golden_tool_calls):
            return 0
        matched_pairs = []
        for i in range(len(golden_tool_calls)):
            if golden_names[i] != pred_names[i]:
                return 0
            matched_pairs.append((i, i))
    elif mode == "soft":
        matched_pairs = []
        p_idx = 0
        for g_idx in range(len(golden_tool_calls)):
            found = False
            while p_idx < len(pred_tool_calls):
                if golden_names[g_idx] == pred_names[p_idx]:
                    matched_pairs.append((g_idx, p_idx))
                    p_idx += 1
                    found = True
                    break
                p_idx += 1
            if not found:
                return 0
    else:
        raise ValueError(f"mode must be 'hard' or 'soft', got '{mode}'")
    for g_idx, p_idx in matched_pairs:
        golden_call = golden_tool_calls[g_idx]
        pred_call = pred_tool_calls[p_idx]
        golden_name = golden_names[g_idx]
        golden_params = extract_params_from_call(golden_call)
        pred_params = extract_params_from_call(pred_call)
        param_result = compare_params(
            pred_params, golden_params, g_idx, simulation_script, golden_tool_calls,
            tool_name=golden_name, emb_model=emb_model
        )
        if param_result["score"] < 1.0:
            return 0
    return 1


def _calculate_parameter_accuracy(pred_tool_calls, golden_tool_calls, simulation_script, emb_model=None):
    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call
    if not golden_tool_calls:
        return 1.0, False, False
    total_score = 0.0
    matched_count = 0
    has_missing = False
    has_wrong_value = False
    for i, golden_call in enumerate(golden_tool_calls):
        golden_name = extract_tool_name_from_call(golden_call)
        if i < len(pred_tool_calls):
            pred_name = extract_tool_name_from_call(pred_tool_calls[i])
            if golden_name == pred_name:
                matched_count += 1
                golden_params = extract_params_from_call(golden_call)
                pred_params = extract_params_from_call(pred_tool_calls[i])
                param_result = compare_params(pred_params, golden_params, i, simulation_script, golden_tool_calls,
                                              tool_name=golden_name, emb_model=emb_model)
                total_score += param_result["score"]
                if param_result["missing"]:
                    has_missing = True
                if param_result["wrong_value"]:
                    has_wrong_value = True
    if matched_count == 0:
        return 0.0, False, False
    return total_score / matched_count, has_missing, has_wrong_value


def _calculate_task_grounding(thoughts, task_description, emb_model):
    if not thoughts or not task_description or emb_model is None:
        return 0.0
    try:
        task_emb = emb_model.encode(task_description)
        similarities = []
        for thought in thoughts:
            if thought and thought.strip():
                thought_emb = emb_model.encode(thought)
                sim = cosine_similarity([thought_emb], [task_emb])[0][0]
                similarities.append(float(sim))
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Error calculating task grounding: {type(e).__name__}")
        return 0.0


def _calculate_reasoning_quality(thoughts, observations, task_description, emb_model):
    if not thoughts or not task_description or emb_model is None:
        return 0.0
    try:
        task_emb = emb_model.encode(task_description)
        quality_scores = []
        for i, thought in enumerate(thoughts):
            if not thought or not thought.strip():
                continue
            thought_emb = emb_model.encode(thought)
            sim_task = cosine_similarity([thought_emb], [task_emb])[0][0]
            if i == 0:
                quality_scores.append(float(sim_task))
            else:
                prev_obs = observations[i-1] if i-1 < len(observations) else ""
                if prev_obs and prev_obs.strip():
                    obs_emb = emb_model.encode(prev_obs)
                    sim_obs = cosine_similarity([thought_emb], [obs_emb])[0][0]
                    score = float(sim_task) - 0.3 * float(sim_obs)
                else:
                    score = float(sim_task)
                quality_scores.append(max(0.0, score))
        if quality_scores:
            return sum(quality_scores) / len(quality_scores)
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Error calculating reasoning quality: {type(e).__name__}")
        return 0.0


def _determine_react_condition(pred_tools, golden_tools, provided_tools,
                               task_success, tool_call_exact, param_accuracy, finished,
                               has_missing_params=False, has_wrong_param_value=False):
    invalid_tools = [t for t in pred_tools if t not in provided_tools and t != ""]
    if invalid_tools:
        return "non_existent_tools"
    if "UnsolvableQuery" in golden_tools and "UnsolvableQuery" not in pred_tools:
        return "solvability_hallu"
    if "UnsolvableQuery" not in golden_tools and "UnsolvableQuery" in pred_tools:
        return "false_unsolvable"
    if task_success == 1:
        if len(pred_tools) > len(golden_tools):
            return "redundant_steps"
        else:
            return "correct"
    golden_set = set(golden_tools)
    pred_set = set(pred_tools)
    missing = golden_set - pred_set
    if missing:
        return "missing_tools"
    if tool_call_exact < 1.0 and param_accuracy >= 1.0:
        return "incorrect_tools"
    if param_accuracy < 1.0:
        if has_missing_params:
            return "missing_params"
        elif has_wrong_param_value:
            return "wrong_param_value"
        else:
            return "wrong_params"
    if tool_call_exact < 1.0 or len(pred_tools) > len(golden_tools):
        return "redundant_steps"
    return "unknown_error"


def _evaluate_reflexion(
        reflexion_response,
        golden_tools,
        provided_tools,
        simulation_script,
        task_description,
        calculate_type="hard",
        emb_model=None
    ):
    from utils.extract_func import extract_tool_name_from_call
    if not reflexion_response or not isinstance(reflexion_response, dict):
        return {
            "task_success_soft": 0,
            "tool_call_exact_soft": 0.0,
            "parameter_accuracy": 0.0,
            "step_efficiency_soft": 0.0,
            "agent_capability_score": 0.0,
            "effective_step_utilization": 0.0,
            "condition": "no_response",
            "details": {},
            "self_correction_rate": 0,
        }
    base_metrics = _evaluate_react(
        react_response=reflexion_response,
        golden_tools=golden_tools,
        provided_tools=provided_tools,
        simulation_script=simulation_script,
        task_description=task_description,
        calculate_type=calculate_type,
        emb_model=emb_model
    )
    trials = reflexion_response.get("trials", [])
    reflections = reflexion_response.get("reflections", [])
    num_trials = reflexion_response.get("num_trials", len(trials))
    improved = reflexion_response.get("improved", False)
    self_correction_rate = 1 if improved else 0
    base_metrics["self_correction_rate"] = self_correction_rate
    base_metrics["details"]["num_trials"] = num_trials
    base_metrics["details"]["num_reflections"] = len(reflections)
    base_metrics["details"]["improved"] = improved
    return base_metrics


def _evaluate_plan_execute(
        pte_response,
        golden_tools,
        provided_tools,
        simulation_script,
        task_description,
        calculate_type="hard",
        emb_model=None
    ):
    from utils.extract_func import extract_tool_name_from_call
    if not pte_response or not isinstance(pte_response, dict):
        return {
            "task_success_soft": 0,
            "tool_call_exact_soft": 0.0,
            "parameter_accuracy": 0.0,
            "step_efficiency_soft": 0.0,
            "agent_capability_score": 0.0,
            "effective_step_utilization": 0.0,
            "condition": "no_response",
            "details": {},
            "plan_quality": 0.0,
            "plan_faithfulness": 0.0,
            "execution_accuracy": 0.0,
            "param_grounding": 0.0,
        }
    base_metrics = _evaluate_react(
        react_response=pte_response,
        golden_tools=golden_tools,
        provided_tools=provided_tools,
        simulation_script=simulation_script,
        task_description=task_description,
        calculate_type=calculate_type,
        emb_model=emb_model
    )
    planned_steps = pte_response.get("plan", [])
    actual_actions = pte_response.get("actions", [])
    golden_names = [extract_tool_name_from_call(t) for t in golden_tools]
    golden_no_finish = [t for t in golden_names if t.lower() != "finish"]
    plan_names = [extract_tool_name_from_call(s) for s in planned_steps]
    plan_no_finish = [t for t in plan_names if t.lower() != "finish"]
    exec_names = [extract_tool_name_from_call(a) for a in actual_actions]
    exec_no_finish = [t for t in exec_names if t.lower() != "finish"]
    plan_quality = calculate_progress_rate(plan_no_finish, golden_no_finish, calculate_type)
    execution_accuracy = calculate_progress_rate(exec_no_finish, golden_no_finish, calculate_type)
    plan_faithfulness = _calculate_plan_faithfulness(plan_no_finish, exec_no_finish)
    param_grounding = _calculate_param_grounding(
        planned_steps, actual_actions, simulation_script
    )
    base_metrics["plan_quality"] = plan_quality
    base_metrics["plan_faithfulness"] = plan_faithfulness
    base_metrics["execution_accuracy"] = execution_accuracy
    base_metrics["param_grounding"] = param_grounding
    base_metrics["details"]["num_planned_steps"] = len(planned_steps)
    base_metrics["details"]["num_executed_steps"] = len(actual_actions)
    return base_metrics


def _calculate_plan_faithfulness(plan_tools, exec_tools):
    if not plan_tools:
        return 1.0 if not exec_tools else 0.0
    matched = 0
    exec_idx = 0
    for plan_tool in plan_tools:
        if exec_idx < len(exec_tools) and exec_tools[exec_idx] == plan_tool:
            matched += 1
            exec_idx += 1
        else:
            exec_idx += 1
    return matched / len(plan_tools)


def _calculate_param_grounding(planned_steps, actual_actions, simulation_script):
    from utils.extract_func import extract_tool_name_from_call, extract_params_from_call
    placeholder_count = 0
    correct_count = 0
    for step_i, planned_call in enumerate(planned_steps):
        if "Step_" not in planned_call or "_Output" not in planned_call:
            continue
        placeholders = re.findall(r'Step_(\d+)_Output', planned_call)
        if not placeholders:
            continue
        if step_i >= len(actual_actions):
            placeholder_count += len(placeholders)
            continue
        exec_params = extract_params_from_call(actual_actions[step_i])
        for ref_step_str in placeholders:
            placeholder_count += 1
            ref_step = int(ref_step_str) - 1
            if ref_step < len(actual_actions):
                ref_tool = extract_tool_name_from_call(actual_actions[ref_step])
                if ref_tool in simulation_script:
                    expected_value = str(simulation_script[ref_tool].get("return_value", ""))
                    if any(expected_value in str(v) for v in exec_params.values()):
                        correct_count += 1
    if placeholder_count == 0:
        return 1.0
    return correct_count / placeholder_count
