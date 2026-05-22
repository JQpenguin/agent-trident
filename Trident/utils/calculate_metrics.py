import numpy as np

from itertools import chain

from tabulate import tabulate

from collections import defaultdict





























all_subtask_types = [

    "single_step", "multi_step_no_rep", "sequential_chain",

    "conditional_planning", "os", "web", "best"

]



subtask_groups = {

    "Solvable": all_subtask_types,

    "Unsolvable": all_subtask_types,

}



subtask_groups["overall"] = all_subtask_types







domain_groups = {

    "Enterprise": ["financial_services", "human_resources", "customer_service", "supply_chain"],

    "Technology": ["software_development", "cybersecurity", "cloud_operations", "data_engineering"],

    "Healthcare": ["clinical_healthcare", "medical_devices"],

    "Industrial": ["smart_manufacturing", "energy_management"],

    "Consumer": ["e_commerce", "smart_home", "transportation", "food_hospitality"],

    "Education": ["edtech", "scientific_research"],

    "Government": ["government_services", "public_safety"]

}





all_domains = list(chain.from_iterable(domain_groups.values()))





domain_display_names = {

    "financial_services": "Financial Services",

    "human_resources": "Human Resources",

    "customer_service": "Customer Service",

    "supply_chain": "Supply Chain",

    "software_development": "Software Dev",

    "cybersecurity": "Cybersecurity",

    "cloud_operations": "Cloud Ops",

    "data_engineering": "Data Engineering",

    "clinical_healthcare": "Clinical Healthcare",

    "medical_devices": "Medical Devices",

    "smart_manufacturing": "Smart Manufacturing",

    "energy_management": "Energy Management",

    "e_commerce": "E-Commerce",

    "smart_home": "Smart Home",

    "transportation": "Transportation",

    "food_hospitality": "Food & Hospitality",

    "edtech": "EdTech",

    "scientific_research": "Scientific Research",

    "government_services": "Government Services",

    "public_safety": "Public Safety"

}



def flatten(data):

    flat_list = []



    def _flatten(item):

        if isinstance(item, (list, tuple, np.ndarray)):

            for sub_item in item:

                _flatten(sub_item)

        else:

            flat_list.append(item)



    _flatten(data)

    return flat_list









def _aggregate_l1_confidence_metrics(samples):

    total = len(samples)

    if total == 0:

        return {

            "exact_match": 0.0,

            "hallucination_rate": 0.0,

            "risk_adjusted_score": 0.0,

            "counts": {"confident_right": 0, "confident_wrong": 0,

                       "uncertain": 0, "no_answer": 0, "total": 0}

        }



    confident_right = sum(1 for s in samples if s.get("pred_category") == "confident_right")

    confident_wrong = sum(1 for s in samples if s.get("pred_category") == "confident_wrong")

    uncertain_count = sum(1 for s in samples if s.get("pred_category") == "uncertain")

    no_answer_count = sum(1 for s in samples if s.get("pred_category") == "no_answer")



    exact_match = np.mean([s.get("metric", 0) for s in samples])





    confident_total = confident_right + confident_wrong

    hallucination_rate = confident_wrong / confident_total if confident_total > 0 else 0.0





    net_correctness = (confident_right - confident_wrong) / total

    confidence_rate = (confident_right + confident_wrong) / total

    ras_score = net_correctness * confidence_rate



    return {

        "exact_match": exact_match,

        "hallucination_rate": hallucination_rate,

        "risk_adjusted_score": ras_score,

        "counts": {

            "confident_right": confident_right,

            "confident_wrong": confident_wrong,

            "uncertain": uncertain_count,

            "no_answer": no_answer_count,

            "total": total,

        }

    }





def calculate_l1_solvability_hallucination(metrics, tasks):

    fp = 0

    fn = 0

    tp = 0

    tn = 0

    uncertain_solvable = 0

    uncertain_unsolvable = 0

    no_answer = 0



    for task in tasks:

        if task not in metrics or "level_1" not in metrics[task]:

            continue

        for m in metrics[task]["level_1"]:

            golden = m.get("golden", "").lower()

            pred = m.get("pred", "").lower()

            cat = m.get("pred_category", "")



            if cat == "no_answer":

                no_answer += 1

            elif pred == "uncertain":

                if golden == "solvable":

                    uncertain_solvable += 1

                else:

                    uncertain_unsolvable += 1

            elif golden == "solvable" and pred == "solvable":

                tp += 1

            elif golden == "solvable" and pred == "unsolvable":

                fn += 1

            elif golden == "unsolvable" and pred == "solvable":

                fp += 1

            elif golden == "unsolvable" and pred == "unsolvable":

                tn += 1





    denom = fp + tn + uncertain_unsolvable

    solvability_hallu_rate = fp / denom if denom > 0 else 0.0



    return {

        "tp": tp, "fp": fp, "tn": tn, "fn": fn,

        "uncertain_solvable": uncertain_solvable,

        "uncertain_unsolvable": uncertain_unsolvable,

        "no_answer": no_answer,

        "solvability_hallucination_rate": solvability_hallu_rate,

    }



def calculate_l2_planning_hallucination(metrics, tasks):

    total_samples = 0

    fabrication_count = 0

    solvability_hallu_count = 0

    unsolvable_total = 0

    false_unsolvable_count = 0

    solvable_total = 0

    redundant_step_count = 0

    success_total = 0



    for task in tasks:

        if task not in metrics or "level_2" not in metrics[task]:

            continue

        for m in metrics[task]["level_2"]:

            total_samples += 1

            condition = m.get("condition", "")

            golden = m.get("golden", "").lower()



            if condition == "non_existent_tools":

                fabrication_count += 1



            if golden == "unsolvable":

                unsolvable_total += 1

                if condition == "solvability_hallu":

                    solvability_hallu_count += 1



            if golden == "solvable":

                solvable_total += 1

                if condition == "false_unsolvable":

                    false_unsolvable_count += 1



            if m.get("task_success_soft", 0) == 1:

                success_total += 1

                pred_count = m.get("pred_tool_count", 0)

                golden_count = m.get("golden_tool_count", 0)

                if pred_count > golden_count:

                    redundant_step_count += 1



    tfr = fabrication_count / total_samples if total_samples > 0 else 0.0

    sbr = solvability_hallu_count / unsolvable_total if unsolvable_total > 0 else 0.0

    fur = false_unsolvable_count / solvable_total if solvable_total > 0 else 0.0

    rsr = redundant_step_count / success_total if success_total > 0 else 0.0



    return {

        "total_samples": total_samples,

        "tool_fabrication_rate": round(tfr, 4),

        "solvability_blindness_rate": round(sbr, 4),

        "false_unsolvable_rate": round(fur, 4),

        "redundant_step_rate": round(rsr, 4),

        "counts": {

            "fabrication": fabrication_count,

            "solvability_hallu": solvability_hallu_count,

            "unsolvable_total": unsolvable_total,

            "false_unsolvable": false_unsolvable_count,

            "solvable_total": solvable_total,

            "redundant_steps": redundant_step_count,

            "success_total": success_total,

        }

    }



def calculate_l3_execution_hallucination(metrics, tasks):

    total_samples = 0

    fabrication_count = 0

    missing_tool_count = 0

    not_finished_count = 0

    correct_count = 0

    correct_with_extra_count = 0



    for task in tasks:

        if task not in metrics or "level_3" not in metrics[task]:

            continue

        for m in metrics[task]["level_3"]:

            total_samples += 1

            condition = m.get("condition", "")



            if condition == "non_existent_tools":

                fabrication_count += 1

            elif condition == "missing_tools":

                missing_tool_count += 1

            elif condition == "not_finished":

                not_finished_count += 1

            elif condition == "correct":

                correct_count += 1

            elif condition == "correct_with_extra_steps":

                correct_with_extra_count += 1



    tfr = fabrication_count / total_samples if total_samples > 0 else 0.0

    mtr = missing_tool_count / total_samples if total_samples > 0 else 0.0

    chr_val = not_finished_count / total_samples if total_samples > 0 else 0.0

    success_total = correct_count + correct_with_extra_count

    esr = correct_with_extra_count / success_total if success_total > 0 else 0.0



    return {

        "total_samples": total_samples,

        "tool_fabrication_rate": round(tfr, 4),

        "missing_tool_rate": round(mtr, 4),

        "completion_hallucination_rate": round(chr_val, 4),

        "extra_step_rate": round(esr, 4),

        "counts": {

            "fabrication": fabrication_count,

            "missing_tools": missing_tool_count,

            "not_finished": not_finished_count,

            "correct": correct_count,

            "correct_with_extra_steps": correct_with_extra_count,

            "success_total": success_total,

        }

    }



def aggregate_token_usage(metrics, tasks, levels=None, solvability=None):

    filter_fn = (lambda m: m.get("golden", "").lower() == solvability) if solvability else (lambda m: True)

    if levels is None:

        available = set()

        for t in tasks:

            if t in metrics:

                available.update(metrics[t].keys())

        levels = [l for l in ["level_1", "level_2", "level_3"] if l in available]



    total_prompt = 0

    total_completion = 0

    count = 0

    for task in tasks:

        if task not in metrics:

            continue

        for level in levels:

            if level not in metrics[task]:

                continue

            for m in metrics[task][level]:

                if not filter_fn(m):

                    continue

                if "prompt_tokens" in m:

                    total_prompt += m["prompt_tokens"]

                    total_completion += m.get("completion_tokens", 0)

                    count += 1



    if count == 0:

        return {}

    return {

        "total_prompt_tokens": total_prompt,

        "total_completion_tokens": total_completion,

        "total_tokens": total_prompt + total_completion,

        "avg_prompt_tokens": round(total_prompt / count, 1),

        "avg_completion_tokens": round(total_completion / count, 1),

        "sample_count": count

    }



def calculate_metrics(metrics, task, level):

    if level == "level_1":

        samples = metrics[task][level]

        return _aggregate_l1_confidence_metrics(samples)

    elif level == "level_2":

        tool_scores_soft = [m["tool_score_soft"] for m in metrics[task][level]]

        param_scores = [m["param_score"] for m in metrics[task][level]]

        task_success_soft = [m["task_success_soft"] for m in metrics[task][level]]

        planning_precision = [m.get("planning_precision", 0.0) for m in metrics[task][level]]

        return {

            "tool_score_soft": np.mean(tool_scores_soft),

            "param_score": np.mean(param_scores),

            "task_success_soft": np.mean(task_success_soft),

            "planning_precision": np.mean(planning_precision),

        }

    elif level == "level_3":



        samples = metrics[task][level]

        task_success_soft = np.mean([m["task_success_soft"] for m in samples])

        tool_call_exact_soft = np.mean([m["tool_call_exact_soft"] for m in samples])

        param_accuracy = np.mean([m["parameter_accuracy"] for m in samples])



        success_effs = [m["step_efficiency_soft"] for m in samples if m.get("task_success_soft", 0) == 1]

        step_efficiency_soft = np.mean(success_effs) if success_effs else 0.0



        acs = np.mean([m.get("agent_capability_score", 0.0) for m in samples])

        result = {

            "task_success_soft": task_success_soft,

            "tool_call_exact_soft": tool_call_exact_soft,

            "parameter_accuracy": param_accuracy,

            "step_efficiency_soft": step_efficiency_soft,

            "agent_capability_score": acs,

            "effective_step_utilization": np.mean([m.get("effective_step_utilization", 0.0) for m in samples]),

        }



        if any("self_correction_rate" in m for m in samples):

            result["self_correction_rate"] = np.mean([m.get("self_correction_rate", 0) for m in samples])



        if any("plan_quality" in m for m in samples):

            result["plan_quality"] = np.mean([m.get("plan_quality", 0.0) for m in samples])

            result["plan_faithfulness"] = np.mean([m.get("plan_faithfulness", 0.0) for m in samples])

            result["execution_accuracy"] = np.mean([m.get("execution_accuracy", 0.0) for m in samples])

            result["param_grounding"] = np.mean([m.get("param_grounding", 0.0) for m in samples])

        return result

    return None



def calculate_group_metrics(metrics, tasks, level, solvability=None):

    filter_fn = (lambda m: m.get("golden", "").lower() == solvability) if solvability else (lambda m: True)

    if level == "level_1":

        samples = [m for task in tasks for m in metrics[task][level] if filter_fn(m)]

        return _aggregate_l1_confidence_metrics(samples)

    elif level == "level_2":

        tool_scores_soft = [m["tool_score_soft"] for task in tasks for m in metrics[task][level] if filter_fn(m)]

        param_scores = [m["param_score"] for task in tasks for m in metrics[task][level] if filter_fn(m)]

        task_success_soft = [m["task_success_soft"] for task in tasks for m in metrics[task][level] if filter_fn(m)]

        planning_precision = [m.get("planning_precision", 0.0) for task in tasks for m in metrics[task][level] if filter_fn(m)]

        return {

            "tool_score_soft": tool_scores_soft, "param_score": param_scores,

            "task_success_soft": task_success_soft,

            "planning_precision": planning_precision,

        }

    elif level == "level_3":

        all_samples = [m for task in tasks for m in metrics[task][level] if filter_fn(m)]

        result = {

            "task_success_soft": [m["task_success_soft"] for m in all_samples],

            "tool_call_exact_soft": [m["tool_call_exact_soft"] for m in all_samples],

            "parameter_accuracy": [m["parameter_accuracy"] for m in all_samples],



            "step_efficiency_soft": [m["step_efficiency_soft"] for m in all_samples if m.get("task_success_soft", 0) == 1],



            "agent_capability_score": [m.get("agent_capability_score", 0.0) for m in all_samples],



            "effective_step_utilization": [m.get("effective_step_utilization", 0.0) for m in all_samples],

        }



        if any("self_correction_rate" in m for m in all_samples):

            result["self_correction_rate"] = [m.get("self_correction_rate", 0) for m in all_samples]



        if any("plan_quality" in m for m in all_samples):

            result["plan_quality"] = [m.get("plan_quality", 0.0) for m in all_samples]

            result["plan_faithfulness"] = [m.get("plan_faithfulness", 0.0) for m in all_samples]

            result["execution_accuracy"] = [m.get("execution_accuracy", 0.0) for m in all_samples]

            result["param_grounding"] = [m.get("param_grounding", 0.0) for m in all_samples]

        return result

    return []



def calculate_group_metrics_embedding(metrics, tasks, level):

    if level == "level_1":

        samples = [m for task in tasks for m in metrics[task][level]]

        return _aggregate_l1_confidence_metrics(samples)

    elif level == "level_2":

        tool_scores_soft = [m["tool_score_soft"] for task in tasks for m in metrics[task][level]]

        param_scores = [m["param_score"] for task in tasks for m in metrics[task][level]]

        task_success_soft = [m["task_success_soft"] for task in tasks for m in metrics[task][level]]

        planning_precision = [m.get("planning_precision", 0.0) for task in tasks for m in metrics[task][level]]

        return {

            "tool_score_soft": np.mean(tool_scores_soft),

            "param_score": np.mean(param_scores),

            "task_success_soft": np.mean(task_success_soft),

            "planning_precision": np.mean(planning_precision),

        }

    elif level == "level_3":



        all_samples = [m for task in tasks for m in metrics[task][level]]

        success_effs = [m["step_efficiency_soft"] for m in all_samples if m.get("task_success_soft", 0) == 1]

        result = {

            "task_success_soft": np.mean([m["task_success_soft"] for m in all_samples]),

            "tool_call_exact_soft": np.mean([m["tool_call_exact_soft"] for m in all_samples]),

            "parameter_accuracy": np.mean([m["parameter_accuracy"] for m in all_samples]),

            "step_efficiency_soft": np.mean(success_effs) if success_effs else 0.0,

            "agent_capability_score": np.mean([m.get("agent_capability_score", 0.0) for m in all_samples]),

            "effective_step_utilization": np.mean([m.get("effective_step_utilization", 0.0) for m in all_samples]),

        }



        if any("self_correction_rate" in m for m in all_samples):

            result["self_correction_rate"] = np.mean([m.get("self_correction_rate", 0) for m in all_samples])



        if any("plan_quality" in m for m in all_samples):

            result["plan_quality"] = np.mean([m.get("plan_quality", 0.0) for m in all_samples])

            result["plan_faithfulness"] = np.mean([m.get("plan_faithfulness", 0.0) for m in all_samples])

            result["execution_accuracy"] = np.mean([m.get("execution_accuracy", 0.0) for m in all_samples])

            result["param_grounding"] = np.mean([m.get("param_grounding", 0.0) for m in all_samples])

        return result

    return None



def calculate_group_metrics_vs(metrics, tasks, level, solvability="solvable"):

    filter_fn = lambda m: m.get("golden", "").lower() == solvability

    if level == "level_1":

        samples = [m for task in tasks for m in metrics[task][level] if filter_fn(m)]

        return _aggregate_l1_confidence_metrics(samples)

    elif level == "level_2":

        tool_scores_soft = [m["tool_score_soft"] for task in tasks for m in metrics[task][level] if filter_fn(m)]

        param_scores = [m["param_score"] for task in tasks for m in metrics[task][level] if filter_fn(m)]

        task_success_soft = [m["task_success_soft"] for task in tasks for m in metrics[task][level] if filter_fn(m)]

        planning_precision = [m.get("planning_precision", 0.0) for task in tasks for m in metrics[task][level] if filter_fn(m)]

        return {

            "tool_score_soft": tool_scores_soft, "param_score": param_scores,

            "task_success_soft": task_success_soft,

            "planning_precision": planning_precision,

        }

    elif level == "level_3":

        all_samples = [m for task in tasks for m in metrics[task][level] if filter_fn(m)]

        result = {

            "task_success_soft": [m["task_success_soft"] for m in all_samples],

            "tool_call_exact_soft": [m["tool_call_exact_soft"] for m in all_samples],

            "parameter_accuracy": [m["parameter_accuracy"] for m in all_samples],



            "step_efficiency_soft": [m["step_efficiency_soft"] for m in all_samples if m.get("task_success_soft", 0) == 1],



            "agent_capability_score": [m.get("agent_capability_score", 0.0) for m in all_samples],



            "effective_step_utilization": [m.get("effective_step_utilization", 0.0) for m in all_samples],

        }



        if any("self_correction_rate" in m for m in all_samples):

            result["self_correction_rate"] = [m.get("self_correction_rate", 0) for m in all_samples]



        if any("plan_quality" in m for m in all_samples):

            result["plan_quality"] = [m.get("plan_quality", 0.0) for m in all_samples]

            result["plan_faithfulness"] = [m.get("plan_faithfulness", 0.0) for m in all_samples]

            result["execution_accuracy"] = [m.get("execution_accuracy", 0.0) for m in all_samples]

            result["param_grounding"] = [m.get("param_grounding", 0.0) for m in all_samples]

        return result

    else:

        return []



def calculate_hallu_analysis(metrics, tasks, level, solvability=None):

    filter_fn = (lambda m: m.get("golden", "").lower() == solvability) if solvability else (lambda m: True)



    count_dict = {

        "non_existent_tools": 0,

        "solvability_hallu": 0,

        "false_unsolvable": 0,

        "wrong_tools": 0,

        "wrong_tool_order": 0,

        "missing_params": 0,

        "wrong_param_value": 0,

        "tool_correct_param_error": 0,

        "no_answer": 0,

        "correct": 0,



        "wrong_unsolvable_index": 0,

        "wrong_reasoning": 0,



        "correct_with_extra_steps": 0,

        "missing_tools": 0,

        "incorrect_tools": 0,

        "wrong_params": 0,

        "not_finished": 0,

        "no_response": 0,

        "unknown_error": 0,

        "redundant_steps": 0

    }

    for m in [m for task in tasks for m in metrics[task][level] if filter_fn(m)]:



        if "condition" in m:

            condition = m["condition"]



        elif "unsolvable" in m and "condition" in m["unsolvable"]:

            condition = m["unsolvable"]["condition"]

        else:

            condition = "unknown"



        if condition in count_dict:

            count_dict[condition] += 1

        else:

            count_dict["unknown_error"] += 1

    return count_dict



def calculate_subtask_results(metrics):

    results_dict = {}

    for sub_task, task_metrics in metrics.items():

        task_dict = {}

        for level, _ in task_metrics.items():

            level_dict = {}

            level_results = calculate_metrics(metrics, sub_task, level)

            if isinstance(level_results, dict):

                level_dict.update({f"{k}": v for k, v in level_results.items()})

            else:

                level_dict = level_results

            task_dict[level] = level_dict

        results_dict[sub_task] = task_dict

    return results_dict



def calculate_group_results(metrics):

    results_dict = {}



    available_levels = set()

    for task_metrics in metrics.values():

        available_levels.update(task_metrics.keys())



    for group, subtasks in subtask_groups.items():



        existing_subtasks = [s for s in subtasks if s in metrics]

        if not existing_subtasks:

            continue



        group_dict = {}



        solvability = None if group == "overall" else group.lower()



        if "level_1" in available_levels:

            l1_data = calculate_group_metrics(metrics, existing_subtasks, "level_1", solvability)

            group_dict["Level 1 Exact Match"] = l1_data.get("exact_match", 0.0)

            group_dict["Level 1 Hallucination Rate"] = l1_data.get("hallucination_rate", 0.0)

            group_dict["Level 1 Risk-Adjusted Score"] = l1_data.get("risk_adjusted_score", 0.0)

            group_dict["Level 1 Counts"] = l1_data.get("counts", {})

        if "level_2" in available_levels:

            result = calculate_group_metrics(metrics, existing_subtasks, "level_2", solvability)

            group_dict["Level 2 Tool Score (Soft)"] = np.mean(result["tool_score_soft"]) if result["tool_score_soft"] else 0.0

            group_dict["Level 2 Param Score"] = np.mean(result["param_score"]) if result["param_score"] else 0.0

            group_dict["Level 2 Task Success (Soft)"] = np.mean(result["task_success_soft"]) if result["task_success_soft"] else 0.0

            group_dict["Level 2 Planning Precision"] = np.mean(result["planning_precision"]) if result["planning_precision"] else 0.0

        if "level_3" in available_levels:

            result = calculate_group_metrics(metrics, existing_subtasks, "level_3", solvability)

            group_dict["Level 3 Task Success (Soft)"] = np.mean(result["task_success_soft"]) if result["task_success_soft"] else 0.0

            group_dict["Level 3 Tool Exact (Soft)"] = np.mean(result["tool_call_exact_soft"]) if result["tool_call_exact_soft"] else 0.0

            group_dict["Level 3 Parameter Accuracy"] = np.mean(result["parameter_accuracy"]) if result["parameter_accuracy"] else 0.0

            group_dict["Level 3 Step Efficiency (Soft)"] = np.mean(result["step_efficiency_soft"]) if result["step_efficiency_soft"] else 0.0

            group_dict["Level 3 Agent Capability Score"] = np.mean(result["agent_capability_score"]) if result["agent_capability_score"] else 0.0

            group_dict["Level 3 Effective Step Utilization"] = np.mean(result["effective_step_utilization"]) if result["effective_step_utilization"] else 0.0



            if "self_correction_rate" in result:

                group_dict["Level 3 Self-Correction Rate"] = np.mean(result["self_correction_rate"]) if result["self_correction_rate"] else 0.0



            if "plan_quality" in result:

                group_dict["Level 3 Plan Quality"] = np.mean(result["plan_quality"]) if result["plan_quality"] else 0.0

                group_dict["Level 3 Plan Faithfulness"] = np.mean(result["plan_faithfulness"]) if result["plan_faithfulness"] else 0.0

                group_dict["Level 3 Execution Accuracy"] = np.mean(result["execution_accuracy"]) if result["execution_accuracy"] else 0.0

                group_dict["Level 3 Param Grounding"] = np.mean(result["param_grounding"]) if result["param_grounding"] else 0.0





        token_stats = aggregate_token_usage(metrics, existing_subtasks, solvability=solvability)

        if token_stats:

            group_dict["token_usage"] = token_stats



        results_dict[group] = group_dict

    return results_dict



def calculate_analysis_results(metrics):

    statistics = {}



    available_levels = set()

    for task_metrics in metrics.values():

        available_levels.update(task_metrics.keys())



    for group, subtasks in subtask_groups.items():



        existing_subtasks = [s for s in subtasks if s in metrics]

        if not existing_subtasks:

            continue





        solvability = None if group == "overall" else group.lower()



        for level in ["level_1", "level_2", "level_3"]:

            if level not in available_levels:

                continue



            level_name = level.replace("l", "L").replace("_", " ")

            group_name = group.replace("overall", "Overall")



            if level == "level_1":



                filter_fn = (lambda m: m.get("golden", "").lower() == solvability) if solvability else (lambda m: True)

                samples = [m for task in existing_subtasks

                           for m in metrics[task].get(level, []) if filter_fn(m)]

                l1_agg = _aggregate_l1_confidence_metrics(samples)

                statistics[f"{level_name} / {group_name}"] = {

                    "Confident Right": l1_agg["counts"]["confident_right"],

                    "Confident Wrong": l1_agg["counts"]["confident_wrong"],

                    "Uncertain": l1_agg["counts"]["uncertain"],

                    "No Answer": l1_agg["counts"]["no_answer"],

                    "Total": l1_agg["counts"]["total"],

                    "Hallucination Rate": round(l1_agg["hallucination_rate"], 4),

                }

                continue



            count_dict = calculate_hallu_analysis(metrics, existing_subtasks, level, solvability)

            statistics[f"{level_name} / {group_name}"] = {}

            statistics[f"{level_name} / {group_name}"]["Non-existent Tools"] = count_dict["non_existent_tools"]

            statistics[f"{level_name} / {group_name}"]["Solvability Hallucination"] = count_dict["solvability_hallu"]

            statistics[f"{level_name} / {group_name}"]["False Unsolvable"] = count_dict["false_unsolvable"]



            statistics[f"{level_name} / {group_name}"]["Missing Tools"] = count_dict["missing_tools"] + count_dict["wrong_tools"] + count_dict["wrong_tool_order"]

            statistics[f"{level_name} / {group_name}"]["Incorrect Tools"] = count_dict["incorrect_tools"]



            statistics[f"{level_name} / {group_name}"]["Missing Params"] = count_dict["missing_params"]

            statistics[f"{level_name} / {group_name}"]["Wrong Param Value"] = count_dict["wrong_param_value"] + count_dict["tool_correct_param_error"]



            statistics[f"{level_name} / {group_name}"]["No Answer"] = count_dict["no_answer"] + count_dict["no_response"]

            statistics[f"{level_name} / {group_name}"]["Correct"] = count_dict["correct"]



            statistics[f"{level_name} / {group_name}"]["Redundant Steps"] = count_dict["correct_with_extra_steps"] + count_dict["redundant_steps"]



            if level == "level_3":



                statistics[f"{level_name} / {group_name}"]["Missing Tools"] += count_dict["not_finished"]



                statistics[f"{level_name} / {group_name}"]["Wrong Param Value"] += count_dict["wrong_params"]

                statistics[f"{level_name} / {group_name}"]["Unknown Error"] = count_dict["unknown_error"]

    return statistics







def extract_domain_from_metrics(metrics):

    domain_data = defaultdict(lambda: defaultdict(list))



    for subtask, level_data in metrics.items():

        for level, samples in level_data.items():

            if not isinstance(samples, list):

                continue

            for sample in samples:

                domain = sample.get("domain", "unknown")

                domain_data[domain][level].append(sample)



    return domain_data





def calculate_domain_metrics(domain_data, level):

    if level not in domain_data or not domain_data[level]:

        return None



    samples = domain_data[level]



    if level == "level_1":

        return _aggregate_l1_confidence_metrics(samples)

    elif level == "level_2":

        tool_scores_soft = [s.get("tool_score_soft", 0) for s in samples]

        param_scores = [s.get("param_score", 0) for s in samples]

        task_success_soft = [s.get("task_success_soft", 0) for s in samples]

        planning_precision = [s.get("planning_precision", 0.0) for s in samples]

        return {

            "tool_score_soft": np.mean(tool_scores_soft),

            "param_score": np.mean(param_scores),

            "task_success_soft": np.mean(task_success_soft),

            "planning_precision": np.mean(planning_precision),

        }

    elif level == "level_3":

        success_effs = [s.get("step_efficiency_soft", 0) for s in samples if s.get("task_success_soft", 0) == 1]

        result = {

            "task_success_soft": np.mean([s.get("task_success_soft", 0) for s in samples]),

            "tool_call_exact_soft": np.mean([s.get("tool_call_exact_soft", 0) for s in samples]),

            "parameter_accuracy": np.mean([s.get("parameter_accuracy", 0) for s in samples]),

            "step_efficiency_soft": np.mean(success_effs) if success_effs else 0.0,

            "agent_capability_score": np.mean([s.get("agent_capability_score", 0.0) for s in samples]),

            "effective_step_utilization": np.mean([s.get("effective_step_utilization", 0.0) for s in samples]),

        }



        if any("self_correction_rate" in s for s in samples):

            result["self_correction_rate"] = np.mean([s.get("self_correction_rate", 0) for s in samples])



        if any("plan_quality" in s for s in samples):

            result["plan_quality"] = np.mean([s.get("plan_quality", 0.0) for s in samples])

            result["plan_faithfulness"] = np.mean([s.get("plan_faithfulness", 0.0) for s in samples])

            result["execution_accuracy"] = np.mean([s.get("execution_accuracy", 0.0) for s in samples])

            result["param_grounding"] = np.mean([s.get("param_grounding", 0.0) for s in samples])

        return result

    return None





def calculate_domain_results(metrics):

    domain_data = extract_domain_from_metrics(metrics)



    results = {}





    available_levels = set()

    for domain, levels in domain_data.items():

        available_levels.update(levels.keys())



    for domain, levels in domain_data.items():

        results[domain] = {

            "sample_count": sum(len(levels.get(lv, [])) for lv in available_levels)

        }

        for level in ["level_1", "level_2", "level_3"]:

            if level in available_levels:

                level_metrics = calculate_domain_metrics(levels, level)

                if level_metrics is not None:

                    results[domain][level] = level_metrics



    return results





def calculate_domain_group_results(metrics):

    domain_results = calculate_domain_results(metrics)

    group_results = {}





    available_levels = set()

    for domain_data in domain_results.values():

        available_levels.update(k for k in domain_data.keys() if k.startswith("level_"))



    for group_name, domains in domain_groups.items():



        group_samples = {"level_1": [], "level_2": [], "level_3": []}

        total_count = 0



        for domain in domains:

            if domain not in domain_results:

                continue

            total_count += domain_results[domain].get("sample_count", 0)



            for level in ["level_1", "level_2", "level_3"]:

                if level in domain_results[domain]:



                    group_samples[level].append(domain_results[domain][level])



        if total_count == 0:

            continue



        group_results[group_name] = {"sample_count": total_count}



        for level in ["level_1", "level_2", "level_3"]:

            if not group_samples[level]:

                continue



            if level == "level_1":



                domain_metrics = group_samples[level]

                group_results[group_name][level] = {

                    "exact_match": np.mean([d["exact_match"] for d in domain_metrics]),

                    "hallucination_rate": np.mean([d["hallucination_rate"] for d in domain_metrics]),

                    "risk_adjusted_score": np.mean([d["risk_adjusted_score"] for d in domain_metrics]),

                }

            elif level == "level_2":

                group_results[group_name][level] = {

                    "tool_score_soft": np.mean([s["tool_score_soft"] for s in group_samples[level]]),

                    "param_score": np.mean([s["param_score"] for s in group_samples[level]]),

                    "task_success_soft": np.mean([s["task_success_soft"] for s in group_samples[level]]),

                    "planning_precision": np.mean([s.get("planning_precision", 0.0) for s in group_samples[level]]),

                }

            elif level == "level_3":

                success_effs = [s["step_efficiency_soft"] for s in group_samples[level] if s.get("task_success_soft", 0) == 1]

                l3_result = {

                    "task_success_soft": np.mean([s["task_success_soft"] for s in group_samples[level]]),

                    "tool_call_exact_soft": np.mean([s["tool_call_exact_soft"] for s in group_samples[level]]),

                    "parameter_accuracy": np.mean([s["parameter_accuracy"] for s in group_samples[level]]),

                    "step_efficiency_soft": np.mean(success_effs) if success_effs else 0.0,

                    "agent_capability_score": np.mean([s.get("agent_capability_score", 0.0) for s in group_samples[level]]),

                    "effective_step_utilization": np.mean([s.get("effective_step_utilization", 0.0) for s in group_samples[level]]),

                }



                if any("self_correction_rate" in s for s in group_samples[level]):

                    l3_result["self_correction_rate"] = np.mean([s.get("self_correction_rate", 0) for s in group_samples[level]])



                if any("plan_quality" in s for s in group_samples[level]):

                    l3_result["plan_quality"] = np.mean([s.get("plan_quality", 0.0) for s in group_samples[level]])

                    l3_result["plan_faithfulness"] = np.mean([s.get("plan_faithfulness", 0.0) for s in group_samples[level]])

                    l3_result["execution_accuracy"] = np.mean([s.get("execution_accuracy", 0.0) for s in group_samples[level]])

                    l3_result["param_grounding"] = np.mean([s.get("param_grounding", 0.0) for s in group_samples[level]])

                group_results[group_name][level] = l3_result



    return group_results





def print_domain_table(domain_results_dict):

    domain_table = []





    for group_name, domains in domain_groups.items():

        for domain in domains:

            if domain not in domain_results_dict:

                continue



            results = domain_results_dict[domain]

            display_name = domain_display_names.get(domain, domain)

            sample_count = results.get("sample_count", 0)





            if 'level_1' in results:

                l1 = results['level_1']

                if isinstance(l1, dict):

                    domain_table.append([display_name, group_name, 'L1 Exact Match',

                                        round(l1.get('exact_match', 0), 2), sample_count])

                    domain_table.append([display_name, group_name, 'L1 Hallu Rate',

                                        round(l1.get('hallucination_rate', 0), 2), sample_count])

                    domain_table.append([display_name, group_name, 'L1 Risk-Adjusted',

                                        round(l1.get('risk_adjusted_score', 0), 2), sample_count])

                else:



                    domain_table.append([display_name, group_name, 'L1 Exact Match',

                                        round(l1, 2), sample_count])





            if 'level_2' in results:

                if isinstance(results['level_2'], dict):

                    domain_table.append([display_name, group_name, 'L2 Tool Score (Soft)',

                                        round(results['level_2']['tool_score_soft'], 2), sample_count])

                    domain_table.append([display_name, group_name, 'L2 Param Score',

                                        round(results['level_2']['param_score'], 2), sample_count])

                    domain_table.append([display_name, group_name, 'L2 Task Success (Soft)',

                                        round(results['level_2']['task_success_soft'], 2), sample_count])

                    domain_table.append([display_name, group_name, 'L2 Planning Precision',

                                        round(results['level_2'].get('planning_precision', 0), 2), sample_count])





            if 'level_3' in results:

                l3 = results['level_3']

                if isinstance(l3, dict):

                    domain_table.append([display_name, group_name, 'L3 Task Success (Soft)',

                                        round(l3.get('task_success_soft', 0), 2), sample_count])

                    domain_table.append([display_name, group_name, 'L3 Tool Exact (Soft)',

                                        round(l3.get('tool_call_exact_soft', 0), 2), sample_count])

                    domain_table.append([display_name, group_name, 'L3 Parameter Accuracy',

                                        round(l3.get('parameter_accuracy', 0), 2), sample_count])

                    domain_table.append([display_name, group_name, 'L3 Step Efficiency (Soft)',

                                        round(l3.get('step_efficiency_soft', 0), 2), sample_count])



    return tabulate(domain_table, headers=['Domain', 'Category', 'Metric', 'Score', 'Samples'], tablefmt='grid')





def print_domain_group_table(group_results_dict):

    group_table = []



    group_order = ["Enterprise", "Technology", "Healthcare", "Industrial", "Consumer", "Education", "Government"]



    for group_name in group_order:

        if group_name not in group_results_dict:

            continue



        results = group_results_dict[group_name]

        sample_count = results.get("sample_count", 0)





        if 'level_1' in results:

            l1 = results['level_1']

            if isinstance(l1, dict):

                group_table.append([group_name, 'L1 Exact Match', round(l1.get('exact_match', 0), 2), sample_count])

                group_table.append([group_name, 'L1 Hallu Rate', round(l1.get('hallucination_rate', 0), 2), sample_count])

                group_table.append([group_name, 'L1 Risk-Adjusted', round(l1.get('risk_adjusted_score', 0), 2), sample_count])

            else:

                group_table.append([group_name, 'L1 Exact Match', round(l1, 2), sample_count])





        if 'level_2' in results:

            if isinstance(results['level_2'], dict):

                group_table.append([group_name, 'L2 Tool Score (Soft)', round(results['level_2']['tool_score_soft'], 2), sample_count])

                group_table.append([group_name, 'L2 Param Score', round(results['level_2']['param_score'], 2), sample_count])

                group_table.append([group_name, 'L2 Task Success (Soft)', round(results['level_2']['task_success_soft'], 2), sample_count])

                group_table.append([group_name, 'L2 Planning Precision', round(results['level_2'].get('planning_precision', 0), 2), sample_count])





        if 'level_3' in results:

            l3 = results['level_3']

            if isinstance(l3, dict):

                group_table.append([group_name, 'L3 Task Success (Soft)', round(l3.get('task_success_soft', 0), 2), sample_count])

                group_table.append([group_name, 'L3 Tool Exact (Soft)', round(l3.get('tool_call_exact_soft', 0), 2), sample_count])

                group_table.append([group_name, 'L3 Parameter Accuracy', round(l3.get('parameter_accuracy', 0), 2), sample_count])

                group_table.append([group_name, 'L3 Step Efficiency (Soft)', round(l3.get('step_efficiency_soft', 0), 2), sample_count])



    return tabulate(group_table, headers=['Domain Group', 'Metric', 'Score', 'Samples'], tablefmt='grid')





def print_table(sub_task_results_dict, group_results_dict, analysis_results_dict):

    sub_task_table = []

    for task, results in sub_task_results_dict.items():



        if 'level_1' in results:

            l1 = results['level_1']

            if isinstance(l1, dict):

                sub_task_table.append([task, 'L1 Exact Match', round(l1.get('exact_match', 0), 2)])

                sub_task_table.append([task, 'L1 Hallucination Rate', round(l1.get('hallucination_rate', 0), 2)])

                sub_task_table.append([task, 'L1 Risk-Adjusted Score', round(l1.get('risk_adjusted_score', 0), 2)])

            else:

                sub_task_table.append([task, 'Level 1 Exact Match', round(l1, 2)])



        if 'level_2' in results:

            if isinstance(results['level_2'], dict):

                sub_task_table.append([task, 'Level 2 Tool Score (Soft)', round(results['level_2']['tool_score_soft'], 2)])

                sub_task_table.append([task, 'Level 2 Param Score', round(results['level_2']['param_score'], 2)])

                sub_task_table.append([task, 'Level 2 Task Success (Soft)', round(results['level_2']['task_success_soft'], 2)])

                sub_task_table.append([task, 'Level 2 Planning Precision', round(results['level_2'].get('planning_precision', 0), 2)])

            else:

                sub_task_table.append([task, 'Level 2 Progress Rate', round(results['level_2'], 2)])



        if 'level_3' in results:

            l3 = results['level_3']

            if isinstance(l3, dict):

                sub_task_table.append([task, 'Level 3 Task Success (Soft)', round(l3.get('task_success_soft', 0), 2)])

                sub_task_table.append([task, 'Level 3 Tool Exact (Soft)', round(l3.get('tool_call_exact_soft', 0), 2)])

                sub_task_table.append([task, 'Level 3 Parameter Accuracy', round(l3.get('parameter_accuracy', 0), 2)])

                sub_task_table.append([task, 'Level 3 Step Efficiency (Soft)', round(l3.get('step_efficiency_soft', 0), 2)])

                sub_task_table.append([task, 'Level 3 Agent Capability Score', round(l3.get('agent_capability_score', 0), 2)])

                sub_task_table.append([task, 'Level 3 Effective Step Utilization', round(l3.get('effective_step_utilization', 0), 2)])



                if 'self_correction_rate' in l3:

                    sub_task_table.append([task, 'Level 3 Self-Correction Rate', round(l3.get('self_correction_rate', 0), 2)])



                if 'plan_quality' in l3:

                    sub_task_table.append([task, 'Level 3 Plan Quality', round(l3.get('plan_quality', 0), 2)])

                    sub_task_table.append([task, 'Level 3 Plan Faithfulness', round(l3.get('plan_faithfulness', 0), 2)])

                    sub_task_table.append([task, 'Level 3 Execution Accuracy', round(l3.get('execution_accuracy', 0), 2)])

                    sub_task_table.append([task, 'Level 3 Param Grounding', round(l3.get('param_grounding', 0), 2)])

            else:



                sub_task_table.append([task, 'Level 3 Score', round(l3, 2)])

    sub_task_table_print = tabulate(sub_task_table, headers=['Subtask', 'Metric', 'Score'], tablefmt='grid')



    group_table = []

    for group, results in group_results_dict.items():

        for metric, value in results.items():

            if metric == "token_usage" and isinstance(value, dict):



                group_table.append([group, "Token: Total Prompt", f"{value.get('total_prompt_tokens', 0):,}"])

                group_table.append([group, "Token: Total Completion", f"{value.get('total_completion_tokens', 0):,}"])

                group_table.append([group, "Token: Total", f"{value.get('total_tokens', 0):,}"])

                group_table.append([group, "Token: Avg Prompt/Sample", value.get('avg_prompt_tokens', 0)])

                group_table.append([group, "Token: Avg Completion/Sample", value.get('avg_completion_tokens', 0)])

            else:

                group_table.append([group, metric, round(value, 2) if isinstance(value, (int, float)) else value])

    group_table_print = tabulate(group_table, headers=['Scenario', 'Metric', 'Score'], tablefmt='grid')



    analysis_table = []

    for level, results in analysis_results_dict.items():

        for error_type, count in results.items():

            analysis_table.append([level, error_type, count])

    analysis_table_print = tabulate(analysis_table, headers=['Level / Scenario', 'Metric', 'Count'], tablefmt='grid')



    return sub_task_table_print, group_table_print, analysis_table_print
