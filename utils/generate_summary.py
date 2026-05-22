

import os

import json

import argparse

import glob

from datetime import datetime

from tabulate import tabulate

import numpy as np

from collections import defaultdict





def load_evaluation_results(eval_dir, level, model_name, calculate_type, strategy=None):

    if strategy:

        subdir = f"level3_{strategy}"

    else:

        subdir = f"level{level}"

    pattern = os.path.join(

        eval_dir, subdir, "eval_results",

        f"Evaluation_results_{model_name}_*_{calculate_type}.json"

    )

    files = glob.glob(pattern)

    if not files:

        return None



    with open(files[0], 'r', encoding='utf-8') as f:

        return json.load(f)





def load_raw_metrics(eval_dir, level, model_name, calculate_type, strategy=None):

    if strategy:

        subdir = f"level3_{strategy}"

    else:

        subdir = f"level{level}"

    pattern = os.path.join(

        eval_dir, subdir, "eval_results",

        f"{model_name}_*_{calculate_type}.json"

    )

    files = glob.glob(pattern)

    if not files:

        return None



    with open(files[0], 'r', encoding='utf-8') as f:

        return json.load(f)





def count_samples(metrics, subtasks):

    count = 0

    for subtask in subtasks:

        if subtask in metrics:

            for level_key, level_data in metrics[subtask].items():

                if isinstance(level_data, list):

                    count += len(level_data)

                    break

    return count





def _compute_gae(ras, l2_task_success, l3_task_success, esu, ppr):

    if ras <= 0:

        return 0.0

    product = l2_task_success * l3_task_success * esu * ppr

    if product <= 0:

        return 0.0

    return ras * (product ** (1 / 4))





def _normalize_level1_data(l1_data):

    if isinstance(l1_data, (int, float)):

        return {

            "exact_match": l1_data,

            "hallucination_rate": 0,

            "risk_adjusted_score": 0,

        }

    if isinstance(l1_data, dict):

        return {

            "exact_match": l1_data.get("exact_match", 0),

            "hallucination_rate": l1_data.get("hallucination_rate", 0),

            "risk_adjusted_score": l1_data.get("risk_adjusted_score", 0),

        }

    return {

        "exact_match": 0,

        "hallucination_rate": 0,

        "risk_adjusted_score": 0,

    }





def generate_overall_summary_horizontal(level_data, model_name):

    headers = ["Model"]

    row = [model_name]





    if level_data.get("level_1"):

        l1 = _normalize_level1_data(level_data["level_1"])

        headers.extend(["L1 EM", "L1 Hallu Rate", "L1 Risk-Adj"])

        row.extend([

            f"{l1.get('exact_match', 0):.2%}",

            f"{l1.get('hallucination_rate', 0):.2%}",

            f"{l1.get('risk_adjusted_score', 0):.2%}",

        ])





    if level_data.get("level_2"):

        l2 = level_data["level_2"]

        headers.extend(["L2 Tool(S)", "L2 Param", "L2 Task Succ(S)", "L2 PPR"])

        row.extend([

            f"{l2.get('tool_score_soft', 0):.2%}",

            f"{l2.get('param_score', 0):.2%}",

            f"{l2.get('task_success_soft', 0):.2%}",

            f"{l2.get('planning_precision', 0):.2%}",

        ])





    ph = level_data.get("planning_hallucination")

    if ph:

        headers.extend(["L2 TFR", "L2 SBR", "L2 FUR", "L2 RSR"])

        row.extend([

            f"{ph.get('tool_fabrication_rate', 0):.2%}",

            f"{ph.get('solvability_blindness_rate', 0):.2%}",

            f"{ph.get('false_unsolvable_rate', 0):.2%}",

            f"{ph.get('redundant_step_rate', 0):.2%}",

        ])





    if level_data.get("level_3"):

        l3 = level_data["level_3"]

        headers.extend([

            "L3 Task Succ(S)", "L3 Tool Exact(S)", "L3 Param Acc", "L3 Step Eff(S)", "L3 ACS", "L3 ESU"

        ])

        row.extend([

            f"{l3.get('task_success_soft', 0):.2%}",

            f"{l3.get('tool_call_exact_soft', 0):.2%}",

            f"{l3.get('parameter_accuracy', 0):.2%}",

            f"{l3.get('step_efficiency_soft', 0):.2%}",

            f"{l3.get('agent_capability_score', 0):.2%}",

            f"{l3.get('effective_step_utilization', 0):.2%}",

        ])



        if "self_correction_rate" in l3:

            headers.append("L3 Self-Corr")

            row.append(f"{l3.get('self_correction_rate', 0):.2%}")



        if "plan_quality" in l3:

            headers.extend(["L3 Plan Qual", "L3 Plan Faith", "L3 Exec Acc", "L3 Param Gnd"])

            row.extend([

                f"{l3.get('plan_quality', 0):.2%}",

                f"{l3.get('plan_faithfulness', 0):.2%}",

                f"{l3.get('execution_accuracy', 0):.2%}",

                f"{l3.get('param_grounding', 0):.2%}",

            ])





    eh = level_data.get("execution_hallucination")

    if eh:

        headers.extend(["L3 TFR", "L3 MTR", "L3 CHR", "L3 ESR"])

        row.extend([

            f"{eh.get('tool_fabrication_rate', 0):.2%}",

            f"{eh.get('missing_tool_rate', 0):.2%}",

            f"{eh.get('completion_hallucination_rate', 0):.2%}",

            f"{eh.get('extra_step_rate', 0):.2%}",

        ])





    l1 = _normalize_level1_data(level_data.get("level_1")) if level_data.get("level_1") else None

    l2_data = level_data.get("level_2")

    l3_data = level_data.get("level_3")

    if l1 and l2_data and l3_data:

        gae = _compute_gae(

            l1.get("risk_adjusted_score", 0),

            l2_data.get("task_success_soft", 0),

            l3_data.get("task_success_soft", 0),

            l3_data.get("effective_step_utilization", 0),

            l2_data.get("planning_precision", 0),

        )

        headers.append("GAE")

        row.append(f"{gae:.2%}")



        s2 = l2_data.get("task_success_soft", 0)

        s3 = l3_data.get("task_success_soft", 0)

        gap = (s3 - s2) / s2 if s2 > 0 else 0.0

        headers.append("GAP")

        row.append(f"{gap:+.2%}")



        gap2 = s3 - s2

        headers.append("GAP2")

        row.append(f"{gap2:+.2%}")



    return tabulate([row], headers=headers, tablefmt='grid')





def generate_level3_strategy_comparison(level_data, strategies):

    if not strategies:

        return ""





    has_self_corr = any("self_correction_rate" in level_data.get(f"level_3_{s}", {}) for s in strategies)

    has_plan_qual = any("plan_quality" in level_data.get(f"level_3_{s}", {}) for s in strategies)



    rows = []

    for strategy in strategies:

        key = f"level_3_{strategy}"

        data = level_data.get(key, {})

        if not data:

            continue

        row = [

            strategy,

            f"{data.get('task_success_soft', 0):.2%}",

            f"{data.get('tool_call_exact_soft', 0):.2%}",

            f"{data.get('parameter_accuracy', 0):.2%}",

            f"{data.get('step_efficiency_soft', 0):.2%}",

            f"{data.get('agent_capability_score', 0):.2%}",

            f"{data.get('effective_step_utilization', 0):.2%}",

        ]



        if has_self_corr:

            if "self_correction_rate" in data:

                row.append(f"{data.get('self_correction_rate', 0):.2%}")

            else:

                row.append("")



        if has_plan_qual:

            if "plan_quality" in data:

                row.extend([

                    f"{data.get('plan_quality', 0):.2%}",

                    f"{data.get('plan_faithfulness', 0):.2%}",

                    f"{data.get('execution_accuracy', 0):.2%}",

                    f"{data.get('param_grounding', 0):.2%}",

                ])

            else:

                row.extend(["", "", "", ""])



        token = data.get("token_usage", {})

        if isinstance(token, dict) and token.get("total_prompt_tokens"):

            row.append(f"{token['total_prompt_tokens']:,}")

            row.append(f"{token['total_completion_tokens']:,}")

        rows.append(row)



    headers = [

        "Strategy", "Task Succ(S)", "Tool Exact(S)", "Param Acc", "Step Eff(S)", "ACS", "ESU"

    ]

    if has_self_corr:

        headers.append("Self-Corr")

    if has_plan_qual:

        headers.extend(["Plan Qual", "Plan Faith", "Exec Acc", "Param Gnd"])

    for strategy in strategies:

        key = f"level_3_{strategy}"

        data = level_data.get(key, {})

        token = data.get("token_usage", {})

        if isinstance(token, dict) and token.get("total_prompt_tokens"):

            headers.extend(["Prompt Tokens", "Completion Tokens"])

            break



    return tabulate(rows, headers=headers, tablefmt='grid')





def generate_scenario_table(by_scenario):

    rows = []

    for scenario in ["Solvable", "Unsolvable"]:

        if scenario not in by_scenario:

            continue

        data = by_scenario[scenario]





        if "level_1" in data:

            l1 = _normalize_level1_data(data["level_1"])

            rows.append([scenario, "Level-1", "Exact Match", f"{l1.get('exact_match', 0):.2%}"])

            rows.append([scenario, "Level-1", "Hallucination Rate", f"{l1.get('hallucination_rate', 0):.2%}"])

            rows.append([scenario, "Level-1", "Risk-Adjusted Score", f"{l1.get('risk_adjusted_score', 0):.2%}"])





        if "level_2" in data:

            rows.append([scenario, "Level-2", "Tool Score (Soft)", f"{data['level_2'].get('tool_score_soft', 0):.2%}"])

            rows.append([scenario, "Level-2", "Param Score", f"{data['level_2'].get('param_score', 0):.2%}"])

            rows.append([scenario, "Level-2", "Task Success (Soft)", f"{data['level_2'].get('task_success_soft', 0):.2%}"])





        if "level_3" in data:

            l3 = data["level_3"]

            rows.append([scenario, "Level-3", "Task Success (Soft)", f"{l3.get('task_success_soft', 0):.2%}"])

            rows.append([scenario, "Level-3", "Tool Call Exact (Soft)", f"{l3.get('tool_call_exact_soft', 0):.2%}"])

            rows.append([scenario, "Level-3", "Parameter Accuracy", f"{l3.get('parameter_accuracy', 0):.2%}"])

            rows.append([scenario, "Level-3", "Step Efficiency (Soft)", f"{l3.get('step_efficiency_soft', 0):.2%}"])

            rows.append([scenario, "Level-3", "Agent Capability Score", f"{l3.get('agent_capability_score', 0):.2%}"])

            rows.append([scenario, "Level-3", "Effective Step Utilization", f"{l3.get('effective_step_utilization', 0):.2%}"])



            if "self_correction_rate" in l3:

                rows.append([scenario, "Level-3", "Self-Correction Rate", f"{l3.get('self_correction_rate', 0):.2%}"])



            if "plan_quality" in l3:

                rows.append([scenario, "Level-3", "Plan Quality", f"{l3.get('plan_quality', 0):.2%}"])

                rows.append([scenario, "Level-3", "Plan Faithfulness", f"{l3.get('plan_faithfulness', 0):.2%}"])

                rows.append([scenario, "Level-3", "Execution Accuracy", f"{l3.get('execution_accuracy', 0):.2%}"])

                rows.append([scenario, "Level-3", "Param Grounding", f"{l3.get('param_grounding', 0):.2%}"])



    return tabulate(rows, headers=["Scenario", "Level", "Metric", "Score"], tablefmt='grid')





def generate_subtask_table(by_subtask):

    rows = []



    subtask_order = [

        "single_step", "multi_step_no_rep",

        "sequential_chain", "conditional_planning",

        "os", "web", "best",

    ]





    subtask_display_names = {"os": "OS", "web": "Web", "best": "Best"}



    for subtask in subtask_order:

        if subtask not in by_subtask:

            continue

        data = by_subtask[subtask]

        display_name = subtask_display_names.get(subtask, subtask.replace("_", " ").title())





        if "level_1" in data:

            l1 = _normalize_level1_data(data["level_1"])

            rows.append([display_name, "Level-1", "Exact Match", f"{l1.get('exact_match', 0):.2%}"])

            rows.append([display_name, "Level-1", "Hallucination Rate", f"{l1.get('hallucination_rate', 0):.2%}"])

            rows.append([display_name, "Level-1", "Risk-Adjusted Score", f"{l1.get('risk_adjusted_score', 0):.2%}"])





        if "level_2" in data:

            rows.append([display_name, "Level-2", "Tool Score (Soft)", f"{data['level_2'].get('tool_score_soft', 0):.2%}"])

            rows.append([display_name, "Level-2", "Param Score", f"{data['level_2'].get('param_score', 0):.2%}"])

            rows.append([display_name, "Level-2", "Task Success (Soft)", f"{data['level_2'].get('task_success_soft', 0):.2%}"])





        if "level_3" in data:

            l3 = data["level_3"]

            rows.append([display_name, "Level-3", "Task Success (Soft)", f"{l3.get('task_success_soft', 0):.2%}"])

            rows.append([display_name, "Level-3", "Tool Call Exact (Soft)", f"{l3.get('tool_call_exact_soft', 0):.2%}"])

            rows.append([display_name, "Level-3", "Param Accuracy", f"{l3.get('parameter_accuracy', 0):.2%}"])

            rows.append([display_name, "Level-3", "Step Efficiency (Soft)", f"{l3.get('step_efficiency_soft', 0):.2%}"])

            rows.append([display_name, "Level-3", "Agent Capability Score", f"{l3.get('agent_capability_score', 0):.2%}"])

            rows.append([display_name, "Level-3", "Effective Step Utilization", f"{l3.get('effective_step_utilization', 0):.2%}"])

            if "self_correction_rate" in l3:

                rows.append([display_name, "Level-3", "Self-Correction Rate", f"{l3.get('self_correction_rate', 0):.2%}"])



    return tabulate(rows, headers=["Subtask", "Level", "Metric", "Score"], tablefmt='grid')











domain_groups_config = {

    "Enterprise": ["financial_services", "human_resources", "customer_service", "supply_chain"],

    "Technology": ["software_development", "cybersecurity", "cloud_operations", "data_engineering"],

    "Healthcare": ["clinical_healthcare", "medical_devices"],

    "Industrial": ["smart_manufacturing", "energy_management"],

    "Consumer": ["e_commerce", "smart_home", "transportation", "food_hospitality"],

    "Education": ["edtech", "scientific_research"],

    "Government": ["government_services", "public_safety"]

}





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





def generate_domain_table(by_domain):

    rows = []





    for group_name, domains in domain_groups_config.items():

        for domain in domains:

            if domain not in by_domain:

                continue



            data = by_domain[domain]

            display_name = domain_display_names.get(domain, domain.replace("_", " ").title())

            sample_count = data.get("sample_count", "-")





            if "level_1" in data:

                l1 = _normalize_level1_data(data["level_1"])

                rows.append([display_name, group_name, "L1", "Exact Match",

                           f"{l1.get('exact_match', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L1", "Hallucination Rate",

                           f"{l1.get('hallucination_rate', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L1", "Risk-Adjusted Score",

                           f"{l1.get('risk_adjusted_score', 0):.2%}", sample_count])





            if "level_2" in data:

                rows.append([display_name, group_name, "L2", "Tool Score (Soft)",

                           f"{data['level_2'].get('tool_score_soft', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L2", "Param Score",

                           f"{data['level_2'].get('param_score', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L2", "Task Success (Soft)",

                           f"{data['level_2'].get('task_success_soft', 0):.2%}", sample_count])





            if "level_3" in data:

                l3 = data["level_3"]

                rows.append([display_name, group_name, "L3", "Task Success (Soft)",

                           f"{l3.get('task_success_soft', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L3", "Tool Exact (Soft)",

                           f"{l3.get('tool_call_exact_soft', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L3", "Param Accuracy",

                           f"{l3.get('parameter_accuracy', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L3", "Step Efficiency (Soft)",

                           f"{l3.get('step_efficiency_soft', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L3", "ACS",

                           f"{l3.get('agent_capability_score', 0):.2%}", sample_count])

                rows.append([display_name, group_name, "L3", "ESU",

                           f"{l3.get('effective_step_utilization', 0):.2%}", sample_count])

                if "self_correction_rate" in l3:

                    rows.append([display_name, group_name, "L3", "Self-Correction Rate",

                               f"{l3.get('self_correction_rate', 0):.2%}", sample_count])



    return tabulate(rows, headers=["Domain", "Category", "Level", "Metric", "Score", "Samples"], tablefmt='grid')





def generate_domain_group_table(by_domain_group):

    rows = []



    group_order = ["Enterprise", "Technology", "Healthcare", "Industrial", "Consumer", "Education", "Government"]



    for group_name in group_order:

        if group_name not in by_domain_group:

            continue



        data = by_domain_group[group_name]

        sample_count = data.get("sample_count", "-")





        if "level_1" in data:

            l1 = _normalize_level1_data(data["level_1"])

            rows.append([group_name, "L1", "Exact Match", f"{l1.get('exact_match', 0):.2%}", sample_count])

            rows.append([group_name, "L1", "Hallucination Rate", f"{l1.get('hallucination_rate', 0):.2%}", sample_count])

            rows.append([group_name, "L1", "Risk-Adjusted Score", f"{l1.get('risk_adjusted_score', 0):.2%}", sample_count])





        if "level_2" in data:

            rows.append([group_name, "L2", "Tool Score (Soft)", f"{data['level_2'].get('tool_score_soft', 0):.2%}", sample_count])

            rows.append([group_name, "L2", "Param Score", f"{data['level_2'].get('param_score', 0):.2%}", sample_count])

            rows.append([group_name, "L2", "Task Success (Soft)", f"{data['level_2'].get('task_success_soft', 0):.2%}", sample_count])





        if "level_3" in data:

            l3 = data["level_3"]

            rows.append([group_name, "L3", "Task Success (Soft)", f"{l3.get('task_success_soft', 0):.2%}", sample_count])

            rows.append([group_name, "L3", "Tool Exact (Soft)", f"{l3.get('tool_call_exact_soft', 0):.2%}", sample_count])

            rows.append([group_name, "L3", "Param Acc", f"{l3.get('parameter_accuracy', 0):.2%}", sample_count])

            rows.append([group_name, "L3", "Step Efficiency (Soft)", f"{l3.get('step_efficiency_soft', 0):.2%}", sample_count])

            rows.append([group_name, "L3", "ACS", f"{l3.get('agent_capability_score', 0):.2%}", sample_count])

            rows.append([group_name, "L3", "ESU", f"{l3.get('effective_step_utilization', 0):.2%}", sample_count])

            if "self_correction_rate" in l3:

                rows.append([group_name, "L3", "Self-Correction Rate", f"{l3.get('self_correction_rate', 0):.2%}", sample_count])



    return tabulate(rows, headers=["Domain Group", "Level", "Metric", "Score", "Samples"], tablefmt='grid')





def generate_error_analysis_table(error_analysis):



    name_mapping = {



        "Wrong Tools": "Missing Tools",

        "Wrong Params": "Wrong Param Value",

        "Unknown Error": "Redundant Steps",

        "Wrong Tool Order": "Missing Tools",

        "Not Finished": "Missing Tools",

        "Correct (Extra Steps)": "Redundant Steps",

    }

    for level_key, level_data in error_analysis.items():

        for scenario, scenario_data in level_data.items():

            for old_name, new_name in name_mapping.items():

                if old_name in scenario_data:

                    scenario_data[new_name] = scenario_data.pop(old_name) + scenario_data.get(new_name, 0)





    strategy_order = ["react", "reflexion", "plan_execute"]

    l3_keys = [f"level_3_{s}" for s in strategy_order if f"level_3_{s}" in error_analysis]





    columns = [("Level-2", ["level_2"])]

    for l3k in l3_keys:



        strategy_name = l3k.replace("level_3_", "")

        display = {"react": "L3-ReAct", "reflexion": "L3-Reflexion", "plan_execute": "L3-PtE"}

        columns.append((display.get(strategy_name, f"L3-{strategy_name}"), [l3k]))





    error_types = set()

    for level_data in error_analysis.values():

        for scenario_data in level_data.values():

            error_types.update(scenario_data.keys())





    error_order = [

        "Correct", "Non-existent Tools", "Solvability Hallucination", "False Unsolvable",

        "Missing Tools", "Incorrect Tools", "Missing Params", "Wrong Param Value",

        "No Answer", "Redundant Steps", "Unknown Error"

    ]

    error_types = (
        [e for e in error_order if e in error_types] +
        [e for e in error_types if e not in error_order]
    )



    def _sum_level(level_keys, error_type):

        total = 0

        for lk in level_keys:

            if lk not in error_analysis:

                continue

            for scenario in ["Solvable", "Unsolvable"]:

                if scenario in error_analysis[lk]:

                    total += error_analysis[lk][scenario].get(error_type, 0)

        return total



    rows = []

    for error_type in error_types:

        row = [error_type]

        for _, keys in columns:

            total = _sum_level(keys, error_type)

            row.append(total if total > 0 else "-")

        rows.append(row)





    rows = [row for row in rows if any(cell != "-" for cell in row[1:])]



    headers = ["Error Type"] + [name for name, _ in columns]

    return tabulate(rows, headers=headers, tablefmt='grid')





def merge_all_levels(eval_dir, model_name, calculate_type):

    summary = {

        "model": model_name,

        "calculate_type": calculate_type,

        "timestamp": datetime.now().isoformat(),

        "overall": {},

        "by_scenario": {},

        "by_subtask": {},

        "by_domain": {},

        "by_domain_group": {},

        "error_analysis": {},

        "level3_strategies": []

    }





    for level in [1, 2]:

        eval_data = load_evaluation_results(eval_dir, level, model_name, calculate_type)

        raw_metrics = load_raw_metrics(eval_dir, level, model_name, calculate_type)



        if not eval_data:

            continue



        level_key = f"level_{level}"

        group_results = eval_data.get("group_results", {})

        sub_task_results = eval_data.get("sub_task_results", {})

        analysis_results = eval_data.get("analysis_results", {})



        if level == 1:

            overall_data = group_results.get("overall", {})

            overall_score = overall_data.get("Level 1 Exact Match", 0)

            summary["overall"]["level_1"] = {

                "exact_match": overall_score,

                "hallucination_rate": overall_data.get("Level 1 Hallucination Rate", 0),

                "risk_adjusted_score": overall_data.get("Level 1 Risk-Adjusted Score", 0),

            }



        elif level == 2:

            overall_data = group_results.get("overall", {})

            all_tool_soft = overall_data.get("Level 2 Tool Score (Soft)", 0)

            all_param = overall_data.get("Level 2 Param Score", 0)

            all_task_success_soft = overall_data.get("Level 2 Task Success (Soft)", 0)



            summary["overall"]["level_2"] = {

                "tool_score_soft": all_tool_soft,

                "param_score": all_param,

                "task_success_soft": all_task_success_soft,

                "planning_precision": overall_data.get("Level 2 Planning Precision", 0),

            }





        for scenario in ["Solvable", "Unsolvable"]:

            if scenario not in summary["by_scenario"]:

                summary["by_scenario"][scenario] = {}

            scenario_data = group_results.get(scenario, {})

            if level == 1:

                summary["by_scenario"][scenario]["level_1"] = {

                    "exact_match": scenario_data.get("Level 1 Exact Match", 0),

                    "hallucination_rate": scenario_data.get("Level 1 Hallucination Rate", 0),

                    "risk_adjusted_score": scenario_data.get("Level 1 Risk-Adjusted Score", 0),

                }

            elif level == 2:

                summary["by_scenario"][scenario]["level_2"] = {

                    "tool_score_soft": scenario_data.get("Level 2 Tool Score (Soft)", 0),

                    "param_score": scenario_data.get("Level 2 Param Score", 0),

                    "task_success_soft": scenario_data.get("Level 2 Task Success (Soft)", 0),

                    "planning_precision": scenario_data.get("Level 2 Planning Precision", 0),

                }





        for subtask, subtask_data in sub_task_results.items():

            if subtask not in summary["by_subtask"]:

                summary["by_subtask"][subtask] = {}

            if level == 1 and "level_1" in subtask_data:

                val = subtask_data["level_1"]

                if isinstance(val, (int, float)):



                    summary["by_subtask"][subtask]["level_1"] = {

                        "exact_match": val,

                        "hallucination_rate": 0,

                        "risk_adjusted_score": 0,

                    }

                elif isinstance(val, dict):

                    summary["by_subtask"][subtask]["level_1"] = {

                        "exact_match": val.get("exact_match", 0),

                        "hallucination_rate": val.get("hallucination_rate", 0),

                        "risk_adjusted_score": val.get("risk_adjusted_score", 0),

                    }

                else:

                    summary["by_subtask"][subtask]["level_1"] = {

                        "exact_match": 0,

                        "hallucination_rate": 0,

                        "risk_adjusted_score": 0,

                    }

            elif level == 2 and "level_2" in subtask_data:

                val = subtask_data["level_2"]

                if isinstance(val, dict):

                    summary["by_subtask"][subtask]["level_2"] = {

                        "tool_score_soft": val.get("tool_score_soft", 0),

                        "param_score": val.get("param_score", 0),

                        "task_success_soft": val.get("task_success_soft", 0),

                        "planning_precision": val.get("planning_precision", 0),

                    }





        if analysis_results and level == 2:

            summary["error_analysis"]["level_2"] = {}

            for key, value in analysis_results.items():

                if "Level 2" in key or "LeveL 2" in key:

                    scenario = key.split(" / ")[-1] if " / " in key else "Overall"

                    if scenario == "Planning Hallucination":

                        continue

                    summary["error_analysis"]["level_2"][scenario] = value



            if "Level 2 / Planning Hallucination" in analysis_results:

                summary["planning_hallucination"] = analysis_results["Level 2 / Planning Hallucination"]





    detected_strategies = []

    for strategy in ["react", "reflexion", "plan_execute"]:

        eval_data = load_evaluation_results(eval_dir, 3, model_name, calculate_type, strategy=strategy)

        if eval_data:

            detected_strategies.append(strategy)



    if not detected_strategies:

        eval_data = load_evaluation_results(eval_dir, 3, model_name, calculate_type)

        if eval_data:

            detected_strategies.append("react")



    summary["level3_strategies"] = detected_strategies



    for strategy in detected_strategies:

        eval_data = load_evaluation_results(eval_dir, 3, model_name, calculate_type, strategy=strategy)

        raw_metrics = load_raw_metrics(eval_dir, 3, model_name, calculate_type, strategy=strategy)



        if not eval_data:

            eval_data = load_evaluation_results(eval_dir, 3, model_name, calculate_type)

            raw_metrics = load_raw_metrics(eval_dir, 3, model_name, calculate_type)

        if not eval_data:

            continue



        group_results = eval_data.get("group_results", {})

        sub_task_results = eval_data.get("sub_task_results", {})

        analysis_results = eval_data.get("analysis_results", {})

        overall_data = group_results.get("overall", {})





        l3_key = f"level_3_{strategy}"

        l3_data = {

            "strategy": strategy,

            "task_success_soft": overall_data.get("Level 3 Task Success (Soft)", 0),

            "tool_call_exact_soft": overall_data.get("Level 3 Tool Exact (Soft)", 0),

            "parameter_accuracy": overall_data.get("Level 3 Parameter Accuracy", 0),

            "step_efficiency_soft": overall_data.get("Level 3 Step Efficiency (Soft)", 0),

            "agent_capability_score": overall_data.get("Level 3 Agent Capability Score", 0),

            "effective_step_utilization": overall_data.get("Level 3 Effective Step Utilization", 0),

        }



        if "Level 3 Self-Correction Rate" in overall_data:

            l3_data["self_correction_rate"] = overall_data.get("Level 3 Self-Correction Rate", 0)



        if "Level 3 Plan Quality" in overall_data:

            l3_data["plan_quality"] = overall_data["Level 3 Plan Quality"]

            l3_data["plan_faithfulness"] = overall_data.get("Level 3 Plan Faithfulness", 0)

            l3_data["execution_accuracy"] = overall_data.get("Level 3 Execution Accuracy", 0)

            l3_data["param_grounding"] = overall_data.get("Level 3 Param Grounding", 0)



        token_usage = overall_data.get("token_usage")

        if isinstance(token_usage, dict):

            l3_data["token_usage"] = token_usage



        summary["overall"][l3_key] = l3_data





        for scenario in ["Solvable", "Unsolvable"]:

            if scenario not in summary["by_scenario"]:

                summary["by_scenario"][scenario] = {}

            scenario_data = group_results.get(scenario, {})

            summary["by_scenario"][scenario][l3_key] = {

                "strategy": strategy,

                "task_success_soft": scenario_data.get("Level 3 Task Success (Soft)", 0),

                "tool_call_exact_soft": scenario_data.get("Level 3 Tool Exact (Soft)", 0),

                "parameter_accuracy": scenario_data.get("Level 3 Parameter Accuracy", 0),

                "step_efficiency_soft": scenario_data.get("Level 3 Step Efficiency (Soft)", 0),

                "agent_capability_score": scenario_data.get("Level 3 Agent Capability Score", 0),

                "effective_step_utilization": scenario_data.get("Level 3 Effective Step Utilization", 0),

            }





        for subtask, subtask_data in sub_task_results.items():

            if subtask not in summary["by_subtask"]:

                summary["by_subtask"][subtask] = {}

            if "level_3" in subtask_data:

                val = subtask_data["level_3"]

                if isinstance(val, dict):

                    summary["by_subtask"][subtask][l3_key] = {

                        "strategy": strategy,

                        "task_success_soft": val.get("task_success_soft", 0),

                        "tool_call_exact_soft": val.get("tool_call_exact_soft", 0),

                        "parameter_accuracy": val.get("parameter_accuracy", 0),

                        "step_efficiency_soft": val.get("step_efficiency_soft", 0),

                        "agent_capability_score": val.get("agent_capability_score", 0),

                        "effective_step_utilization": val.get("effective_step_utilization", 0),

                    }





        if analysis_results:

            summary["error_analysis"][l3_key] = {}

            for key, value in analysis_results.items():

                if "Level 3" in key or "LeveL 3" in key:

                    scenario_name = key.split(" / ")[-1] if " / " in key else "Overall"

                    if scenario_name == "Execution Hallucination":

                        continue

                    summary["error_analysis"][l3_key][scenario_name] = value



            if "Level 3 / Execution Hallucination" in analysis_results:

                summary["execution_hallucination"] = analysis_results["Level 3 / Execution Hallucination"]







    _l3_common_keys = [

        "task_success_soft", "tool_call_exact_soft", "parameter_accuracy",

        "step_efficiency_soft", "agent_capability_score", "effective_step_utilization",

    ]

    if detected_strategies:



        l3_values = [summary["overall"][f"level_3_{s}"] for s in detected_strategies if f"level_3_{s}" in summary["overall"]]

        if l3_values:

            summary["overall"]["level_3"] = {

                k: np.mean([v.get(k, 0) for v in l3_values]) for k in _l3_common_keys

            }





    l1_overall = summary["overall"].get("level_1", {})

    l2_overall = summary["overall"].get("level_2", {})

    l3_overall = summary["overall"].get("level_3", {})

    if l1_overall and l2_overall and l3_overall:

        if isinstance(l1_overall, (int, float)):

            l1_overall = {"exact_match": l1_overall, "risk_adjusted_score": 0}

        summary["overall"]["gae"] = _compute_gae(

            l1_overall.get("risk_adjusted_score", 0),

            l2_overall.get("task_success_soft", 0),

            l3_overall.get("task_success_soft", 0),

            l3_overall.get("effective_step_utilization", 0),

            l2_overall.get("planning_precision", 0),

        )



        s2 = l2_overall.get("task_success_soft", 0)

        s3 = l3_overall.get("task_success_soft", 0)

        summary["overall"]["gap_score"] = (s3 - s2) / s2 if s2 > 0 else 0.0



        summary["overall"]["gap2_score"] = s3 - s2





    if detected_strategies:

        for scenario in ["Solvable", "Unsolvable"]:

            sc = summary["by_scenario"].get(scenario, {})

            sc_values = [sc[f"level_3_{s}"] for s in detected_strategies if f"level_3_{s}" in sc]

            if sc_values:

                sc["level_3"] = {

                    k: np.mean([v.get(k, 0) for v in sc_values]) for k in _l3_common_keys

                }



        for subtask, st_data in summary["by_subtask"].items():

            st_values = [st_data[f"level_3_{s}"] for s in detected_strategies if f"level_3_{s}" in st_data]

            if st_values:

                st_data["level_3"] = {

                    k: np.mean([v.get(k, 0) for v in st_values]) for k in _l3_common_keys

                }





    for level in [1, 2]:

        raw_metrics = load_raw_metrics(eval_dir, level, model_name, calculate_type)

        if not raw_metrics:

            continue



        level_key = f"level_{level}"





        domain_samples = defaultdict(lambda: defaultdict(list))



        for subtask, subtask_data in raw_metrics.items():

            if level_key not in subtask_data:

                continue

            for sample in subtask_data[level_key]:

                domain = sample.get("domain", "unknown")

                domain_samples[domain][level_key].append(sample)





        for domain, levels_data in domain_samples.items():

            if domain not in summary["by_domain"]:

                summary["by_domain"][domain] = {"sample_count": 0}



            samples = levels_data.get(level_key, [])

            if not samples:

                continue



            summary["by_domain"][domain]["sample_count"] += len(samples)



            if level == 1:

                summary["by_domain"][domain]["level_1"] = {

                    "exact_match": np.mean([s.get("metric", 0) for s in samples]),

                    "hallucination_rate": np.mean([s.get("hallucination_rate", 0) for s in samples]),

                    "risk_adjusted_score": np.mean([s.get("risk_adjusted_score", 0) for s in samples]),

                }

            elif level == 2:

                summary["by_domain"][domain]["level_2"] = {

                    "tool_score_soft": np.mean([s.get("tool_score_soft", 0) for s in samples]),

                    "param_score": np.mean([s.get("param_score", 0) for s in samples]),

                    "task_success_soft": np.mean([s.get("task_success_soft", 0) for s in samples]),

                    "planning_precision": np.mean([s.get("planning_precision", 0.0) for s in samples]),

                }





    if detected_strategies:



        all_strategy_domain_metrics = defaultdict(list)



        for strat in detected_strategies:

            raw_metrics = load_raw_metrics(eval_dir, 3, model_name, calculate_type, strategy=strat)

            if not raw_metrics:

                raw_metrics = load_raw_metrics(eval_dir, 3, model_name, calculate_type)

            if not raw_metrics:

                continue



            domain_samples_strat = defaultdict(list)

            for subtask, subtask_data in raw_metrics.items():

                if "level_3" not in subtask_data:

                    continue

                for sample in subtask_data["level_3"]:

                    domain = sample.get("domain", "unknown")

                    domain_samples_strat[domain].append(sample)



            for domain, samples in domain_samples_strat.items():

                if not samples:

                    continue

                strat_metrics = {

                    "task_success_soft": np.mean([s.get("task_success_soft", 0) for s in samples]),

                    "tool_call_exact_soft": np.mean([s.get("tool_call_exact_soft", 0) for s in samples]),

                    "parameter_accuracy": np.mean([s.get("parameter_accuracy", 0) for s in samples]),

                    "step_efficiency_soft": np.mean([s.get("step_efficiency_soft", 0) for s in samples]),

                    "agent_capability_score": np.mean([s.get("agent_capability_score", 0) for s in samples]),

                    "effective_step_utilization": np.mean([s.get("effective_step_utilization", 0) for s in samples]),

                    "sample_count": len(samples),

                }

                all_strategy_domain_metrics[domain].append(strat_metrics)





        for domain, strat_list in all_strategy_domain_metrics.items():

            if domain not in summary["by_domain"]:

                summary["by_domain"][domain] = {"sample_count": 0}



            summary["by_domain"][domain]["sample_count"] += sum(m["sample_count"] for m in strat_list)

            summary["by_domain"][domain]["level_3"] = {

                k: np.mean([m[k] for m in strat_list]) for k in _l3_common_keys

            }





    for group_name, domains in domain_groups_config.items():

        group_data = {"sample_count": 0}

        level_aggregates = {"level_1": [], "level_2": [], "level_3": []}



        for domain in domains:

            if domain not in summary["by_domain"]:

                continue

            domain_data = summary["by_domain"][domain]

            group_data["sample_count"] += domain_data.get("sample_count", 0)



            for lv in ["level_1", "level_2", "level_3"]:

                if lv in domain_data:

                    level_aggregates[lv].append(domain_data[lv])



        if group_data["sample_count"] == 0:

            continue





        if level_aggregates["level_1"]:

            group_data["level_1"] = {

                "exact_match": np.mean([d.get("exact_match", 0) for d in level_aggregates["level_1"]]),

                "hallucination_rate": np.mean([d.get("hallucination_rate", 0) for d in level_aggregates["level_1"]]),

                "risk_adjusted_score": np.mean([d.get("risk_adjusted_score", 0) for d in level_aggregates["level_1"]]),

            }

        if level_aggregates["level_2"]:

            group_data["level_2"] = {

                "tool_score_soft": np.mean([d["tool_score_soft"] for d in level_aggregates["level_2"]]),

                "param_score": np.mean([d["param_score"] for d in level_aggregates["level_2"]]),

                "task_success_soft": np.mean([d["task_success_soft"] for d in level_aggregates["level_2"]]),

            }

        if level_aggregates["level_3"]:

            group_data["level_3"] = {

                "task_success_soft": np.mean([d["task_success_soft"] for d in level_aggregates["level_3"]]),

                "tool_call_exact_soft": np.mean([d["tool_call_exact_soft"] for d in level_aggregates["level_3"]]),

                "parameter_accuracy": np.mean([d["parameter_accuracy"] for d in level_aggregates["level_3"]]),

                "step_efficiency_soft": np.mean([d["step_efficiency_soft"] for d in level_aggregates["level_3"]]),

                "agent_capability_score": np.mean([d.get("agent_capability_score", 0) for d in level_aggregates["level_3"]]),

                "effective_step_utilization": np.mean([d.get("effective_step_utilization", 0) for d in level_aggregates["level_3"]]),

            }



        summary["by_domain_group"][group_name] = group_data



    return summary





def generate_summary_report(summary, model_name, calculate_type):

    lines = []





    lines.append("=" * 80)

    lines.append("                    ARES EVALUATION SUMMARY REPORT")

    lines.append("=" * 80)

    lines.append(f"Model:          {model_name}")

    lines.append(f"Calculate Type: {calculate_type}")

    lines.append(f"Generated:      {summary['timestamp']}")

    lines.append("=" * 80)

    lines.append("")





    lines.append("-" * 80)

    lines.append("1. OVERALL PERFORMANCE")

    lines.append("-" * 80)

    lines.append(generate_overall_summary_horizontal(summary["overall"], model_name))

    lines.append("")





    strategies = summary.get("level3_strategies", [])

    if len(strategies) > 1:

        lines.append("-" * 80)

        lines.append("1.5 LEVEL-3 STRATEGY COMPARISON")

        lines.append("-" * 80)

        lines.append(generate_level3_strategy_comparison(summary["overall"], strategies))

        lines.append("")





    lines.append("-" * 80)

    lines.append("2. PERFORMANCE BY SCENARIO")

    lines.append("-" * 80)

    lines.append(generate_scenario_table(summary["by_scenario"]))

    lines.append("")





    lines.append("-" * 80)

    lines.append("3. PERFORMANCE BY SUBTASK")

    lines.append("-" * 80)

    lines.append(generate_subtask_table(summary["by_subtask"]))

    lines.append("")





    if summary["error_analysis"]:

        lines.append("-" * 80)

        lines.append("4. ERROR ANALYSIS")

        lines.append("-" * 80)

        lines.append(generate_error_analysis_table(summary["error_analysis"]))

        lines.append("")





    ph = summary.get("planning_hallucination")

    if ph:

        lines.append("-" * 80)

        lines.append("5. PLANNING HALLUCINATION ANALYSIS (Level-2)")

        lines.append("-" * 80)

        lines.append(f"  Tool Fabrication Rate (TFR):      {ph.get('tool_fabrication_rate', 0):.2%}")

        lines.append(f"  Solvability Blindness Rate (SBR): {ph.get('solvability_blindness_rate', 0):.2%}")

        lines.append(f"  False Unsolvable Rate (FUR):      {ph.get('false_unsolvable_rate', 0):.2%}")

        lines.append(f"  Redundant Step Rate (RSR):        {ph.get('redundant_step_rate', 0):.2%}")

        counts = ph.get("counts", {})

        lines.append(f"  (Fabrications: {counts.get('fabrication', 0)}, "

                     f"Solvability Hallu: {counts.get('solvability_hallu', 0)}/{counts.get('unsolvable_total', 0)}, "

                     f"False Unsolvable: {counts.get('false_unsolvable', 0)}/{counts.get('solvable_total', 0)}, "

                     f"Redundant Steps: {counts.get('redundant_steps', 0)}/{counts.get('success_total', 0)})")

        lines.append("")





    eh = summary.get("execution_hallucination")

    if eh:

        lines.append("-" * 80)

        lines.append("6. EXECUTION HALLUCINATION ANALYSIS (Level-3)")

        lines.append("-" * 80)

        lines.append(f"  Tool Fabrication Rate (TFR):          {eh.get('tool_fabrication_rate', 0):.2%}")

        lines.append(f"  Missing Tool Rate (MTR):              {eh.get('missing_tool_rate', 0):.2%}")

        lines.append(f"  Completion Hallucination Rate (CHR):  {eh.get('completion_hallucination_rate', 0):.2%}")

        lines.append(f"  Extra Step Rate (ESR):                {eh.get('extra_step_rate', 0):.2%}")

        counts = eh.get("counts", {})

        lines.append(f"  (Fabrications: {counts.get('fabrication', 0)}, "

                     f"Missing Tools: {counts.get('missing_tools', 0)}, "

                     f"Not Finished: {counts.get('not_finished', 0)}, "

                     f"Extra Steps: {counts.get('correct_with_extra_steps', 0)}/{counts.get('success_total', 0)})")

        lines.append("")



    lines.append("=" * 80)

    lines.append("                           END OF REPORT")

    lines.append("=" * 80)



    return "\n".join(lines)





def main():

    parser = argparse.ArgumentParser(description="Generate summary report for Ares evaluation")

    parser.add_argument("--eval_dir", type=str, required=True, help="Base evaluation directory")

    parser.add_argument("--model", type=str, default=None, help="Model name")

    parser.add_argument("--calculate_type", type=str, default="hard", help="Calculate type (hard/soft)")

    parser.add_argument("--leaderboard", action="store_true", help="Generate cross-model leaderboard only")

    args = parser.parse_args()



    if args.leaderboard:

        if args.calculate_type == "all":

            for ct in ["hard", "soft"]:

                generate_leaderboard(args.eval_dir, ct)

        else:

            generate_leaderboard(args.eval_dir, args.calculate_type)

        return



    if not args.model:

        parser.error("--model is required when not using --leaderboard")





    calc_types = ["hard", "soft"] if args.calculate_type == "all" else [args.calculate_type]



    for calc_type in calc_types:

        print(f"Generating summary report for {args.model} ({calc_type})...")

        _generate_model_summary(args.eval_dir, args.model, calc_type)





def _generate_model_summary(eval_dir, model_name, calculate_type):



    summary = merge_all_levels(eval_dir, model_name, calculate_type)





    output_dir = os.path.join(eval_dir, "eval_all")

    os.makedirs(output_dir, exist_ok=True)



    suffix = f"{model_name}_{calculate_type}"





    overall_table = generate_overall_summary_horizontal(summary["overall"], model_name)

    overall_path = os.path.join(output_dir, f"1_Overall_Performance_{suffix}.txt")

    with open(overall_path, 'w', encoding='utf-8') as f:

        f.write("=" * 80 + "\n")

        f.write("OVERALL PERFORMANCE\n")

        f.write("=" * 80 + "\n\n")

        f.write(overall_table)

    print(f"Saved: {overall_path}")





    strategies = summary.get("level3_strategies", [])

    if len(strategies) > 1:

        strategy_table = generate_level3_strategy_comparison(summary["overall"], strategies)

        strategy_path = os.path.join(output_dir, f"1.5_Level3_Strategy_Comparison_{suffix}.txt")

        with open(strategy_path, 'w', encoding='utf-8') as f:

            f.write("=" * 80 + "\n")

            f.write("LEVEL-3 STRATEGY COMPARISON\n")

            f.write("=" * 80 + "\n\n")

            f.write(strategy_table)

        print(f"Saved: {strategy_path}")





    scenario_table = generate_scenario_table(summary["by_scenario"])

    scenario_path = os.path.join(output_dir, f"2_Performance_By_Scenario_{suffix}.txt")

    with open(scenario_path, 'w', encoding='utf-8') as f:

        f.write("=" * 80 + "\n")

        f.write("PERFORMANCE BY SCENARIO\n")

        f.write("=" * 80 + "\n\n")

        f.write(scenario_table)

    print(f"Saved: {scenario_path}")





    subtask_table = generate_subtask_table(summary["by_subtask"])

    subtask_path = os.path.join(output_dir, f"3_Performance_By_Subtask_{suffix}.txt")

    with open(subtask_path, 'w', encoding='utf-8') as f:

        f.write("=" * 80 + "\n")

        f.write("PERFORMANCE BY SUBTASK\n")

        f.write("=" * 80 + "\n\n")

        f.write(subtask_table)

    print(f"Saved: {subtask_path}")





    if summary["error_analysis"]:

        error_table = generate_error_analysis_table(summary["error_analysis"])

        error_path = os.path.join(output_dir, f"4_Error_Analysis_{suffix}.txt")

        with open(error_path, 'w', encoding='utf-8') as f:

            f.write("=" * 80 + "\n")

            f.write("ERROR ANALYSIS\n")

            f.write("=" * 80 + "\n\n")

            f.write(error_table)

        print(f"Saved: {error_path}")





    if summary["by_domain_group"]:

        domain_group_table = generate_domain_group_table(summary["by_domain_group"])

        domain_group_path = os.path.join(output_dir, f"5_Performance_By_Domain_Group_{suffix}.txt")

        with open(domain_group_path, 'w', encoding='utf-8') as f:

            f.write("=" * 80 + "\n")

            f.write("PERFORMANCE BY DOMAIN GROUP (7 Categories)\n")

            f.write("=" * 80 + "\n\n")

            f.write(domain_group_table)

        print(f"Saved: {domain_group_path}")





    if summary["by_domain"]:

        domain_table = generate_domain_table(summary["by_domain"])

        domain_path = os.path.join(output_dir, f"6_Performance_By_Domain_{suffix}.txt")

        with open(domain_path, 'w', encoding='utf-8') as f:

            f.write("=" * 80 + "\n")

            f.write("PERFORMANCE BY DOMAIN (20 Domains)\n")

            f.write("=" * 80 + "\n\n")

            f.write(domain_table)

        print(f"Saved: {domain_path}")





    json_path = os.path.join(output_dir, f"Summary_All_Levels_{suffix}.json")

    with open(json_path, 'w', encoding='utf-8') as f:

        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved: {json_path}")





    print("")

    print("=" * 80)

    print("                    ARES EVALUATION SUMMARY")

    print("=" * 80)

    print(f"Model:          {model_name}")

    print(f"Calculate Type: {calculate_type}")

    print(f"Output Dir:     {output_dir}")

    print("=" * 80)

    print("")

    print(overall_table)



    if len(summary.get("level3_strategies", [])) > 1:

        print("")

        print("Level-3 Strategy Comparison:")

        print(generate_level3_strategy_comparison(summary["overall"], summary["level3_strategies"]))

    print("")

    print(f"Detailed tables saved to: {output_dir}/")









_LEADERBOARD_HEADERS = [

    "Model",

    "GAE",

    "L1 HR", "L1 RAS",

    "S2", "L2 PPR",

    "S3", "L3 PPR",

    "GAP", "GAP2",

]



_LEADERBOARD_METRIC_MAP = {

    "L1 HR":   ("level_1", "hallucination_rate"),

    "L1 RAS":  ("level_1", "risk_adjusted_score"),

    "S2":      ("level_2", "task_success_soft"),

    "L2 PPR":  ("level_2", "planning_precision"),

    "S3":      ("level_3", "task_success_soft"),

    "L3 PPR":  ("level_3", "effective_step_utilization"),

}





def generate_leaderboard(results_dir, calculate_type="soft"):

    pattern = os.path.join(results_dir, "eval_*", "eval_all",

                           f"Summary_All_Levels_*_{calculate_type}.json")

    rows = []

    for filepath in sorted(glob.glob(pattern)):

        with open(filepath, "r", encoding="utf-8") as f:

            summary = json.load(f)

        model_name = summary.get("model", os.path.basename(filepath))

        overall = summary.get("overall", {})

        row_values = {}

        for header, (section, key) in _LEADERBOARD_METRIC_MAP.items():

            section_data = overall.get(section, {})

            row_values[header] = section_data.get(key, 0)



        l1 = overall.get("level_1", {})

        l2 = overall.get("level_2", {})

        l3 = overall.get("level_3", {})

        if l1 and l2 and l3:

            row_values["GAE"] = _compute_gae(

                l1.get("risk_adjusted_score", 0),

                l2.get("task_success_soft", 0),

                l3.get("task_success_soft", 0),

                l3.get("effective_step_utilization", 0),

                l2.get("planning_precision", 0),

            )

        else:

            row_values["GAE"] = 0



        s2 = row_values.get("S2", 0)

        s3 = row_values.get("S3", 0)

        row_values["GAP"] = (s3 - s2) / s2 if s2 > 0 else 0.0



        row_values["GAP2"] = s3 - s2

        rows.append((model_name, row_values))



    if not rows:

        print(f"No model Summary JSON found in: {results_dir}")

        return





    rows.sort(key=lambda r: r[1].get("GAE", 0), reverse=True)



    table_rows = []

    for model_name, values in rows:

        row = [model_name]

        for header in _LEADERBOARD_HEADERS[1:]:

            val = values.get(header, 0)

            if header in ("GAP", "GAP2"):

                row.append(f"{val:+.2%}")

            else:

                row.append(f"{val:.2%}")

        table_rows.append(row)



    table_str = tabulate(table_rows, headers=_LEADERBOARD_HEADERS, tablefmt="grid")



    output_path = os.path.join(results_dir, f"Leaderboard_{calculate_type}.txt")

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")

        f.write("ALL MODELS LEADERBOARD\n")

        f.write(f"Sorted by: GAE (descending) | Models: {len(rows)}\n")

        f.write("=" * 80 + "\n\n")

        f.write(table_str + "\n")

    print(f"Leaderboard saved: {output_path}")





if __name__ == "__main__":

    main()

