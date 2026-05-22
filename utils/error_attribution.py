

from collections import Counter, defaultdict

from tabulate import tabulate

import numpy as np















CONDITION_TO_CATEGORY = {



    "non_existent_tools": ("Tool Selection Error", "Fabrication"),

    "missing_tools":      ("Tool Selection Error", "Tool Mismatch"),

    "incorrect_tools":    ("Tool Selection Error", "Tool Mismatch"),

    "wrong_tools":        ("Tool Selection Error", "Tool Mismatch"),

    "wrong_tool_order":   ("Tool Selection Error", "Tool Mismatch"),





    "missing_params":           ("Tool Usage Error", "Param Error"),

    "wrong_param_value":        ("Tool Usage Error", "Param Error"),

    "tool_correct_param_error": ("Tool Usage Error", "Param Error"),

    "wrong_params":             ("Tool Usage Error", "Param Error"),





    "solvability_hallu": ("Constraint Error", "Solvability Blind"),

    "false_unsolvable":  ("Constraint Error", "False Unsolvable"),





    "redundant_steps":          ("Redundant Steps", "Redundant"),

    "correct_with_extra_steps": ("Redundant Steps", "Redundant"),





    "no_response": ("No Response", "No Response"),

    "no_answer":   ("No Response", "No Response"),

    "not_finished": ("Tool Selection Error", "Tool Mismatch"),





    "correct": ("Correct", "Correct"),





    "unknown_error":    ("Unknown", "Unknown"),

    "wrong_reasoning":  ("Tool Selection Error", "Tool Mismatch"),

    "wrong_unsolvable_index": ("Constraint Error", "Solvability Blind"),

}





CATEGORY_ORDER = [

    "Tool Selection Error",

    "Tool Usage Error",

    "Constraint Error",

    "Redundant Steps",

    "No Response",

    "Unknown",

]





SUBTYPE_ORDER = [

    "Fabrication", "Tool Mismatch",

    "Param Error",

    "Solvability Blind", "False Unsolvable",

    "Redundant",

    "No Response",

    "Unknown",

]





L1_FAILURE_ORDER = ["L1-FP", "L1-FN", "L1-UNC", "L1-NA"]



L1_FAILURE_NAMES = {

    "L1-FP":  "Solvability Hallucination",

    "L1-FN":  "Over-Caution",

    "L1-UNC": "Decision Avoidance",

    "L1-NA":  "Format Failure",

}













def _classify_l1(sample):

    pred_cat = sample.get("pred_category", "no_answer")

    golden = sample.get("golden", "").lower()



    if pred_cat == "confident_right":

        return "L1-TP" if golden == "solvable" else "L1-TN"

    elif pred_cat == "confident_wrong":

        return "L1-FP" if golden == "unsolvable" else "L1-FN"

    elif pred_cat == "uncertain":

        return "L1-UNC"

    else:

        return "L1-NA"





def _classify_l2l3(sample):

    condition = sample.get("condition", "unknown_error")

    return CONDITION_TO_CATEGORY.get(condition, ("Unknown", "Unknown"))















def _flatten_raw(raw, level_key):

    flat = {}

    for subtask, level_data in raw.items():

        if isinstance(level_data, dict) and level_key in level_data:

            flat[subtask] = level_data[level_key]

        elif isinstance(level_data, list):



            flat[subtask] = level_data

    return flat





def run_attribution(raw_metrics_by_level):

    l1_raw = _flatten_raw(raw_metrics_by_level.get("level_1", {}), "level_1")

    l2_raw = _flatten_raw(raw_metrics_by_level.get("level_2", {}), "level_2")

    l3_raw = _flatten_raw(raw_metrics_by_level.get("level_3", {}), "level_3")



    result = {}





    if l1_raw:

        result["l1_attribution"] = _compute_l1_attribution(l1_raw)





    if l2_raw:

        result["l2_attribution"] = _compute_l2l3_attribution(l2_raw, "level_2")





    if l3_raw:

        result["l3_attribution"] = _compute_l2l3_attribution(l3_raw, "level_3")





    if l2_raw:

        result["l2_subtask_matrix"] = _compute_subtask_matrix(l2_raw)

    if l3_raw:

        result["l3_subtask_matrix"] = _compute_subtask_matrix(l3_raw)





    if l2_raw:

        result["l2_solvability"] = _compute_solvability_comparison(l2_raw)

    if l3_raw:

        result["l3_solvability"] = _compute_solvability_comparison(l3_raw)



    return result





def _compute_l1_attribution(l1_raw):

    all_codes = []

    total = 0

    for subtask, samples in l1_raw.items():

        for s in samples:

            code = _classify_l1(s)

            all_codes.append(code)

            total += 1



    counts = Counter(all_codes)

    success = counts.get("L1-TP", 0) + counts.get("L1-TN", 0)

    failures = total - success



    failure_detail = {}

    for code in L1_FAILURE_ORDER:

        c = counts.get(code, 0)

        failure_detail[code] = {

            "count": c,

            "pct_of_failures": round(c / failures * 100, 1) if failures > 0 else 0.0,

        }



    return {

        "total": total,

        "success": success,

        "failures": failures,

        "failure_detail": failure_detail,

        "all_counts": dict(counts),

    }





def _compute_l2l3_attribution(raw, level_key):



    category_counts = Counter()

    total = 0

    correct = 0

    failure_samples = []



    for subtask, samples in raw.items():

        for s in samples:

            total += 1

            cat, sub = _classify_l2l3(s)

            if cat == "Correct":

                correct += 1

                continue

            category_counts[(cat, sub)] += 1

            failure_samples.append(s)



    failures = total - correct





    detail = {}

    for cat in CATEGORY_ORDER:

        cat_subtypes = [(c, st) for (c, st) in category_counts if c == cat]

        if not cat_subtypes:

            continue

        subtypes = {}

        for _, st in sorted(cat_subtypes, key=lambda x: SUBTYPE_ORDER.index(x[1]) if x[1] in SUBTYPE_ORDER else 99):

            c = category_counts[(cat, st)]

            subtypes[st] = {

                "count": c,

                "pct_of_failures": round(c / failures * 100, 1) if failures > 0 else 0.0,

            }

        cat_total = sum(v["count"] for v in subtypes.values())

        detail[cat] = {

            "total": cat_total,

            "pct_of_failures": round(cat_total / failures * 100, 1) if failures > 0 else 0.0,

            "subtypes": subtypes,

        }



    return {

        "total": total,

        "correct": correct,

        "failures": failures,

        "detail": detail,

    }





def _compute_subtask_matrix(raw):

    matrix = {}

    subtask_failures = {}



    for subtask, samples in raw.items():

        cat_counts = Counter()

        fail_count = 0

        for s in samples:

            cat, sub = _classify_l2l3(s)

            if cat == "Correct":

                continue

            cat_counts[cat] += 1

            fail_count += 1



        subtask_failures[subtask] = fail_count

        matrix[subtask] = {}

        for cat in CATEGORY_ORDER:

            c = cat_counts.get(cat, 0)

            matrix[subtask][cat] = {

                "count": c,

                "pct": round(c / fail_count * 100, 1) if fail_count > 0 else 0.0,

            }



    return {"matrix": matrix, "subtask_failures": subtask_failures}





def _compute_solvability_comparison(raw):

    groups = {"solvable": Counter(), "unsolvable": Counter()}

    group_failures = {"solvable": 0, "unsolvable": 0}



    for subtask, samples in raw.items():

        for s in samples:

            golden = s.get("golden", "").lower()

            if golden not in groups:

                continue

            cat, sub = _classify_l2l3(s)

            if cat == "Correct":

                continue

            groups[golden][(cat, sub)] += 1

            group_failures[golden] += 1



    result = {}

    for solv in ["solvable", "unsolvable"]:

        total_f = group_failures[solv]

        detail = {}

        for cat in CATEGORY_ORDER:

            pairs = [(c, st) for (c, st) in groups[solv] if c == cat]

            if not pairs:

                continue

            cat_total = sum(groups[solv][(c, st)] for c, st in pairs)

            detail[cat] = {

                "count": cat_total,

                "pct": round(cat_total / total_f * 100, 1) if total_f > 0 else 0.0,

            }

        result[solv] = {"failures": total_f, "detail": detail}



    return result













def generate_attribution_report(attribution):

    lines = []

    lines.append("=" * 80)

    lines.append("                    DEEP ERROR ATTRIBUTION ANALYSIS")

    lines.append("=" * 80)

    lines.append("")





    for level, key, label in [

        (1, "l1_attribution", "Level-1 Solvability Judgment"),

        (2, "l2_attribution", "Level-2 Planning"),

        (3, "l3_attribution", "Level-3 Execution"),

    ]:

        if key not in attribution:

            continue

        lines.append("-" * 80)

        lines.append(f"TABLE 1-{level}: FAILURE DISTRIBUTION - {label}")

        lines.append("-" * 80)

        if level == 1:

            lines.append(_render_l1_failure_table(attribution[key]))

        else:

            lines.append(_render_l2l3_failure_table(attribution[key], f"L{level}"))

        lines.append("")





    for level, key, label in [

        (2, "l2_subtask_matrix", "Level-2"),

        (3, "l3_subtask_matrix", "Level-3"),

    ]:

        if key not in attribution:

            continue

        lines.append("-" * 80)

        lines.append(f"TABLE 2-{level}: FAILURE x SUBTASK MATRIX - {label}")

        lines.append("-" * 80)

        lines.append(_render_subtask_matrix(attribution[key]))

        lines.append("")





    for level, key, label in [

        (2, "l2_solvability", "Level-2"),

        (3, "l3_solvability", "Level-3"),

    ]:

        if key not in attribution:

            continue

        lines.append("-" * 80)

        lines.append(f"TABLE 3-{level}: SOLVABLE vs UNSOLVABLE - {label}")

        lines.append("-" * 80)

        lines.append(_render_solvability_table(attribution[key]))

        lines.append("")



    lines.append("=" * 80)

    lines.append("                    END OF ERROR ATTRIBUTION")

    lines.append("=" * 80)

    return "\n".join(lines)





def _render_l1_failure_table(l1_attr):

    total = l1_attr["total"]

    failures = l1_attr["failures"]

    detail = l1_attr["failure_detail"]



    rows = []

    for code in L1_FAILURE_ORDER:

        d = detail.get(code, {"count": 0, "pct_of_failures": 0.0})

        rows.append([

            code,

            L1_FAILURE_NAMES.get(code, code),

            d["count"],

            f"{d['pct_of_failures']:.1f}%",

        ])



    header = f"L1 Failures (Failed={failures} / Total={total})\n\n"

    table = tabulate(rows, headers=["Code", "Error Type", "Count", "% of Failures"],

                     tablefmt="grid")

    return header + table





def _render_l2l3_failure_table(attr, level_prefix):

    total = attr["total"]

    failures = attr["failures"]

    detail = attr["detail"]



    rows = []

    for cat in CATEGORY_ORDER:

        if cat not in detail:

            continue

        cat_data = detail[cat]

        subtypes = cat_data["subtypes"]

        first = True

        for st_name, st_data in subtypes.items():

            cat_display = cat if first else ""

            rows.append([

                cat_display,

                st_name,

                st_data["count"],

                f"{st_data['pct_of_failures']:.1f}%",

            ])

            first = False



    header = f"{level_prefix} Failures (Failed={failures} / Total={total})\n\n"

    table = tabulate(rows, headers=["Category", "Sub-type", "Count", "% of Failures"],

                     tablefmt="grid")

    return header + table





def _render_subtask_matrix(matrix_data):

    matrix = matrix_data["matrix"]

    subtask_failures = matrix_data["subtask_failures"]



    if not matrix:

        return "(No data)\n"



    subtasks = sorted(matrix.keys())



    active_cats = [cat for cat in CATEGORY_ORDER

                   if any(matrix[st].get(cat, {}).get("count", 0) > 0 for st in subtasks)]



    if not active_cats:

        return "(No failures)\n"



    rows = []

    for cat in active_cats:

        row = [cat]

        for st in subtasks:

            d = matrix[st].get(cat, {"count": 0, "pct": 0.0})

            if subtask_failures.get(st, 0) > 0:

                row.append(f"{d['pct']:.1f}%")

            else:

                row.append("-")

        rows.append(row)





    total_row = ["TOTAL FAILURES"]

    for st in subtasks:

        total_row.append(str(subtask_failures.get(st, 0)))

    rows.append(total_row)





    display_names = {

        "single_step": "single",

        "multi_step_no_rep": "multi",

        "sequential_chain": "seq_chain",

        "conditional_planning": "cond_plan",

        "os": "os",

        "web": "web",

        "best": "best",

    }

    headers = ["Error Type"] + [display_names.get(st, st) for st in subtasks]

    return tabulate(rows, headers=headers, tablefmt="grid")





def _render_solvability_table(solv_data):



    active_cats = set()

    for solv in ["solvable", "unsolvable"]:

        for cat in solv_data.get(solv, {}).get("detail", {}):

            active_cats.add(cat)



    rows = []

    for cat in CATEGORY_ORDER:

        if cat not in active_cats:

            continue

        row = [cat]

        for solv in ["solvable", "unsolvable"]:

            d = solv_data.get(solv, {}).get("detail", {}).get(cat)

            if d is None:

                row.append("-")

            else:

                row.append(f"{d['pct']:.1f}%")

        rows.append(row)



    s_fail = solv_data.get("solvable", {}).get("failures", 0)

    u_fail = solv_data.get("unsolvable", {}).get("failures", 0)

    headers = ["Error Type", f"Solvable (N={s_fail})", f"Unsolvable (N={u_fail})"]

    return tabulate(rows, headers=headers, tablefmt="grid")

