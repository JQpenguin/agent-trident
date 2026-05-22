import re



def remove_finish_if_last(tools):

    if tools and extract_tool_name_from_call(tools[-1]).lower() == 'finish':

        return tools[:-1]

    return tools





def split_concatenated_tool_calls(line):

    if not line or not line.strip():

        return []



    line = line.strip()

    calls = []

    pos = 0



    while pos < len(line):



        while pos < len(line) and line[pos] in (' ', '\t'):

            pos += 1

        if pos >= len(line):

            break





        func_match = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', line[pos:])

        if not func_match:



            remainder = line[pos:].strip()

            if remainder:

                calls.append(remainder)

            break





        paren_start = pos + func_match.end() - 1

        depth = 0

        in_string = False

        string_char = None

        escape_next = False

        found_end = False



        for i in range(paren_start, len(line)):

            if escape_next:

                escape_next = False

                continue



            ch = line[i]



            if ch == '\\':

                escape_next = True

                continue



            if ch in ('"', "'") and not in_string:

                in_string = True

                string_char = ch

            elif in_string and ch == string_char:

                in_string = False

                string_char = None

            elif not in_string:

                if ch == '(':

                    depth += 1

                elif ch == ')':

                    depth -= 1

                    if depth == 0:

                        calls.append(line[pos:i + 1].strip())

                        pos = i + 1

                        found_end = True

                        break



        if not found_end:



            remainder = line[pos:].strip()

            if remainder:

                calls.append(remainder)

            break



    return calls





def extract_all_answer_blocks(response, answer_pattern=r"<answer>(.*?)</answer>"):

    if not response:

        return []





    matches = re.findall(answer_pattern, response, re.DOTALL)

    if not matches:

        return []



    all_calls = []

    for block_content in matches:



        lines = [l.strip() for l in block_content.strip().split("\n") if l.strip()]

        for line in lines:



            line = re.sub(r'^\d+[\.\)]\s*', '', line)

            line = re.sub(r'^[-\*]\s+', '', line)



            split_calls = split_concatenated_tool_calls(line)

            all_calls.extend(split_calls)



    return all_calls





def _extract_subgoal_plantool_del_lastfinish(output):

    pattern = r"(Subgoal \d+[:：]\s*.*?)(?= Planned tool[:：]) Planned tool[:：]\s*([\w\s\+\-]+)(?=\n|$)"

    matches = re.findall(pattern, output)



    subgoal_texts, planned_tools = [], []

    for match in matches:

        subgoal_text, planned_tool = match

        subgoal_texts.append(subgoal_text.strip())

        planned_tools.append(planned_tool.strip())



    if planned_tools and planned_tools[-1].lower() == 'finish':

        planned_tools = planned_tools[:-1]

        subgoal_texts = subgoal_texts[:-1]



    return subgoal_texts, planned_tools









def extract_tool_name_from_call(tool_call_str):

    if not tool_call_str:

        return ""



    tool_call_str = tool_call_str.strip()





    match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', tool_call_str)

    if match:

        return match.group(1).strip()





    return tool_call_str.strip()





def extract_tool_name_from_definition(tool_def_str):

    if not tool_def_str:

        return ""



    tool_def_str = tool_def_str.strip()





    match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', tool_def_str)

    if match:

        return match.group(1).strip()





    match = re.match(r'^(?:\d+\.\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*[:：]', tool_def_str)

    if match:

        return match.group(1).strip()



    return tool_def_str.strip()





def _extract_balanced_paren_content(text):

    start = text.find('(')

    if start == -1:

        return None



    depth = 0

    in_string = False

    string_char = None

    escape_next = False



    for i in range(start, len(text)):

        if escape_next:

            escape_next = False

            continue

        ch = text[i]

        if ch == '\\':

            escape_next = True

            continue

        if ch in ('"', "'") and not in_string:

            in_string = True

            string_char = ch

        elif in_string and ch == string_char:

            in_string = False

            string_char = None

        elif not in_string:

            if ch == '(':

                depth += 1

            elif ch == ')':

                depth -= 1

                if depth == 0:

                    return text[start + 1:i]





    match = re.search(r'\(([^)]*)\)', text)

    return match.group(1) if match else None





def extract_params_from_call(tool_call_str):

    if not tool_call_str:

        return {}





    params_str = _extract_balanced_paren_content(tool_call_str)

    if params_str is None:

        return {}



    params_str = params_str.strip()

    if not params_str:

        return {}



    params = {}





    param_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\)]+))"

    for m in re.finditer(param_pattern, params_str):

        key = m.group(1)



        value = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))

        if value is not None:

            params[key] = value.strip()



    return params





def normalize_param_value(val):

    return str(val).strip().strip("'\"")





def is_step_output_placeholder(value):

    if not value:

        return False

    return bool(re.match(r'^Step_\d+_Output$', str(value).strip()))





def extract_step_number_from_placeholder(value):

    if not value:

        return None

    match = re.match(r'^Step_(\d+)_Output$', str(value).strip())

    if match:

        return int(match.group(1))

    return None





def extract_tool_names_list(tool_calls):

    return [extract_tool_name_from_call(tc) for tc in tool_calls]
