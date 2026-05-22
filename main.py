import os

import re

import json

import datetime

import time

import argparse

from tqdm import tqdm

from utils.load_save import load_results, save_results, save_table

from utils.calculate_metrics import (calculate_subtask_results, calculate_group_results,

                                     calculate_analysis_results, print_table,

                                     calculate_domain_results, calculate_domain_group_results,

                                     print_domain_table, print_domain_group_table,

                                     calculate_l1_solvability_hallucination,

                                     calculate_l2_planning_hallucination,

                                     calculate_l3_execution_hallucination)

from utils.process import process_all_tasks_eval, process_all_tasks_infer, process_all_tasks_infer_parallel, process_all_tasks_recover, convert_hf_data

from utils.generation_prompt import _detecting_en, _diagnosing_en, _planning_en



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/ares_benchmark.json")

    parser.add_argument("--output_dictory", type=str, default="./results/en")

    parser.add_argument("--results_dictory", type=str, default="./results/20240609/infer_results")

    parser.add_argument("--results_path", type=str, default="./results/20240609/infer_results/results_0608_iter_gpt4o.json")

    parser.add_argument("--embedding_model", type=str, default="minilm", choices=["minilm", "gemini"])



    parser.add_argument("--mode", type=str, default="eval", choices=["infer", "eval", "recover"])

    parser.add_argument("--recover_path", type=str, default="", help="Recover from unfinished results file")

    parser.add_argument("--model_type", type=str, default="vllm", choices=["api", "vllm"])

    parser.add_argument("--api_key", type=str, default="your_api_key")

    parser.add_argument("--model_name", type=str, default="gemini", choices=["gemini", "openai", "deepseek", "kimi", "qwen", "nvidia", "anthropic"])

    parser.add_argument("--deepseek_url", type=str, default="https://api.deepseek.com", help="DeepSeek base URL (optional)")

    parser.add_argument("--kimi_url", type=str, default="https://api.moonshot.cn/v1", help="Kimi/Moonshot base URL")

    parser.add_argument("--qwen_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Qwen/DashScope base URL")

    parser.add_argument("--nvidia_url", type=str, default="https://integrate.api.nvidia.com/v1", help="NVIDIA Build base URL")

    parser.add_argument("--anthropic_url", type=str, default="", help="Optional Anthropic API base URL")

    parser.add_argument("--model", type=str, default="models/gemini-1.5-pro")

    parser.add_argument("--model_name_save", type=str, default="")



    parser.add_argument("--level_1", dest='level_1', action="store_true", help="Enable level 1 (default)")

    parser.add_argument("--no-level_1", dest='level_1', action="store_false", help="Disable level 1")



    parser.add_argument("--calculate_type", type=str, default="hard", choices=["hard", "soft", "all"],

                        help="Calculate metrics for hard, soft, or all calculation levels (default: hard)")



    parser.add_argument("--level_2", dest='level_2', action="store_true", help="Enable level 2 (default)")

    parser.add_argument("--no-level_2", dest='level_2', action="store_false", help="Disable level 2")



    parser.add_argument("--level_3", dest='level_3', action="store_true", help="Enable level 3 (default)")

    parser.add_argument("--no-level_3", dest='level_3', action="store_false", help="Disable level 3")



    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"])

    parser.add_argument("--detecting_prompt", type=str, default=_detecting_en)

    parser.add_argument("--planning_prompt", type=str, default=_planning_en)

    parser.add_argument("--diagnosing_prompt", type=str, default=_diagnosing_en)

    parser.add_argument("--answer_pattern", type=str, default=r"<answer>(.*?)</answer>",

                        help="Pattern to extract answer from the response, depending on the prompt type.")



    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/chat/completions", help="Default for vllm server.")





    parser.add_argument("--parallel", action="store_true", help="Enable parallel inference")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (default: 4)")





    parser.add_argument("--agent_strategy", type=str, default="react", choices=["react", "reflexion", "plan_execute"],

                        help="Agent reasoning strategy for Level-3: 'react' (single-pass), 'reflexion' (self-reflective retry), or 'plan_execute' (plan first, then execute)")

    parser.add_argument("--max_reflexion_trials", type=int, default=2,

                        help="Maximum number of Reflexion retry trials (default: 2, only used when agent_strategy=reflexion)")

    parser.add_argument("--react_results_path", type=str, default=None,

                        help="Path to prior ReAct inference results JSON. When provided with "

                             "agent_strategy=reflexion, the first trial reuses these results "

                             "instead of re-running ReAct from scratch.")



    parser.set_defaults(level_1=True, level_2=True, level_3=True)

    args = parser.parse_args()



    start_time = time.time()



    if args.mode != "eval":

        if args.model_type == "api":

            if args.model_name == 'gemini':

                from utils.generation_func import GeminiGeneration

                generation = GeminiGeneration(api_key=args.api_key, model_name=args.model)

                generation_func = generation.generation_gemini

            elif args.model_name == 'deepseek':

                from utils.generation_func import DeepSeekGeneration

                generation = DeepSeekGeneration(api_key=args.api_key, model=args.model, api_url=args.deepseek_url)

                generation_func = generation.generation_deepseek

            elif args.model_name == 'qwen':

                from utils.generation_func import DeepSeekGeneration

                generation = DeepSeekGeneration(api_key=args.api_key, model=args.model, api_url=args.qwen_url)

                generation_func = generation.generation_deepseek

            elif args.model_name == 'kimi':

                from utils.generation_func import KimiGeneration

                generation = KimiGeneration(api_key=args.api_key, model=args.model, api_url=args.kimi_url)

                generation_func = generation.generation_kimi

            elif args.model_name == 'nvidia':

                from utils.generation_func import NvidiaGeneration

                generation = NvidiaGeneration(api_key=args.api_key, model=args.model, api_url=args.nvidia_url)

                generation_func = generation.generation_nvidia

            elif args.model_name == 'anthropic':

                from utils.generation_func import AnthropicGeneration

                generation = AnthropicGeneration(api_key=args.api_key, model=args.model, api_url=args.anthropic_url or None)

                generation_func = generation.generation_anthropic

            else:

                from utils.generation_func import OpenAIGeneration

                generation = OpenAIGeneration(api_key=args.api_key, model=args.model)

                generation_func = generation.generation_openai



        else:

            from utils.generation_func import VllmGeneration

            generation = VllmGeneration(api_url=args.api_url, model=args.model)

            generation_func = generation.generation_vllm





    if args.model_name_save == "":

        args.model_name_save = args.model.replace("/", "-")



    with open(args.data_path, "r") as f:

        items = json.load(f)

    items = convert_hf_data(items)



    if args.level_2 or args.level_3:

        if args.mode != "infer":

            from utils.generation_func import get_tools_embeddings

            if args.embedding_model == "minilm":

                from sentence_transformers import SentenceTransformer

                args.emb_model = SentenceTransformer("all-MiniLM-L6-v2")

                tools_embedding = get_tools_embeddings(items, args)

                args.tools_embedding = tools_embedding

            elif args.embedding_model == "gemini":

                from utils.generation_func import GeminiEmbedding

                embedding = GeminiEmbedding(api_key=args.api_key)

                embedding_func = embedding.get_embedding_gemini

                args.emb_model = embedding_func

                tools_embedding = get_tools_embeddings(items, args)

                args.tools_embedding = tools_embedding



    if args.mode == "eval" and args.results_dictory is not None:

        import glob

        generation_func = ""

        file_pattern = os.path.join(args.results_dictory, "results*.json")

        results_files = glob.glob(file_pattern)

        calculate_types = ["hard", "soft"] if args.calculate_type == "all" else [args.calculate_type]

        for results_file in tqdm(results_files, desc=f"Evaluating file"):

            args.model_name_save = results_file.split("results_")[-1].split(".json")[0]

            results = load_results(results_file)

            for calculate_type in calculate_types:

                args.calculate_type = calculate_type

                metrics = process_all_tasks_eval(items, args, results)

                os.makedirs(os.path.join(args.output_dictory, f"{args.mode}_results"), exist_ok=True)

                save_results(os.path.join(args.output_dictory, f"{args.mode}_results", f"{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.json"), metrics)



                sub_task_results_dict = calculate_subtask_results(metrics)

                group_results_dict = calculate_group_results(metrics)

                analysis_results_dict = calculate_analysis_results(metrics)





                if getattr(args, 'level_1', True):

                    all_tasks = list(metrics.keys())

                    l1_solv_hallu = calculate_l1_solvability_hallucination(metrics, all_tasks)

                    analysis_results_dict["Level 1 / Solvability Hallucination"] = l1_solv_hallu





                if getattr(args, 'level_2', True):

                    all_tasks = list(metrics.keys())

                    l2_plan_hallu = calculate_l2_planning_hallucination(metrics, all_tasks)

                    analysis_results_dict["Level 2 / Planning Hallucination"] = l2_plan_hallu





                if getattr(args, 'level_3', True):

                    all_tasks = list(metrics.keys())

                    l3_exec_hallu = calculate_l3_execution_hallucination(metrics, all_tasks)

                    analysis_results_dict["Level 3 / Execution Hallucination"] = l3_exec_hallu



                sub_task_table_print, group_table_print, analysis_table_print = print_table(sub_task_results_dict, group_results_dict, analysis_results_dict)

                os.makedirs(os.path.join(args.output_dictory, "table_results"), exist_ok=True)

                save_table(os.path.join(args.output_dictory, "table_results", f"Subtask_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), sub_task_table_print)

                save_table(os.path.join(args.output_dictory, "table_results", f"Scenario_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), group_table_print)

                save_table(os.path.join(args.output_dictory, "table_results", f"Analysis_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), analysis_table_print)





                domain_results_dict = calculate_domain_results(metrics)

                domain_group_results_dict = calculate_domain_group_results(metrics)



                if domain_results_dict:

                    domain_table_print = print_domain_table(domain_results_dict)

                    domain_group_table_print = print_domain_group_table(domain_group_results_dict)

                    save_table(os.path.join(args.output_dictory, "table_results", f"Domain_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), domain_table_print)

                    save_table(os.path.join(args.output_dictory, "table_results", f"DomainGroup_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), domain_group_table_print)



                all_evaluation_results = {

                    "sub_task_results": sub_task_results_dict,

                    "group_results": group_results_dict,

                    "analysis_results": analysis_results_dict,

                    "domain_results": domain_results_dict,

                    "domain_group_results": domain_group_results_dict

                }

                os.makedirs(os.path.join(args.output_dictory, f"{args.mode}_results"), exist_ok=True)

                save_results(os.path.join(args.output_dictory, f"{args.mode}_results", f"Evaluation_results_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.json"), all_evaluation_results)



    else:

        if args.mode == "recover":

            print(f"Recovering results from {args.recover_path}")

            recover_results = load_results(args.recover_path)

            results = process_all_tasks_recover(items, args, generation_func, recover_results)

        elif args.mode == "eval":

            results = load_results(args.results_path)

        else:



            if getattr(args, 'parallel', False):

                print(f"Using parallel inference with {args.num_workers} workers")

                results = process_all_tasks_infer_parallel(items, args, generation_func)

            else:

                results = process_all_tasks_infer(items, args, generation_func)

        end_time = time.time()

        elapsed = end_time - start_time

        print(f"Total time for {args.model_type}-{args.model_name_save}: {datetime.timedelta(seconds=elapsed)}")





        if args.mode in ("infer", "recover"):

            try:

                token_usage = generation.total_usage

                total_prompt = token_usage.get("prompt_tokens", 0)

                total_completion = token_usage.get("completion_tokens", 0)

                total_tokens = total_prompt + total_completion

                print(f"Token usage - prompt: {total_prompt:,}, completion: {total_completion:,}, total: {total_tokens:,}")





                token_report = {

                    "model": args.model_name_save,

                    "agent_strategy": getattr(args, 'agent_strategy', 'react'),

                    "total_prompt_tokens": total_prompt,

                    "total_completion_tokens": total_completion,

                    "total_tokens": total_tokens,

                    "elapsed_seconds": round(elapsed, 2)

                }

                token_dir = os.path.join(args.output_dictory, "infer_results")

                os.makedirs(token_dir, exist_ok=True)

                token_path = os.path.join(token_dir, f"token_usage_{args.model_name_save}.json")

                with open(token_path, 'w') as f:

                    json.dump(token_report, f, indent=2)

                print(f"Token usage saved to: {token_path}")

            except (AttributeError, Exception) as e:

                print(f"Token tracking not available: {e}")



        if args.mode == "infer":

            os.makedirs(os.path.join(args.output_dictory, f"{args.mode}_results"), exist_ok=True)

            save_results(os.path.join(args.output_dictory, f"{args.mode}_results", f"results_{args.model_name_save}.json"), results)

        else:

            metrics = process_all_tasks_eval(items, args, results)



            sub_task_results_dict = calculate_subtask_results(metrics)

            group_results_dict = calculate_group_results(metrics)

            analysis_results_dict = calculate_analysis_results(metrics)





            if getattr(args, 'level_1', True):

                all_tasks = list(metrics.keys())

                l1_solv_hallu = calculate_l1_solvability_hallucination(metrics, all_tasks)

                analysis_results_dict["Level 1 / Solvability Hallucination"] = l1_solv_hallu





            if getattr(args, 'level_2', True):

                all_tasks = list(metrics.keys())

                l2_plan_hallu = calculate_l2_planning_hallucination(metrics, all_tasks)

                analysis_results_dict["Level 2 / Planning Hallucination"] = l2_plan_hallu





            if getattr(args, 'level_3', True):

                all_tasks = list(metrics.keys())

                l3_exec_hallu = calculate_l3_execution_hallucination(metrics, all_tasks)

                analysis_results_dict["Level 3 / Execution Hallucination"] = l3_exec_hallu



            sub_task_table_print, group_table_print, analysis_table_print = print_table(sub_task_results_dict, group_results_dict, analysis_results_dict)

            os.makedirs(os.path.join(args.output_dictory, "table_results"), exist_ok=True)

            save_table(os.path.join(args.output_dictory, "table_results", f"Subtask_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), sub_task_table_print)

            save_table(os.path.join(args.output_dictory, "table_results", f"Scenario_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), group_table_print)

            save_table(os.path.join(args.output_dictory, "table_results", f"Analysis_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), analysis_table_print)





            domain_results_dict = calculate_domain_results(metrics)

            domain_group_results_dict = calculate_domain_group_results(metrics)



            if domain_results_dict:

                domain_table_print = print_domain_table(domain_results_dict)

                domain_group_table_print = print_domain_group_table(domain_group_results_dict)

                save_table(os.path.join(args.output_dictory, "table_results", f"Domain_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), domain_table_print)

                save_table(os.path.join(args.output_dictory, "table_results", f"DomainGroup_table_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.txt"), domain_group_table_print)



            all_evaluation_results = {

                "sub_task_results": sub_task_results_dict,

                "group_results": group_results_dict,

                "analysis_results": analysis_results_dict,

                "domain_results": domain_results_dict,

                "domain_group_results": domain_group_results_dict

            }

            os.makedirs(os.path.join(args.output_dictory, f"{args.mode}_results"), exist_ok=True)

            save_results(os.path.join(args.output_dictory, f"{args.mode}_results", f"Evaluation_results_{args.model_name_save}_{args.embedding_model}_{args.calculate_type}.json"), all_evaluation_results)





if __name__ == "__main__":

    main()

