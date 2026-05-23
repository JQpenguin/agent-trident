import argparse
import datetime
import json
import os
import random
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


DOMAIN_CONFIG = {
    "financial_services": {"display": "Financial Services", "roles": ["Risk Analyst", "Compliance Officer", "Fraud Investigator", "Investment Advisor", "Treasury Manager", "Credit Analyst"], "category": "Enterprise"},
    "human_resources": {"display": "Human Resources", "roles": ["HR Business Partner", "Talent Acquisition Specialist", "Compensation Analyst", "Employee Relations Manager", "HRIS Administrator"], "category": "Enterprise"},
    "customer_service": {"display": "Customer Service", "roles": ["Customer Success Manager", "Support Team Lead", "Escalation Specialist", "Quality Assurance Analyst", "CX Operations Manager"], "category": "Enterprise"},
    "supply_chain": {"display": "Supply Chain Management", "roles": ["Logistics Coordinator", "Procurement Specialist", "Inventory Analyst", "Warehouse Manager", "Demand Planner", "Freight Manager"], "category": "Enterprise"},
    "software_development": {"display": "Software Development", "roles": ["DevOps Engineer", "Release Manager", "Build Engineer", "Platform Engineer", "Site Reliability Engineer", "QA Lead"], "category": "Technology"},
    "cybersecurity": {"display": "Cybersecurity", "roles": ["SOC Analyst", "Threat Intelligence Analyst", "Incident Responder", "Security Architect", "Penetration Tester", "Forensic Analyst"], "category": "Technology"},
    "cloud_operations": {"display": "Cloud Operations", "roles": ["Cloud Engineer", "Infrastructure Architect", "Kubernetes Administrator", "Database Administrator", "Network Engineer"], "category": "Technology"},
    "data_engineering": {"display": "Data Engineering", "roles": ["Data Engineer", "ETL Developer", "Data Pipeline Architect", "Analytics Engineer", "Data Quality Manager"], "category": "Technology"},
    "clinical_healthcare": {"display": "Clinical Healthcare", "roles": ["Clinical Coordinator", "Medical Records Manager", "Lab Technician", "Pharmacy Coordinator", "Radiology Technician", "Triage Nurse"], "category": "Healthcare"},
    "medical_devices": {"display": "Medical Devices", "roles": ["Biomedical Engineer", "Equipment Calibration Specialist", "Clinical Applications Specialist", "Regulatory Affairs Manager"], "category": "Healthcare"},
    "smart_manufacturing": {"display": "Smart Manufacturing", "roles": ["Production Planner", "Quality Control Engineer", "Maintenance Technician", "Process Engineer", "Plant Supervisor"], "category": "Industrial"},
    "energy_management": {"display": "Energy Management", "roles": ["Grid Operator", "Energy Analyst", "Renewable Energy Technician", "Utility Manager", "Power Systems Engineer"], "category": "Industrial"},
    "e_commerce": {"display": "E-Commerce", "roles": ["Marketplace Manager", "Fulfillment Coordinator", "Product Catalog Manager", "Pricing Analyst", "Returns Specialist"], "category": "Consumer"},
    "smart_home": {"display": "Smart Home IoT", "roles": ["Home Automation Specialist", "IoT Support Engineer", "Security System Technician", "Energy Optimization Consultant"], "category": "Consumer"},
    "transportation": {"display": "Transportation and Logistics", "roles": ["Fleet Manager", "Route Planner", "Dispatch Coordinator", "Traffic Operations Manager", "Mobility Solutions Analyst"], "category": "Consumer"},
    "food_hospitality": {"display": "Food and Hospitality", "roles": ["Restaurant Operations Manager", "Kitchen Manager", "Reservation Coordinator", "Inventory Control Specialist", "Guest Services Manager"], "category": "Consumer"},
    "edtech": {"display": "Educational Technology", "roles": ["Learning Management Administrator", "Instructional Designer", "Academic Technology Specialist", "Student Success Analyst"], "category": "Education"},
    "scientific_research": {"display": "Scientific Research", "roles": ["Lab Manager", "Research Coordinator", "Data Scientist", "Equipment Specialist", "Grant Administrator"], "category": "Education"},
    "government_services": {"display": "Government Services", "roles": ["Case Manager", "Benefits Administrator", "Permit Processor", "Records Manager", "Compliance Auditor"], "category": "Government"},
    "public_safety": {"display": "Public Safety", "roles": ["Emergency Dispatcher", "Incident Commander", "Resource Coordinator", "Public Information Officer", "Safety Inspector"], "category": "Government"},
}


class GeminiGeneration:
    def __init__(self, api_key, model):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt, temperature=0.7, max_tokens=4096, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        candidate_count=1,
                    ),
                )
                return getattr(response, "text", "") or ""
            except Exception as exc:
                if attempt == max_retries - 1:
                    print(f"Generation failed after {max_retries} attempts: {exc}")
                    return ""
                wait_time = 2 * (attempt + 1)
                print(f"Generation error on attempt {attempt + 1}: {exc}. Retrying in {wait_time}s.")
                time.sleep(wait_time)


class OpenAIGeneration:
    def __init__(self, api_key, model, api_url=None):
        from openai import OpenAI

        kwargs = {"api_key": api_key, "timeout": 120.0}
        if api_url:
            kwargs["base_url"] = api_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(self, prompt, temperature=0.7, max_tokens=4096, max_retries=3):
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                if attempt == max_retries - 1:
                    print(f"Generation failed after {max_retries} attempts: {exc}")
                    return ""
                wait_time = 2 * (attempt + 1)
                print(f"Generation error on attempt {attempt + 1}: {exc}. Retrying in {wait_time}s.")
                time.sleep(wait_time)


class AnthropicGeneration:
    def __init__(self, api_key, model, api_url=None):
        import anthropic

        kwargs = {"api_key": api_key, "timeout": 120.0}
        if api_url:
            kwargs["base_url"] = api_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model

    def generate(self, prompt, temperature=0.7, max_tokens=4096, max_retries=3):
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
            except Exception as exc:
                if attempt == max_retries - 1:
                    print(f"Generation failed after {max_retries} attempts: {exc}")
                    return ""
                wait_time = 2 * (attempt + 1)
                print(f"Generation error on attempt {attempt + 1}: {exc}. Retrying in {wait_time}s.")
                time.sleep(wait_time)


def load_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as handle:
        return handle.read()


def extract_sample_content(response):
    match = re.search(r"----Begin sign----\s*(.*?)\s*----End sign----", response, re.DOTALL)
    if not match:
        print("Warning: response did not contain valid sample markers.")
        print(f"Response preview: {response[:500]}...")
        return None
    return match.group(1).strip()


def remove_json_comments(text):
    return re.sub(r"//.*?(?=\n|$)", "", text, flags=re.MULTILINE)


def parse_sample_to_dict(sample_content, domain):
    try:
        cleaned = remove_json_comments(sample_content)
        cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            cleaned = remove_json_comments(sample_content)
            cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
            parsed = json.loads("{" + cleaned.strip().strip("{}") + "}")
        except Exception as exc:
            print(f"Warning: failed to parse generated JSON: {exc}")
            return None
    sample = OrderedDict()
    sample["domain"] = domain
    sample["subtask"] = "single_step"
    for key in ["Agent_role", "Env", "task", "tools", "simulation_script", "planning_tools", "Solvability"]:
        sample[key] = parsed.get(key)
    return sample


def tool_name(call_text):
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", call_text or "")
    return match.group(1) if match else ""


def validate_single_step_sample(sample):
    required = ["domain", "subtask", "Agent_role", "Env", "task", "tools", "simulation_script", "planning_tools", "Solvability"]
    if not all(key in sample and sample[key] not in (None, "", []) for key in required):
        return False
    if sample["subtask"] != "single_step" or sample["Solvability"] != "Solvable":
        return False
    if not isinstance(sample["tools"], list) or not (4 <= len(sample["tools"]) <= 8):
        return False
    if not isinstance(sample["simulation_script"], dict) or not sample["simulation_script"]:
        return False
    if not isinstance(sample["planning_tools"], list) or len(sample["planning_tools"]) != 2:
        return False
    first_tool = tool_name(sample["planning_tools"][0])
    second_tool = tool_name(sample["planning_tools"][1])
    if not first_tool or second_tool != "finish":
        return False
    if first_tool in {"finish", "UnsolvableQuery"}:
        return False
    return first_tool in sample["simulation_script"]


def build_single_step_prompt(base_prompt, sample_id, domain):
    rng = random.Random(f"{sample_id}-{time.time_ns()}")
    domain_info = DOMAIN_CONFIG[domain]
    role = rng.choice(domain_info["roles"])
    scenario = rng.choice(["Routine Operation", "Critical Emergency", "Complex Investigation", "Long-term Project", "Audit and Compliance Check", "Troubleshooting", "Strategic Planning", "Crisis Management"])
    time_context = rng.choice(["Q1 2025, start of fiscal year", "Q3 2024, mid-year review", "End of month closing", "Weekend on-call shift", "Holiday coverage period", "Annual audit season", "System migration phase", "Product launch week", "Post-incident review"])
    organization = rng.choice(["startup with 10 to 50 employees", "mid-size company with more than 500 employees", "enterprise with more than 5000 employees", "government agency", "multinational corporation"])
    reference_id = f"{rng.choice(['PROJ', 'CASE', 'REQ', 'TICKET'])}-{rng.randint(1000, 9999)}"
    location_code = f"{rng.choice(['Building', 'Floor', 'Zone', 'Sector'])}-{rng.choice('ABCDEFGHIJK')}{rng.randint(1, 9)}"
    tool_ranges = [(2, 2, 1, 2), (2, 3, 1, 2), (3, 4, 2, 3), (4, 5, 2, 3)]
    min_tools, max_tools, min_params, max_params = tool_ranges[sample_id % len(tool_ranges)]
    return f"""{base_prompt}

MANDATORY SINGLE_STEP CONSTRAINTS FOR SAMPLE #{sample_id + 1}:

Domain Key: {domain}
Industry: {domain_info["display"]}
Category: {domain_info["category"]}
Role Title: {role}
Scenario Category: {scenario}
Time Setting: {time_context}
Organization Type: {organization}
Reference ID: {reference_id}
Location Code: {location_code}

The generated sample must follow the paper-aligned single_step structure:
1. The sample is Solvable.
2. The task requires exactly one domain-specific tool call.
3. planning_tools has exactly two entries: one domain tool call and one finish call.
4. The tools list contains {min_tools} to {max_tools} domain-specific tools plus UnsolvableQuery and finish.
5. Every domain-specific tool has {min_params} to {max_params} parameters and never more than 4 parameters.
6. simulation_script defines realistic return values for every domain-specific tool in tools.
7. Use exact parameter values from Env or task.

Generate exactly one valid JSON sample between the required Begin and End markers."""


def build_review_prompt(sample):
    sample_json = json.dumps({key: value for key, value in sample.items() if key not in {"id", "domain", "subtask"}}, ensure_ascii=False, indent=2)
    return f"""Review and, if needed, correct this TRIDENT single_step candidate.

The corrected sample must satisfy:
1. Solvability is exactly "Solvable".
2. planning_tools contains exactly two entries.
3. The first planning_tools entry is one domain-specific tool call.
4. The second planning_tools entry is finish(...).
5. The planned domain tool is present in simulation_script.
6. Every parameter in the planned tool call is grounded in Env or task.
7. The tools list includes UnsolvableQuery and finish.
8. The output contains exactly the keys Agent_role, Env, task, tools, simulation_script, planning_tools, and Solvability.

Return only the corrected JSON object between ----Begin sign---- and ----End sign----.

Candidate:
----Begin sign----
{sample_json}
----End sign----"""


def review_sample(sample, reviewer, temperature, max_tokens, domain):
    if reviewer is None:
        return sample
    response = reviewer.generate(build_review_prompt(sample), temperature=temperature, max_tokens=max_tokens)
    if not response:
        return sample
    sample_content = extract_sample_content(response)
    if not sample_content:
        return sample
    reviewed = parse_sample_to_dict(sample_content, domain)
    if reviewed and validate_single_step_sample(reviewed):
        return reviewed
    return sample


def generate_single_sample(sample_id, base_prompt, generator, reviewer, generation_temperature, review_temperature, max_tokens, domain):
    prompt = build_single_step_prompt(base_prompt, sample_id, domain)
    response = generator.generate(prompt, temperature=generation_temperature, max_tokens=max_tokens)
    if not response:
        print(f"Sample {sample_id} returned an empty response.")
        return None
    sample_content = extract_sample_content(response)
    if not sample_content:
        return None
    sample = parse_sample_to_dict(sample_content, domain)
    if not sample:
        return None
    sample = review_sample(sample, reviewer, review_temperature, max_tokens, domain)
    if not validate_single_step_sample(sample):
        print(f"Sample {sample_id} failed single_step validation.")
        return None
    print(f"Sample {sample_id} completed: {domain}")
    return sample


def build_domain_assignments(num_samples, fixed_domain=None):
    if fixed_domain:
        return [fixed_domain] * num_samples
    domains = list(DOMAIN_CONFIG.keys())
    assignments = [domains[index % len(domains)] for index in range(num_samples)]
    random.shuffle(assignments)
    return assignments


def generate_dataset(prompt, generator, reviewer, num_samples, generation_temperature, review_temperature, max_tokens, batch_size, domain):
    assignments = build_domain_assignments(num_samples, domain)
    results = []
    effective_batch_size = batch_size or min(10, max(1, num_samples))
    print(f"Starting single_step generation for {num_samples} samples.")
    print(f"Batch size: {effective_batch_size}")
    for batch_start in range(0, num_samples, effective_batch_size):
        batch_end = min(batch_start + effective_batch_size, num_samples)
        batch_ids = range(batch_start, batch_end)
        print(f"Batch {batch_start // effective_batch_size + 1}: samples {batch_start}-{batch_end - 1}")
        with ThreadPoolExecutor(max_workers=effective_batch_size) as executor:
            future_to_id = {
                executor.submit(
                    generate_single_sample,
                    index,
                    prompt,
                    generator,
                    reviewer,
                    generation_temperature,
                    review_temperature,
                    max_tokens,
                    assignments[index],
                ): index
                for index in batch_ids
            }
            for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Batch"):
                result = future.result()
                if result is not None:
                    results.append(result)
        if batch_end < num_samples:
            time.sleep(1)
    normalized = []
    for index, sample in enumerate(results):
        row = OrderedDict()
        row["id"] = index
        for key, value in sample.items():
            if key != "id":
                row[key] = value
        normalized.append(row)
    print(f"Generated {len(normalized)}/{num_samples} valid samples.")
    return normalized


def save_results(output_path, results):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")


def env_key_for_provider(provider):
    if provider == "gemini":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    return None


def build_model_client(provider, api_key, model, api_url=None):
    if provider == "gemini":
        return GeminiGeneration(api_key=api_key, model=model)
    if provider == "openai":
        return OpenAIGeneration(api_key=api_key, model=model, api_url=api_url)
    if provider == "anthropic":
        return AnthropicGeneration(api_key=api_key, model=model, api_url=api_url)
    raise ValueError(f"Unsupported provider: {provider}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TRIDENT single_step data.")
    parser.add_argument("--generator_provider", type=str, default=os.getenv("GENERATOR_PROVIDER", "gemini"), choices=["gemini", "openai", "anthropic"])
    parser.add_argument("--generator_model", type=str, default=os.getenv("GENERATOR_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--generator_api_key", type=str, default=os.getenv("GENERATOR_API_KEY"))
    parser.add_argument("--generator_api_url", type=str, default=os.getenv("GENERATOR_API_URL"))
    parser.add_argument("--review_provider", type=str, default=os.getenv("REVIEW_PROVIDER", "openai"), choices=["gemini", "openai", "anthropic"])
    parser.add_argument("--review_model", type=str, default=os.getenv("REVIEW_MODEL", "o3"))
    parser.add_argument("--review_api_key", type=str, default=os.getenv("REVIEW_API_KEY"))
    parser.add_argument("--review_api_url", type=str, default=os.getenv("REVIEW_API_URL"))
    parser.add_argument("--disable_review", action="store_true")
    parser.add_argument("--prompt_path", type=str, default="./prompt/single_step.txt")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--generation_temperature", type=float, default=0.7)
    parser.add_argument("--review_temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--subtask", type=str, default="single_step", choices=["single_step"])
    parser.add_argument("--domain", type=str, default=None, choices=list(DOMAIN_CONFIG.keys()))
    return parser.parse_args()


def main():
    args = parse_args()
    generator_key = args.generator_api_key or env_key_for_provider(args.generator_provider)
    if not generator_key:
        raise SystemExit(f"Set an API key for generator provider {args.generator_provider}.")
    reviewer = None
    if not args.disable_review:
        review_key = args.review_api_key or env_key_for_provider(args.review_provider)
        if not review_key:
            raise SystemExit(f"Set an API key for review provider {args.review_provider}, or pass --disable_review.")
        reviewer = build_model_client(args.review_provider, review_key, args.review_model, args.review_api_url)
    prompt = load_prompt(args.prompt_path)
    generator = build_model_client(args.generator_provider, generator_key, args.generator_model, args.generator_api_url)
    print(f"Generator: {args.generator_provider}/{args.generator_model}")
    if reviewer is not None:
        print(f"Reviewer:  {args.review_provider}/{args.review_model}")
    start_time = time.time()
    results = generate_dataset(
        prompt=prompt,
        generator=generator,
        reviewer=reviewer,
        num_samples=args.num_samples,
        generation_temperature=args.generation_temperature,
        review_temperature=args.review_temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        domain=args.domain,
    )
    output_name = args.output_name or f"single_step_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(os.path.join(args.output_dir, output_name), results)
    elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
    print(f"Generation completed in {elapsed}.")


if __name__ == "__main__":
    main()
