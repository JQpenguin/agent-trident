import os

import re

import pickle

import time

import requests

from openai import OpenAI

try:

    import google.generativeai as genai

    from google.generativeai.types import HarmCategory, HarmBlockThreshold

except (ImportError, AttributeError):

    genai = None

    HarmCategory = None

    HarmBlockThreshold = None



def rate_limited(max_per_minute):

    min_interval = 60.0 / float(max_per_minute)

    def decorate(func):

        last_called = [0.0]

        def rate_limited_function(*args, **kwargs):

            elapsed = time.time() - last_called[0]

            left_to_wait = min_interval - elapsed

            if left_to_wait > 0:

                time.sleep(left_to_wait)

            last_called[0] = time.time()

            return func(*args, **kwargs)

        return rate_limited_function

    return decorate



def get_tools_embeddings(items, args):

    tools_embedding = {}

    os.makedirs(f"./tools_emb/{args.embedding_model}", exist_ok=True)

    for sub_task in list(items.keys()):

        sub_task_pkl_path = f"./tools_emb/{args.embedding_model}/{sub_task}_task_tools_emb.pkl"

        if os.path.exists(sub_task_pkl_path):

            with open(sub_task_pkl_path, "rb") as f:

                tools_embedding[sub_task] = pickle.load(f)

        else:

            task_name, task_desc = [], []

            for item in items[sub_task]:

                tools = item["tools"]





                if isinstance(tools, list):



                    for tool_str in tools:



                        match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*[–\-]\s*(.*)$', tool_str)

                        if match:

                            tool_name = match.group(1).strip()

                            tool_desc = match.group(2).strip()

                            task_name.append(tool_name)

                            task_desc.append(tool_desc)

                else:



                    pattern = re.compile(r'\d+\.\s*([^\:：]+)[:：]\s*(.*)')

                    matches = pattern.findall(tools)

                    for match in matches:

                        tool_name, tool_desc = match

                        task_name.append(tool_name)

                        task_desc.append(tool_desc)



            if args.embedding_model == "minilm":

                tool_embeddings = args.emb_model.encode(task_desc)

            elif args.embedding_model == "gemini":

                tool_embeddings = []

                for desc in task_desc:

                    embedding = args.emb_model(desc)["embedding"]

                    tool_embeddings.append(embedding)

            else:

                raise Exception("Wrong embedding type")



            tools_embedding[sub_task] = {

                    "name": task_name,

                    "desc": task_desc,

                    "embeddings": tool_embeddings

                }

            with open(sub_task_pkl_path, "wb") as fw:

                pickle.dump(tools_embedding[sub_task], fw)



    return tools_embedding





class GeminiGeneration:

    def __init__(self, api_key, model_name):

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)

        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}



    @rate_limited(59)

    def generation_gemini(self, prompt):

        try:

            gemini_response = self.model.generate_content(

                prompt,

                generation_config=genai.types.GenerationConfig(

                    candidate_count=1,

                    max_output_tokens=8192,

                    temperature=0.0),

                safety_settings={

                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,

                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,

                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,

                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

                    },

                )

            try:

                usage = gemini_response.usage_metadata

                self.total_usage["prompt_tokens"] += getattr(usage, "prompt_token_count", 0)

                self.total_usage["completion_tokens"] += getattr(usage, "candidates_token_count", 0)

            except Exception:

                pass

            return gemini_response.text

        except Exception as e:

            if "429" in str(e):

                print("Need to switched key due to 429 Error.")

            else:

                print(f'Unknown error occurred: {e}')

            return ""



class GeminiEmbedding:

    def __init__(self, api_key, model_name="models/text-embedding-004"):

        genai.configure(api_key=api_key)

        self.model = model_name



    def get_embedding_gemini(self, prompt):

        result = genai.embed_content(

            model=self.model,

            content=prompt,

            task_type="semantic_similarity"

        )

        return result





class OpenAIGeneration:

    def __init__(self, api_key, model):

        self.client = OpenAI(api_key=api_key)

        self.model = model

        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}



    def generation_openai(self, prompt):

        try:

            completion = self.client.chat.completions.create(

                model=self.model,

                messages=[

                    {"role": "user", "content": prompt}

                ],

                temperature = 0.0,

                max_tokens = 8192

            )

            try:

                self.total_usage["prompt_tokens"] += completion.usage.prompt_tokens

                self.total_usage["completion_tokens"] += completion.usage.completion_tokens

            except Exception:

                pass

            return completion.choices[0].message.content

        except Exception as e:

            print(f'Unknown error occurred: {e}')

            return ""





import time

import threading







class RateLimiter:

    def __init__(self, requests_per_second=2):

        self.min_interval = 1.0 / requests_per_second

        self.last_request_time = 0

        self.lock = threading.Lock()



    def wait(self):

        with self.lock:

            now = time.time()

            elapsed = now - self.last_request_time

            if elapsed < self.min_interval:

                time.sleep(self.min_interval - elapsed)

            self.last_request_time = time.time()





_rate_limiter = RateLimiter(requests_per_second=5)





class DeepSeekGeneration:

    def __init__(self, api_key, model='deepseek-chat', api_url=None):



        if api_url:



            self.client = OpenAI(api_key=api_key, base_url=api_url)

        else:

            self.client = OpenAI(api_key=api_key, base_url='https://api.deepseek.com')

        self.model = model

        self.max_retries = 5

        self.base_delay = 2.0

        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}



    def generation_deepseek(self, prompt):

        for attempt in range(self.max_retries):

            try:



                _rate_limiter.wait()







                is_reasoner = ('reasoner' in self.model

                               or self.model.startswith('o1')

                               or self.model.startswith('o3')

                               or self.model.startswith('o4'))

                kwargs = dict(

                    model=self.model,

                    messages=[

                        {"role": "user", "content": prompt}

                    ],

                    max_tokens=8192,

                    timeout=300 if is_reasoner else 120,

                )

                if not is_reasoner:

                    kwargs["temperature"] = 0.0



                completion = self.client.chat.completions.create(**kwargs)

                try:

                    self.total_usage["prompt_tokens"] += completion.usage.prompt_tokens

                    self.total_usage["completion_tokens"] += completion.usage.completion_tokens

                except Exception:

                    pass

                return completion.choices[0].message.content

            except Exception as e:

                delay = self.base_delay * (2 ** attempt)

                print(f'DeepSeek error (attempt {attempt + 1}/{self.max_retries}): {e}')

                if attempt < self.max_retries - 1:

                    print(f'Retrying in {delay:.1f} seconds...')

                    time.sleep(delay)

                else:

                    print(f'Max retries reached, returning empty response')

                    return ""

        return ""





class NvidiaGeneration:

    def __init__(self, api_key, model='minimaxai/minimax-m2.5',

                 api_url='https://integrate.api.nvidia.com/v1'):

        self.client = OpenAI(api_key=api_key, base_url=api_url)

        self.model = model

        self.max_retries = 5

        self.base_delay = 3.0

        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}



        self._rate_limiter = RateLimiter(requests_per_second=38.0 / 60.0)



    def generation_nvidia(self, prompt):

        for attempt in range(self.max_retries):

            try:

                self._rate_limiter.wait()



                completion = self.client.chat.completions.create(

                    model=self.model,

                    messages=[

                        {"role": "user", "content": prompt}

                    ],

                    temperature=0.0,

                    max_tokens=8192,

                    timeout=180

                )

                try:

                    self.total_usage["prompt_tokens"] += completion.usage.prompt_tokens

                    self.total_usage["completion_tokens"] += completion.usage.completion_tokens

                except Exception:

                    pass

                return completion.choices[0].message.content

            except Exception as e:

                delay = self.base_delay * (2 ** attempt)

                print(f'NVIDIA error (attempt {attempt + 1}/{self.max_retries}): {e}')

                if "429" in str(e) or "rate" in str(e).lower():

                    delay = max(delay, 10.0)

                    print(f'Rate limit hit, backing off {delay:.1f}s...')

                if attempt < self.max_retries - 1:

                    print(f'Retrying in {delay:.1f} seconds...')

                    time.sleep(delay)

                else:

                    print(f'Max retries reached, returning empty response')

                    return ""

        return ""





class KimiGeneration:

    def __init__(self, api_key, model='moonshot-v1-128k', api_url='https://api.moonshot.cn/v1'):



        self.client = OpenAI(api_key=api_key, base_url=api_url)

        self.model = model

        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}



    def generation_kimi(self, prompt):

        try:

            completion = self.client.chat.completions.create(

                model=self.model,

                messages=[

                    {"role": "user", "content": prompt}

                ],

                temperature=0.0,

                max_tokens=8192

            )

            try:

                self.total_usage["prompt_tokens"] += completion.usage.prompt_tokens

                self.total_usage["completion_tokens"] += completion.usage.completion_tokens

            except Exception:

                pass

            return completion.choices[0].message.content

        except Exception as e:

            print(f'Kimi error occurred: {e}')

            return ""





class AnthropicGeneration:

    def __init__(self, api_key, model='claude-opus-4-6', api_url=None):

        import anthropic

        kwargs = {"api_key": api_key}

        if api_url:

            kwargs["base_url"] = api_url

        self.client = anthropic.Anthropic(**kwargs)

        self.model = model

        self.max_retries = 5

        self.base_delay = 2.0

        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}



    def generation_anthropic(self, prompt):

        for attempt in range(self.max_retries):

            try:

                _rate_limiter.wait()



                message = self.client.messages.create(

                    model=self.model,

                    max_tokens=8192,

                    temperature=0.0,

                    messages=[

                        {"role": "user", "content": prompt}

                    ]

                )

                try:

                    self.total_usage["prompt_tokens"] += message.usage.input_tokens

                    self.total_usage["completion_tokens"] += message.usage.output_tokens

                except Exception:

                    pass



                return "".join(block.text for block in message.content if block.type == "text")

            except Exception as e:

                delay = self.base_delay * (2 ** attempt)

                print(f'Anthropic error (attempt {attempt + 1}/{self.max_retries}): {e}')

                if "rate" in str(e).lower() or "429" in str(e):

                    delay = max(delay, 10.0)

                    print(f'Rate limit hit, backing off {delay:.1f}s...')

                if attempt < self.max_retries - 1:

                    print(f'Retrying in {delay:.1f} seconds...')

                    time.sleep(delay)

                else:

                    print(f'Max retries reached, returning empty response')

                    return ""

        return ""





class VllmGeneration:

    def __init__(self, api_url, model='model'):

        self.api_url = api_url

        self.model = model

        self.max_retries = 3

        self.base_delay = 2.0

        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}



    def generation_vllm(self, prompt):

        headers = {'Content-Type': 'application/json'}

        post_data = {

                'model': self.model,

                'messages': [{"role": "user", "content": prompt}],

                'max_tokens': 8192,

                'temperature': 0.0,

                'top_p': 0.95,

            }

        for attempt in range(self.max_retries):

            try:

                resp = requests.post(self.api_url, headers=headers, json=post_data, timeout=300)

                response = resp.json()





                if "error" in response:

                    raise RuntimeError(f"API error: {response['error']}")



                try:

                    usage = response.get("usage", {})

                    self.total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)

                    self.total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

                except Exception:

                    pass



                if "choices" not in response:

                    raise KeyError(f"'choices' not in response. Keys: {list(response.keys())}. "

                                   f"Response (first 500 chars): {str(response)[:500]}")



                content = response['choices'][0]['message']['content']





                if content and '<think>' in content:

                    import re

                    cleaned = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()

                    if cleaned:

                        content = cleaned



                return content

            except Exception as e:

                delay = self.base_delay * (2 ** attempt)

                print(f'vLLM error (attempt {attempt + 1}/{self.max_retries}): {e}')

                if attempt < self.max_retries - 1:

                    print(f'Retrying in {delay:.1f} seconds...')

                    time.sleep(delay)

                else:

                    print(f'Max retries reached, returning empty response')

                    return ""

        return ""

