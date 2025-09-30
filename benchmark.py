import math
import os
import re
import random
from argparse import ArgumentParser
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import (
    Any,
    Dict,
    Sequence
)

from dataset import (
    Dataset,
    Datasets
)
from engine import (
    Engine,
    Engines
)
from languages import Languages, LANGUAGE_TO_CODE
from metric import (
    Metric,
    Metrics
)
from normalizer import (
    SUPPORTED_PUNCTUATION_SET,
    EnglishNormalizer,
    Normalizer
)
from optim import (
    prompt_optimization,
    nbest_optimization,
    evolutionary_prompt_optimization,
    RIF_ASR,
)

WorkerResult = namedtuple("WorkerResult", ["metric", "num_errors", "num_tokens", "audio_sec", "process_sec"])
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")


def process(
    engine_name: Engines,
    engine_params: Dict[str, Any],
    language: Languages,
    punctuation: bool,
    punctuation_set: str,
    dataset_name: Datasets,
    dataset_folder: str,
    indices: Sequence[int],
    metric_names: Sequence[Metrics],
    optimization: str,
    llm: str = "qwen",
    num_candidate: int = 5,
    num_iteration: int = 3,
    neighbor: bool = True,
    rule: bool = True,
    num_sample: int = 80,
    device: str = "cuda",
    cache: bool = False,
) -> Sequence[WorkerResult]:
    engine = Engine.create(engine_name, language=language, device=device, cache=cache, **engine_params)
    dataset = Dataset.create(
        dataset_name,
        folder=dataset_folder,
        language=language,
        punctuation=punctuation,
        punctuation_set=punctuation_set,
    )
    
    normalizer = Normalizer.create(language=language, keep_punctuation=punctuation, punctuation_set=punctuation_set)

    metrics = {m: Metric.create(m) for m in metric_names}
    results = {m: {"num_errors": 0, "num_tokens": 0} for m in metric_names}
    
    for index in tqdm(indices[:num_sample]):
        audio_path, ref_transcript = dataset.get(index)
        transcript = engine.transcribe(audio_path)
        if language == Languages.ZH:
            import opencc
            cc = opencc.OpenCC('t2s')
            transcript = cc.convert(transcript)
        corrected_transcript = transcript
        
        if optimization == "prompt":
            corrected_transcript = prompt_optimization(transcript, llm=llm)
        elif optimization == "evolutionary_prompt":
            corrected_transcript = evolutionary_prompt_optimization(transcript, llm=llm)
        elif optimization == "nbest":
            corrected_transcript = nbest_optimization(transcript, llm=llm, num_candidate=num_candidate)
        elif optimization == "RIF_ASR":
            corrected_transcript = RIF_ASR(transcript, 
                                            llm=llm, 
                                            language=LANGUAGE_TO_CODE[language].split('-')[0].upper(),
                                            num_candidate=num_candidate,
                                            num_iteration=num_iteration,
                                            neighbor=neighbor,
                                            rule=rule,
                                            normalizer=normalizer,
                                            #verbose=True
                                            )
        
        def diff_words(a, b, c):
            if language in [Languages.ZH]:
                import jieba
                a = ''.join(re.findall(r'[\u4E00-\u9FFF0-9]+', a))
                b = ''.join(re.findall(r'[\u4E00-\u9FFF0-9]+', b))
                c = ''.join(re.findall(r'[\u4E00-\u9FFF0-9]+', c))
                a_words = list(normalizer.normalize(a).upper())
                b_words = list(normalizer.normalize(b).upper())
                c_words = list(normalizer.normalize(c).upper())
            else:
                a_words = normalizer.normalize(a).upper().split()
                b_words = normalizer.normalize(b).upper().split()
                c_words = normalizer.normalize(c).upper().split()
            out_a, out_b = [], []
            for i in range(max(len(a_words), len(b_words))):
                w_a = a_words[i] if i < len(a_words) else ""
                w_b = b_words[i] if i < len(b_words) else ""
                if w_a != w_b:
                    out_a.append(f"[{w_a}]")
                    out_b.append(f"[{w_b}]")
                else:
                    out_a.append(w_a)
                    out_b.append(w_b)
            return "标签: " + " ".join(out_a) + "\n矫正: " + " ".join(out_b) + "\n原始: " + " ".join(c_words)
        print(diff_words(ref_transcript, corrected_transcript, transcript))
        norm_transcript = normalizer.normalize(corrected_transcript)
        
        if language in [Languages.ZH]:
            import jieba
            ref_transcript = ''.join(re.findall(r'[\u4E00-\u9FFF0-9]+', ref_transcript))
            norm_transcript = ''.join(re.findall(r'[\u4E00-\u9FFF0-9]+', norm_transcript))
            ref_sentence = ' '.join(list(jieba.cut(normalizer.normalize(ref_transcript))))
            transcribed_sentence = ' '.join(list(jieba.cut(normalizer.normalize(norm_transcript))))
        else:
            ref_sentence = ref_transcript.strip("\n ").lower()
            transcribed_sentence = norm_transcript.strip("\n ").lower()
        
        for metric_name, metric in metrics.items():
            num_errors, num_tokens = metric.calculate(prediction=transcribed_sentence, reference=ref_sentence)
            results[metric_name]["num_errors"] += num_errors
            results[metric_name]["num_tokens"] += num_tokens

    engine.delete()

    worker_results = []
    for metric_name in metric_names:
        worker_results.append(
            WorkerResult(
                metric=metric_name.value,
                num_errors=results[metric_name]["num_errors"],
                num_tokens=results[metric_name]["num_tokens"],
                audio_sec=engine.audio_sec(),
                process_sec=engine.process_sec(),
            )
        )

    return worker_results


def main():
    parser = ArgumentParser()
    parser.add_argument("--engine", required=True, choices=[x.value for x in Engines])
    parser.add_argument("--dataset", required=True, choices=[x.value for x in Datasets])
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--language", required=True, choices=[x.value for x in Languages])
    parser.add_argument("--punctuation", action="store_true")
    parser.add_argument("--punctuation-set", type=str, default="")
    parser.add_argument("--optimization", type=str, default="none")
    parser.add_argument("--aws-profile")
    parser.add_argument("--azure-speech-key")
    parser.add_argument("--azure-speech-location")
    parser.add_argument("--google-application-credentials")
    parser.add_argument("--picovoice-access-key")
    parser.add_argument("--picovoice-model-path", default=None)
    parser.add_argument("--picovoice-library-path", default=None)
    parser.add_argument("--watson-speech-to-text-api-key")
    parser.add_argument("--watson-speech-to-text-url")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--njob", type=int, default=os.cpu_count())
    parser.add_argument("--llm", type=str, default="Qwen3-235B",
                                           choices=["Qwen3-235B"],
                                           help="LLM model to use for optimization: qwen, k2, claude, gemini, chatgpt3.5, chatgpt4")
    parser.add_argument("--num-candidate", type=int, default=5)
    parser.add_argument("--num-iteration", type=int, default=3)
    parser.add_argument("--neighbor", type=str, default="True")
    parser.add_argument("--rule", type=str, default="True")
    parser.add_argument("--num-sample", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache", type=str, default=False)
    args = parser.parse_args()
    
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    engine = Engines(args.engine)
    dataset_type = Datasets(args.dataset)
    language = Languages(args.language)
    punctuation = args.punctuation
    punctuation_set = args.punctuation_set
    dataset_folder = args.dataset_folder
    num_examples = args.num_examples
    njob = args.njob
    optimization = args.optimization
    llm = args.llm
    num_sample = args.num_sample
    num_candidate = args.num_candidate
    num_iteration = args.num_iteration
    neighbor = args.neighbor == "True"
    rule = args.rule == "True"
    cache = args.cache == "True"

    engine_params = dict()
    if engine == Engines.AMAZON_TRANSCRIBE:
        if args.aws_profile is None:
            raise ValueError("`aws-profile` is required")
        os.environ["AWS_PROFILE"] = args.aws_profile
    elif engine == Engines.AZURE_SPEECH_TO_TEXT:
        if args.azure_speech_key is None or args.azure_speech_location is None:
            raise ValueError("`azure-speech-key` and `azure-speech-location` are required")
        engine_params["azure_speech_key"] = args.azure_speech_key
        engine_params["azure_speech_location"] = args.azure_speech_location
    elif engine == Engines.GOOGLE_SPEECH_TO_TEXT or engine == Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED:
        if args.google_application_credentials is None:
            raise ValueError("`google-application-credentials` is required")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_application_credentials
    elif engine == Engines.PICOVOICE_CHEETAH:
        if args.picovoice_access_key is None:
            raise ValueError("`picovoice-access-key` is required")
        if args.picovoice_model_path is None and args.language != Languages.EN.value:
            raise ValueError("`picovoice-model-path` is required for non-EN languages")
        engine_params["access_key"] = args.picovoice_access_key
        engine_params["model_path"] = args.picovoice_model_path
        engine_params["library_path"] = args.picovoice_library_path
        engine_params["punctuation"] = punctuation
    elif engine == Engines.PICOVOICE_LEOPARD:
        if args.picovoice_access_key is None:
            raise ValueError("`picovoice-access-key` is required")
        if args.picovoice_model_path is None and args.language != Languages.EN.value:
            raise ValueError("`picovoice-model-path` is required for non-EN languages")
        engine_params["access_key"] = args.picovoice_access_key
        engine_params["model_path"] = args.picovoice_model_path
        engine_params["library_path"] = args.picovoice_library_path
        engine_params["punctuation"] = punctuation
    elif engine == Engines.IBM_WATSON_SPEECH_TO_TEXT:
        if args.watson_speech_to_text_api_key is None or args.watson_speech_to_text_url is None:
            raise ValueError("`watson-speech-to-text-api-key` and `watson-speech-to-text-url` are required")
        engine_params["watson_speech_to_text_api_key"] = args.watson_speech_to_text_api_key
        engine_params["watson_speech_to_text_url"] = args.watson_speech_to_text_url

    for p in punctuation_set:
        if p not in SUPPORTED_PUNCTUATION_SET:
            raise ValueError(f"`{p}` is not a supported punctuation character")

    dataset = Dataset.create(
        dataset_type,
        folder=dataset_folder,
        language=language,
        punctuation=punctuation,
        punctuation_set=punctuation_set,
    )
    indices = list(range(dataset.size()))
    #random.shuffle(indices)
    if args.num_examples is not None:
        indices = indices[:num_examples]

    chunk = math.ceil(len(indices) / njob)

    metrics = [Metrics.PER] if punctuation else [Metrics.WER, Metrics.CER]

    print(f"Processing {len(indices)} examples...")
    results = []
    if njob > 1:
        futures = []
        with ProcessPoolExecutor(njob) as executor:
            for i in range(njob):
                future = executor.submit(
                    process,
                    engine_name=engine,
                    engine_params=engine_params,
                    language=language,
                    punctuation=punctuation,
                    punctuation_set=punctuation_set,
                    dataset_name=dataset_type,
                    dataset_folder=dataset_folder,
                    indices=indices[i * chunk : (i + 1) * chunk],
                    metric_names=metrics,
                    optimization=optimization,
                    llm=llm,
                    num_candidate=num_candidate,
                    num_iteration=num_iteration,
                    neighbor=neighbor,
                    rule=rule,
                    num_sample=num_sample,
                    device=args.device,
                    cache=cache,
                )
                futures.append(future)
        
        for future in futures:
            results.extend(future.result())

    else:
        results.extend(
            process(
                engine_name=engine,
                engine_params=engine_params,
                language=language,
                punctuation=punctuation,
                punctuation_set=punctuation_set,
                dataset_name=dataset_type,
                dataset_folder=dataset_folder,
                indices=indices,
                metric_names=metrics,
                optimization=optimization,
                llm=llm,
                num_candidate=num_candidate,
                num_iteration=num_iteration,
                neighbor=neighbor,
                rule=rule,
                num_sample=num_sample,
                device=args.device,
                cache=cache,
            )
        )
    metric_results = {}
    for result in results:
        if result.metric not in metric_results:
            metric_results[result.metric] = []
        metric_results[result.metric].append(result)

    rtf = sum(x.process_sec for x in results) / (sum(x.audio_sec for x in results) + 1e-8)

    results_log_path = os.path.join(RESULTS_FOLDER, language.value, dataset_type.value, f"{str(engine)}.log")
    os.makedirs(os.path.dirname(results_log_path), exist_ok=True)
    with open(results_log_path, "w") as f:
        for metric_name, metric_results in metric_results.items():
            num_errors = sum(x.num_errors for x in metric_results)
            num_tokens = sum(x.num_tokens for x in metric_results)
            error_rate = 100 * float(num_errors) / num_tokens

            f.write(f"{metric_name}: {str(error_rate)}\n")
            print(f"{metric_name}: {error_rate:.2f}")

        f.write(f"RTF: {str(rtf)}\n")
        print(f"RTF: {rtf}")
        print("engine: ", engine)
        print("llm: ", llm)
        print("language: ", language)
        print("optimization: ", optimization)
        print("num_candidate: ", num_candidate)
        print("num_iteration: ", num_iteration)
        print("neighbor: ", neighbor)
        print("rule: ", rule)


if __name__ == "__main__":
    main()
