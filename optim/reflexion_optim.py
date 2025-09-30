from .utils import *


def generate_candidates(transcript: str,
                        state: str,
                        llm: str = "Qwen3-235B",
                        language: str = "ZH",
                        num_candidate: int = 5, 
                        neighbor: bool = True,
                        iteration: int = 0,
                        normalizer: Normalizer = None,
                        verbose: bool = False) -> str:
    candidates = []
    import jieba
    words = list(jieba.cut(transcript))
    num_words = len(words)
        
    if state == 0:
        num_candidate = 1
    for i in range(num_candidate):
        if neighbor and state > 0:
            modified_words = words.copy()
            if language in ["ZH"]:
                selected_indices = list(range(i, num_words, num_candidate))
            else:
                selected_indices = list(range(i, num_words, num_candidate*3))
            if state == 2:
                selected_indices = [idx for j in range(i, num_words, num_candidate) for idx in (j, j + 1) if idx < num_words]
            for idx in selected_indices:
                word = words[idx]
                if language in ["ZH"]:
                    pinyin = pypinyin.lazy_pinyin(word)
                    if state == 2:
                        for j in range(len(pinyin)):
                            for group in pinyin_confusion_set:
                                if pinyin[j] in group:
                                    alt = random.choice([g for g in group])
                                    pinyin[j] = alt
                    pinyin = "][".join(pinyin)
                    modified_words[idx] = f"[{pinyin}]"

            if language in ["ZH"]:
                transcript_neighbor = "".join(modified_words)
            else:
                transcript_neighbor = " ".join(modified_words)
        else:
            transcript_neighbor = transcript
        
        prompt = f"""
        You will be provided with short {language} ASR (Automatic Speech Recognition) sentences. Your task is to correct these sentences to a standard, error-free form, fixing typos, homophone errors, or grammar mistakes. 
        
        Instructions:
        1. If the sentence has no errors, return the original sentence in uppercase, wrapped in [] and preceded by #original#.
        2. If the sentence has errors, correct it in uppercase, wrapped in [] and preceded by #change#.
        3. Follow these reasoning steps internally (do not output steps, only the final [] result):
        a. Locate defective phrase(s)
        b. Determine pronunciation of defective phrase
        c. Generate multiple candidate corrections based on pronunciation
        d. Choose the candidate that fits the context
        4. You may attempt corrections up to 3 times if necessary, but only output the final corrected sentence.
        5. Only return the [] result; do not include any extra text, explanation, or notes.
        6. Ensure the corrected sentence is in the **same language** as the input.
        7. Do Not change the sentence structure or add new information; only correct existing errors.
        8. Special handling for phonetic transcriptions or Pinyin: If you encounter words/phrases enclosed in square brackets [] (which represent phonetic transcriptions or Pinyin), you need to convert them back to the correct words in the target language.
        
        Example 1:
        Input: BECAUSE HE WANTED TO BREAD A NEW GENERATION OF BAKERS
        Result: #change#[BECAUSE HE WANTED TO BREED A NEW GENERATION OF BAKERS]
        
        Example 2:
        Input: DUE TO THEE THEIR PRAISE OF MAIDEN PURE OF TEEMING MOTHERHOOD
        Result: #original#[DUE TO THEE THEIR PRAISE OF MAIDEN PURE OF TEEMING MOTHERHOOD]
        
        Example 3:
        Input: I SAW A [dɔg] IN THE PARK
        Result: #change#[I SAW A DOG IN THE PARK]
        
        Example 4 (Chinese):
        Input: 正是[yin]为你，她们才赞美这位纯洁少女母性的伟大
        Result: #original#[正是因为你，她们才赞美这位纯洁少女母性的伟大]
        
        Now process this ASR sentence:
        Input: {transcript_neighbor}
        Result:
        """
        model_name = MODEL_MAP.get(llm, MODEL_MAP["Qwen3-235B"])
        client_correct = CLIENT_MAP.get(llm, client_ali)
        response1 = client_correct.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": "You are an expert ASR correction assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
            top_p=0.9,
            max_tokens=512
        )
        raw_output1 = response1.choices[0].message.content.strip()

        match = re.search(r'#(?:original|change)#\[(.*?)\]$', raw_output1.strip(), re.DOTALL)
        try:
            final_sentence = normalizer.normalize(match.group(1).strip())
            candidates.append(final_sentence)
        except:
            pass
        
        if verbose:
            print(f"第{iteration+1}轮迭代，当前输入: {transcript_neighbor}")
            print(f"第{iteration+1}轮迭代，当前输出: {final_sentence}")
    return candidates

def fusion(transcript:str,
           original_transcript: str,
           candidates: list,
           llm: str = "Qwen3-235B",
           language: str = "ZH",
           normalizer: Normalizer = None,
           verbose: bool = False) -> str:
    fusion_prompt = f"""
    You are an expert ASR correction assistant in {language}. You will be provided with the original transcript and multiple candidate corrections 
    of an ASR transcript. Your task is to carefully read ALL candidates and then produce ONE fused 
    correction that combines their strengths, avoids their errors, and preserves the original meaning 
    in the most natural and grammatically correct way.

    Requirements:
    - Use the same language as the candidates.
    - Do NOT add extra content not present in any candidate.
    - Prefer words/phrases that are consistent across multiple candidates.
    - If there is a conflict, choose the one that best preserves semantic correctness and fluency.
    - Only output the fused correction in plain text (no explanations, no special formatting).
    
    Original Transcript: {original_transcript}
    Candidates:
    {json.dumps(candidates, ensure_ascii=False, indent=2)}

    Now produce the single fused correction:
    """
    model_name = MODEL_MAP.get(llm, MODEL_MAP["Qwen3-235B"])
    client_fuse = CLIENT_MAP.get(llm, client_ali)
    resp_fusion = client_fuse.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an ASR correction fusion expert."},
            {"role": "user", "content": fusion_prompt}
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=256
    )
    try:
        fused_candidate = normalizer.normalize(resp_fusion.choices[0].message.content.strip())
    except:
        fused_candidate = transcript
    return fused_candidate
    
def rule_filter(transcript: str,
                candidates: list,
                language: str = "ZH",
                verbose: bool = False) -> str:
    for i in reversed(range(len(candidates))):
        if language in ["ZH"]:
            output = jiwer.process_characters(
                transcript.upper(), 
                candidates[i].upper()
            )
            substitutions, insertions, deletions = jiwer.collect_error_counts(output)
            replacements = [
                (c1, c2)
                for key, count in substitutions.items()
                for _ in range(count)
                for c1, c2 in zip(key[0], key[1])
            ]
            inserted_words = [word for word, count in insertions.items() for _ in range(count)] 
            deleted_words = [word for word, count in deletions.items() for _ in range(count)]
            if len(inserted_words) != 0 or len(deleted_words) != 0:
                del candidates[i]
                continue
            
            def phone_similarity(word1, word2):
                phone1 = ''.join(pypinyin.lazy_pinyin(word1))
                phone2 = ''.join(pypinyin.lazy_pinyin(word2))
                similarity = 1 - Levenshtein.distance(phone1, phone2) / max(len(phone1), len(phone2), 1)
                return similarity
            similarity_threshold = 0.35
            final_replacements = [] 
            for orig, repl in replacements: 
                similarity = phone_similarity(orig, repl)
                if similarity < similarity_threshold:
                    final_replacements.append((orig, repl))
                else: 
                    final_replacements.append((orig, orig))
            for orig, repl in final_replacements: 
                candidates[i] = candidates[i].replace(repl, orig)
    return candidates

def scoring(transcript: str,
            candidates: list,
            llm: str = "Qwen3-235B") -> str:
    model_name = MODEL_MAP.get(llm, MODEL_MAP["Qwen3-235B"])
    client_used = CLIENT_MAP.get(llm, client_ali)

    # Use json.dumps to safely inject the candidate list into the prompt
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    prompt = f"""
    You are an expert evaluator of ASR correction candidates. You will be given a list of candidate corrections. For each candidate, give a single numeric SCORE between 0.0 and 1.0
    (one decimal place, e.g., 0.8) that reflects how well the candidate:
    - preserves the meaning of the original audio (semantic match), and
    - is grammatically correct and natural.

    Return a STRICT JSON object only (no extra text) with this exact shape:

    {{ 
    "scores": [
        {{ "candidate": "<candidate string>", "score": 0.0, "note": "short explanation (optional)" }},
        ...
    ]
    }}

    Rules:
    - Provide one entry per candidate in the same order as the list.
    - Score must be a numeric value between 0.0 and 1.0 (one decimal place).
    - Do not output anything outside the JSON object.
    - DO NOT modify candidate strings; evaluate them as-is.

    Candidates (JSON array):
    {candidates_json}

    Now produce the JSON described above.
    """

    resp = client_used.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a strict evaluator that only returns the requested JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=512
    )

    raw = resp.choices[0].message.content.strip()
    m = re.search(r'(\{[\s\S]*\})', raw)
    parsed = None
    if m:
        try:
            parsed = json.loads(m.group(1))
        except Exception:
            try:
                raw_fixed = m.group(1).replace("'", '"')
                parsed = json.loads(raw_fixed)
            except Exception:
                parsed = None

    if parsed is None:
        parsed = {"scores": []}
        lines = raw.splitlines()
        for cand in candidates:
            found_score = None
            for line in lines:
                if cand in line and re.search(r'([01](?:\.\d)?)', line):
                    num_m = re.search(r'([01](?:\.\d)?)', line)
                    if num_m:
                        found_score = float(num_m.group(1))
                        break
            if found_score is None:
                found_score = 0.0
            parsed["scores"].append({"candidate": cand, "score": round(found_score, 1), "note": "heuristic-extracted"})

    scored_list = []
    for item in parsed.get("scores", []):
        try:
            cand_text = item.get("candidate")
            score_val = float(item.get("score"))
        except Exception:
            cand_text = item.get("candidate", "")
            score_val = 0.0
        scored_list.append({"candidate": cand_text, "score": round(score_val, 1), "note": item.get("note", "")})
    
    best = max(scored_list, key=lambda x: x["score"]) if scored_list else (candidates[0] if candidates else transcript)
    best_candidate = best["candidate"]
    best_comment = best["note"]
    return scored_list, best_candidate, best_comment

def heuristic_search(transcript: str,
                    original_transcript: str,
                    comment: str,
                    state: str,
                    llm: str = "Qwen3-235B",
                    language: str = "ZH",
                    num_candidate: int = 5, 
                    neighbor: bool = True, 
                    rule: bool = True,
                    iteration: int = 0,
                    normalizer: Normalizer = None,
                    verbose: bool = False) -> str:
    candidates = generate_candidates(transcript, state, llm, language, num_candidate, neighbor, iteration, normalizer, verbose)
    if verbose:
        print(f"第{iteration+1}轮迭代，生成候选集: {candidates}")
    
    fused_candidate = fusion(transcript, original_transcript, candidates, llm, language, normalizer, verbose)
    if verbose:
        print(f"第{iteration+1}轮迭代，融合候选: {fused_candidate}")
    candidates.append(fused_candidate)
        
    if rule:
        candidates = rule_filter(transcript, candidates, language, verbose)
    else:
        candidates = candidates
    candidates.append(transcript)
    candidates_unique = []
    for i in range(len(candidates)):
        if candidates[i] not in candidates_unique:
            candidates_unique.append(candidates[i])
    candidates = candidates_unique
    
    if len(candidates) == 1:
        return candidates[0], ""
    
    if verbose:
        print(f"第{iteration+1}轮迭代，当前规则过滤候选集: {candidates}")

    scored_list, best_candidate, best_comment = scoring(transcript, candidates, llm)
    if verbose:
        print(f"第{iteration+1}轮迭代，当前打分结果: {scored_list}")
    return best_candidate, best_comment


def RIF_ASR(transcript: str, 
            llm: str = "Qwen3-235B",
            language: str = "ZH",
            num_candidate: int = 5,
            num_iteration: int = 3,
            neighbor: bool = True,
            rule: bool = True,
            normalizer: Normalizer = None,
            verbose: bool = False) -> str:
    original_transcript = transcript
    state = 0
    comment = ""
    history_state = [-1]
    for i in range(num_iteration):
        if verbose:
            print(f"第{i+1}轮迭代，当前状态: {state}")
        history_state.append(state)
        transcript_current, comment = heuristic_search(transcript, original_transcript, comment, state, llm, language, num_candidate, neighbor, rule, i, normalizer, verbose)
        if language in ["ZH"]:
            import opencc
            cc = opencc.OpenCC('t2s')
            transcript_current = cc.convert(transcript_current)
        if transcript_current == transcript:
            if state == 2:
                if history_state[-1] == 2 and history_state[-2] == 2:
                    state = 3
                else:
                    state = 2
            else:
                state += 1
        else:
            state = 1
        if history_state[-1] == history_state[-2] == history_state[-3] == 1:
            state = 2
        if state == 3:
            break
        transcript = transcript_current
        if verbose:
            print(f"第{i+1}轮迭代，当前结果: {transcript_current}")
    return transcript