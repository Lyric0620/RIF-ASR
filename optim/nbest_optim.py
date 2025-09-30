from .utils import *

def nbest_optimization(transcript: str,
                        llm: str = "DeepSeek-V3.1",
                        language: str = "EN",
                        num_candidate: int = 3,
                        verbose: bool = False) -> str:
    candidates = []
    for i in range(num_candidate):
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
        
        Example 5 (Cantonese):
        Input: 我今日食咗[AiC]啲嘢
        Result: #change#[我今日食咗餸啲嘢]

        Example 6 (Japanese):
        Input: 私は[でんしゃ]に乗った
        Result: #change#[私は電車に乗った]
        
        Now process this ASR sentence:
        Input: {transcript}
        Result:
        """
        model_name = MODEL_MAP.get(llm, MODEL_MAP["DeepSeek-V3.1"])
        client_used = CLIENT_MAP.get(llm, client_modelscope)
        response1 = client_used.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": "You are an expert ASR correction assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.8,
            top_p=0.6,
            max_tokens=512
        )
        raw_output1 = response1.choices[0].message.content.strip()

        match = re.search(r'#(?:original|change)#\[(.*?)\]$', raw_output1.strip(), re.DOTALL)
        try:
            final_sentence = match.group(1).strip()
            candidates.append(final_sentence)
        except:
            pass
    if candidates[0] != candidates[1] or candidates[0] != candidates[2]:
        print(f"候选集: {candidates}")
    
    # --- 融合步骤 ---
    fusion_prompt = f"""
    You are an expert ASR correction assistant. You will be provided with the original transcript and multiple candidate corrections 
    of an ASR transcript. Your task is to carefully read ALL candidates and then produce ONE fused 
    correction that combines their strengths, avoids their errors, and preserves the original meaning 
    in the most natural and grammatically correct way.

    Requirements:
    - Use the same language as the candidates.
    - Do NOT add extra content not present in any candidate.
    - Prefer words/phrases that are consistent across multiple candidates.
    - If there is a conflict, choose the one that best preserves semantic correctness and fluency.
    - Only output the fused correction in plain text (no explanations, no special formatting).
    
    Original Transcript: {transcript}
    Candidates:
    {json.dumps(candidates, ensure_ascii=False, indent=2)}

    Now produce the single fused correction:
    """

    resp_fusion = client_used.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an ASR correction fusion expert."},
            {"role": "user", "content": fusion_prompt}
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=256
    )
    fused_candidate = resp_fusion.choices[0].message.content.strip()
    return fused_candidate