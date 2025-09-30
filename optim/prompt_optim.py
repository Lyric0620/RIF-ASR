from .utils import *

def prompt_optimization(transcript: str, llm: str = "DeepSeek-V3.1") -> str:
    prompt = f"""
    You will be provided with short ASR (Automatic Speech Recognition) sentences. Your task is to correct these sentences to a standard, error-free form, fixing typos, homophone errors, or grammar mistakes. 
    
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
    
    Example 1:
    Input: BECAUSE HE WANTED TO BREAD A NEW GENERATION OF BAKERS
    Result: #change#[BECAUSE HE WANTED TO BREED A NEW GENERATION OF BAKERS]
    
    Example 2:
    Input: DUE TO THEE THEIR PRAISE OF MAIDEN PURE OF TEEMING MOTHERHOOD
    Result: #original#[DUE TO THEE THEIR PRAISE OF MAIDEN PURE OF TEEMING MOTHERHOOD]
    
    Example 3 (simple chinese):
    Input: 正是因为你，她们才赞美这位纯洁少女母性的伟大
    Result: #original#[正是因为你，她们才赞美这位纯洁少女母性的伟大]
    
    Now process this ASR sentence:
    Input: {transcript}
    Result:
    """
    model_name = MODEL_MAP.get(llm, MODEL_MAP["DeepSeek-V3.1"])
    client_used = CLIENT_MAP.get(llm, client_modelscope)
    response = client_used.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "You are an expert ASR correction assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        max_tokens=512
    )
    raw_output = response.choices[0].message.content.strip()

    match = re.search(r'\[(.*?)\]', raw_output)
    if match:
        return match.group(1).strip()
    else:
        return raw_output

def evolutionary_prompt_optimization(transcript: str, llm: str = "DeepSeek-V3.1") -> str:
    prompt = f"""
    As a linguistic expert specializing in speech-to-text analysis, your task is to refine multiple transcription hypotheses into a single, precise representation of an audio recording. Examine five distinct interpretations, weighing factors such as semantic coherence, contextual appropriateness, and idiomatic accuracy. Synthesize these inputs to produce one polished sentence that best captures the audio's true content. Your output should demonstrate impeccable grammar, natural flow, and fidelity to the original meaning, without any mention of the analysis process or source hypotheses. Prioritize clarity and concision in your final transcription, ensuring it stands as a seamless, standalone representation of the spoken content.
    
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
    
    Example 1:
    Input: BECAUSE HE WANTED TO BREAD A NEW GENERATION OF BAKERS
    Result: #change#[BECAUSE HE WANTED TO BREED A NEW GENERATION OF BAKERS]
    
    Example 2:
    Input: DUE TO THEE THEIR PRAISE OF MAIDEN PURE OF TEEMING MOTHERHOOD
    Result: #original#[DUE TO THEE THEIR PRAISE OF MAIDEN PURE OF TEEMING MOTHERHOOD]
    
    Example 3 (simple chinese):
    Input: 正是因为你，她们才赞美这位纯洁少女母性的伟大
    Result: #original#[正是因为你，她们才赞美这位纯洁少女母性的伟大]
    
    Now process this ASR sentence:
    Input: {transcript}
    Result:
    """
    model_name = MODEL_MAP.get(llm, MODEL_MAP["DeepSeek-V3.1"])
    client_used = CLIENT_MAP.get(llm, client_modelscope)
    response = client_used.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "You are an expert ASR correction assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        max_tokens=512
    )
    raw_output = response.choices[0].message.content.strip()

    match = re.search(r'\[(.*?)\]', raw_output)
    if match:
        return match.group(1).strip()
    else:
        return raw_output