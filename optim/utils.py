import re
import json
import math
import random
import jiwer
import pypinyin
import Levenshtein
import torchaudio
import time
import opencc
import jieba
from typing import List, Tuple, Dict, Any, Optional
from openai import OpenAI
from openai import APIError, APIConnectionError, APIStatusError

from normalizer import Normalizer

client_ali = OpenAI(
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=''
)

MODEL_MAP = {
    "Qwen3-235B": "qwen3-235b-a22b-instruct-2507",
}

CLIENT_MAP = {
    "Qwen3-235B": client_ali
}

from g2p_en import G2p
import epitran
g2p_en = G2p()
g2p_en_1 = epitran.Epitran('eng-Latn')

pinyin_confusion_set = [["quan", "xuan"],
                        ["gou", "ge"],
                        ["xu", "qu"],
                        ['ai', 'a'],
                        ["ning", "ni"],
                        ["lin", "ling"],
                        ["sheng", "shen"],
                        ["hun", "huan"],
                        ["sui", "shen"],
                        ["pian", "bian"],
                        ["yi", "yu"],
                        ["huang", "wang"],
                        ["geng", "gen"],
                        ["chong", "cheng"],
                        ["you", "jiu"],
                        ["zhi", "shi"],
                        ["dai", "zai"],
                        ["chang", "chan"],
                        ["li", "di"],
                        ["bian", "bin"],
                        ["li", "bi"],
                        ["hou", "he"],
                        ["pin", "bin"],
                        ["mo", "mu"],
                        ["dai", "zai"],
                        ["xin", "xi"],
                        ["mu", "fu"],
                        ["xian", "xia"],
                        ["pu", "bu"],
                        ["liao", "yao"],
                        ["pin", "bin"],
                        ["qing", "xin"],
                        ["cheng", "chen"],
                        ["xing", "xin"],
                        ["yu", "ju"],
                        ["ju", "ji"],
                        ["lie", "nie"],
                        ["huo", "he"],
                        ["lu", "ru"]
                        ]