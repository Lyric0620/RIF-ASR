import re
import re
import string
import unicodedata

import inflect

from languages import Languages

SUPPORTED_PUNCTUATION_SET = ",.?'"


class Normalizer(object):
    def __init__(self, keep_punctuation: bool, punctuation_set: str = SUPPORTED_PUNCTUATION_SET) -> None:
        self._keep_punctuation = keep_punctuation
        self._punctuation_set = punctuation_set

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool) -> str:
        raise NotImplementedError()

    @classmethod
    def create(
        cls,
        language: Languages,
        keep_punctuation: bool,
        punctuation_set: str = SUPPORTED_PUNCTUATION_SET,
    ):
        if language == Languages.EN:
            return EnglishNormalizer(keep_punctuation, punctuation_set)
        elif language in [Languages.ZH, Languages.YUE]:
            return ChineseNormalizer(keep_punctuation, punctuation_set)
        elif language in [
            Languages.DE,
            Languages.ES,
            Languages.FR,
            Languages.IT,
            Languages.PT_PT,
            Languages.PT_BR,
        ]:
            return DefaultNormalizer(keep_punctuation, punctuation_set)
        elif language == Languages.JA:
            return JapaneseNormalizer(keep_punctuation, punctuation_set)
        elif language == Languages.BO:
            return TibetanNormalizer(keep_punctuation, punctuation_set)
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type `{language}`")


class DefaultNormalizer(Normalizer):
    """
    Adapted from: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
    """

    ADDITIONAL_DIACRITICS = {
        "œ": "oe",
        "Œ": "OE",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "AE",
        "ß": "ss",
        "ẞ": "SS",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "th",
        "ł": "l",
        "Ł": "L",
    }

    def _remove_symbols_and_diacritics(self, s: str) -> str:
        return "".join(
            (
                DefaultNormalizer.ADDITIONAL_DIACRITICS[c]
                if c in DefaultNormalizer.ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else (
                        " "
                        if unicodedata.category(c)[0] in "MS"
                        or (unicodedata.category(c)[0] == "P" and c not in SUPPORTED_PUNCTUATION_SET)
                        else c
                    )
                )
            )
            for c in unicodedata.normalize("NFKD", s)
        )

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        sentence = sentence.lower()
        sentence = re.sub(r"[<\[][^>\]]*[>\]]", "", sentence)
        sentence = re.sub(r"\(([^)]+?)\)", "", sentence)
        sentence = sentence.replace("!", ".")
        sentence = sentence.replace("...", "")
        sentence = self._remove_symbols_and_diacritics(sentence).lower()

        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        sentence = re.sub(r"\s+", " ", sentence)

        return sentence


class EnglishNormalizer(Normalizer):
    AMERICAN_SPELLINGS = {
        "acknowledgement": "acknowledgment",
        "analogue": "analog",
        "armour": "armor",
        "ascendency": "ascendancy",
        "behaviour": "behavior",
        "behaviourist": "behaviorist",
        "cancelled": "canceled",
        "catalogue": "catalog",
        "centre": "center",
        "centres": "centers",
        "colour": "color",
        "coloured": "colored",
        "colourist": "colorist",
        "colourists": "colorists",
        "colours": "colors",
        "cosier": "cozier",
        "counselled": "counseled",
        "criticised": "criticized",
        "crystallise": "crystallize",
        "defence": "defense",
        "discoloured": "discolored",
        "dishonour": "dishonor",
        "dishonoured": "dishonored",
        "encyclopaedia": "Encyclopedia",
        "endeavour": "endeavor",
        "endeavouring": "endeavoring",
        "favour": "favor",
        "favourite": "favorite",
        "favours": "favors",
        "fibre": "fiber",
        "flamingoes": "flamingos",
        "fulfill": "fulfil",
        "grey": "gray",
        "harmonised": "harmonized",
        "honour": "honor",
        "honourable": "honorable",
        "honourably": "honorably",
        "honoured": "honored",
        "honours": "honors",
        "humour": "humor",
        "islamised": "islamized",
        "labour": "labor",
        "labourers": "laborers",
        "levelling": "leveling",
        "luis": "lewis",
        "lustre": "luster",
        "manoeuvring": "maneuvering",
        "marshall": "marshal",
        "marvellous": "marvelous",
        "merchandising": "merchandizing",
        "milicent": "millicent",
        "moustache": "mustache",
        "moustaches": "mustaches",
        "neighbour": "neighbor",
        "neighbourhood": "neighborhood",
        "neighbouring": "neighboring",
        "neighbours": "neighbors",
        "omelette": "omelet",
        "organisation": "organization",
        "organiser": "organizer",
        "practise": "practice",
        "pretence": "pretense",
        "programme": "program",
        "realise": "realize",
        "realised": "realized",
        "recognised": "recognized",
        "shrivelled": "shriveled",
        "signalling": "signaling",
        "skilfully": "skillfully",
        "smouldering": "smoldering",
        "specialised": "specialized",
        "sterilise": "sterilize",
        "sylvia": "silvia",
        "theatre": "theater",
        "theatres": "theaters",
        "travelled": "traveled",
        "travellers": "travelers",
        "travelling": "traveling",
        "vapours": "vapors",
        "wilful": "willful",
    }

    ABBREVIATIONS = {
        "junior": "jr",
        "senior": "sr",
        "okay": "ok",
        "doctor": "dr",
        "mister": "mr",
        "missus": "mrs",
        "saint": "st",
    }

    APOSTROPHE_REGEX = r"(?<!\w)\'|\'(?!\w)"  # Apostrophes that are not part of a contraction

    @staticmethod
    def to_american(sentence: str) -> str:
        return " ".join(
            [
                (EnglishNormalizer.AMERICAN_SPELLINGS[x] if x in EnglishNormalizer.AMERICAN_SPELLINGS else x)
                for x in sentence.split()
            ]
        )

    @staticmethod
    def normalize_abbreviations(sentence: str) -> str:
        return " ".join(
            [
                (EnglishNormalizer.ABBREVIATIONS[x] if x in EnglishNormalizer.ABBREVIATIONS else x)
                for x in sentence.split()
            ]
        )

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        p = inflect.engine()

        sentence = sentence.lower()

        for c in "-/–—":
            sentence = sentence.replace(c, " ")

        for c in '‘":;“”`()[]':
            sentence = sentence.replace(c, "")

        sentence = sentence.replace("!", ".")
        sentence = sentence.replace("...", "")

        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        sentence = sentence.replace("’", "'").replace("&", "and")

        sentence = re.sub(self.APOSTROPHE_REGEX, "", sentence)

        def num2txt(y):
            return p.number_to_words(y).replace("-", " ").replace(",", "") if any(x.isdigit() for x in y) else y

        sentence = " ".join(num2txt(x) for x in sentence.split())

        if raise_error_on_invalid_sentence:
            valid_characters = " '" + self._punctuation_set if self._keep_punctuation else " '"
            if not all(c in valid_characters + string.ascii_lowercase for c in sentence):
                raise RuntimeError()
            if any(x.startswith("'") for x in sentence.split()):
                raise RuntimeError()

        return sentence
    
    
class ChineseNormalizer(Normalizer):
    """
    A simple Chinese normalizer: unify punctuation, remove invalid symbols, 
    and optionally keep or drop punctuation.
    """

    PUNCTUATION_MAP = {
        "，": ",",
        "。": ".",
        "？": "?",
        "！": "!",
        "：": ":",
        "；": ";",
        "、": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "",
        "》": "",
    }

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        for zh_punc, en_punc in ChineseNormalizer.PUNCTUATION_MAP.items():
            sentence = sentence.replace(zh_punc, en_punc)

        sentence = re.sub(r"\s+", "", sentence)

        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        if raise_error_on_invalid_sentence:
            if not all(
                (u'\u4e00' <= ch <= u'\u9fff') or ch in self._punctuation_set or ch.isascii()
                for ch in sentence
            ):
                raise RuntimeError(f"Invalid character found: {sentence}")

        return sentence



class JapaneseNormalizer(Normalizer):
    """
    A simple Japanese normalizer: unify punctuation, remove invalid symbols,
    and optionally keep or drop punctuation.
    """

    PUNCTUATION_MAP = {
        "、": ",",
        "。": ".",
        "？": "?",
        "！": "!",
        "：": ":",
        "；": ";",
        "「": "'",
        "」": "'",
        "『": "'",
        "』": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "",
        "》": "",
        "〜": "~",
        "～": "~",
        "ー": "-",
        " ": "",
        "-": "",
        "・": ""
    }

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        # Replace Japanese punctuation with standard punctuation
        for ja_punc, en_punc in JapaneseNormalizer.PUNCTUATION_MAP.items():
            sentence = sentence.replace(ja_punc, en_punc)

        # Remove extra whitespace
        sentence = re.sub(r"\s+", " ", sentence).strip()

        # Handle punctuation based on keep_punctuation flag
        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        # Validate characters if required
        if raise_error_on_invalid_sentence:
            # Japanese characters range: Hiragana, Katakana, Kanji, and common symbols
            valid_ranges = [
                (0x3040, 0x309F),  # Hiragana
                (0x30A0, 0x30FF),  # Katakana
                (0x4E00, 0x9FFF),  # Kanji
                (0xFF00, 0xFFEF),  # Full-width ASCII
            ]
            
            def is_valid_char(ch):
                code = ord(ch)
                return (any(start <= code <= end for start, end in valid_ranges) or
                        ch in self._punctuation_set or
                        ch.isascii())
            
            if not all(is_valid_char(ch) for ch in sentence):
                raise RuntimeError(f"Invalid character found: {sentence}")

        return sentence


class TibetanNormalizer(Normalizer):
    """
    A simple Tibetan normalizer: unify punctuation, remove invalid symbols,
    and optionally keep or drop punctuation.
    """

    # 藏文标点符号映射到标准标点符号
    PUNCTUATION_MAP = {
        "་": " ",  # 藏文音节分隔符 -> 空格
        "།": ",",  # 藏文逗号 -> 英文逗号
        "༎": ".",  # 藏文句号 -> 英文句号
        "？": "?",  # 问号
        "！": "!",
        "：": ":",
        "；": ";",
        "「": "'",
        "」": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "",
        "》": "",
        "～": "~",
        "ー": "-",
    }

    def normalize(self, sentence: str, raise_error_on_invalid_sentence: bool = False) -> str:
        # 替换藏文标点符号为标准标点符号
        for bo_punc, en_punc in TibetanNormalizer.PUNCTUATION_MAP.items():
            sentence = sentence.replace(bo_punc, en_punc)

        # 移除多余的空白字符
        sentence = re.sub(r"\s+", " ", sentence).strip()

        # 根据keep_punctuation标志处理标点符号
        if self._keep_punctuation:
            removable_punctuation = "".join(set(SUPPORTED_PUNCTUATION_SET) - set(self._punctuation_set))
        else:
            removable_punctuation = SUPPORTED_PUNCTUATION_SET

        for c in removable_punctuation:
            sentence = sentence.replace(c, "")

        # 如果需要，验证字符
        if raise_error_on_invalid_sentence:
            # 藏文字符范围：主要包括藏文字母、数字和常见符号
            valid_ranges = [
                (0x0F00, 0x0FFF),  # 藏文字母、数字和符号
                (0x11400, 0x1147F),  # 藏文扩展-A
                (0x11480, 0x114DF),  # 藏文扩展-B
            ]
            
            def is_valid_char(ch):
                code = ord(ch)
                return (any(start <= code <= end for start, end in valid_ranges) or
                        ch in self._punctuation_set or
                        ch.isascii())
            
            if not all(is_valid_char(ch) for ch in sentence):
                raise RuntimeError(f"Invalid character found: {sentence}")

        return sentence


__all__ = ["Normalizer"]
