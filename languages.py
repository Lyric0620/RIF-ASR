from enum import Enum


class Languages(Enum):
    EN = "EN"
    DE = "DE"
    ES = "ES"
    FR = "FR"
    IT = "IT"
    ZH = "ZH"
    BO = "BO"
    PT_PT = "PT-PT"
    PT_BR = "PT-BR"
    YUE = "YUE"
    JA = "JA"
    KO = "KO"


LANGUAGE_TO_CODE = {
    Languages.EN: "en-US",
    Languages.DE: "de-DE",
    Languages.YUE: "yue",
    Languages.JA: "ja-jp",
    Languages.KO: "kor",
    Languages.BO: "bo-ZH",
    Languages.ES: "es-ES",
    Languages.FR: "fr-FR",
    Languages.IT: "it-IT",
    Languages.ZH: "zh-ZH",
    Languages.PT_PT: "pt-PT",
    Languages.PT_BR: "pt-BR",
}

__all__ = [
    "LANGUAGE_TO_CODE",
    "Languages",
]
