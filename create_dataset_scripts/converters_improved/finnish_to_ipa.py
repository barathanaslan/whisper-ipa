import re

class Finnish2IPA:
    pron_dict = {"a": "ɑ",
            "aa": "ɑː",
            "b": "b",
            "bb": "bː",
            "d": "d",
            "dd": "dː",
            "e": "e",
            "ee": "eː",
            "f": "f",
            "ff": "fː",
            "g": "g",
            "h": "h",
            "hh": "hː",
            "i": "i",
            "ii": "iː",
            "j": "j",
            "jj": "jː",
            "k": "k",
            "kk": "kː",
            "l": "l",
            "ll": "lː",
            "m": "m",
            "mm": "mː",
            "n": "n",
            "nn": "nː",
            "ng": "ŋː",
            "nk": "ŋk",
            "o": "o",
            "oo": "oː",
            "p": "p",
            "pp": "pː",
            "r": "r",
            "rr": "rː",
            "s": "s",
            "ss": "sː",
            "t": "t",
            "tt": "tː",
            "u": "u",
            "uu": "uː",
            "v": "ʋ",
            "w": "w",
            "y": "y",
            "z": "z",
            "ä": "æ",
            "ää": "æː",
            "ö": "ø",
            "öö": "øː"
            }

    def remove_punct(self, sent):
        non_punct = r"[\s\w]"
        sent = re.findall(non_punct, sent.lower(), re.MULTILINE)
        sent = "".join(sent)
        return sent

    def convert_ipa(self, sent):
        two = {k: v for k, v in self.pron_dict.items() if len(k) == 2}
        one = {k: v for k, v in self.pron_dict.items() if len(k) == 1}
        for k, v in two.items():
            sent = sent.replace(k, v)
        for k, v in one.items():
            sent = sent.replace(k, v)
        return sent

    @classmethod
    def finnish_generate_ipa(cls, sent):
        sent = cls.remove_punct(cls, sent)
        sent = cls.convert_ipa(cls, sent)
        return sent
