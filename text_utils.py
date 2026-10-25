# coding: utf-8
"""Phoneme alphabet for Bengali PL-BERT.

The symbol table below is copied VERBATIM from StyleTTS2's meldataset.py.
Do not edit it. If these two alphabets ever disagree, PL-BERT's embeddings are
indexed by different phonemes than the TTS model uses and the whole transfer
silently produces garbage.
"""

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

assert len(symbols) == 178, f"symbol table must stay at 178, got {len(symbols)}"


class TextCleaner:
    """Map a phoneme string to symbol indices (same rules as StyleTTS2)."""

    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        text = text.replace("\u0964", ".")
        indexes = []
        for char in text:
            if char == "\u0303":
                continue
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(f"[UNK CHAR] {repr(char)} in: {text}")
        return indexes
