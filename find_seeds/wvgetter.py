from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from langdetect import detect
from babel.languages import get_territory_language_info
from collections import Counter
import os
import shutil
from gensim.models import FastText as ft


class WordVectorGetter:

    def __init__(self, country_code=None, text_samples=None, lang=None,
                 tmp_dir=""):

        # Check whether the argument combination is valid
        situation_1 = ((country_code is None)
                       and (text_samples is None)
                       and (lang is not None))
        situation_2 = ((country_code is not None)
                       and (text_samples is not None)
                       and (lang is None))
        assert situation_1 or situation_2, "Invalid argument combination."

        # If the language is not specified then identify it
        if lang is None:
            # Generate the list of official languages for this country
            lang_info = get_territory_language_info(country_code)
            off_langs = [lang for lang, info in lang_info.items()
                         if info["official_status"] is not None]
            # Check that a language was found
            err_msg = "No official languages found for {}".format(country_code)
            assert len(off_langs) > 0, err_msg

            # Identify the language from multiple samples
            langs = [detect(t) for t in text_samples]

            # Check whether the result is valid
            lang, _ = Counter(langs).most_common()[0]
            err_msg = "Language '{}' not in {}".format(lang, off_langs)
            assert lang in off_langs, err_msg
            print(("Decided that '{}' is the "
                   "language of this text").format(lang))

        # Prepare the zip url and filename strings
        self.zip_url = ("https://s3-us-west-1.amazonaws.com/"
                        "fasttext-vectors/wiki.{}.zip").format(lang)
        self.filename = "wiki.{}.bin".format(lang)
        self.path = os.path.join(tmp_dir, self.filename)

    def get_wordvector(self):
        print("Downloading...")
        r = urlopen(self.zip_url)
        with BytesIO(r.read()) as b:
            print("Extracting...")
            with ZipFile(b) as z:
                with z.open(self.filename) as zf, open(self.path, "wb") as f:
                    shutil.copyfileobj(zf, f)
        print("Loading...")
        model = ft.load_fasttext_format(self.path)
        os.remove(self.path)
        return model
