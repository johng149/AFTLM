from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from tqdm.auto import tqdm
from typing import Union, List


class MC4DatasetBuilder:

    def __init__(self):
        self.available_languages = set(['af', 'am', 'ar', 'az', 'be', 'bg', 'bg-Latn', 'bn', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'el-Latn', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hi-Latn', 'hmn', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn',
                                       'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'und', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zh-Latn', 'zu'])
        self.recommended = set(
            ["en", "es", "de", "fr", "nl", "it", "pt", "hu", "zh", "ja", "ko", "ru"])

    def _build_single(self, language):
        """
        Builds a single pair of training and validation dataset for the given language

        If the language is not available, both will be `None`
        """
        if language not in self.available_languages:
            return None, None

        train_dataset = load_dataset(
            "mc4", language, split="train", streaming=True)
        val_dataset = load_dataset(
            "mc4", language, split="validation", streaming=True)

        return train_dataset, val_dataset

    def build(self, languages, data_col, interleave=False):
        """
        Creates a list of training and validation datasets for the given languages

        If a language is not available, it will be skipped, and will be added to the
        list of skipped languages

        The datasets will not be interleaved by default, but can be by setting the
        `interleave` parameter to `True`

        The data column of the dataset will be renamed to the given `data_col` parameter
        """
        train_datasets = []
        val_datasets = []
        skipped_languages = []

        for language in (pbar := tqdm(languages)):
            pbar.set_description(f"Building dataset for {language}")
            train_dataset, val_dataset = self._build_single(language)
            if train_dataset is None:
                skipped_languages.append(language)
                continue
            else:
                train_dataset = train_dataset.remove_columns(
                    ["timestamp", "url"])
                val_dataset = val_dataset.remove_columns(["timestamp", "url"])
                if data_col != "text":
                    train_dataset = train_dataset.rename_column(
                        "text", data_col)
                    val_dataset = val_dataset.rename_column("text", data_col)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        train_dataset = interleave_datasets(
            train_datasets) if interleave else train_datasets
        val_dataset = interleave_datasets(
            val_datasets) if interleave else val_datasets

        return train_dataset, val_dataset, skipped_languages

    def build_recommended(self, data_col, interleave=False):
        """
        Creates a list of training and validation datasets for the recommended languages

        The datasets will not be interleaved by default, but can be by setting the
        `interleave` parameter to `True`

        The data column of the dataset will be renamed to the given `data_col` parameter
        """
        return self.build(self.recommended, data_col, interleave)


class GutenbergMultilingDatasetBuilder:

    # this is much simpler than the MC4DatasetBuilder, because the Gutenberg dataset
    # doesn't have different splits for each language (they are all together)
    # furthermore, they only have "train" split
    def __init__(self):
        pass

    def build(self, data_col):
        """
        Creates the gutenberg multiling dataset

        The data column of the dataset will be renamed to the given `data_col` parameter
        """
        dataset = load_dataset(
            "sedthh/gutenberg_multilang", streaming=True)["train"]
        if data_col != "TEXT":
            dataset = dataset.rename_column("TEXT", data_col)
        dataset = dataset.remove_columns(["SOURCE", "METADATA"])
        return dataset


class GutenbergEnglishDatasetBuilder:

    def __init__(self):
        pass

    def build(self, data_col):
        """
        Creates the gutenberg english dataset

        The data column of the dataset will be renamed to the given `data_col` parameter
        """
        dataset = load_dataset("sedthh/gutenberg_english",
                               streaming=True)["train"]
        if data_col != "TEXT":
            dataset = dataset.rename_column("TEXT", data_col)
        dataset = dataset.remove_columns(["SOURCE", "METADATA"])
        return dataset


class DatasetBuilder:
    """
    Follows the builder design pattern to create a dataset with
    the given parameters
    """

    def __init__(self, data_col):
        self.trainings = []
        self.validations = []
        self.data_col = data_col
        self.errors = []
        self.warnings = []
        self.counter = 0

    # languages can be a list or a bool
    def add_mc4(self, languages: Union[List[str], bool]):
        """
        param languages:
            list of languages to add to the dataset, or `True` to add
            recommended languages or `False` to add all languages
        """
        if languages is True:
            languages = MC4DatasetBuilder().recommended
        elif languages is False:
            languages = MC4DatasetBuilder().available_languages
        train, val, skipped = MC4DatasetBuilder().build(languages, self.data_col)
        self.trainings.extend(train)
        self.validations.extend(val)
        if len(skipped) > 0:
            self.errors.append(
                f"{self.counter} time: When adding MC4, the following languages were skipped: {skipped}")
        self.counter += 1

    def add_gutenberg_multiling(self):
        train = GutenbergMultilingDatasetBuilder().build(self.data_col)
        self.trainings.append(train)
        self.warnings.append(
            f"{self.counter} time: Added Gutenberg Multiling dataset but it does not have validation data")
        self.counter += 1

    def add_gutenberg_english(self):
        train = GutenbergEnglishDatasetBuilder().build(self.data_col)
        self.trainings.append(train)
        self.warnings.append(
            f"{self.counter} time: Added Gutenberg English dataset but it does not have validation data")
        self.counter += 1

    def build(self, interleave=True):
        """
        param interleave:
            Whether to interleave the datasets or not
        """
        train_dataset = interleave_datasets(self.trainings).with_format(
            "torch") if interleave else self.trainings
        val_dataset = interleave_datasets(self.validations).with_format(
            "torch") if interleave else self.validations
        return train_dataset, val_dataset

    def auto(self, interleave=True, take=None):
        """
        Automatically adds all available datasets to the builder
        MC4 will use recommended languages

        param interleave:
            Whether to interleave the datasets or not
        param take:
            none or int
            If not None and datasets are interleaved, then take a subset of the dataset
        """
        self.add_mc4(True)
        self.add_gutenberg_multiling()
        self.add_gutenberg_english()
        train, val = self.build(interleave)

        # if take is not None, and train is not a list, then take a subset of the dataset
        if take is not None and not isinstance(train, list):
            train = train.take(take)
            val = val.take(take)
        return train, val

    def get_errors(self):
        return self.errors

    def get_warnings(self):
        return self.warnings

    def get_counter(self):
        return self.counter
