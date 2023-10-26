import datasets
import langdetect
import tqdm

_BASE_DATASET = "timdettmers/openassistant-guanaco"
# Samples that contain any of these words are removed.
_FORBIDDEN_WORDS = ["python", "script", "ascii"]


def get_dataset(split: str) -> datasets.Dataset:
    base = datasets.load_dataset(_BASE_DATASET, split=split)
    rows = []
    for row in tqdm.tqdm(base["text"]):
        try:
            if any(word in row.lower() for word in _FORBIDDEN_WORDS):
                continue

            # Try to prevent detecting English because of these words.
            row_replaced = row.replace("### Human:", "")
            row_replaced = row_replaced.replace("### Assistant:", "")
            if langdetect.detect(row_replaced) != "en":
                continue

            rows.append(row)

        except:
            continue

    return datasets.Dataset.from_dict({"text": rows})
