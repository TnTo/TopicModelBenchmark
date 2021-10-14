from itertools import chain
from os import environ, system

import adso
import dask
import nltk
from dask.distributed import Client
from pymongo import MongoClient
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(
    MongoObserver(
        url=f"mongodb://{environ['MONGO_INITDB_ROOT_USERNAME']}:{environ['MONGO_INITDB_ROOT_PASSWORD']}@mongo:27017/{environ['MONGO_DATABASE']}?authSource=admin",
        db_name=environ["MONGO_DATABASE"],
    )
)


@ex.config
def default_config():
    dataset_name = ""
    model_name = ""
    adso_seed = 0
    dataset_func = lambda: None
    dataset_kwargs = {}
    vect_kwargs = {}
    n_topic = 0
    model_func = lambda: None
    model_kwargs = {}
    model_args = []


@ex.main
def main(
    dataset_name,
    dataset_func,
    dataset_kwargs,
    vect_kwargs,
    n_topic,
    model_name,
    model_func,
    model_kwargs,
    model_args,
    adso_seed,
):

    adso.set_seed(adso_seed)

    try:
        print("Load...")
        dataset = adso.data.LabeledDataset.load(f"/data/benchmark/{dataset_name}")
    except FileNotFoundError:
        print("Create dataset...")
        dataset = dataset_func(dataset_name, **dataset_kwargs)
        dataset.set_vectorizer_params(**vect_kwargs)

    print("Init model...")
    model = model_func(*model_args, **model_kwargs)

    print("Fit!")
    return model.fit_transform(
        dataset, f"{dataset_name}_{model_name}_{adso_seed}_{n_topic}"
    )


adso.data.common.nltk_download("punkt")


def my_tokenizer(doc):
    return list(
        filter(
            lambda s: s.isalpha() and len(s) >= 3,
            adso.data.common.tokenize_and_stem(doc),
        )
    )


def simple_tokenizer(doc):
    tokenizer = nltk.tokenize.word_tokenize
    return list(filter(lambda s: s.isalpha() and len(s) >= 3, tokenizer(doc),))


stopwords = adso.data.common.get_nltk_stopwords()


def simple_filter(word):
    if word in stopwords:
        return False
    return word.isalpha() and len(word) >= 3


if __name__ == "__main__":

    print("Start!")

    adso.set_adso_dir("/data")
    adso.set_project_name("benchmark")

    dask.config.set({"temporary_directory": str(adso.common.ADSODIR / "dask")})
    dask_client = Client()

    # gc.set_threshold(200, 10, 10)

    adso.data.common.nltk_download("punkt")

    mongo_client = MongoClient(
        f"mongodb://{environ['MONGO_INITDB_ROOT_USERNAME']}:{environ['MONGO_INITDB_ROOT_PASSWORD']}@mongo:27017/{environ['MONGO_DATABASE']}?authSource=admin"
    )
    db = mongo_client.sacred

    for dataset_name in ["20newsgroups", "wos"]:
        for model_name in [
            "NMF",
            "LDAVB",
            "LDAVBlong",
            "LDAGS",
            "LDAGSlong",
            "HDPVB",
            "HDPGS",
            "TM",
            "hSBM",
        ]:
            for adso_seed in range(1000, 1005):
                config = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "adso_seed": adso_seed,
                }
                config["vect_kwargs"] = {
                    "min_df": 5,
                    "tokenizer": my_tokenizer,
                    "stop_words": adso.data.common.get_nltk_stopwords(),
                    "strip_accents": "unicode",
                }
                # DATASET
                if dataset_name == "20newsgroups":
                    config["dataset_func"] = adso.corpora.get_20newsgroups
                    config["dataset_kwargs"] = {}
                    topic_range = chain(range(10, 31), [5], range(10, 101, 10))
                elif dataset_name == "wos":
                    config["dataset_func"] = adso.corpora.get_wos
                    config["dataset_kwargs"] = {}
                    topic_range = chain(range(5, 21), range(10, 101, 10))
                for n_topic in topic_range:
                    config["n_topic"] = n_topic
                    # ALGO
                    if model_name == "NMF":
                        config["model_func"] = adso.algorithms.NMF
                        config["model_kwargs"] = {"max_iter": 2000}
                        config["model_args"] = [n_topic]
                    elif model_name == "LDAVBlong":
                        config["model_func"] = adso.algorithms.LDAVB
                        config["model_kwargs"] = {"passes": 5, "iterations": 250}
                        config["model_args"] = [n_topic]
                    elif model_name == "LDAGSlong":
                        config["model_func"] = adso.algorithms.LDAGS
                        config["model_kwargs"] = {
                            "memory": "12G",
                            "mallet_args": {
                                "num-iterations": 2000,
                                "optimize-interval": 50,
                            },
                        }
                        config["model_args"] = [n_topic]
                    elif model_name == "LDAVB":
                        config["model_func"] = adso.algorithms.LDAVB
                        config["model_kwargs"] = {}
                        config["model_args"] = [n_topic]
                    elif model_name == "LDAGS":
                        config["model_func"] = adso.algorithms.LDAGS
                        config["model_kwargs"] = {"memory": "12G"}
                        config["model_args"] = [n_topic]
                    elif model_name == "HDPVB":
                        config["n_topic"] = 0
                        config["model_func"] = adso.algorithms.HDPVB
                        config["model_kwargs"] = {}
                        config["model_args"] = []
                    elif model_name == "HDPGS":
                        config["n_topic"] = 0
                        config["model_func"] = adso.algorithms.HDPGS
                        config["model_kwargs"] = {}
                        config["model_args"] = []
                    elif model_name == "TM":
                        config["n_topic"] = 0
                        config["model_func"] = adso.algorithms.TopicMapping
                        config["model_kwargs"] = {}
                        config["model_args"] = []
                    elif model_name == "hSBM":
                        config["n_topic"] = 0
                        config["model_func"] = adso.algorithms.hSBM
                        config["model_kwargs"] = {}
                        config["model_args"] = []
                    #
                    #  config ready
                    #
                    print(
                        f"Config: {config['dataset_name']} {config['model_name']} {config['adso_seed']} {config['n_topic']}"
                    )

                    if (
                        db.runs.find_one(
                            {
                                "config.dataset_name": config["dataset_name"],
                                "config.model_name": config["model_name"],
                                "config.adso_seed": config["adso_seed"],
                                "config.n_topic": config["n_topic"],
                                "status": "COMPLETED",
                            },
                            max_time_ms=None,
                        )
                        is None
                    ):
                        print("Run!")
                        ex.run(config_updates=config)
                        print("Dump db")
                        system(
                            f"mongodump --forceTableScan --uri mongodb://{environ['MONGO_INITDB_ROOT_USERNAME']}:{environ['MONGO_INITDB_ROOT_PASSWORD']}@mongo:27017/{environ['MONGO_DATABASE']}?authSource=admin --gzip --out /data/dump"
                        )
                    else:
                        print("Skip!")

                    if config["n_topic"] == 0:
                        break
