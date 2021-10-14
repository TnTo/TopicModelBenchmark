#%%
# import
import gzip
import os
import re

import adso
import bson
import dask
import numpy as np
import pandas as pd
from dask.distributed import Client
from scipy.sparse import csc_matrix as csr
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

adso.set_adso_dir("/data")
adso.set_project_name("benchmark")

#%%
# aux func
def n_nnz(row):
    c = adso.data.LabeledDataset.load(
        "/data/benchmark/" + row.config_dataset_name
    ).get_count_matrix()
    s = 0
    n = 0
    for r in c:
        s += np.sum(r)
        n += np.count_nonzero(r)
    return (s, n)


def get_nlevel(row):
    if row.config_model_name in hierarchical:
        return row["result_py/tuple"][1]["py/tuple"][0]
    else:
        return 1


def get_ll(row):
    if "NMF" in row.config_model_name:
        return row["result_py/tuple"][1]["py/tuple"][1]["value"]
    elif "LDAVB" in row.config_model_name:
        return float(
            re.search(r"dnuob drow\-rep ([0-9]+\.[0-9]+\-)", row.captured_out[::-1])[1][
                ::-1
            ]
        )
    elif "LDAGS" in row.config_model_name:
        return float(
            re.search(r"([0-9]{1,5}\.[0-9]+\-) \:nekot\/LL", row.captured_out[::-1])[1][
                ::-1
            ]
        )
    else:
        return np.nan


def get_perpl(row):
    if "LDAVB" in row.config_model_name:
        return float(
            re.search(r"ytixelprep ([0-9]+\.[0-9]+)", row.captured_out[::-1])[1][::-1]
        )
    else:
        return np.nan


def get_nt(row):
    if row.config_model_name in nonparametric:
        return row["result_py/tuple"][1]["py/tuple"][0]
    elif row.config_model_name in hierarchical:
        return (
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path
            )[row.level]
            .get_doc_topic_matrix()
            .shape[1]
        )
    else:
        return np.nan


def get_ant(row):
    if row.config_model_name in (parametric + nonparametric):
        return len(
            np.unique(
                adso.data.topicmodel.TopicModel.load(
                    "/data/benchmark/" + row.path
                ).get_labels()
            )
        )
    elif row.config_model_name in hierarchical:
        return len(
            np.unique(
                adso.data.topicmodel.HierarchicalTopicModel.load(
                    "/data/benchmark/" + row.path
                )[row.level].get_cluster_labels()
            )
        )
    else:
        return np.nan


def compute_nmi(row, soft=False):
    nmi = adso.metrics.supervised.softNMI if soft else adso.metrics.supervised.NMI
    if row.config_model_name not in hierarchical:
        return nmi(
            adso.data.LabeledDataset.load("/data/benchmark/" + row.config_dataset_name),
            adso.data.topicmodel.TopicModel.load("/data/benchmark/" + row.path),
        )
    else:
        return cluster_nmi(row, soft=soft)


def cluster_nmi(row, soft=False):
    if row.config_model_name in hierarchical:
        if not soft:
            return normalized_mutual_info_score(
                adso.data.LabeledDataset.load(
                    "/data/benchmark/" + row.config_dataset_name
                ).get_labels_vect(),
                adso.data.topicmodel.HierarchicalTopicModel.load(
                    "/data/benchmark/" + row.path
                ).get_cluster_labels(l=row.level),
            )
        else:
            c = csr(
                adso.data.LabeledDataset.load(
                    "/data/benchmark/" + row.config_dataset_name
                ).get_labels_matrix()
            ).T @ csr(
                adso.data.topicmodel.HierarchicalTopicModel.load(
                    "/data/benchmark/" + row.path
                ).get_doc_cluster_matrix(l=row.level, normalize=True)
            )
            norm = (entropy(c.sum(axis=0), axis=1) + entropy(c.sum(axis=1))).item()
            mi = mutual_info_score(
                None,
                None,
                contingency=c,
            )
            return mi / norm
    else:
        return np.nan


def auto_nmi(row, level="level"):
    if row.config_model_name in hierarchical:
        return normalized_mutual_info_score(
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_x
            )[row["level_x"]].get_cluster_labels(),
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_y
            )[row["level_y"]].get_cluster_labels(),
        )
    else:
        return normalized_mutual_info_score(
            adso.data.topicmodel.TopicModel.load(
                "/data/benchmark/" + row.path_x
            ).get_labels(),
            adso.data.topicmodel.TopicModel.load(
                "/data/benchmark/" + row.path_y
            ).get_labels(),
        )


def auto_softnmi(row):
    if row.config_model_name in hierarchical:
        c = csr(
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_x
            )[row["level_x"]].get_doc_cluster_matrix(normalize=True)
        ).T @ csr(
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_y
            )[row["level_y"]].get_doc_cluster_matrix(normalize=True)
        )
    else:
        c = csr(
            adso.data.topicmodel.TopicModel.load(
                "/data/benchmark/" + row.path_x
            ).get_doc_topic_matrix(normalize=True)
        ).T @ csr(
            adso.data.topicmodel.TopicModel.load(
                "/data/benchmark/" + row.path_y
            ).get_doc_topic_matrix(normalize=True)
        )
    norm = (entropy(c.sum(axis=0), axis=1) + entropy(c.sum(axis=1))).item()
    mi = mutual_info_score(
        None,
        None,
        contingency=c,
    )
    return mi / norm


def reverse_auto_nmi(row):
    if row.config_model_name in hierarchical:
        return normalized_mutual_info_score(
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_x
            )[row["level"]].get_cluster_labels(),
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_y
            )[row["level"]].get_cluster_labels(),
        )
    else:
        return np.nan


def reverse_auto_softnmi(row):
    if row.config_model_name in hierarchical:
        c = csr(
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_x
            )[row["level"]].get_doc_cluster_matrix(normalize=True)
        ).T @ csr(
            adso.data.topicmodel.HierarchicalTopicModel.load(
                "/data/benchmark/" + row.path_y
            )[row["level"]].get_doc_cluster_matrix(normalize=True)
        )
        norm = (entropy(c.sum(axis=0), axis=1) + entropy(c.sum(axis=1))).item()
        mi = mutual_info_score(
            None,
            None,
            contingency=c,
        )
        return mi / norm
    else:
        return np.nan


#%%
# some constant
old = ["hSBM", "LDAGSlong"]
spectral = ["NMF"]
lda = ["LDAVB", "LDAVBlong", "LDAGS", "LDAGSlonger"]
parametric = spectral + lda
nonparametric = ["HDPVB", "HDPGS", "TM"]
hierarchical = ["hSBMv2"]
order = parametric + nonparametric + hierarchical

if __name__ == "__main__":

    # prelude
    try:
        os.mkdir("data")
    except FileExistsError:
        pass

    dask.config.set({"temporary_directory": str(adso.common.ADSODIR / "dask")})
    dask_client = Client()

    # additional dataset
    try:
        adso.data.LabeledDataset.load("/data/benchmark/wos2")
    except FileNotFoundError:

        def my_tokenizer(doc):
            return list(
                filter(
                    lambda s: s.isalpha() and len(s) >= 3,
                    adso.data.common.tokenize_and_stem(doc),
                )
            )

        adso.corpora.get_wos("wos2", subfields=True).set_vectorizer_params(
            min_df=5,
            tokenizer=my_tokenizer,
            stop_words=adso.data.common.get_nltk_stopwords(),
            strip_accents="unicode",
        )

    # load bson
    data_file = gzip.open("/data/dump/sacred/runs.bson.gz")

    # filter completed runs
    data_bson = bson.decode_file_iter(data_file)
    data = filter(lambda item: item["status"] == "COMPLETED", data_bson)
    # pprint(next(data))

    # load into pandas
    df = pd.DataFrame.from_dict(pd.json_normalize(data, max_level=1, sep="_"))
    df = df[~df.config_model_name.isin(old)]  # discard old hSBM
    df.to_pickle("data/df_clean.pkl")
    # pprint(df.columns)

    # path
    df["path"] = df.apply(
        lambda row: "_".join(
            [
                row.config_dataset_name,
                row.config_model_name,
                str(row.config_adso_seed),
                str(row.config_n_topic),
            ]
        ),
        axis=1,
    )

    # add additional labels to dataset
    og_dataset = df.config_dataset_name.unique()

    df = df.append(
        df[df.config_dataset_name == "wos"].replace(to_replace="wos", value="wos2")
    ).reset_index(drop=True)

    # columns
    df["time"] = (df.stop_time - df.start_time).dt.seconds
    df["nlevel"] = df.apply(get_nlevel, axis=1)
    df["level"] = df.apply(lambda row: list(range(row.nlevel)), axis=1)
    df = df.explode("level")
    df["plotlevel"] = df.nlevel - df.level  # inverse ordered levels
    df["ll"] = df.apply(get_ll, axis=1)
    df["perpl"] = df.apply(get_perpl, axis=1)

    # slow columns

    # nmi
    df["nmi"] = df.apply(
        lambda row: compute_nmi(row),
        axis=1,
    )

    # softnmi
    df["softnmi"] = df.apply(
        lambda row: compute_nmi(row, soft=True),
        axis=1,
    )

    # nt
    df["nt"] = df.apply(get_nt, axis=1)

    # assigned nt
    df["ant"] = df.apply(get_ant, axis=1)

    # save
    df.to_pickle("data/df.pkl")

    # buld list of autopairs
    df_light = df[
        [
            "config_dataset_name",
            "config_model_name",
            "config_adso_seed",
            "config_n_topic",
            "level",
            "plotlevel",
            "path",
        ]
    ]
    df_auto = df_light.merge(
        df_light,
        on=["config_dataset_name", "config_model_name", "config_n_topic", "plotlevel"],
        how="outer",
    )

    df_auto = df_auto[df_auto.config_adso_seed_x < df_auto.config_adso_seed_y]

    # autonmi
    df_auto["autonmi"] = df_auto.apply(auto_nmi, axis=1)

    # autosoftnmi
    df_auto["autosoftnmi"] = df_auto.apply(auto_softnmi, axis=1)

    # save
    df_auto.to_pickle("data/auto.pkl")

    df_rev = df_light.merge(
        df_light,
        on=["config_dataset_name", "config_model_name", "config_n_topic", "level"],
        how="outer",
    )

    df_rev = df_rev[df_rev.config_adso_seed_x < df_rev.config_adso_seed_y]

    df_rev["reverseautonmi"] = df_rev.apply(reverse_auto_nmi, axis=1)
    df_rev["reverseautosoftnmi"] = df_rev.apply(reverse_auto_softnmi, axis=1)

    df_rev.to_pickle("data/rev.pkl")

    # dataset
    dataset = pd.DataFrame(df.config_dataset_name.drop_duplicates())
    dataset["DW"] = dataset.apply(
        lambda row: adso.data.LabeledDataset.load(
            "/data/benchmark/" + row.config_dataset_name
        ).get_shape(),
        axis=1,
    )
    dataset["D"] = dataset.apply(lambda row: row.DW[0], axis=1)
    dataset["W"] = dataset.apply(lambda row: row.DW[1], axis=1)
    dataset["Ls"] = dataset.apply(
        lambda row: np.unique(
            adso.data.LabeledDataset.load(
                "/data/benchmark/" + row.config_dataset_name
            ).get_labels()
        ),
        axis=1,
    )
    dataset["nL"] = dataset.apply(lambda row: len(row.Ls), axis=1)
    dataset["N_NNZ"] = dataset.apply(lambda row: n_nnz(row), axis=1)
    dataset["n"] = dataset.apply(lambda row: row.N_NNZ[0], axis=1)
    dataset["nnz"] = dataset.apply(lambda row: row.N_NNZ[1], axis=1)
    dataset.to_pickle("data/dataset.pkl")
