import time
import logging
from multiprocessing import Pool
import re
from pathlib import Path

import pandas as pd

logging.basicConfig()
logger = logging.getLogger("climatechange")
DATA_D = Path(
    "/projectnb/llamagrp/davidat/projects/climatechange/scraping/scrapped_data_tmp"
)


pattern = re.compile(r"\n+")


def get_article_df(
    guid: str, has_image: bool, url: str, text_f: Path,
):
    """
    Returns
    -------
        `pd.DataFrame[str, bool, int, str]`:
            paragraph number(starts at 1), and paragraph contents
    """
    with open(text_f) as f:
        txt = f.read()
    df = pd.DataFrame(
        (enumerate(pattern.split(txt), start=1)),
        index=None,
        columns=["paragraph_num", "paragraph_txt"],
    )
    df["guid"] = guid
    df["url"] = url
    df["has_image"] = bool(has_image)
    df = df[["guid", "url", "has_image", "paragraph_num", "paragraph_txt"]]
    return df


def process_row(guid, has_image):
    text_f = (DATA_D / "text") / (guid + ".txt")
    url = urls_df.loc[guid]
    if len(url) > 1:
        logger.info("Taking first of duplicates {}".format(url["URL"].tolist()))
        url = url.iloc[0].item()
    else:
        url = url.item()
    article_df = get_article_df(guid, has_image, url, text_f)
    return article_df


def main():
    global urls_df

    logger.setLevel(logging.DEBUG)
    with open("./news_articles.csv") as f:
        urls_df = pd.read_csv(f, index_col=0, header=0)

    with open(DATA_D / "image_success.csv") as f:
        all_pairs_df = pd.read_csv(f, header=None)
    logger.debug(
        "Opened image_success.csv with shape %s. First few:\n%s"
        % (all_pairs_df.shape, all_pairs_df.iloc[:3])
    )

    dfs_per_article = []
    print("Breaking into paragraphs")
    all_pairs_tuples = all_pairs_df.values.tolist()
    t = time.time()
    with Pool(60) as p:
        dfs_per_article = p.starmap(process_row, all_pairs_tuples)
    e = time.time()
    avg_para_per_article = sum(map(len, dfs_per_article)) / len(dfs_per_article)
    print(
        "Done in {:.2f} seconds. Average of {:.2f} paragraphs per article.".format(
            (e - t), avg_para_per_article
        )
    )

    out_d = DATA_D / "split_by_paragraph"
    out_d.mkdir(exist_ok=True)
    # one_chunk_size = len(dfs_per_article) // int(avg_para_per_article)
    one_chunk_size = len(dfs_per_article)
    total_len = 0
    for i in range(0, len(dfs_per_article), one_chunk_size):
        articles_grp = dfs_per_article[i : i + one_chunk_size]
        out_f = out_d / "clm_chg_article_paras_grp_{}.csv".format(
            i // one_chunk_size + 1
        )
        concat_df = pd.concat(articles_grp)
        print(
            "Outputing to {} {} articles containing {} paragraphs.".format(
                out_f, len(articles_grp), len(concat_df)
            )
        )
        concat_df.to_csv(out_f, index=False)
        total_len += len(articles_grp)
    print("Outputted a total of {} articles".format(total_len))


if __name__ == "__main__":
    main()
