from argparse import ArgumentParser
import logging
import requests
from pathlib import Path
import typing as T
import pandas as pd
from multiprocessing import Pool

from newspaper.article import Article, ArticleDownloadState
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import (
    Features,
    MetadataOptions,
)
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


ARTICLES_LIST = Path(
    "/projectnb/llamagrp/davidat/projects/climatechange/scraping/news_articles.xlsx"
)

OUTPUT_DIR = Path(
    "/projectnb/llamagrp/davidat/projects/climatechange/scraping/scrapped_data_tmp/"
)
TEXTS_DIR = OUTPUT_DIR / "text"
IMAGES_DIR = OUTPUT_DIR / "images"


logging.basicConfig()
logger = logging.getLogger("clmchng_scraper")
logger.setLevel(logging.DEBUG)


def load_urls(file_: Path = ARTICLES_LIST) -> T.List[T.List[str]]:
    """
    Returns: list with GUID and URL
    """
    with open(file_, "rb") as f:
        df = pd.read_excel(f)
    print("Loaded {} urls".format(df.shape[0]))
    return df[["GUID", "URL"]].values.tolist()


class Scraper:
    def __init__(self, texts_dir, images_dir):
        self.texts_dir = texts_dir
        self.images_dir = images_dir

    def get_all_articles_and_images(self, urls: T.List[T.List[str]]) -> None:
        pool_size = max(1, len(urls) // 300)
        # Uncomment three lines below(and comment *out* the following line)
        # to enable multiprocessing.
        # logger.info(f"Using {pool_size} parallel workers.")
        # with Pool(pool_size) as p:
        # result = p.map(self.get_article_and_image, urls)
        result = list(map(self.get_article_and_image, urls))
        print(
            "Final report\n"
            "-----------\n"
            "Total: {}\n"
            "Text successful: {}\n"
            "Image successful: {}".format(
                len(result),
                sum([1 if row[0] else 0 for row in result]),
                sum([1 if row[1] else 0 for row in result]),
            )
        )

    def get_article_and_image(self, args) -> T.Tuple[bool, bool]:
        """A convinient function for Pool.map"""
        return self._get_article_and_image(*args)

    def _get_article_and_image(self, guid: str, url: str) -> T.Tuple[bool, bool]:
        """

        Returns
        -------
            (text_downloaded, image_downloaded)
        """

        text_file = self.texts_dir / (guid + ".txt")
        image_file = self.images_dir / (guid + ".jpg")

        if text_file.exists() and image_file.exists():
            logger.info("Text and image downloaded for {} already".format(url))
            return (True, True)
        else:
            text, image = self._fetch_article_and_image(guid, url)

        logger.info("{}: Fetched {}".format(guid, url))

        if image is not None:
            with open(image_file, "wb") as fb:
                fb.write(image)
        else:
            logger.warning("No image found for {}".format(url))

        if text is not None:
            with open(text_file, "w") as f:
                f.write(text)

        return text is not None, image is not None

    def _fetch_article_and_image(
        self, guid: str, url: str, exclude_image: bool = False
    ):
        raise NotImplementedError()


class NewspaperApiScraper(Scraper):
    def _fetch_article_and_image(
        self, guid: str, url: str, exclude_image: bool = False
    ) -> T.Tuple[T.Optional[str], T.Optional[bytes]]:
        """"

        Args
        ----
            url:

        Returns
        -------
            (downloaded_text, downloaded_image)
        """
        text = None
        article = Article(url)
        article.download()
        if article.download_state == ArticleDownloadState.FAILED_RESPONSE:
            logger.warning("{}: {}".format(url, article.download_exception_msg,))
        else:
            article.parse()
            text = article.text

        image = None
        if not exclude_image:
            if not article.top_image:
                logger.warning("{}: No image found.".format(url))
            else:
                pass
            try:
                r = requests.get(article.top_image, stream=True)
                image = r.content

            except requests.RequestException as e:
                logger.warning("{}: {}".format(url, str(e)))
        return (text, image)


class WatsonNLUApiScrapper(Scraper):
    def __init__(self, *args):
        super().__init__(*args)

        # Authenticate using ibm credentials

        authenticator = IAMAuthenticator(
            apikey="ewb7APMXClxjqZTuOaPuz0Snl2qx_4PAwoGj-SWGsO02"
        )
        self.service = NaturalLanguageUnderstandingV1(
            "2019-07-12", authenticator=authenticator
        )
        self.service.set_service_url(
            "https://api.us-east.natural-language-understanding.watson.cloud.ibm.com/instances/d59ffe68-b8e6-4220-b241-886b674a3a99"
        )

    def _fetch_article_and_image(
        self, guid: str, url: str, exclude_image: bool = False
    ) -> T.Tuple[T.Optional[str], T.Optional[bytes]]:
        """"

        Args
        ----
            url:

        Returns
        -------
            (downloaded_text, downloaded_image)
        """
        text = None
        image = None
        try:
            response = self.service.analyze(
                url=url,
                return_analyzed_text=True,
                language="en",
                features=Features(metadata=MetadataOptions()),
            ).get_result()
        except Exception as e:
            logger.warn(f"{guid}: {url}: Failed with {e}")
            return None, None

        text = response["analyzed_text"]
        image_url = response["metadata"]["image"]

        if not exclude_image:
            if not image_url:
                logger.warning("{}: No image found.".format(url))
            try:
                r = requests.get(image_url, stream=True)
                image = r.content
            except requests.RequestException as e:
                logger.warning("{}: {}".format(url, str(e)))

        return (text, image)


def main():

    parser = ArgumentParser()
    parser.add_argument("--verbose", "-v", action="count", default=0)

    # Set log level
    # Jargs = parser.parse_args()
    # jlog_level = (5 - min(5, args.verbose)) * 10
    # Jprint("setting log_level={}, logging.DEBUG={}".format(log_level, logging.DEBUG))
    # logger.setLevel(log_level)

    OUTPUT_DIR.mkdir(exist_ok=True)
    TEXTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    urls = load_urls()
    scraper = WatsonNLUApiScrapper(TEXTS_DIR, IMAGES_DIR)
    scraper.get_all_articles_and_images(urls)


if __name__ == "__main__":
    main()
