# batchpaper
A script to fetch the text content and images for a bunch of urls on some xlsx sheet).

## NewspaperAPI vs Watson NLU
There are two implemenations of the `Scraper` class: `NewspaperApiScrapper` and `WatsonNLUApiScrapper`. `NewspaperApiScrapper` is better(a  subjective opinion based on cursory glances of scrapped data).

## Multiprocessing
If you are going to scrape a lot of articles/images, it makes a lot of sense to paraellize. There's some lines you can uncomment to enable that.

### What's the `segment_para.py` file?
You can ignore it. I had to split a table of articles into a table of paragraphs and article ids.
