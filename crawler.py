from dotenv import load_dotenv

load_dotenv()

from langchain_tavily.tavily_crawl import TavilyCrawl

tavily_crawl = TavilyCrawl()


async def main():
    res = tavily_crawl.invoke(
        {"url": "https://langchain.com", "max_depth": 5, "extract_depth": "advanced"}
    )
    all_docs = res["results"]
    pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
