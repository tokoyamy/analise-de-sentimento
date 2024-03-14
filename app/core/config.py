from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    TITLE = os.getenv("TITLE", "Default Title")
    VERSION = os.getenv("VERSION", "1.0")
    DESCRIPTION = os.getenv("DESCRIPTION", "Default Description")
    SPACY_MODEL_NAME = "pt_core_news_sm"

    # Adicione mais configurações conforme necessário
