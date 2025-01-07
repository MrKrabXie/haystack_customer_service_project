from haystack.schema import Document
from pydantic import ConfigDict

class CustomDocument(Document):
    model_config = ConfigDict(arbitrary_types_allowed=True)