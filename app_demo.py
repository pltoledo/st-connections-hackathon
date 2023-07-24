# Running this because of sqlite incompatible version
__import__("pysqlite3")  # noqa: E402
import sys  # noqa: E402

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # noqa: E402

import streamlit as st  # noqa: E402
from langchain.document_loaders import TextLoader  # noqa: E402
from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: E402

from chroma_connection import ChromaDBConnection  # noqa: E402

st.set_page_config(page_title="ChromaDB Connection Demo", page_icon="🔗")

st.title("ChromaDB Connection Demo")
st.info("`ChromaDBConnection` makes it easy to connect and manage chormadb collections.", icon="💡")
st.write("## Setup")
st.write(
    """In this demo, there will be examples on how to use chroma on two of the three available deployment modes, while also giving instructions on how to test the third mode aswell."""
)
st.write("First, it is necessary to install the `chromadb` and `langchain` packages:")
st.code(
    """pip install chromadb
pip install langchain # for loading data in the examples"""
)
st.write(
    "Chroma has also a dependency on `sqlite3 >= 3.35.0`. If your python version does not support this, you can follow the troubleshooting instructions [here](https://docs.trychroma.com/troubleshooting#sqlite)."
)
st.divider()
st.write("## Configuring the connection")
st.write(
    """It is posible to configure the client connection directly in the `ChromaDBConnection` constructor, or to use the `.streamlit/secrets.toml` file to store the connection parameters.
In this demo, we will use the both methods."""
)
st.info("The `kwargs` in the constructor takes precedence over the `secrets.toml` file.", icon="💡")
st.write(
    """In the `.streamlit/secrets.toml` file, the connection parameters should be stored in a `[chroma]` section, for example:"""
)
st.code(
    """[connections.chroma]
host = "localhost"
port = "8000"
ssl = false
headers = {}
""",
    language="toml",
)
st.write("This is the same as using the constructor with the following parameters:")
st.code(
    """ChromaDBConnection(mode="client", host="localhost", port="8000", ssl=False, headers={})""",
    language="python",
)
st.write(
    """The `settings` param can be defined in the `secrets.toml` file within chroma section:"""
)
st.code(
    """[connections.chroma]
settings.anonymized_telemetry = false
""",
    language="toml",
)
st.write("Or by using the constructor directly:")
st.code(
    """from chromadb import Settings
ChromaDBConnection(settings=Settings(anonymized_telemetry=False))""",
    language="python",
)
st.write(
    """Finally, the `mode` parameter is required and it is used to specify which chroma deployment mode to use. It can be one of the following:
* `in-memory`: This is the default mode. It creates a new chroma instance in memory, and all data is lost when the session is closed. Uses the `chromadb.EphemeralClient` class.
* `client`: This mode connects to an already stabilished chroma server. Uses the `chromadb.HttpClient` class.
* `persistent`: This mode creates a new chroma instance in memory, but saving the data in the specified path. Uses the `chromadb.PersistentClient` class.
"""
)
st.divider()
st.write("## Examples")
st.write(
    """For the examples, we will be using the "What I Worked On" Paul Graham essay, release on July 2023. We will use `langchain` to load and prepare the data:"""
)
st.code(
    """from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

full_text = TextLoader('./data/paul_graham_essay.txt').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
    add_start_index = True,
)
documents = text_splitter.split_documents(full_text)
""",
    language="python",
)

full_text = TextLoader("./data/paul_graham_essay.txt").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
)
documents = text_splitter.split_documents(full_text)

st.write(
    "Now we have a list of documents, each with a chunk off approximately 1000 characters from the original text. A single document looks like this:"
)
st.code("""Document(page_content = "...", metadata = {})""")
st.write(
    "Now we can use the `ChromaDBConnection` to create a new collection and add the documents to it."
)
st.info(
    "Important to note that the documents generated from `langchain` already follow a format close to what is implemented in Chroma.",
    icon="💡",
)
st.write("### In-memory mode")
conn = st.experimental_connection("chroma", type=ChromaDBConnection, mode="in-memory")
st.write(conn.cursor)
