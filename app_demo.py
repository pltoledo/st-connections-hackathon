# Running this because of sqlite incompatible version
__import__("pysqlite3")  # noqa: E402
import sys  # noqa: E402

from requests.exceptions import ConnectionError  # noqa: E402

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from langchain.document_loaders import TextLoader  # noqa: E402
from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: E402

from streamlit_chroma_connection.chroma_connection import ChromaDBConnection  # noqa: E402

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
pip install langchain # for loading data in the examples
pip install pandas # for structuring the query results"""
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
    """ChromaDBConnection(mode="client", host="localhost", port="8000", ssl=False, headers={})"""
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
ChromaDBConnection(settings=Settings(anonymized_telemetry=False))"""
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
"""
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
tabs = st.tabs(["In-memory", "Persistent", "Client"])
with tabs[0]:
    st.write("#### Creating a collection")
    st.write("To code below creates an in-memory client for chroma:")
    st.code(
        """conn = st.connection("chroma", type=ChromaDBConnection, mode="in-memory")"""
    )
    conn = st.connection("chroma", type=ChromaDBConnection, mode="in-memory")
    st.write("To create a new collection, we can use the `create` method:")
    st.code("""conn.create('paul_graham_essay', distance_metric='cosine')""")
    st.write(
        "The `distance_metric` param defines what metric should be used to compute distances between embeddings in the collection."
    )
    st.info(
        """The **embedding_function** param must be used every time one wants to retrieve a collection. Because of this, it is recommended to always save which embedding function was used to create each collection.
    \nFor simplicity, we will be using the default Chroma function, but its always posible to implement custom ones. More information [here](https://docs.trychroma.com/embeddings).
    """,
        icon="💡",
    )
    conn.create("paul_graham_essay", distance_metric="cosine")
    st.write("#### Inserting data")
    st.write("After it is created, we can add the documents to the collection:")
    st.code(
        """document_contents = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    conn.insert(
        'paul_graham_essay',
        documents=document_contents,
        metadatas=metadatas,
    )
    """
    )
    st.write(
        "You can pass your own `ids` list to the function, but if you do not, it will generate a random id for each document using the `uuid` package."
    )
    document_contents = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    if conn.count("paul_graham_essay") == 0:
        conn.insert("paul_graham_essay", documents=document_contents, metadatas=metadatas)
    st.write("#### Querying")
    st.write("To query the collection, we can use the `query` method:")
    st.code(
        """
    result_docs = conn.query('paul_graham_essay', query_vector = ["<query text>"])
    """
    )
    st.write(
        "With some transformation using pandas, we have a dataframe with the semantically closest documents to the query:"
    )
    st.code(
        """
    results_transformed = [
        [id, doc, meta]
        for id, doc, meta
        in zip(result_docs['ids'][0], result_docs['documents'][0], result_docs['metadatas'][0])
    ]
    results_df = pd.DataFrame(results_transformed, columns=['ids', 'documents', 'metadatas'])
    """
    )
    st.write("**Try it out:**")
    query_text = st.text_input("Input query", "What is the secret to success?")
    result_docs = conn.query("paul_graham_essay", query_vector=[query_text])
    results_transformed = [
        [id, doc, meta]
        for id, doc, meta in zip(
            result_docs["ids"][0], result_docs["documents"][0], result_docs["metadatas"][0]
        )
    ]
    results_df = pd.DataFrame(results_transformed, columns=["ids", "documents", "metadatas"])
    st.write(results_df)
    st.write(
        "It is also posible to query by embedding vector instead of text by using `query_type = 'embeddings'`."
    )
    st.info(
        "The `query` method accepts every other parameter from the `query` method in the `chromadb.Collection` class, including the `where` and `where_documents` used for filtering documents by metadata and content, respectively.",
        icon="💡",
    )
    st.write("#### Other operations")
    st.write("Some of the other available methods are:")
    st.code(
        """
    conn.count("paul_graham_essay") # counts the total number of documents in the collection
    conn.peek("paul_graham_essay", limit=5) # get the first few results in the database up to limit
    conn.rename("paul_graham_essay", "paul_graham_essay_renamed") # renames the collection
    conn.update("paul_graham_essay", ids=ids, documents=...) # updates the documents by id
    conn.upsert("paul_graham_essay", ids=ids, documents=...) # updates the documents by id if they already exists, otherwise insert them
    conn.delete("paul_graham_essay", ids=ids) # deletes the documents by id. It is also posible to use where and where_documents filters
    conn.drop("paul_graham_essay") # drops the collection permanently
    """
    )
    st.write("**Try it out:**")
    peek_limit = st.number_input("Number of documents to peek", 1, 10)
    peeked_docs = conn.peek("paul_graham_essay", limit=peek_limit)
    st.write(pd.DataFrame(peeked_docs)[["ids", "documents", "metadatas"]])
    cols = st.columns(2)
    show_count = cols[0].radio(
        "Show count",
        [True, False],
        index=1,
        format_func=lambda x: "Yes" if x else "No",
        horizontal=True,
    )
    if show_count:
        cols[1].metric("Number of documents in collection", conn.count("paul_graham_essay"))

with tabs[1]:
    st.write(
        """The persistent mode only differs from the in-memory in the way it persists the data on disk.
The directory in which the data will be saved can be controlled by the `path` parameter."""
    )
    st.write("To create an instance of the persistent Chroma client, one can do the following:")
    st.code(
        """
    persist_conn = st.connection("chroma", type=ChromaDBConnection, mode="persistent", path="./chroma")
"""
    )
    persist_conn = st.connection(
        "chroma", type=ChromaDBConnection, mode="persistent", path="./chroma"
    )
    st.write(
        "Then, once we create a collection and populate it, the data will be saved in the specified path."
    )
    st.code(
        """
    persist_conn.create("persisted_paul_graham_essay", distance_metric="cosine")

    document_contents = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    persist_conn.insert(
        'persisted_paul_graham_essay',
        documents=document_contents,
        metadatas=metadatas,
    )
"""
    )
    persist_conn.create("persisted_paul_graham_essay", distance_metric="cosine")
    document_contents = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    if persist_conn.count("persisted_paul_graham_essay") == 0:
        persist_conn.insert(
            "persisted_paul_graham_essay", documents=document_contents, metadatas=metadatas
        )
    st.write("The `path` parameter can also be specified in the `secrets.toml` file.")
with tabs[2]:
    st.write(
        """To use Chroma in client mode, we need to have a running instance of Chroma to act as the server.
            This demo cannot provide this, but one can follow guidelines below to test it locally.
            It is the recommend way to use docker containers, as said by the maintainers [here](https://github.com/chroma-core/chroma/issues/797)."""
    )
    st.write(
        """Assuming you already have docker and docker compose installed, follow this setup steps:

```
git clone https://github.com/chroma-core/chroma.git # clone the repository
cd chroma
docker-compose up -d --build
```

After the build, run the following:
```
docker logs <id of the container>
```
It should show something like:
```
2023-07-25 03:18:53 INFO     uvicorn.error   Started reloader process [1] using WatchFiles
2023-07-25 03:18:53 INFO     chromadb.telemetry.posthog Anonymized telemetry enabled. See https://docs.trychroma.com/telemetry for more information.
2023-07-25 03:18:53 DEBUG    chromadb.config Starting component System
2023-07-25 03:18:53 DEBUG    chromadb.config Starting component Posthog
2023-07-25 03:18:53 DEBUG    chromadb.config Starting component SqliteDB
2023-07-25 03:18:53 DEBUG    chromadb.config Starting component LocalSegmentManager
2023-07-25 03:18:53 DEBUG    chromadb.config Starting component SegmentAPI
2023-07-25 03:18:53 INFO     uvicorn.error   Started server process [8]
2023-07-25 03:18:53 INFO     uvicorn.error   Waiting for application startup.
2023-07-25 03:18:53 INFO     uvicorn.error   Application startup complete.
```
Then it is good to go."""
    )
    st.write("To create an instance of Chroma in client mode:")
    st.code(
        """
    client_conn = st.connection("chroma", type=ChromaDBConnection, mode="client")
"""
    )
    st.write(
        "As in previous examples, we can rely both on the `secrets.toml` file or the constructor parameters to configure the connection."
    )
    client_conn = st.connection("chroma", type=ChromaDBConnection, mode="client")
    try:
        client_conn.create("client_paul_graham_essay", distance_metric="cosine")
        st.write(
            "Then, once we create a collection and populate it, the data will be saved in server:"
        )
        st.code(
            """
        client_conn.create("client_paul_graham_essay", distance_metric="cosine")

        document_contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        client_conn.insert(
            'client_paul_graham_essay',
            documents=document_contents,
            metadatas=metadatas,
        )
    """
        )
        document_contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if client_conn.count("client_paul_graham_essay") == 0:
            client_conn.insert(
                "client_paul_graham_essay", documents=document_contents, metadatas=metadatas
            )
        st.write("Still, it is posible to query the collection normally:")
        query_text = st.text_input(
            "Input query", "What is the secret to success?", key="client-input"
        )
        result_docs = client_conn.query("client_paul_graham_essay", query_vector=[query_text])
        results_transformed = [
            [id, doc, meta]
            for id, doc, meta in zip(
                result_docs["ids"][0], result_docs["documents"][0], result_docs["metadatas"][0]
            )
        ]
        results_df = pd.DataFrame(results_transformed, columns=["ids", "documents", "metadatas"])
        st.write(results_df)
    except (ConnectionError, Exception):  # Chroma throws bare Exceptions for some reason
        pass

st.write("")
st.write("")
st.write("")
cols = st.columns(10)
cols[4].markdown(
    "[![Repo](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/pltoledo/st-connections-hackathon)"
)
cols[5].markdown(
    "[![linkedin](https://img.icons8.com/color/48/linkedin.png)](https://linkedin.com/in/pedro-toledo)"
)