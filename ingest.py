import io
import zipfile
import requests
import frontmatter

from minsearch import Index


def read_repo_data(repo_owner, repo_name):
    """
    Download and parse all .md / .mdx files from a GitHub repository.
    """
    url = f"https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main"
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename_full = file_info.filename
        filename_lower = filename_full.lower()

        if not (filename_lower.endswith(".md") or filename_lower.endswith(".mdx")):
            continue

        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode("utf-8", errors="ignore")
                post = frontmatter.loads(content)
                data = post.to_dict()

                # Remove the zip root folder prefix, e.g. "claude-howto-main/"
                _, filename_repo = filename_full.split("/", maxsplit=1)
                data["filename"] = filename_repo

                # Normalize content field
                if "content" not in data:
                    data["content"] = ""

                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename_full}: {e}")
            continue

    zf.close()
    return repository_data


def sliding_window(seq, size, step):
    """
    Split a long string into overlapping chunks.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []

    for i in range(0, n, step):
        batch = seq[i:i + size]
        result.append({
            "start": i,
            "content": batch
        })

        if i + size >= n:
            break

    return result


def chunk_documents(docs, size=2000, step=1000):
    """
    Chunk document content into overlapping windows.
    """
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content", "")

        if not isinstance(doc_content, str) or not doc_content.strip():
            continue

        doc_chunks = sliding_window(doc_content, size=size, step=step)

        for chunk in doc_chunks:
            chunk.update(doc_copy)

        chunks.extend(doc_chunks)

    return chunks


def index_data(
    repo_owner,
    repo_name,
    filter_func=None,
    chunk=True,
    chunking_params=None,
):
    """
    Full pipeline:
    read repo -> optional filter -> optional chunk -> build minsearch index
    """
    docs = read_repo_data(repo_owner, repo_name)

    if filter_func is not None:
        docs = [doc for doc in docs if filter_func(doc)]

    if chunk:
        if chunking_params is None:
            chunking_params = {"size": 2000, "step": 1000}
        docs = chunk_documents(docs, **chunking_params)

    index = Index(
        text_fields=["content", "filename", "title", "description"],
        keyword_fields=[]
    )

    index.fit(docs)
    return index, docs