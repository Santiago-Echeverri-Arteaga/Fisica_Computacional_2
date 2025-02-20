import shutil
import tarfile
import zipfile
import IPython
import os

from pathlib import Path
from urllib.parse import urlparse
from typing import List, Union, Optional, Iterable, Generator
from tqdm.auto import tqdm


__all__ = [
    "InvalidURLException",
    "download_dataset",
    "download",
    "read_dataset",
    "read",
    "prepare_dataset",
    "prepare",
]

DEFAULT_CHUNK_SIZE = 8 << 10


class InvalidURLException(Exception):
    """
    Raised if URL is invalid.
    """

    def __init__(self, url):
        self.url = url
        self.message = f"'{self.url}' is not a valid URL."
        super().__init__(self.message)


def _is_jupyterlite() -> bool:
    """
    :returns: True if running in jupyterlite environment, False otherwise.
    """
    return IPython.sys.platform == "emscripten"


def _is_url_valid(url: str) -> bool:
    """
    :param url: URL which is checked for validity.
    :returns: True if url is valid URL, False otherwise.
    """
    try:
        result = urlparse(url)
        # Assume it's a valid URL if a URL scheme and netloc are successfully parsed
        return all([result.scheme, result.netloc])
    except Exception:
        # If urlparse chokes on something, assume it's an invalid URL
        return False


async def _get_chunks(url: str, chunk_size: int) -> Generator[bytes, None, None]:
    """
    Generator that yields consecutive chunks of bytes from URL 'url'
    :param url: The URL containing the data file to be read
    :param chunk_size: The size of each chunk (in no. of bytes).
    :returns: Generator yielding chunks of bytes from file at URL until done.
    :raise InvalidURLException: When URL is invalid.
    :raise Exception: When Exception encountered when reading from URL.
    """
    if not _is_url_valid(url):
        raise InvalidURLException(url)
    desc = f"Downloading {Path(urlparse(url).path).name}"
    if _is_jupyterlite():
        from js import fetch  # pyright: ignore
        from pyodide import JsException  # pyright: ignore

        try:
            response = await fetch(url)
            reader = response.body.getReader()
            pbar = tqdm(
                mininterval=1,
                desc=desc,
                total=int(response.headers.get("content-length", 0)),
            )
            while True:
                res = (await reader.read()).to_py()
                value, done = res["value"], res["done"]
                if done:
                    break
                value = value.tobytes()
                yield value
                pbar.update(len(value))
            pbar.close()
        except JsException:
            raise Exception(f"Failed to read dataset at '{url}'.") from None
    else:
        import requests  # pyright: ignore
        from requests.exceptions import ConnectionError  # pyright: ignore

        try:
            with requests.get(url, stream=True) as response:
                # If requests.get fails, it will return readable error
                if response.status_code >= 400:
                    raise Exception(
                        f"received status code {response.status_code} from '{url}'."
                    )
                pbar = tqdm(
                    miniters=1,
                    desc=desc,
                    total=int(response.headers.get("content-length", 0)),
                )
                for chunk in response.iter_content(chunk_size=chunk_size):
                    yield chunk
                    pbar.update(len(chunk))
                pbar.close()
        except ConnectionError:
            raise Exception(f"Failed to read dataset at '{url}'.") from None


def _rmrf(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _verify_files_dont_exist(
    paths: Iterable[Path], remove_if_exist: bool = False
) -> None:
    """
    Verifies all paths in 'paths' don't exist.
    :param paths: A iterable of pathlib.Path s.
    :param remove_if_exist=False: Remove each file at each path in paths if they already exist.
    :returns: None
    :raises FileExistsError: On the first path found that already exists if remove_if_exist is False.
    """
    for path in paths:
        # Could be a broken symlink => path.exists() is False
        if path.exists() or path.is_symlink():
            if remove_if_exist:
                while path.is_symlink():
                    temp = Path(os.readlink(path))
                    path.unlink()
                    path = temp
                if path.exists():
                    _rmrf(path)
            else:
                raise FileExistsError(f"Error: File '{path}' already exists.")


def _is_file_to_symlink(path: Path) -> bool:
    """
    :param path: path to check.
    :returns: True if file should be symlinked, False otherwise.
    """
    # Don't symlink "._" junk
    return not (path.name.startswith("._") or path.name in ["__MACOSX"])


async def download(
    url: str,
    path: Optional[str] = None,
    verbose: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """
    Downloads file located at URL to path.

    >>> import skillsnetwork
    >>> path = "./my_file.txt"
    >>> await skillsnetwork.download("https://example.com/myfile", path)
    Saved as './my_file.txt'
    >>> with open(path, "r") as f:
    >>>     content = f.read()

    :param url: The URL where the file is located.
    :param path: The path to which the file at URL should be downloaded. Auto-generated from url by default.
    :param chunk_size: The number of bytes to read from url at a time.
    :raise FileNotFoundError: If path is invalid.
    :raise skillsnetwork.InvalidURLException: If URL is invalid.
    """
    filename = Path(urlparse(url).path).name
    if path is None:
        path = Path.cwd() / filename
    else:
        path = Path(path)
        if path.is_dir():
            path /= filename
    with open(path, "wb") as f:  # Will raise FileNotFoundError if invalid path
        async for chunk in _get_chunks(url, chunk_size):
            f.write(chunk)
    if verbose:
        print(f"Saved as '{os.path.relpath(path.resolve())}'")


async def read(url: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    """
    Reads file at URL into bytes

    >>> import skillsnetwork
    >>> content = await skillsnetwork.read("https://example.com/myfile") # Is bytes
    >>> content_str = content.decode()                                   # Is str

    :param url: The URL where the file is located.
    :param chunk_size: Number of bytes to read from url at a time.
    :returns: bytes containing file located at URL
    :raise FileNotFoundError: If path is invalid.
    :raise InvalidURLException: If URL is invalid.
    """
    return b"".join([chunk async for chunk in _get_chunks(url, chunk_size)])


async def prepare(
    url: str, path: Optional[str] = None, verbose: bool = True, overwrite: bool = False
) -> None:
    """
    Prepares a dataset for learners. Downloads a dataset from the given url,
    decompresses it if necessary. If not using jupyterlite, will extract to
    /tmp and and symlink it so it's available at the desired path.

    >>> import skillsnetwork
    >>> await skillsnetwork.prepare("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/images/images.tar.gz")
    Saved to '.'

    :param url: The URL to download the dataset from.
    :param path: The path the dataset will be available at. Current working directory by default.
    :param verbose=True: Prints saved path if True.
    :param overwrite=False: If True, overwrites any existing files at destination.
    :raise InvalidURLException: When URL is invalid.
    :raise FileExistsError: When a file to be symlinked already exists and overwrite is False.
    :raise ValueError: When requested path is in /tmp, or cannot be saved to path.
    """

    filename = Path(urlparse(url).path).name
    path = Path.cwd() if path is None else Path(path)
    # Check if path contains /tmp
    if Path("/tmp") in path.parents:
        raise ValueError("path must not be in /tmp.")
    elif path.is_file():
        raise ValueError("Datasets must be prepared to directories, not files.")
    # Create the target path if it doesn't exist yet
    path.mkdir(exist_ok=True)

    # For avoiding collisions with any other files the user may have downloaded to /tmp/
    dname = f"skills-network-{hash(url)}"
    # The file to extract data to. If not jupyterlite, to be symlinked to as well
    extract_dir = path if _is_jupyterlite() else Path(f"/tmp/{dname}")
    # The file to download the (possibly) compressed data to
    tmp_download_file = Path(f"/tmp/{dname}-{filename}")
    # Download the dataset to tmp_download_file file
    # File will be overwritten if it already exists
    await download(url, tmp_download_file, verbose=False)

    # Delete extract_dir directory if it already exists
    if not _is_jupyterlite():
        if extract_dir.is_dir():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir()

    try:
        if tarfile.is_tarfile(tmp_download_file):
            with tarfile.open(tmp_download_file) as tf:
                _verify_files_dont_exist(
                    [
                        path / child.name
                        for child in map(Path, tf.getnames())
                        if len(child.parents) == 1 and _is_file_to_symlink(child)
                    ],  # Only check if top-level fileobject
                    remove_if_exist=overwrite,
                )
                pbar = tqdm(iterable=tf.getmembers(), total=len(tf.getmembers()))
                pbar.set_description(f"Extracting {filename}")
                for member in pbar:
                    tf.extract(member=member, path=extract_dir)
            tmp_download_file.unlink()
        elif zipfile.is_zipfile(tmp_download_file):
            with zipfile.ZipFile(tmp_download_file) as zf:
                _verify_files_dont_exist(
                    [
                        path / child.name
                        for child in map(Path, zf.namelist())
                        if len(child.parents) == 1 and _is_file_to_symlink(child)
                    ],  # Only check if top-level fileobject
                    remove_if_exist=overwrite,
                )
                pbar = tqdm(iterable=zf.infolist(), total=len(zf.infolist()))
                pbar.set_description(f"Extracting {filename}")
                for member in pbar:
                    zf.extract(member=member, path=extract_dir)
            tmp_download_file.unlink()
        else:
            _verify_files_dont_exist([path / filename], remove_if_exist=overwrite)
            shutil.move(tmp_download_file, extract_dir / filename)
    except FileExistsError as e:
        raise FileExistsError(
            str(e)
            + "\nIf you want to overwrite any existing files, use prepare(..., overwrite=True)."
        ) from None

    # If in jupyterlite environment, the extract_dir = path, so the files are already there.
    if not _is_jupyterlite():
        # If not in jupyterlite environment, symlink top-level file objects in extract_dir
        for child in filter(_is_file_to_symlink, extract_dir.iterdir()):
            if (path / child.name).is_symlink() and overwrite:
                (path / child.name).unlink()
            (path / child.name).symlink_to(child, target_is_directory=child.is_dir())

    if verbose:
        print(f"Saved to '{os.path.relpath(path.resolve())}'")


def setup() -> None:
    if _is_jupyterlite():
        tqdm.monitor_interval = 0


setup()


# For backwards compatibility
download_dataset = download
read_dataset = read
prepare_dataset = prepare
