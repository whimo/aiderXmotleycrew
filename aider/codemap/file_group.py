import os
import os.path
import logging
from pathlib import Path
from typing import Callable, List
import re
from collections import defaultdict

from aider.repo import GitRepo
from diskcache import Cache


class FileGroup:
    """
    A FileGroup is a collection of files that we are parsing and monitoring for changes.
    This might be a git repo or a directory. If new files appear in it,
    we will see that as well using the get_all_filenames method.
    """

    CACHE_VERSION = 4
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    def __init__(self, repo: GitRepo | None, root: str | None = None, filename_filter=None):
        # TODO: support other kinds of locations
        self.repo = repo
        if self.repo is None:
            if os.path.isdir(root):
                self.root = root
            else:
                raise ValueError("Must supply either a GitRepo or a valid root directory")
        else:
            self.root = self.repo.root

        if filename_filter is None:
            self.filename_filter = lambda x: x.endswith(".py")
        else:
            self.filename_filter = filename_filter

        self.load_tags_cache()
        self.warned_files = set()

    def abs_root_path(self, path):
        "Gives an abs path, which safely returns a full (not 8.3) windows path"
        res = Path(self.root) / path
        res = Path(res).resolve()
        return res

    def get_all_filenames(self):
        """
        Get all the filenames in the group, including new files.
        :return: List of unique absolute file paths
        """
        if self.repo:
            files = self.repo.get_tracked_files()
            files = [self.abs_root_path(fname) for fname in files]
            files = [str(fname) for fname in files if fname.is_file()]

        else:
            files = [str(f) for f in Path(self.root).rglob("*") if f.is_file()]

        files = [f for f in files if self.filename_filter(f)]

        return sorted(set(files))

    def validate_fnames(self, fnames: List[str]) -> List[str]:
        cleaned_fnames = []
        for fname in fnames:
            # TODO: skip files that are obviously not source code, eg .zip files
            if Path(fname).is_file():
                cleaned_fnames.append(str(fname))
            else:
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        logging.error(f"Repo-map can't include {fname}, it is not a normal file")
                    else:
                        logging.error(
                            f"Repo-map can't include {fname}, it doesn't exist (anymore?)"
                        )

                self.warned_files.add(fname)

        return cleaned_fnames

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        if not path.exists():
            logging.warning(f"Tags cache not found, creating: {path}")
        self.TAGS_CACHE = Cache(str(path))

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            logging.error(f"File not found error: {fname}")

    def cached_function_call(self, fname: str, function: Callable, key: str | None = None):
        """
        Cache the result of a function call, refresh the cache if the file has changed.
        :param fname: the file to monitor for changes
        :param function: the function to apply to the file
        :param key: the key to use in the cache, if None, the function name is used
        :return: the function's result
        """
        # Check if the file is in the cache and if the modification time has not changed
        # TODO: this should be a decorator?
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname + "::" + (key or function.__name__)
        if cache_key in self.TAGS_CACHE and self.TAGS_CACHE[cache_key]["mtime"] == file_mtime:
            return self.TAGS_CACHE[cache_key]["data"]

        # miss!
        data = function(fname)

        # Update the cache
        self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
        self.save_tags_cache()
        return data

    def get_file_mentions(self, content, abs_added_fnames):
        words = set(word for word in content.split())

        # drop sentence punctuation from the end
        words = set(word.rstrip(",.!;:") for word in words)

        # strip away all kinds of quotes
        quotes = "".join(['"', "'", "`"])
        words = set(word.strip(quotes) for word in words)

        all_files = self.get_all_filenames()
        other_files = set(all_files) - set(abs_added_fnames)
        addable_rel_fnames = [self.get_rel_fname(f) for f in other_files]

        mentioned_rel_fnames = set()
        fname_to_rel_fnames = {}
        for rel_fname in addable_rel_fnames:
            if rel_fname in words:
                mentioned_rel_fnames.add(str(rel_fname))

            fname = os.path.basename(rel_fname)

            # Don't add basenames that could be plain words like "run" or "make"
            if "/" in fname or "." in fname or "_" in fname or "-" in fname:
                if fname not in fname_to_rel_fnames:
                    fname_to_rel_fnames[fname] = []
                fname_to_rel_fnames[fname].append(rel_fname)

        for fname, rel_fnames in fname_to_rel_fnames.items():
            if len(rel_fnames) == 1 and fname in words:
                mentioned_rel_fnames.add(rel_fnames[0])

        return mentioned_rel_fnames


def get_ident_filename_matches(idents, all_rel_fnames: List[str]):
    all_fnames = defaultdict(set)
    for fname in all_rel_fnames:
        base = Path(fname).with_suffix("").name.lower()
        if len(base) >= 5:
            all_fnames[base].add(fname)

    matches = set()
    for ident in idents:
        if len(ident) < 5:
            continue
        matches.update(all_fnames[ident.lower()])

    return matches


def get_ident_mentions(text):
    # Split the string on any character that is not alphanumeric
    # \W+ matches one or more non-word characters (equivalent to [^a-zA-Z0-9_]+)
    words = set(re.split(r"\W+", text))
    return words


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files
