from typing import List
import os
from importlib import resources

from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tree_sitter import Tree, Query, Node
from tree_sitter_languages import get_language
from grep_ast import filename_to_lang
from tree_sitter_languages import get_parser  # noqa: E402

import logging

from dataclasses import dataclass


@dataclass
class Tag:
    rel_fname: str
    line: int
    name: str
    kind: str
    fname: str
    text: str
    byte_range: tuple[int, int]
    parent_names: tuple[str, ...] = ()

    @property
    def full_name(self):
        if self.kind == "ref":
            return self.name
        else:
            return tuple(list(self.parent_names) + [self.name])

    def to_tuple(self):
        return (
            self.rel_fname,
            self.line,
            self.name,
            self.kind,
            self.fname,
            self.text,
            self.byte_range,
            self.parent_names,
        )

    def __getitem__(self, item):
        return self.to_tuple()[item]

    def __len__(self):
        return len(self.to_tuple())

    def __hash__(self):
        return hash(self.to_tuple())


def get_query(lang: str) -> Query | None:
    language = get_language(lang)
    # Load the tags queries
    here = os.path.dirname(__file__)
    scm_fname = os.path.realpath(os.path.join(here, "../queries", f"tree-sitter-{lang}-tags.scm"))
    if not os.path.exists(scm_fname):
        return None

    with open(scm_fname, "r") as file:
        query_scm = file.read()

    # Run the tags queries
    query = language.query(query_scm)
    return query


def tree_to_tags(tree: Tree, query: Query, rel_fname: str, fname: str) -> List[Tag]:
    # TODO: extract docstrings and comments to do RAG on
    captures = list(query.captures(tree.root_node))
    defs = []
    refs = []
    names = []

    for node, tag in captures:
        if tag.startswith("name"):
            names.append(node)
        elif tag.startswith("reference"):
            refs.append((node, "ref"))
        elif tag.startswith("definition"):
            defs.append((node, "def"))
        else:
            continue

    out = []
    for node, kind in defs + refs:
        name_node = node2namenode(node, names)
        if name_node is None:
            continue

        parent_defs = get_def_parents(node, [d[0] for d in defs])
        parent_names = tuple([namenode2name(node2namenode(d, names)) for d in parent_defs])

        out.append(
            Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=namenode2name(name_node),
                parent_names=parent_names,
                kind=kind,
                line=name_node.start_point[0],
                text=node.text.decode("utf-8"),
                byte_range=node.byte_range,
            )
        )

    return out


def node2namenode(node: Node, name_nodes: List[Node]) -> Node | None:
    tmp = [n for n in name_nodes if n in node.children]

    if len(tmp) > 0:
        return tmp[0]

    # method calls
    tmp = [n for n in node.children if n.type == "attribute"]
    if len(tmp) == 0:
        logging.warning(f"Could not find name node for {node}")
        return None
    # method name
    tmp = [n for n in name_nodes if n in tmp[0].children]

    if len(tmp) == 0:
        logging.warning(f"Could not find name node for {node}")
        return None

    return tmp[0]


def namenode2name(node: Node | None) -> str:
    return node.text.decode("utf-8") if node else ""


def get_def_parents(node: Node, defs: List[Node]) -> List[Node]:
    dp = []
    while node.parent is not None:
        if node.parent in defs:
            dp.append(node.parent)
        node = node.parent
    return tuple(reversed(dp))


def refs_from_lexer(rel_fname, fname, code):
    try:
        lexer = guess_lexer_for_filename(fname, code)
    except ClassNotFound:
        return []

    tokens = list(lexer.get_tokens(code))
    tokens = [token[1] for token in tokens if token[0] in Token.Name]

    out = [
        Tag(
            rel_fname=rel_fname,
            fname=fname,
            name=token,
            kind="ref",
            line=-1,
        )
        for token in tokens
    ]
    return out


def get_tags_raw(fname, rel_fname, code) -> list[Tag]:
    lang = filename_to_lang(fname)
    if not lang:
        return []

    parser = get_parser(lang)

    if not code:
        return []

    tree = parser.parse(bytes(code, "utf-8"))
    query = get_query(lang)
    if not query:
        return []

    pre_tags = tree_to_tags(tree, query, rel_fname, fname)

    saw = set([tag.kind for tag in pre_tags])
    if "ref" in saw or "def" not in saw:
        return pre_tags

    # We saw defs, without any refs
    # Some tags files only provide defs (cpp, for example)
    # Use pygments to backfill refs
    refs = refs_from_lexer(rel_fname, fname, code)
    return pre_tags + refs


def read_text(filename: str, encoding: str = "utf-8") -> str | None:
    try:
        with open(str(filename), "r", encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"{filename}: file not found error")
        return
    except IsADirectoryError:
        logging.error(f"{filename}: is a directory")
        return
    except UnicodeError as e:
        logging.error(f"{filename}: {e}")
        logging.error("Use encoding parameter to set the unicode encoding.")
        return
