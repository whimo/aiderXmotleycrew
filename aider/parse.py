from collections import namedtuple
from typing import List, Set
from importlib import resources
from functools import total_ordering

from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tree_sitter import Tree, Query, Node
from tree_sitter_languages import get_language
import logging

from dataclasses import dataclass


@dataclass
class Tag:
    rel_fname: str
    fname: str
    line: int
    name: str
    kind: str
    parent_names: tuple[str] = ()

    @property
    def full_name(self):
        if self.kind == "ref":
            return self.name
        else:
            return tuple(list(self.parent_names) + [self.name])

    def to_tuple(self):
        return (self.rel_fname, self.fname, self.line, self.name, self.kind, self.parent_names)

    def __getitem__(self, item):
        return self.to_tuple()[item]

    def __len__(self):
        return len(self.to_tuple())

    def __hash__(self):
        return hash(self.to_tuple())


def get_query(lang: str) -> Query | None:
    language = get_language(lang)
    # Load the tags queries
    try:
        scm_fname = resources.files(__package__).joinpath("queries", f"tree-sitter-{lang}-tags.scm")
    except KeyError:
        return None
    query_scm = scm_fname
    if not query_scm.exists():
        return None
    query_scm = query_scm.read_text()

    # Run the tags queries
    query = language.query(query_scm)
    return query


def tree_to_tags(tree: Tree, query: Query, rel_fname: str, fname: str) -> List[Tag]:

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

    out_nodes = defs + refs

    tmp = []
    out = []
    for node, kind in out_nodes:
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
