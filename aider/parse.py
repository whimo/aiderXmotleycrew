from collections import namedtuple
from typing import List
from importlib import resources

from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tree_sitter import Tree, Query
from tree_sitter_languages import get_language

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


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

    captures = query.captures(tree.root_node)

    captures = list(captures)

    out = []
    for node, tag in captures:
        if tag.startswith("name.definition."):
            kind = "def"
        elif tag.startswith("name.reference."):
            kind = "ref"
        else:
            continue

        out.append(
            Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )
        )

    return out


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
