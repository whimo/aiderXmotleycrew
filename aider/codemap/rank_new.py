from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np

from aider.codemap.tag import Tag


def rank_tags_new(
    tag_graph: nx.MultiDiGraph,
    mentioned_fnames,
    mentioned_idents,
    chat_fnames,
    other_rel_fnames,
    search_terms,
) -> List[Tag | tuple]:
    tag_weights = defaultdict(float)
    # process mentioned_idents
    for tag in tag_graph.nodes:
        if tag.kind == "def" and tag.name in mentioned_idents:
            tag_weights[tag] += 1.0

    # process mentioned_fnames
    fname_counts = defaultdict(int)
    for tag in tag_graph.nodes:
        if tag.kind == "def" and tag.fname in mentioned_fnames:
            fname_counts[tag.fname] += 1

    # Normalize the weights to take into account what's typical in the codebase
    typical_count = np.median(np.array(list(fname_counts.values())))
    for tag in tag_graph.nodes:
        if tag.fname in fname_counts and tag.kind == "def":
            tag_weights[tag] += 0.1 * typical_count / fname_counts[tag.fname]

    # process search_terms:
    tag_matches = defaultdict(set)
    for tag in tag_graph.nodes:
        for term in search_terms:
            if tag.kind == "def" and term in tag.text:
                tag_matches[term].add(tag)

    typical_search_count = np.median([len(tags) for tags in tag_matches.values()])
    for term, tags in tag_matches.items():
        for tag in tags:
            tag_weights[tag] += typical_search_count / len(tags)

    # TODO: propagate these weights through the graph to references

    # Order the tags by weight
    tags = sorted([(t[1], t[0]) for t in tag_weights.items()], key=lambda x: -tag_weights[x[1]])

    # TODO: do we need to handle chat_fnames here too?
    # Probably yes, once we have some.
    return [t[1] for t in tags]
