from collections import defaultdict, Counter
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


def rank_tags(
    tags: List[Tag],
    mentioned_fnames: List[str],
    mentioned_idents: List[str],
    chat_fnames: List[str],
    other_rel_fnames: List[str],
) -> List[tuple]:
    """
    The original aider ranking algorithm
    :param tags:
    :param mentioned_fnames:
    :param mentioned_idents:
    :param chat_fnames:
    :param other_rel_fnames:
    :return:
    """
    defines = defaultdict(set)
    references = defaultdict(list)
    definitions = defaultdict(set)

    cleaned_fnames = set([(tag.fname, tag.rel_fname) for tag in tags])

    for tag in tags:
        if tag.kind == "def":
            defines[tag.name].add(tag.rel_fname)
            definitions[(tag.rel_fname, tag.name)].add(tag)

        if tag.kind == "ref":
            references[tag.name].append(tag.rel_fname)

    # now construct the graph

    chat_rel_fnames = set()
    personalization = dict()
    # Default personalization for unspecified files is 1/num_nodes
    # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
    personalize = 10 / (len(cleaned_fnames) + 1)

    for fname, rel_fname in cleaned_fnames:
        if fname in chat_fnames:
            personalization[rel_fname] = personalize
            chat_rel_fnames.add(rel_fname)

        if fname in mentioned_fnames:
            personalization[rel_fname] = personalize

    if not references:
        references = dict((k, list(v)) for k, v in defines.items())

    idents = set(defines.keys()).intersection(set(references.keys()))

    G = nx.MultiDiGraph()

    for ident in idents:
        definers = defines[ident]
        if ident in mentioned_idents:
            mul = 10
        else:
            mul = 1
        for referencer, num_refs in Counter(references[ident]).items():
            for definer in definers:
                # if referencer == definer:
                #    continue
                G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)

    if personalization:
        pers_args = dict(personalization=personalization, dangling=personalization)
    else:
        pers_args = dict()

    try:
        ranked = nx.pagerank(G, weight="weight", **pers_args)
    except ZeroDivisionError:
        return []

    # distribute the rank from each source node, across all of its out edges
    ranked_definitions = defaultdict(float)
    for src in G.nodes:
        src_rank = ranked[src]
        total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
        # dump(src, src_rank, total_weight)
        for _src, dst, data in G.out_edges(src, data=True):
            data["rank"] = src_rank * data["weight"] / total_weight
            ident = data["ident"]
            ranked_definitions[(dst, ident)] += data["rank"]

    ranked_tags = []
    ranked_definitions = sorted(ranked_definitions.items(), reverse=True, key=lambda x: x[1])

    # dump(ranked_definitions)

    # First collect the definitions in rank order
    # Do NOT include the chat-added files - is that because they'll be added in their entirety?
    for (fname, ident), rank in ranked_definitions:
        # print(f"{rank:.03f} {fname} {ident}")
        if fname in chat_rel_fnames:
            continue
        ranked_tags += list(definitions.get((fname, ident), []))

    rel_other_fnames_without_tags = set(other_rel_fnames)

    fnames_already_included = set(rt.rel_fname for rt in ranked_tags)

    # Then go through the __files__ ranked earlier, and add them in rank order
    # These are just files with references, without definitions, presumably
    top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
    for rank, fname in top_rank:
        if fname in rel_other_fnames_without_tags:
            rel_other_fnames_without_tags.remove(fname)
        if fname not in fnames_already_included:
            ranked_tags.append((fname,))

    # At the very tail of the list, append the files that have no tags at all
    for fname in rel_other_fnames_without_tags:
        ranked_tags.append((fname,))

    return ranked_tags
