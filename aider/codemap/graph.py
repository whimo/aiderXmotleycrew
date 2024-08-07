from typing import List, Optional, Dict
from collections import defaultdict

import networkx as nx

from motleycrew.common import logger

from aider.codemap.tag import Tag
from aider.codemap.render import RenderCode


class TagGraph(nx.MultiDiGraph):
    def __init__(self):
        super().__init__()
        self.code_renderer = RenderCode()

    @property
    def filenames(self):
        return set([tag.fname for tag in self.nodes])

    def get_parents(self, tag: Tag) -> List[Tag] | str:
        """
        Get the parent tags of a tag in the same file, eg the class def for a method name
        :param tag:
        :return: list of parent tags
        """
        if not len(tag.parent_names):
            return []

        parents = [t for t in self.predecessors(tag) if t.kind == "def"]
        if not parents:
            logger.warning(f"No parent found for {tag} with nonempty parent names!")
            return ".".join(tag.parent_names) + "." + tag.name + ":"
        parent = parents[0]

        if len(parent.parent_names):
            predecessors = self.get_parents(parent)
        else:
            predecessors = []

        return predecessors + [parent]

    def get_tag_representation(self, tag: Tag, parent_details: bool = False, max_lines=200) -> str:
        if tag is None:
            return None
        if tag not in self.nodes:
            raise ValueError(f"The tag {tag} is not in the tag graph")

        tag_repr = [tag.rel_fname + ":"]
        if not parent_details:
            if len(tag.parent_names):
                tag_repr.append(".".join(tag.parent_names) + "." + tag.name + ":")
        else:
            parents = self.get_parents(tag)
            if parents:
                if isinstance(parents, str):
                    tag_repr.append(parents)
                else:
                    # if there are parents, this will include the filename
                    tag_repr = [self.code_renderer.to_tree(parents)]

        tag_repr.append(RenderCode.text_with_line_numbers(tag))
        tag_repr = "\n".join(tag_repr)

        if len(tag_repr.split("\n")) <= max_lines:
            # if the full text hast at most 200 lines, put it all in the summary
            children = []
            for _, c in self.out_edges(tag):
                if (  # If the child is included in the parent's full text anyway, skip it
                    c.fname == tag.fname
                    and c.byte_range[0] >= tag.byte_range[0]
                    and c.byte_range[1] <= tag.byte_range[1]
                ):
                    continue
                children.append(c)

            out = [tag_repr]
            if children:
                out.extend(
                    [
                        "Referenced entities summary:",
                        self.code_renderer.to_tree(children),
                    ]
                )
            return "\n".join(out)
        else:
            # if the full text is too long, send a summary of it and its children
            children = list(self.successors(tag))
            tag_repr = self.code_renderer.to_tree([tag] + children)
            return tag_repr

    def get_tag_from_filename_lineno(
        self, fname: str, line_no: int, try_next_line=True
    ) -> Tag | None:
        files = [f for f in self.filenames if fname in f]
        if not files:
            raise ValueError(f"File {fname} not found in the file group")
        this_file_nodes = [node for node in self.nodes if node.fname in files]
        if not this_file_nodes:
            raise ValueError(f"File {fname} not found in the tag graph")
        for node in this_file_nodes:
            if node.line == line_no - 1:
                return node
        # If we got this far, we didn't find the tag
        # Let's look in the next line, sometimes that works
        if try_next_line:
            return self.get_tag_from_filename_lineno(fname, line_no + 1, try_next_line=False)

        return None

    def get_tags_from_entity_name(
        self, entity_name: Optional[str] = None, file_name: Optional[str] = None
    ) -> List[Tag]:

        if entity_name is None:
            assert file_name is not None, "Must supply at least one of entity_name, file_name"
            return [t for t in self.nodes if file_name in t.fname]

        # Composite, like `file.py:method_name`
        if file_name is not None:
            preselection: List[Tag] = [t for t in self.nodes if file_name in t.fname]
            test = [
                t for t in preselection if t.name == entity_name.split(".")[-1] and t.kind == "def"
            ]
            if not test:
                logger.warning(
                    f"Definition of entity {entity_name} not found in file {file_name}, searching globally"
                )
                preselection: List[Tag] = list(self.nodes)
        else:
            preselection: List[Tag] = list(self.nodes)

        orig_tags: List[Tag] = [
            t for t in preselection if t.name == entity_name.split(".")[-1] and t.kind == "def"
        ]

        # do fancier name resolution
        re_tags = [t for t in orig_tags if match_entity_name(entity_name, t)]

        if len(re_tags) > 1:
            logger.warning(f"Multiple definitions found for {entity_name}: {re_tags}")
        return re_tags


def match_entity_name(entity_name: str, tag: Tag) -> bool:
    entity_name = entity_name.split(".")
    if entity_name[-1] != tag.name:
        return False

    # Simple reference, with no dots, and names are the same
    # or the tag has no parent names, and the dots are just package names
    if len(entity_name) == 1 or len(tag.parent_names) == 0:
        return True

    # Check if the parent names match if they exist
    if tag.parent_names == tuple((entity_name[:-1])[-len(tag.parent_names) :]):
        return True

    # TODO: do fancier resolution here, potentially returning match scores to rank matches

    # If entity name includes package name, check that
    fn_parts = tag.fname.split("/")
    fn_parts[-1] = fn_parts[-1].replace(".py", "")

    potential_parents = fn_parts + list(tag.parent_names)
    clipped_parents = potential_parents[-len(entity_name) - 1 :]

    if tuple(clipped_parents) == tuple(entity_name[:-1]):
        return True

    return False


def build_tag_graph(tags: List[Tag], code_map: Dict[str, str]) -> TagGraph:
    """
    Build a graph of tags, with edges from references to definitions
    And with edges from parent definitions to child definitions in the same file
    :param tags:
    :return:
    """
    # Build a map from entity names to their definitions
    # There may be multiple definitions for a single name in different scopes,
    # for now we don't bother resolving them
    G = TagGraph()
    G.code_renderer.code_map = code_map

    def_map = defaultdict(set)

    for tag in tags:
        if tag.kind == "def":
            def_map[tag.name].add(tag)
        elif tag.kind == "file":
            # Just add all the parsed files to the graph
            G.add_node(tag, kind=tag.kind)

    # Add all tags as nodes
    # Add edges from references to definitions
    for tag in tags:
        G.add_node(tag, kind=tag.kind)
        if tag.kind == "def":
            # Look for any references to other entities inside that definition
            for ref_tag in tags:
                if ref_tag.kind == "ref" and ref_tag.fname == tag.fname:
                    if (
                        ref_tag.byte_range[0] >= tag.byte_range[0]
                        and ref_tag.byte_range[1] <= tag.byte_range[1]
                    ):
                        G.add_edge(tag, ref_tag)

        elif tag.kind == "ref":
            G.add_node(tag, kind=tag.kind)
            for def_tag in def_map[tag.name]:
                # point to any definitions that might have been meant
                # would probably need a language server for unique resolution,
                # don't bother with that here
                G.add_edge(tag, def_tag)

        # Build up definition hierarchy
        # A parent definition for a tag must:
        # - be in the same file
        # - Have matching tail of parent names
        if len(tag.parent_names):
            parent_name = tag.parent_names[-1]
            candidates = [
                c
                for c in def_map[parent_name]
                if c.fname == tag.fname and c.parent_names == tag.parent_names[:-1]
            ]
            for c in candidates:
                G.add_edge(c, tag)

    return G


def only_defs(tag_graph: TagGraph) -> TagGraph:
    """
    Return a graph with only the def nodes and the edges between them
    If a def node has a reference node as a child, add an edge to the reference node's definition
    :param tag_graph: A graph generated by build_tag_graph
    :return: A graph with only def nodes and edges between them
    """

    G = TagGraph()
    G.code_renderer.code_map = tag_graph.code_renderer.code_map

    for tag in tag_graph.nodes:
        if tag.kind == "def":
            G.add_node(tag)
    for u, v, data in tag_graph.edges(data=True):
        if u.kind == "def" and v.kind == "def":
            G.add_edge(u, v, **data)
    # Also add edges betweend defs and their two-hop descendant defs
    # TODO: should we look for more than two-hop def descendants? Can these ever happen?
    for u, v, data in tag_graph.edges(data=True):
        if u.kind == "def" and v.kind != "def":
            for _, v_desc in tag_graph.out_edges(v):
                if v_desc.kind == "def":
                    G.add_edge(u, v_desc, **data)
    return G


# TODO: rework the below function to get def descendants up to an arbitrary depth
def get_descendants_up_to_depth(G, start_node, max_depth):
    """
    Get all descendants of a start_node in a MultiDiGraph G up to a certain depth.

    Parameters:
    G (nx.MultiDiGraph): The graph.
    start_node: The node to start the search from.
    max_depth (int): The maximum depth to search.

    Returns:
    set: A set of descendants up to the given depth.
    """
    # Initialize the set of descendants
    descendants = set()
    # Initialize the queue for BFS, storing (node, depth) tuples
    queue = [(start_node, 0)]

    while queue:
        current_node, current_depth = queue.pop(0)

        # If the current depth exceeds max_depth, skip processing this node
        if current_depth > max_depth:
            continue

        # Process the current node
        for _, neighbor in G.out_edges(current_node):
            if neighbor not in descendants:
                descendants.add(neighbor)
                queue.append((neighbor, current_depth + 1))

    return descendants
