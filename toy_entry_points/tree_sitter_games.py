from tree_sitter_languages import get_language, get_parser
from grep_ast import filename_to_lang
from aider.codemap.parse import get_query, tree_to_tags
from aider.codemap.graph import build_tag_graph

lang = filename_to_lang("dummy.py")
language = get_language(lang)
parser = get_parser(lang)

# PY_LANGUAGE = Language(name=tspython.language())
#
# parser = Parser(PY_LANGUAGE)
tree = parser.parse(
    bytes(
        """
def foo(a: int, 
        b: str) -> None:
    '''
    This is a docstring
    a: int - an integer
    b: str - a string
    
    This is a multiline docstring
    ''' 
    if bar:
        baz()
        
class A:
    def __init__(self):
        foo(1,"2")
        
    def beep(self):
        return self.boop()
        
class B(BaseModel):
    a: int
    b: str
    def blah(self):
        c = A()
        return c.beep()
""",
        "utf8",
    )
)
query = get_query(lang)
tags = tree_to_tags(tree, query, "dummy.py", "dummy.py")
graph = build_tag_graph(tags)

import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(G):
    # Create a color map based on the 'kind' attribute
    color_map = []
    for node in G:
        if G.nodes[node]["kind"] == "def":
            color_map.append("blue")
        else:
            color_map.append("green")

    # Create a dictionary of labels
    labels = {node: node.name for node in G.nodes}

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=color_map, labels=labels, with_labels=True)

    # Show the graph
    plt.show()


visualize_graph(graph)
print("yay!")
