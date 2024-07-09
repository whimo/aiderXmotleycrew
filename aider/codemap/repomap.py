import colorsys
import os
import random
import sys
import warnings
from typing import List, Set

from pathlib import Path

import networkx as nx
from tqdm import tqdm


from aider.codemap.parse import get_tags_raw, Tag, read_text  # noqa: F402
from aider.codemap.graph import rank_tags, rank_tags_directly, build_tag_graph  # noqa: F402
from aider.codemap.file_group import (
    FileGroup,
    find_src_files,
    get_ident_mentions,
    get_ident_filename_matches,
)
from aider.codemap.render import RenderCode
from aider.dump import dump  # noqa: F402,E402

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)


class RepoMap:
    CACHE_VERSION = 3
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    cache_missing = False

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        file_group: FileGroup = None,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

        # self.load_tags_cache()

        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window

        self.token_count = main_model.token_count
        self.repo_content_prefix = repo_content_prefix
        self.file_group = file_group
        self.code_renderer = RenderCode(text_encoding=self.io.encoding)

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        add_prefix: bool = True,
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        # With no files in the chat, give a bigger view of the entire repo
        MUL = 16
        padding = 4096
        if self.max_map_tokens and self.max_context_window:
            target = min(self.max_map_tokens * MUL, self.max_context_window - padding)
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files, other_files, self.max_map_tokens, mentioned_fnames, mentioned_idents
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        num_tokens = self.token_count(files_listing)
        if self.verbose:
            self.io.tool_output(f"Repo-map: {num_tokens/1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix and add_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_tag(self, fname: str, line_no: int, tag_graph: nx.MultiDiGraph):
        files = [f for f in self.file_group.get_all_filenames() if fname in f]
        for node in tag_graph.nodes:
            if node.fname in files:
                if node.line == line_no - 1:
                    return node
        return None

    def get_tag_representation(self, tag: Tag, tag_graph: nx.MultiDiGraph) -> str:
        if tag is None:
            return None
        if tag not in tag_graph.nodes:
            raise ValueError(f"The tag {tag} is not in the tag graph")
        children = []
        for c in tag_graph.successors(tag):
            if (  # If the child is included in the parent's full text anyway, skip it
                c.fname == tag.fname
                and c.byte_range[0] >= tag.byte_range[0]
                and c.byte_range[1] <= tag.byte_range[1]
            ):
                continue
            children.append(c)

        out = [tag.rel_fname + ":", self.code_renderer.text_with_line_numbers(tag)]
        if children:
            out.extend(
                [
                    "Referenced entities summary:",
                    self.code_renderer.to_tree(children),
                ]
            )
        return "\n".join(out)

    def get_tag_graph(self, abs_fnames: List[str] | None = None) -> nx.MultiDiGraph:
        if not abs_fnames:
            abs_fnames = self.file_group.get_all_filenames()
        clean_fnames = self.file_group.validate_fnames(abs_fnames)
        tags = sum([self.tags_from_filename(fname) for fname in clean_fnames], [])
        return build_tag_graph(tags)

    def tags_from_filename(self, fname):
        def get_tags_raw_function(fname):
            code = read_text(fname, self.io.encoding)
            rel_fname = self.file_group.get_rel_fname(fname)
            data = get_tags_raw(fname, rel_fname, code)
            assert isinstance(data, list)
            return data

        return self.file_group.cached_function_call(fname, get_tags_raw_function)

    def get_ranked_tags(self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents):

        # Check file names for validity
        fnames = sorted(set(chat_fnames).union(set(other_fnames)))

        # Do better filename matching
        clean_mentioned_filenames = []
        for name in mentioned_fnames:
            for fname in fnames:
                if name in fname:
                    clean_mentioned_filenames.append(fname)
                    break
        mentioned_fnames = clean_mentioned_filenames

        # What does that do?
        if self.cache_missing:
            fnames = tqdm(fnames)
        self.cache_missing = False

        cleaned = self.file_group.validate_fnames(fnames)

        # All the source code parsing happens here
        tag_graph = self.get_tag_graph(cleaned)
        tags = list(tag_graph.nodes)

        # this constructs the graph and ranks the tags based on it
        other_rel_fnames = [self.file_group.get_rel_fname(fname) for fname in other_fnames]

        ranked_tags = rank_tags(
            tags, mentioned_fnames, mentioned_idents, chat_fnames, other_rel_fnames
        )
        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        """
        Does a binary search over the number of tags to include in the map,
        to find the largest map that fits within the token limit.
        :param chat_fnames:
        :param other_fnames:
        :param max_map_tokens:
        :param mentioned_fnames:
        :param mentioned_idents:
        :return:
        """
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        ranked_tags = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = [self.file_group.get_rel_fname(fname) for fname in chat_fnames]

        # Guess a small starting number to help with giant repos
        middle = min(max_map_tokens // 25, num_tags)

        self.tree_cache = dict()

        while lower_bound <= upper_bound:
            used_tags = [tag for tag in ranked_tags[:middle] if tag[0] not in chat_rel_fnames]
            tree = self.code_renderer.to_tree(used_tags)
            num_tokens = self.token_count(tree)

            if num_tokens < max_map_tokens and num_tokens > best_tree_tokens:
                best_tree = tree
                best_tree_tokens = num_tokens

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        return best_tree

    # tree_cache = dict()

    def repo_map_from_message(
        self, message: str, abs_added_fnames: Set[str] | None = None, add_prefix: bool = False
    ) -> str:
        if not abs_added_fnames:
            abs_added_fnames = set()

        cur_msg_text = message
        all_files = self.file_group.get_all_filenames()
        other_files = set(all_files) - set(abs_added_fnames)

        mentioned_fnames = self.file_group.get_file_mentions(cur_msg_text, abs_added_fnames)
        mentioned_idents = get_ident_mentions(cur_msg_text)

        all_rel_fnames = [self.file_group.get_rel_fname(f) for f in all_files]
        mentioned_fnames.update(get_ident_filename_matches(mentioned_idents, all_rel_fnames))

        repo_content = self.get_repo_map(
            abs_added_fnames,
            other_files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            add_prefix=add_prefix,
        )

        # fall back to global repo map if files in chat are disjoint from rest of repo
        if not repo_content:
            repo_content = self.get_repo_map(
                set(),
                set(all_files),
                mentioned_fnames=mentioned_fnames,
                mentioned_idents=mentioned_idents,
                add_prefix=add_prefix,
            )

        # fall back to completely unhinted repo
        if not repo_content:
            repo_content = self.get_repo_map(
                set(),
                set(all_files),
                add_prefix=add_prefix,
            )

        return repo_content


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = ""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())
    for lang, ext in data:
        res += "<tr>"
        res += f'<td style="text-align: center;">{lang:20}</td>\n'
        res += f'<td style="text-align: center;">{ext:20}</td>\n'
        res += "</tr>"
    return res


if __name__ == "__main__":
    fnames = sys.argv[1:]

    chat_fnames = []
    other_fnames = []
    for fname in sys.argv[1:]:
        if Path(fname).is_dir():
            chat_fnames += find_src_files(fname)
        else:
            chat_fnames.append(fname)

    rm = RepoMap(root=".")
    repo_map = rm.get_ranked_tags_map(chat_fnames, other_fnames)

    dump(len(repo_map))
    print(repo_map)
