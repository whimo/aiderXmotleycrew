import os.path
from typing import TYPE_CHECKING

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import List, Optional

from motleycrew.common import logger
from motleycrew.tools import MotleyTool

from aider.codemap.repomap import RepoMap
from aider.codemap.render import RenderCode
from aider.codemap.tag import Tag

if TYPE_CHECKING:
    from .motleycrew_coder import MotleyCrewCoder


class InspectObjectToolInput(BaseModel):
    entity_name: Optional[str] = Field(description="Name of the entity to inspect.", default=None)
    file_name: Optional[str] = Field(
        description="Full or partial name of the file(s) to inspect", default=None
    )


class InspectEntityTool(MotleyTool):
    def __init__(
        self,
        repo_map: RepoMap,
        show_other_files: bool = False,
        max_lines_long=200,
        max_lines_short=25,
    ):
        self.repo_map = repo_map
        self.show_other_files = show_other_files
        self.max_lines_long = max_lines_long
        self.max_lines_short = max_lines_short
        self.requested_tags = set()

        langchain_tool = StructuredTool.from_function(
            func=self.get_object_summary,
            name="inspect_entity",
            description=""""Get the code of the entity with a given name (if that's not too long), 
            including summary of the entities it references. Valid entities 
            are function names, class names, method names prefixed with class, like `Foo.bar`. 
            You can restrict your search to specific files by supplying the file_name argument,
            but ONLY supply the file name/relative path if you need it to disambiguate the entity name,
            or if you want to inspect a whole file; in all other cases, just supply the entity name.
            You can also supply a partial file or directory name to get all files whose relative paths
            contain the partial name you supply.
            You can also request a whole file by name by omitting the entity name.
            You MUST supply at least one of entity_name or file_name.
            """,
            args_schema=InspectObjectToolInput,
        )
        super().__init__(langchain_tool)

    def get_object_summary(
        self, entity_name: Optional[str] = None, file_name: Optional[str] = None
    ) -> str:
        if entity_name is not None:
            entity_name = entity_name.replace("()", "")

        if (entity_name, file_name) in self.requested_tags:
            return "You've already requested that one!"
        else:
            self.requested_tags.add((entity_name, file_name))

        tag_graph = self.repo_map.get_tag_graph()

        # TODO: if file_name is a directory, just list the files in it?
        re_tags = tag_graph.get_tags_from_entity_name(entity_name, file_name)

        if not re_tags:  # maybe it was an explicit import?
            if entity_name is not None:
                out = f"Definition of entity {entity_name} not found in the repo"
                if file_name is None or self.to_dir(file_name) is None:
                    return out  # Absolutely no directories to work with
                else:
                    candidate_dirs = [self.to_dir(file_name)]
            else:
                return f"File {file_name} not found in the repo"
        elif len(re_tags) == 1:
            out = tag_graph.get_tag_representation(
                re_tags[0], parent_details=True, max_lines=self.max_lines_long
            )
            candidate_dirs = [self.to_dir(re_tags[0].fname)]
        else:  # Can get multiple tags eg when requesting a whole file
            # TODO: this could be neater
            repr = "\n".join(
                [
                    tag_graph.get_tag_representation(
                        t, parent_details=False, max_lines=self.max_lines_short
                    )
                    for t in re_tags
                ]
            )
            if len(repr.split("\n")) < self.max_lines_long:
                out = repr
            else:
                out = tag_graph.code_renderer.to_tree(re_tags)

            candidate_dirs = list(set([self.to_dir(t.fname) for t in re_tags]))

        files = set(
            sum(
                [self.repo_map.file_group.get_rel_fnames_in_directory(d) for d in candidate_dirs],
                [],
            )
        )

        mentioned_fnames = set([t.fname for t in re_tags])
        other_fnames = files - mentioned_fnames
        if other_fnames and self.show_other_files:
            out += "\nOther files in same directory(s):\n" + "\n".join(sorted(list(other_fnames)))
        return out

    def to_dir(self, rel_fname: str) -> str:
        abs_dir = self.repo_map.file_group.abs_root_path(rel_fname)
        if os.path.isfile(abs_dir):
            abs_dir = os.path.dirname(abs_dir)
        if os.path.isdir(abs_dir):
            return abs_dir
        else:
            return None
