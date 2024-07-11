from typing import TYPE_CHECKING

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import List

from motleycrew.common import logger
from motleycrew.tools import MotleyTool

from aider.codemap.repomap import RepoMap
from aider.codemap.tag import Tag

if TYPE_CHECKING:
    from .motleycrew_coder import MotleyCrewCoder


class InspectObjectToolInput(BaseModel):
    entity_name: str = Field(description="Name of the entity to inspect.")


class InspectEntityTool(MotleyTool):
    def __init__(self, repo_map: RepoMap):
        self.repo_map = repo_map
        self.requested_tags = set()

        langchain_tool = StructuredTool.from_function(
            func=self.get_object_summary,
            name="Inspect_Entity",
            description="Get the full code of the entity with a given name, "
            "including summary of the entities it references",
            args_schema=InspectObjectToolInput,
        )
        super().__init__(langchain_tool)

    def get_object_summary(self, entity_name: str) -> str:
        entity_name = entity_name.replace("()", "")
        if entity_name in self.requested_tags:
            return "You've already requested that one!"
        else:
            self.requested_tags.add(entity_name)

        tag_graph = self.repo_map.get_tag_graph()

        re_tags = tag_graph.get_tags_from_entity_name(entity_name)

        if not re_tags:  # maybe it was an explicit import?
            return f"Definition of entity {entity_name} not found in the repo"

        else:
            repr = "\n".join([tag_graph.get_tag_representation(t) for t in re_tags])
            return repr
