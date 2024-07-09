from typing import TYPE_CHECKING

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from motleycrew.common import logger
from motleycrew.tools import MotleyTool

from aider.codemap.repomap import RepoMap

if TYPE_CHECKING:
    from .motleycrew_coder import MotleyCrewCoder


class InspectObjectToolInput(BaseModel):
    filename: str = Field(description="The filename to inspect.")
    line_number: int = Field(description="The line number where the object definition starts.")


class InspectObjectTool(MotleyTool):
    def __init__(self, repo_map: RepoMap):
        self.repo_map = repo_map

        langchain_tool = StructuredTool.from_function(
            func=self.get_object_summary,
            name="Inspect Entity",
            description="Get the full code of the entity whose definition starts at"
            " the specified location, including summary of the entities it references",
            args_schema=InspectObjectToolInput,
        )
        super().__init__(langchain_tool)

    def get_object_summary(self, filename: str, line_number: int) -> str:
        tag_graph = self.repo_map.get_tag_graph()
        tag = self.repo_map.get_tag(filename, line_number, tag_graph)
        repr = self.repo_map.get_tag_representation(tag, tag_graph)
        return repr
