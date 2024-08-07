from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from motleycrew.tools import MotleyTool


class GetModifiableFilesToolInput(BaseModel):
    pass


class GetModifiableFilesTool(MotleyTool):
    def __init__(self, coder: "MotleyCrewCoder", name: str = "get_modifiable_files"):

        langchain_tool = StructuredTool.from_function(
            func=self.get_modifiable_files,
            name=name,
            description="Get the relative paths files that can be modified.",
            args_schema=GetModifiableFilesToolInput,
        )
        super().__init__(langchain_tool)
        self.coder = coder

    def get_modifiable_files(self) -> List[str]:
        files = self.coder.abs_fnames
        return [self.coder.get_rel_fname(abs_fname) for abs_fname in files]
