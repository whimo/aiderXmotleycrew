from typing import TYPE_CHECKING

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from aider.utils import is_image_file
from motleycrew.common import logger
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    from .motleycrew_coder import MotleyCrewCoder


class AddFilesToolInput(BaseModel):
    files: list[str] = Field(description="List of file paths to add to the chat.")


class AddFilesTool(MotleyTool):
    def __init__(self, name: str, coder: "MotleyCrewCoder"):
        self.coder = coder

        langchain_tool = StructuredTool.from_function(
            func=self.add_files,
            name=name,
            description="Ask the user to add files to the chat.",
            args_schema=AddFilesToolInput,
        )
        super().__init__(langchain_tool)

    def add_files(self, files: list[str]):
        if not self.coder.io.confirm_ask("Add these files to the chat?"):
            return "The user declined to add the files."

        for path in files:
            logger.info(f"Adding file to the chat: {path}")
            self.coder.add_rel_fname(path)

        files_content_prompt = self.make_files_content_prompt(files)
        return files_content_prompt

    def make_files_content_prompt(self, files):
        prompt = self.coder.gpt_prompts.files_content_prefix
        for filename, content in self.get_files_content(files):
            if not is_image_file(filename):
                prompt += "\n"
                prompt += filename

                prompt += f"\n```\n"
                prompt += content
                prompt += f"```\n"

        return prompt

    def get_files_content(self, files: list[str]):
        for filename in files:
            abs_filename = self.coder.abs_root_path(filename)
            content = self.read_text_file(abs_filename)

            if content is None:
                logger.warning(f"Error reading {filename}, dropping it from the chat.")
                self.coder.abs_fnames.remove(abs_filename)
            else:
                yield filename, content

    def read_text_file(self, filename: str):
        try:
            with open(str(filename), "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"{filename}: file not found error")
            return
        except IsADirectoryError:
            logger.error(f"{filename}: is a directory")
            return
        except UnicodeError as e:
            logger.error(f"{filename}: {e}")
            return
