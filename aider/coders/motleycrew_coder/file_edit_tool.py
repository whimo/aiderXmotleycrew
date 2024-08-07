import traceback
from typing import TYPE_CHECKING

import git
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from aider import urls
from motleycrew.common import logger
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    from .motleycrew_coder import MotleyCrewCoder


class FileEditToolInput(BaseModel):
    file_path: str = Field(description="The file path to edit.")
    language: str = Field(description="The programming language of the file.")
    search: str = Field(description="The SEARCH block.")
    replace: str = Field(description="The REPLACE block.")


class FileEditTool(MotleyTool):
    def __init__(self, coder: "MotleyCrewCoder", name: str = "edit_file"):
        # TODO: replace coder with specific components
        self.coder = coder

        langchain_tool = StructuredTool.from_function(
            func=self.edit_file,
            name=name,
            description="Make changes to a file using a *SEARCH/REPLACE* block.",
            args_schema=FileEditToolInput,
        )
        super().__init__(langchain_tool)

    def edit_file(self, file_path: str, language: str, search: str, replace: str) -> str:
        error_message = self.edit_file_inner(file_path, search, replace)
        # TODO: num_reflections belongs in the calling agent, not in the tool!
        if error_message:
            if self.coder.num_reflections < self.coder.max_reflections:
                self.coder.num_reflections += 1
                return error_message
            else:
                logger.warning(f"Only {self.coder.max_reflections} reflections allowed, stopping.")
        return self.coder.gpt_prompts.file_edit_success.format(file_path=file_path)

    def edit_file_inner(self, file_path: str, search: str, replace: str) -> str:
        if not search or search[-1] != "\n":
            search += "\n"
        if not replace or replace[-1] != "\n":
            replace += "\n"

        abs_path = self.coder.abs_root_path(file_path)
        if abs_path not in self.coder.abs_fnames:
            return f"""Cannot edit {file_path} yet. You need to add it to the list
            of editable files using the Add_files_to_be_modified tool first."""

        # here it checks for all sorts of other possible issues
        allowed_to_edit = self.coder.allowed_to_edit(file_path)
        if not allowed_to_edit:
            return f"""Cannot edit {file_path}. """

        try:
            self.coder.dirty_commit()  # Add the file to the repo if it's not already there
            self.coder.apply_edits([(file_path, search, replace)])
        except ValueError as err:
            self.coder.num_malformed_responses += 1

            err = err.args[0]

            logger.warning("The LLM did not conform to the edit format.")
            logger.warning(urls.edit_errors)
            logger.warning(str(err))
            return str(err)
        except git.exc.GitCommandError as err:
            message = f"Git error while editing file {file_path}:{str(err)}"
            logger.warning(message)
            return message
        except Exception as err:
            logger.warning("Exception while updating files:")
            logger.warning(str(err), strip=False)

            traceback.print_exc()
            return str(err)

        self.coder.aider_edited_files.append(file_path)

        if self.coder.auto_lint:
            errors = self.coder.linter.lint(self.coder.abs_root_path(file_path))
            if errors:
                logger.error(f"Lint errors in {file_path}: {errors}")
                ok = self.coder.io.confirm_ask("Attempt to fix lint errors?")
                if ok:
                    return errors

        if self.coder.dry_run:
            return f"Did not apply edit to {file_path} (--dry-run)"
        else:
            return f"Applied edit to {file_path}"
