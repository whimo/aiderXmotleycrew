import traceback
from typing import Optional, Any

import git
import langchain_core.messages.utils as msg_utils
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.tools.render import render_text_description
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.tools import StructuredTool

from aider import urls
from aider.utils import format_content, is_image_file
from motleycrew.agents import MotleyOutputHandler
from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.common import LLMFramework
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.common.llms import init_llm
from motleycrew.common.utils import print_passthrough
from motleycrew.tools import MotleyTool
from motleycrew.tracking import add_default_callbacks_to_langchain_config
from . import EditBlockCoder
from .motleycrew_prompts import MotleyCrewPrompts
from ..dump import dump  # noqa: F401

ADD_FILES_TOOL_NAME = "add_files"
FILE_EDIT_TOOL_NAME = "edit_file"
RETURN_TO_USER_TOOL_NAME = "return_to_user"


class MotleyCrewCoder(EditBlockCoder):
    max_reflections = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gpt_prompts = MotleyCrewPrompts()
        self.agent = CoderAgent(coder=self, verbose=self.verbose)

    def run(self, with_message=None):
        while True:
            self.init_before_message()

            try:
                if with_message:
                    new_user_message = with_message
                    self.io.user_input(with_message)
                else:
                    new_user_message = self.run_loop()

                if new_user_message:
                    self.reflected_message = None
                    response = self.send_new_user_message(new_user_message)

                    if with_message:
                        return response

            except KeyboardInterrupt:
                self.keyboard_interrupt()
            except EOFError:
                return

    def send_new_user_message(self, inp):
        self.aider_edited_files = []

        self.cur_messages = [
            dict(role="user", content=inp),
        ]

        messages = self.format_messages()

        self.io.log_llm_history("TO LLM", messages)

        # if self.verbose:
        #    utils.show_messages(messages, functions=self.functions)
        # TODO: display PromptTemplates here
        # if self.verbose:
        #     utils.show_messages(messages, functions=self.functions)

        try:
            # TODO: function calls
            # yield from self.send(messages, functions=self.functions)
            return self.agent.invoke({"prompt": messages})
        except KeyboardInterrupt:
            self.keyboard_interrupt()
        # except ExhaustedContextWindow:
        #     exhausted = True
        # except litellm.exceptions.BadRequestError as err:
        #     if "ContextWindowExceededError" in err.message:
        #         exhausted = True
        #     else:
        #         self.io.tool_error(f"BadRequestError: {err}")
        #         return
        # except openai.BadRequestError as err:
        #     if "maximum context length" in str(err):
        #         exhausted = True
        #     else:
        #         raise err
        except Exception as err:
            # TODO: exhausted context windows
            self.io.tool_error(f"Unexpected error: {err}")
            raise  # return

    def format_messages(self):
        self.choose_fence()
        messages = [self.gpt_prompts.main_system]

        example_messages = []
        if self.main_model.examples_as_sys_msg:
            raise NotImplementedError("Please use tool calling models with MotleyCrewCoder.")

        for msg in self.gpt_prompts.example_messages:
            assert isinstance(
                msg, BaseMessage
            ), "MotleyCrewCoder expects BaseMessage objects in example_messages."
            example_messages.append(msg)
        if self.gpt_prompts.example_messages:
            example_messages += [
                HumanMessage(
                    "I switched to a new code base. Please don't consider the above files"
                    " or try to edit them any longer."
                ),
                AIMessage("Ok."),
            ]

        messages += [self.gpt_prompts.system_reminder]
        messages += example_messages

        self.summarize_end()
        messages += msg_utils.convert_to_messages(self.done_messages)
        messages += msg_utils.convert_to_messages(self.get_files_messages())

        # TODO: count tokens
        # messages_tokens = self.main_model.token_count(messages)
        # reminder_tokens = self.main_model.token_count(reminder_message)
        # cur_tokens = self.main_model.token_count(self.cur_messages)
        #
        # if None not in (messages_tokens, reminder_tokens, cur_tokens):
        #     total_tokens = messages_tokens + reminder_tokens + cur_tokens
        # else:
        #     # add the reminder anyway
        #     total_tokens = 0
        total_tokens = 0

        messages += msg_utils.convert_to_messages(self.cur_messages)

        max_input_tokens = self.main_model.info.get("max_input_tokens")
        # Add the reminder prompt if we still have room to include it.
        if max_input_tokens is None or total_tokens < max_input_tokens:
            messages += [self.gpt_prompts.system_reminder]

        return self.create_and_fill_prompt_template(messages)

    def create_and_fill_prompt_template(self, messages: list):
        lazy_prompt = self.gpt_prompts.lazy_prompt if self.main_model.lazy else ""

        prompt_template = ChatPromptTemplate.from_messages(messages)
        prompt_template = prompt_template.partial(lazy_prompt=lazy_prompt)
        return prompt_template


class AddFilesToolInput(BaseModel):
    files: list[str] = Field(description="List of file paths to add to the chat.")


class FileEditToolInput(BaseModel):
    file_path: str = Field(description="The file path to edit.")
    language: str = Field(description="The programming language of the file.")
    search: str = Field(description="The SEARCH block.")
    replace: str = Field(description="The REPLACE block.")


class CoderOutputHandlerInput(BaseModel):
    question: Optional[str] = Field(
        default=None,
        description="The question to be sent to the user. "
        "Leave empty if you only want to submit your edits.",
    )


class AddFilesTool(MotleyTool):
    def __init__(self, coder: MotleyCrewCoder):
        self.coder = coder

        langchain_tool = StructuredTool.from_function(
            func=self.add_files,
            name=ADD_FILES_TOOL_NAME,
            description="Ask the user to add files to the chat.",
            args_schema=AddFilesToolInput,
        )
        super().__init__(langchain_tool)

    def add_files(self, files: list[str]):
        for path in files:
            self.coder.io.tool_output(path)

        if not self.coder.io.confirm_ask("Add these files to the chat?"):
            return "The user declined to add the files."

        for path in files:
            self.coder.add_rel_fname(path)

        files_content_prompt = self.make_files_content_prompt(files)
        return files_content_prompt

    def make_files_content_prompt(self, files):
        prompt = self.coder.gpt_prompts.files_content_prefix
        for filename, content in self.get_files_content(files):
            if not is_image_file(filename):
                prompt += "\n"
                prompt += filename

                prompt += f"\n{self.coder.fence[0]}\n"
                prompt += content
                prompt += f"{self.coder.fence[1]}\n"

        return prompt

    def get_files_content(self, files: list[str]):
        for filename in files:
            abs_filename = self.coder.abs_root_path(filename)
            content = self.coder.io.read_text(abs_filename)

            if content is None:
                self.coder.io.tool_error(f"Dropping {filename} from the chat.")
                self.coder.abs_fnames.remove(abs_filename)
            else:
                yield filename, content


class FileEditTool(MotleyTool):
    def __init__(self, coder: MotleyCrewCoder):
        self.coder = coder

        langchain_tool = StructuredTool.from_function(
            func=self.edit_file,
            name=FILE_EDIT_TOOL_NAME,
            description="Make changes to a file using a *SEARCH/REPLACE* block.",
            args_schema=FileEditToolInput,
        )
        super().__init__(langchain_tool)

    def edit_file(self, file_path: str, language: str, search: str, replace: str):
        error_message = self.edit_file_inner(file_path, search, replace)
        if error_message:
            if self.coder.num_reflections < self.coder.max_reflections:
                self.coder.num_reflections += 1
                return error_message
            else:
                self.coder.io.tool_error(
                    f"Only {self.coder.max_reflections} reflections allowed, stopping."
                )
        return self.coder.gpt_prompts.file_edit_success.format(file_path=file_path)

    def edit_file_inner(self, file_path: str, search: str, replace: str):
        allowed_to_edit = self.coder.allowed_to_edit(file_path)
        if not allowed_to_edit:
            return f"Cannot edit {file_path}."

        try:
            self.coder.dirty_commit()  # Add the file to the repo if it's not already there
            self.coder.apply_edits([(file_path, search, replace)])
        except ValueError as err:
            self.coder.num_malformed_responses += 1

            err = err.args[0]

            self.coder.io.tool_error("The LLM did not conform to the edit format.")
            self.coder.io.tool_error(urls.edit_errors)
            self.coder.io.tool_error()
            self.coder.io.tool_error(str(err), strip=False)

            self.coder.reflected_message = str(err)
            return str(err)
        except git.exc.GitCommandError as err:
            self.coder.io.tool_error(str(err))
            return  # I see no point in returning the error to the agent (the user is aware)
        except Exception as err:
            self.coder.io.tool_error("Exception while updating files:")
            self.coder.io.tool_error(str(err), strip=False)

            traceback.print_exc()
            return str(err)

        self.coder.aider_edited_files.append(file_path)

        if self.coder.auto_lint:
            errors = self.coder.linter.lint(self.coder.abs_root_path(file_path))
            if errors:
                self.coder.io.tool_error(errors)
                ok = self.coder.io.confirm_ask("Attempt to fix lint errors?")
                if ok:
                    return errors

        if self.coder.dry_run:
            self.coder.io.tool_output(f"Did not apply edit to {file_path} (--dry-run)")
        else:
            self.coder.io.tool_output(f"Applied edit to {file_path}")
        return


class CoderOutputHandler(MotleyOutputHandler):
    _name = RETURN_TO_USER_TOOL_NAME
    _description = (
        "Call this tool when you have finished your edits "
        "OR when you need to ask the user a question. "
        "ONLY COMMUNICATE WITH THE USER THROUGH THIS TOOL. Direct messages will be ignored."
    )
    _args_schema = CoderOutputHandlerInput

    def __init__(self, coder: EditBlockCoder):
        super().__init__()

        self.coder = coder

    def handle_output(self, question: Optional[str] = None):
        question = question or "OK"  # Empty question == OK
        self.coder.partial_response_content = question
        self.coder.partial_response_function_call = None

        self.coder.io.ai_output(question)

        self.coder.io.tool_output()
        self.coder.io.log_llm_history("LLM RESPONSE", format_content("ASSISTANT", question))

        reflected_message = self.process_output_and_reflect()

        if reflected_message:
            if self.coder.num_reflections < self.coder.max_reflections:
                self.coder.num_reflections += 1
                return InvalidOutput(reflected_message)
            else:
                self.coder.io.tool_error(
                    f"Only {self.coder.max_reflections} reflections allowed, stopping."
                )

        if question:
            return question

    def process_output_and_reflect(self):
        edited_files = self.coder.aider_edited_files

        if edited_files and self.coder.auto_test:
            test_errors = self.coder.commands.cmd_test(self.coder.test_cmd)
            self.coder.test_outcome = not test_errors
            if test_errors:
                ok = self.coder.io.confirm_ask("Attempt to fix test errors?")
                if ok:
                    return test_errors


coder_prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="original_prompt"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="additional_notes", optional=True),
    ]
)


class CoderAgent(LangchainMotleyAgent):
    def __init__(
        self,
        coder: MotleyCrewCoder,
        llm: BaseChatModel | None = None,
        verbose: bool = False,
    ):
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        add_files_tool = AddFilesTool(coder=coder)
        file_edit_tool = FileEditTool(coder=coder)
        output_handler = CoderOutputHandler(coder=coder)

        def agent_factory(tools: dict) -> AgentExecutor:
            assert ADD_FILES_TOOL_NAME in tools, f"Tool {ADD_FILES_TOOL_NAME} is required."
            assert FILE_EDIT_TOOL_NAME in tools, f"Tool {FILE_EDIT_TOOL_NAME} is required."
            assert (
                RETURN_TO_USER_TOOL_NAME in tools
            ), f"Tool {RETURN_TO_USER_TOOL_NAME} is required."

            tools_for_langchain = []
            for tool in tools.values():
                tool_for_langchain = tool.to_langchain_tool()
                tool_for_langchain.handle_tool_error = True
                tool_for_langchain.handle_validation_error = True
                tools_for_langchain.append(tool_for_langchain)

            llm_with_tools = llm.bind_tools(tools=tools_for_langchain)
            tools_description = render_text_description(tools_for_langchain)

            def prepare_prompt(input: dict):
                messages_template = input["input"]
                messages_with_tools = messages_template.invoke(
                    {"tools": tools_description}
                ).to_messages()

                prompt = coder_prompt_template.invoke(
                    dict(
                        original_prompt=messages_with_tools,
                        agent_scratchpad=format_to_tool_messages(input["intermediate_steps"]),
                        additional_notes=input.get("additional_notes") or [],
                    )
                )
                return prompt

            agent = (
                RunnableLambda(print_passthrough)
                | RunnableLambda(prepare_prompt)
                | llm_with_tools
                | ToolsAgentOutputParser()
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools_for_langchain,
                handle_parsing_errors=True,
                verbose=verbose,
            )
            return agent_executor

        super().__init__(
            description="Coder agent",
            name="coder",
            agent_factory=agent_factory,
            tools=[add_files_tool, file_edit_tool],
            output_handler=output_handler,
            verbose=True,
            chat_history=False,  # History is managed by the coder for now
        )
        self.coder = coder

    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        self.materialize()

        if isinstance(self.output_handler, MotleyOutputHandler):
            self.output_handler.agent = self
            self.output_handler.agent_input = input

        config = add_default_callbacks_to_langchain_config(config)
        if self.get_session_history_callable:
            config["configurable"] = config.get("configurable") or {}
            config["configurable"]["session_id"] = (
                config["configurable"].get("session_id") or "default"
            )

        output = self.agent.invoke(
            {"input": input["prompt"]},
            config,
        )

        return output
