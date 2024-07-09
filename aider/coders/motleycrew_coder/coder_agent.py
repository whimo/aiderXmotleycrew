from typing import Optional, Any, TYPE_CHECKING

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.tools import render_text_description

from motleycrew.agents import LangchainMotleyAgent, MotleyOutputHandler
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.common.utils import print_passthrough
from motleycrew.tools import MotleyTool
from motleycrew.tracking import add_default_callbacks_to_langchain_config
from .add_files_tool import AddFilesTool
from .coder_output_handler import CoderOutputHandler
from .file_edit_tool import FileEditTool

if TYPE_CHECKING:
    from .motleycrew_coder import MotleyCrewCoder


ADD_FILES_TOOL_NAME = "add_files"
FILE_EDIT_TOOL_NAME = "edit_file"
RETURN_TO_USER_TOOL_NAME = "return_to_user"


CODER_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="original_prompt"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="additional_notes", optional=True),
    ]
)


class CoderAgent(LangchainMotleyAgent):
    def __init__(
        self,
        coder: "MotleyCrewCoder",
        llm: BaseChatModel | None = None,
        verbose: bool = False,
    ):
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        add_files_tool = AddFilesTool(name=ADD_FILES_TOOL_NAME, coder=coder)
        file_edit_tool = FileEditTool(name=FILE_EDIT_TOOL_NAME, coder=coder)
        output_handler = CoderOutputHandler(name=RETURN_TO_USER_TOOL_NAME, coder=coder)

        def agent_factory(tools: dict[str, MotleyTool]) -> AgentExecutor:
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

                prompt = CODER_PROMPT_TEMPLATE.invoke(
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
