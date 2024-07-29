import langchain_core.messages.utils as msg_utils
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate

from aider.coders import EditBlockCoder
from aider.dump import dump  # noqa: F401
from .coder_agent import CoderAgent
from .motleycrew_prompts import MotleyCrewPrompts


class MotleyCrewCoder(EditBlockCoder):
    max_reflections = 5  # increased number because each edit tool error counts as a reflection

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
                    self.aider_edited_files = []

                    prompt = self.create_prompt(new_user_message)
                    response = self.agent.invoke({"prompt": prompt})

                    if with_message:
                        return response

            except KeyboardInterrupt:
                self.keyboard_interrupt()
            except EOFError:
                return

    def create_prompt(self, user_message: str) -> ChatPromptTemplate:
        messages = [self.gpt_prompts.main_system, self.gpt_prompts.system_reminder]

        messages += self.gpt_prompts.example_messages
        messages += [
            HumanMessage(
                "I switched to a new code base. Please don't consider the above files"
                " or try to edit them any longer."
            ),
            AIMessage("Ok."),
        ]

        messages += msg_utils.convert_to_messages(self.get_files_messages())
        messages += [HumanMessage(user_message)]
        messages += [self.gpt_prompts.system_reminder]

        return self.create_and_fill_prompt_template(messages)

    def create_and_fill_prompt_template(self, messages: list) -> ChatPromptTemplate:
        lazy_prompt = self.gpt_prompts.lazy_prompt if self.main_model.lazy else ""

        prompt_template = ChatPromptTemplate.from_messages(messages)
        prompt_template = prompt_template.partial(lazy_prompt=lazy_prompt)
        return prompt_template
