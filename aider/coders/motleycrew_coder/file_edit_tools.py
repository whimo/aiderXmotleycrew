from .file_edit_tool import FileEditTool
from .add_files_tool import AddFilesTool
from .get_modifiable_files_tool import GetModifiableFilesTool


def get_file_edit_tools(coder):
    return AddFilesTool(coder), GetModifiableFilesTool(coder), FileEditTool(coder)
