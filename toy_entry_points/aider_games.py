from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

# This is a list of files to add to the chat
fnames = ["greeting.py"]

model = Model("gpt-4o")

# Create a coder object
io = InputOutput(yes=True)
coder = Coder.create(main_model=model, fnames=fnames, io=io)

files = coder.repo_map.file_group.get_all_filenames()
tag_graph = coder.repo_map.get_tag_graph(files)
repo_map = coder.get_repo_map()
test_tag = coder.repo_map.get_tag("agent_interface.py", line_no=14, tag_graph=tag_graph)

# This will execute one instruction on those files and then return
coder.run("make a script that prints hello world, save it as greeting.py")

# Send another instruction
coder.run("make it say goodbye")

print("Done!")
