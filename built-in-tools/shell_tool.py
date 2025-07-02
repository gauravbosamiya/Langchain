from langchain_community.tools import ShellTool

shell_tool = ShellTool()
results = shell_tool.invoke('dir')
print(results)