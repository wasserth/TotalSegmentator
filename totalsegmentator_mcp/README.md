# TotalSegmentator MCP server

The Model Context Protocol (MCP) server gives AI clients access to TotalSegmentator tools. It supports stdio and Streamable HTTP.

Configure either stdio or Streamable HTTP for each client.


## Install

Install TotalSegmentator in an isolated environment with uv or pipx. Choose one method.

With uv:

```bash
uv tool install \
  'TotalSegmentator[mcp] @ git+https://github.com/wasserth/TotalSegmentator.git'
```

With pipx:

```bash
pipx install \
  'TotalSegmentator[mcp] @ git+https://github.com/wasserth/TotalSegmentator.git'
```

Make sure that the command is available:

```bash
totalseg_mcp --help
```


## Install the agent skill

The corresponding skill teaches compatible agents how to select TotalSegmentator tasks, classes, and run options. Execute the following command and choose the agent(s) of your choice:

```bash
npx skills add wasserth/TotalSegmentator@totalsegmentator --global
```

The MCP server works without the skill.


## Connect with stdio

Stdio is the recommended transport for local use. The client starts and stops the server process.

### Codex

```bash
codex mcp add TotalSegmentator -- totalseg_mcp --transport stdio
```



### Claude Code

```bash
claude mcp add --scope user --transport stdio TotalSegmentator \
  -- totalseg_mcp --transport stdio
```



### Cursor

Add this configuration to `~/.cursor/mcp.json` for global use:

```json
{
  "mcpServers": {
    "TotalSegmentator": {
      "type": "stdio",
      "command": "totalseg_mcp",
      "args": ["--transport", "stdio"]
    }
  }
}
```

For project-only use, add the same configuration to `.cursor/mcp.json` in the project.

## Connect with Streamable HTTP

Start the server:

```bash
totalseg_mcp --transport http --host 127.0.0.1 --port 8000
```

While clients use the server, keep this process active. The MCP endpoint is `http://127.0.0.1:8000/mcp`.

### Codex

```bash
codex mcp add TotalSegmentator --url http://127.0.0.1:8000/mcp
```



### Claude Code

```bash
claude mcp add --scope user --transport http \
  TotalSegmentator http://127.0.0.1:8000/mcp
```



### Cursor

Add this configuration to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "TotalSegmentator": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```



## Available tools

- `list_all_classes`: List output classes by modality.
- `get_class_options`: Find tasks that provide specified classes.
- `list_all_tasks`: List tasks by modality.
- `get_task_details`: Get classes, modality, and license information for specified tasks.
- `get_modality`: Predict whether a NIfTI image is CT or MR.
- `detect_contrast_phase`: Estimate the contrast phase of a CT image.
- `estimate_body_statistics`: Estimate body measurements and patient attributes.
- `run_segmentation`: Run segmentation and return a machine-readable report.



## Notes

- The first inference can download model weights.
- Some tasks require a TotalSegmentator license.
- Segmentation can exceed the default tool timeout of an MCP client.
- Input and output paths refer to the machine that runs the server.
- The HTTP server is currently not authenticated.
- If a GUI client cannot find `totalseg_mcp`, use the absolute path from `which totalseg_mcp`.

Model outputs are estimates and are not definitive clinical findings. TotalSegmentator is not a medical device.