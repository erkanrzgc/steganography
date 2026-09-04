# Third-party tools in the `full` image

The core image and Python package are MIT licensed and do not bundle the tools
below. The optional `full` image executes them as separate, resource-limited
processes and reports their availability explicitly.

| Tool | Version | License | Source |
|---|---:|---|---|
| zsteg | 0.2.13 | MIT | https://github.com/zed-0xff/zsteg |
| Stegseek | 0.6 | GPL-3.0 | https://github.com/RickdeJager/stegseek |
| ExifTool | Debian bookworm package | Artistic/GPL | https://exiftool.org/ |

Container builds retain the upstream programs' license obligations. No model
weights or third-party datasets are redistributed.
