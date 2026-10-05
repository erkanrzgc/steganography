# Third-party tools in the `full` image

Optional research extra `jpeg-sim` uses [conseal 2025.11](https://github.com/uibk-uncover/conseal)
(MPL-2.0) for embedding simulation, not end-to-end message extraction. It is not
a base-wheel dependency or bundled dataset/model. Respect upstream and original
dataset usage conditions; source attribution alone does not grant redistribution.

The core image and Python package are MIT licensed and do not bundle the tools
below. The optional `full` image executes them as separate, resource-limited
processes and reports their availability explicitly.

| Tool | Version | License | Source |
|---|---:|---|---|
| Binwalk | 2.3.4 (Debian bookworm) | MIT | https://github.com/ReFirmLabs/binwalk |
| zsteg | 0.2.14 | MIT | https://github.com/zed-0xff/zsteg |
| iostruct | 0.7.0 | MIT | zsteg runtime dependency |
| zpng | 0.4.6 | MIT | zsteg runtime dependency |
| rainbow | 3.1.1 | MIT | zsteg runtime dependency |
| prime | 0.1.4 | Ruby/BSD-2-Clause | zsteg runtime dependency |
| Stegseek | 0.6 | GPL-3.0 | https://github.com/RickdeJager/stegseek |
| Steghide | 0.5.1 (Debian bookworm) | GPL-2.0+ | https://steghide.sourceforge.net/ |
| OutGuess | 0.4 (Debian bookworm) | BSD-4-Clause | https://github.com/resurrecting-open-source-projects/outguess |
| ExifTool | 12.57 (Debian bookworm) | Artistic/GPL | https://exiftool.org/ |
| pngcheck | 3.0.3 (Debian bookworm) | zlib | http://www.libpng.org/pub/png/apps/pngcheck.html |
| FFmpeg | 5.1.x (Debian bookworm security) | GPL/LGPL | https://ffmpeg.org/ |
| SoX | 14.4.2 (Debian bookworm) | GPL/LGPL | https://sox.sourceforge.net/ |
| gifsicle | 1.93 (Debian bookworm) | GPL-2.0 | https://www.lcdf.org/gifsicle/ |
| ZBar | 0.23.92 (Debian bookworm) | LGPL-2.1 | https://github.com/mchehab/zbar |
| OpenStego | 0.8.6 | GPL-2.0 | https://www.openstego.com/ |

Container builds retain the upstream programs' license obligations. No model
weights or third-party datasets are redistributed.

Downloaded release artifacts are verified with the SHA-256 values pinned in
`Dockerfile.full`. Debian packages are verified by APT's signed Release and
package checksum chain and their installed versions are checked during full
image E2E.
