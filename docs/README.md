# 文档 · Docs

技术说明在这里。项目介绍见仓库根目录的 [README.md](../README.md)。  
Technical notes live here. The project introduction is the root [README.md](../README.md).

| 文档 | 内容 · What it covers |
| --- | --- |
| [architecture.md](architecture.md) | 整体结构：浏览器只连页面，转写和翻译留在内网。 Architecture: the browser only talks to the page; ASR and translate stay on the private network. |
| [deploy.md](deploy.md) | 一台 2 核小服务器上的 Compose 部署、口令、内存、swap，以及结果目录怎么挂载。 Deploy on one 2-core host: Compose, the access token, memory, swap, and how the results directory is mounted. |
| [results.md](results.md) | 完成的双语字幕如何落盘、保留几天、重启后还在，以及页面上的再次打开、下载和删除。 How finished bilingual captions are stored, how long they are kept, that they survive a restart, and how the page reopens, downloads, or deletes them. |
| [asr.md](asr.md) | CPU 英文转写（faster-whisper）的环境变量和启动方式。 The CPU English transcription service (faster-whisper): settings and how to run it. |
| [glossary.md](glossary.md) | 转写进程如何只读课程术语表，不改文件。 How ASR reads a course glossary and does not write it. |
| [translate-contract.md](translate-contract.md) | 字幕页如何把英文片段交给翻译服务。 How the captions page sends English segments to the translate service. |
