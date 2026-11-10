# translator-edu

跟上英文课的中英字幕。  
Bilingual captions for English-taught classes.

English → 中文 · 一台小服务器  
English → 中文 · one small server

| 中文 | English |
| --- | --- |
| 给 Tim 自己用的听课工具。英文授课的大学课，在一个页面上看中英对照字幕。 | A personal tool for Tim. Follow an English-taught university class with Chinese and English captions on one page. |

## 主要功能 · Main features

| 中文 | English |
| --- | --- |
| 把课堂上的英文转写成文字（faster-whisper） | Transcribe the English spoken in class (faster-whisper) |
| 再逐段译成中文（opus-mt-en-zh，通过 CTranslate2） | Then translate each segment into Chinese (opus-mt-en-zh via CTranslate2) |
| 每门课一份术语表，同一个专业词始终用同一种译法。第一门课：深度学习与神经网络 | One glossary per course, so each technical term always comes out the same way. First course: deep learning and neural networks |
| 上传一节课的录音，中英字幕随进度一段段出现，并可下载 `.txt` 与 `.srt` | Upload a class recording. Chinese–English captions appear segment by segment, with progress, and can be downloaded as `.txt` or `.srt` |
| 实时麦克风字幕：需要 HTTPS，之后加入 | Live mic captions: need HTTPS, coming later |

跑在一台小服务器上。技术细节见 [docs/](docs/)。  
It runs on a single small server. Technical details are in [docs/](docs/).
