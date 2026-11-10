# Architecture

translator-edu runs on one 2-vCPU host in mainland China. There is no GPU. v1 is an uploaded recording over plain HTTP. Live mic is still in the page, and it is not the default path.

```
lecturer's browser
        │  HTTPS or http://<web-host>:8080   (the only public port)
        ▼
   apps/web          proxies, does not translate
        │                         │
        │  ws://asr:8001          │  http://translate:8080
        ▼                         ▼
   services/asr              services/translate   (Anna)
   faster-whisper            opus-mt-en-zh + glossary
        │                         │
        └──── glossary/<course>.csv (read-only mount) ────┘
```

The browser never opens ASR or translate. Those processes have no authentication, and translate allows every origin. On Alibaba Cloud ECS the security group should allow the web port only. See [deploy.md](deploy.md).

Live audio is 16 kHz mono PCM in 4-second windows. Each finished English segment is sent to translate on its own, with the same `id`, `start`, and `end` the page is showing. A recording upload is the fallback when live transcription runs slower than real time: the browser keeps the audio and posts the file when you stop.

The translate request and response are defined in [services/translate/README.md](../services/translate/README.md). The page's path through that API is [translate-contract.md](translate-contract.md).
