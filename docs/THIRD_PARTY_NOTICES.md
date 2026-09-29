# Third-party notices

The Windows runtime core archive carries these license files alongside the
project's `LICENSE`:

- `LICENSE-ggml.txt` — ggml backend and core, MIT License, copyright (c) 2023–2026 The ggml authors.
- `LICENSE-cpp-httplib.txt` — cpp-httplib, MIT License, copyright (c) 2017 yhirose.
- `LICENSE-yyjson.txt` — yyjson, MIT License, copyright (c) 2020 YaoYuan.

The separate CUDA runtime archive includes NVIDIA's EULA. Model weights are not
included in these runtime packages; each model's license and download terms
remain with its model repository.

## RoyalCities Foundation prompt engine

`src/sat/foundation_prompt.h` adapts the weighted vocabulary and M1/T1 structure from
[RoyalCities/RC-stable-audio-tools](https://github.com/RoyalCities/RC-stable-audio-tools).
The C++ random stream is independent and is not intended to reproduce Python's
`random.Random` byte-for-byte.

The source repository distributes this code under the MIT License:

> MIT License
>
> Copyright (c) 2023 Stability AI
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify,
> merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
> persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
> PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
> FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.
