# Model advisor

Kerniq plugin. This folder is the whole plugin. It is not part of the Kerniq install. Publish this folder as its own repository. Kerniq loads it only after it is installed into `plugins/`.

`model_advice` reads the quantization, parameter count, and context window from a model name (or from the arguments you pass) and compares that with the machine Kerniq is running on. The presets are `balance`, `performance`, and `context`. If you do not name a model, the tool uses the model list of the provider already selected in Kerniq. That list is read on this computer. If the provider does not answer, the tool still returns a size, a quant, and a context that fit the machine.

Installed, the plugin adds a Model advisor page to Kerniq settings, the same way the benchmark adds its page. The page does not ask you to type models. It uses the provider selected in Kerniq and that provider's models, including each model's parameter count and context window.

Kerniq collects the operating system, CPU, RAM, and, when `nvidia-smi` is present, GPU memory. Those readings stay on this computer. This plugin does not send them anywhere. The weight and KV-cache figures are estimates, not a promise that a server will load the model.
