# Model advisor

Kerniq plugin. This folder is the whole plugin. It is not part of the Kerniq install. Publish this folder as its own repository. Kerniq loads it only after it is installed into `plugins/`.

`model_advice` reads the quantization, parameter count, and context window from a model name (or from the arguments you pass) and compares that with the machine Kerniq is running on. The presets are `balance`, `performance`, and `context`. If you do not name a model, the tool uses the model list of the provider already selected in Kerniq. That list is read on this computer. If the provider does not answer, the tool still returns a size, a quant, and a context that fit the machine.

A name that does not contain a parameter count, such as `7b` or `32b`, is not given one. The tool reports what this computer can hold, and, separately, the best installed model whose size is actually known. It does not average the list, and it does not treat every unnamed model as the same size.

Installed, the plugin adds a Model advisor page to Kerniq settings. The page does not ask you to type models. It uses the provider selected in Kerniq and that provider's models, including each model's parameter count and context window when the provider sent them.

Kerniq collects the operating system, CPU, RAM, and, when `nvidia-smi` is present, GPU memory. Those readings stay on this computer. This plugin does not send them anywhere. The weight and KV-cache figures are estimates, not a promise that a server will load the model.
