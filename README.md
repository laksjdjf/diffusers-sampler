# diffusers-sampler

このリポジトリは、diffusersで使えるサンプラーの簡易実装をまとめたものです。

方針は以下のような感じです。

1. 共通コードをまとめた親クラスの実装
2. 一般的に使われない条件分岐や引数を省略

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

# replace scheduler
from ddpm import SimpleDDPMScheduler
pipe.scheduler = SimpleDDPMScheduler()

seed = 4545
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=20, generator = torch.Generator("cuda").manual_seed(seed)).images[0]  
    
image.save("astronaut_rides_horse.png")
```
