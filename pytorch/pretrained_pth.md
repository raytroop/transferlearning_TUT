## 0. [How to extract features of an image from a trained model](https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/7)
### 0.0 [https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/13](https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/13)
@alexis-jacq I wouldn’t recommend that. It’s better to keep your models stateless i.e. not hold any of the intermediate states. Otherwise, if you don’t pay enough attention to them, you might end up with problems when you’ll have references to the graphs you don’t need, and they will be only taking up memory.

If you really want to do something like that, I’d recommend this:
```
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]
```
This unfortunately uses a private member _modules, but I don’t expect it to change in the near future, and we’ll probably expose an API for iterating over modules with names soon.


### 0.1 [https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/49](https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/49)
Because of the way Inception_v3 is structured 59, you would need to do some more manual work to get the layers that you want.
Something on the lines of
```
class MyInceptionFeatureExtractor(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(MyInceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_3a_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_3x3
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        # stop where you want, copy paste from the model def

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # copy paste from model definition, just stopping where you want
        return x

inception = torchvision.models['inception_v3_google']
my_inception = MyInceptionFeatureExtractor(inception)
```

## 1. [How to load part of pre trained model?](https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113)
### 1.0 [https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3](https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3),
[https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16](https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/16)

You can remove all keys that don’t match your model from the state dict and use it to load the weights afterwards:

```
pretrained_dict = ...
model_dict = model.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)
```