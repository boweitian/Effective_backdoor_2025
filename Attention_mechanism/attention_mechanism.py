import torch
import numpy as np

class GradCam():
    def __init__(self, model, target_layer, use_cuda):
        self.model = model.eval()
        self.target_layer = target_layer
        self.use_cuda = use_cuda
        self.feature_map = 0
        self.grad = 0

        if self.use_cuda:
            self.model = self.model.cuda()

        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.save_feature_map)
                module[1].register_backward_hook(self.save_grad)

    def save_feature_map(self, module, input, output):
        self.feature_map = output.detach()

    def save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def __call__(self, x, output_size, index=None):
        x = x.clone()
        if self.use_cuda:
            x = x.cuda()

        output = self.model(x)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad_()
        if self.use_cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()

        one_hot.backward()
        # print('-------------------------')
        self.feature_map = self.feature_map.cpu().numpy()[0]
        # print('feature map size', self.feature_map.size)
        self.weights = np.mean(self.grad.cpu().numpy(), axis=(2, 3))[0, :]
        # print('weight size', self.weights.size)
        mask = np.sum(self.feature_map * self.weights[:, None, None], axis=0)
        # print('mask', mask.shape)

        # mask = np.sum(self.feature_map, axis=0)
        mask = np.maximum(mask, 0)
        mask = cv2.resize(mask, output_size)
        mask = mask - np.min(mask)

        return mask, output