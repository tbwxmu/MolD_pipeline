import torch
import torch.nn as nn
from torchvision.utils import save_image

class SOM(nn.Module):
    def __init__(self, input_size, out_size=(10, 10), lr=0.3, sigma=None):
        '''

        :param input_size:
        :param out_size:
        :param lr:
        :param sigma:
        '''
        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size

        self.lr = lr
        if sigma is None:
            self.sigma = max(out_size) / 2 #半径
        else:
            self.sigma = float(sigma)
        #d,x*y
        self.weight = nn.Parameter(torch.randn(input_size, out_size[0] * out_size[1]), requires_grad=False)#input_dim,map_size
        print(f'self.weight.size=={self.weight.size()}')
        self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)#https://blog.csdn.net/mingtian715/article/details/51163986
        #p=2 范数  就 l2, 不同维度的向量， [2]-->[2,2,2]
        # 3维以上是 按列向量来处理了 第2维dim=1 被挤掉
    def get_map_index(self):
        '''Two-dimensional mapping function'''
        for x in range(self.out_size[0]):
            for y in range(self.out_size[1]):
                yield (x, y)

    def _neighborhood_fn(self, input, current_sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(current_sigma ** 2)#input 直接change
        input.neg_()
        input.exp_()

        return input

    def forward(self, input):
        '''
        Find the location of best matching unit.
        :param input: data
        :return: location of best matching unit, loss
        '''
        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)#batch_size,d,1
        batch_weight = self.weight.expand(batch_size, -1, -1)#expand 重复 weight 相当于 b,d,x*y
        #print(f'input={input.size()}\nbatch_weight={batch_weight.size()}')
        dists = self.pdist_fn(input, batch_weight)
        #p=2 范数  就 l2, 不同维度的向量， [2]-->[2,2,2]
        # 3维以上是 按列向量来处理了 第2维dim=1 讲被挤掉 pdist_fn  pytorch pairwise_distance
        #torch.Size([32, 784, 400]) 784 是维度  400 是 map 神经元个数
        #torch.Size([32, 784, 1])
        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        bmu_locations = self.locations[bmu_indexes]

        return bmu_locations, losses.sum().div_(batch_size).item()

    def self_organizing(self, input, current_iter, max_iter):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''
        batch_size = input.size()[0]
        #Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction

        #Find best matching unit
        bmu_locations, loss = self.forward(input)#

        distance_squares = self.locations.float() - bmu_locations.float()#b,map_size,2(i_j posizition)
        #print(f'distance_squares.size={distance_squares.size()},{distance_squares}')#[32, 400, 2])
        distance_squares.pow_(2)
        #print(f'distance_squares.size={distance_squares.size()},pow_2{distance_squares}')
        distance_squares = torch.sum(distance_squares, dim=2) #32, 400

        lr_locations = self._neighborhood_fn(distance_squares, sigma)#lr_locations: e^(-(input / sigma^2))
        lr_locations.mul_(lr).unsqueeze_(1)#32,1,400
        #print(f'lr_locations.size={lr_locations.size()}')#32,1,400
        delta = lr_locations * (input.unsqueeze(2) - self.weight)#input.unsqueeze(2)=torch.Size([32, 784, 1]),self.weight=torch.Size([784, 400])delta.size=torch.Size([32, 784, 400])
        #print(f'delta.size={delta.size()},input.unsqueeze(2)={input.unsqueeze(2).size()},self.weight={self.weight.size()}')
        delta = delta.sum(dim=0)
        delta.div_(batch_size)#delta 是平均 化的 距离！！！todo check 更新W 是基于平均距离的
        self.weight.data.add_(delta)
        #print(f'self.weight size={self.weight.size()}')#self.weight size=torch.Size([784, 400]) todo input_size, out_size[0] * out_size[1]

        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])

        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])
