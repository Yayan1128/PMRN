import math
import random

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.擦除面积与输入图像的最小比例
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.擦除面积的最小纵横比
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:#随机生成一个数，若>=self.probability，就不在随机擦除
            return img

        for attempt in range(100):#循环100次为了让擦拭能够进行，即 w < img.size()[2] and h < img.size()[1]
            area = img.size()[1] * img.size()[2]#img.size()=tensor.Size([3,300,400])  area=120000,
            # print(random.uniform(self.sl, self.sh))
            target_area = random.uniform(self.sl, self.sh) * area#random.uniform(self.sl, self.sh)=0.087,target_area=43925
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)#=1.851555500982092

            h = int(round(math.sqrt(target_area * aspect_ratio)))#224
            w = int(round(math.sqrt(target_area / aspect_ratio)))#156
            # print(h,w)
            #
            if w < img.size()[2] and h < img.size()[1]:#宽400，高300
                x1 = random.randint(0, img.size()[1] - h)#37
                y1 = random.randint(0, img.size()[2] - w)#90
                if img.size()[0] == 3:#图片为三个通道，则三个通道都赋值
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]#img[0, x1:x1 + h, y1:y1 + w].shape=torch.Size([224, 156]),这么大区域全部附上0.4914这个值
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img