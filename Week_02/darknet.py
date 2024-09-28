import torch
import torch.nn as nn

# Darknet 네트워크 구조 설정
architecture_config = [
    (7, 64, 2, 3),  # (kernel size, number of filters of output, stride, padding)   ## (kyu) tuple, 단일 conv layer
    "M",  # Max-pooling 2x2 stride = 2   ## (kyu) string, max-pooling layer
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # [Conv1, Conv2, repeat times]   ## (kyu) list, conv layer 2개 + 반복 횟수
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# CNN 블록 정의
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        ## TODO
        ## 위에서 정의한 layer이용해서 구현하기

        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)

        return x

# Darknet 백본 네트워크 정의
class Darknet(nn.Module):
    def __init__(self, architecture, in_channels=3):
        super(Darknet, self).__init__()
        self.architecture = architecture   ## (kyu) architecture에 architecture.config를 대입하면 됨
        self.in_channels = in_channels
        self.layers = self._create_conv_layers(self.architecture)

    def forward(self, x):
        return self.layers(x)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        ## TODO
        # 위에서 정의한 CNNBlock을 이용해서 각 type에 맞는 layer를 정의해주세요!
        # hint: architecture_config 참고하기!!
        for x in architecture:
            if type(x) == tuple:  
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size = x[0], stride = x[2], padding = x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str: 
                # hint: maxpoolinglayer일 경우입니다!
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]

            elif type(x) == list:
                for _ in range(x[2]):
                    layers += [
                        CNNBlock(
                            in_channels, x[0][1], kernel_size = x[0][0], stride = x[0][2], padding = x[0][3]
                        ),
                        CNNBlock(
                            x[0][1], x[1][1], kernel_size = x[1][0], stride = x[1][2], padding = x[1][3]
                        )
                    ]
                    in_channels = x[1][1]
                
        return nn.Sequential(*layers)



def test():
    model = Darknet(architecture_config)

    # 더미 입력 데이터 생성를 생성합니다. (batchsize, channel, height, width)
    dummy_input = torch.randn(1, 3, 448, 448)

    # 모델에 데이터 전달 및 출력 확인
    output = model(dummy_input)
    print("Output shape:", output.shape)

    # print:[1, 1024, 7, 7]

if __name__ == "__main__":
    test()

