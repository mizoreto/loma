import os
import sys
import ctypes
import numpy as np
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler

if __name__ == '__main__':
    with open('dragon_ball_render.py') as f:
        structs, lib = compiler.compile(f.read(),
                                        target='c',
                                        output_filename='_code/dragon_ball_render')

    # 图像尺寸
    w = 1280
    h = 720

    # 输入图像（环境贴图），这里生成一个简单的渐变背景替代贴图加载
    environment = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            environment[y, x, 0] = x / w           # R: 横向渐变
            environment[y, x, 1] = y / h           # G: 纵向渐变
            environment[y, x, 2] = 0.2             # B: 常数值
            environment[y, x, 3] = 1.0             # A: 不透明

    # 输出图像
    output = np.zeros((h, w, 4), dtype=np.float32)

    # 获取 Loma 函数句柄
    mainImage = lib.mainImage
    Vec3 = structs['Vec3']
    Vec4 = structs['Vec4']

    # 渲染循环
    for y in range(h):
        for x in range(w):
            fragCoord = Vec3(x, y, 0.0)
            fragColor = Vec4(0.0, 0.0, 0.0, 0.0)
            mainImage(fragCoord,
                      float(w),
                      float(h),
                      0.0,  # iTime = 0
                      environment.ctypes.data_as(ctypes.POINTER(Vec4)),
                      w,
                      h,
                      ctypes.byref(fragColor))
            output[y, x, 0] = fragColor.x
            output[y, x, 1] = fragColor.y
            output[y, x, 2] = fragColor.z
            output[y, x, 3] = fragColor.w

    plt.imshow(np.clip(output[:, :, :3], 0.0, 1.0))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('dragon_ball_output.png', dpi=300)
    plt.show()
