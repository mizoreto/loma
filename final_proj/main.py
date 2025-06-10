import os
import sys
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler

def render_scene(environment, mainImage, Vec3, Vec4, w, h):
    output = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            fragCoord = Vec3(x, y, 0.0)
            fragColor = Vec4(0.0, 0.0, 0.0, 0.0)
            mainImage(fragCoord,
                      float(w), float(h), 0.0,
                      environment.ctypes.data_as(ctypes.POINTER(Vec4)),
                      w, h,
                      ctypes.byref(fragColor))
            output[y, x, 0] = fragColor.x
            output[y, x, 1] = fragColor.y
            output[y, x, 2] = fragColor.z
            output[y, x, 3] = fragColor.w
    return output


if __name__ == '__main__':

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

    #### render the initial image ####
    with open('dragon_ball_render.py') as f:
        structs, lib = compiler.compile(f.read(),
                                        target='c',
                                        output_filename='_code/dragon_ball_render')

    mainImage = lib.mainImage
    Vec3 = structs['Vec3']
    Vec4 = structs['Vec4']

    initial_output = render_scene(environment, mainImage, Vec3, Vec4, w, h)

    plt.imshow(np.clip(initial_output[:, :, :3], 0.0, 1.0))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('dragon_ball_initial.png', dpi=300)

    #### optimization ####
    grad_fn = lib.grad_loss_fn
    loss_fn = lib.loss_fn

    target_image = np.array(Image.open('target.png').resize((w, h))) / 255.0
    if target_image.shape[2] == 3:
        alpha = np.ones((h, w, 1))
        target_image = np.concatenate([target_image, alpha], axis=-1)
    target = target_image.astype(np.float32)
    target_flat = target.reshape((-1, 4))

    g_env = np.zeros_like(environment)
    g_target = np.zeros_like(target_flat)

    lr = 5e-3
    losses = []

    # 训练循环
    for step in range(300):
        grad_fn(
            environment.ctypes.data_as(ctypes.POINTER(Vec4)),
            g_env.ctypes.data_as(ctypes.POINTER(Vec4)),
            w,
            h,
            target_flat.ctypes.data_as(ctypes.POINTER(Vec4)),
            g_target.ctypes.data_as(ctypes.POINTER(Vec4))
        )
        g_env[:, :, 3] = 0.0

        # 梯度下降
        environment[:, :, :3] -= lr * g_env[:, :, :3]
        environment = np.clip(environment, 0.0, 1.0)
        g_env.fill(0.0)

        # 当前 loss
        loss = loss_fn(
            environment.ctypes.data_as(ctypes.POINTER(Vec4)),
            w, h,
            target_flat.ctypes.data_as(ctypes.POINTER(Vec4)),
        )
        losses.append(loss)
        print(f"[Step {step}] Loss = {loss:.6f}")

        if step % 10 == 9:
            intermediate_output = render_scene(environment, mainImage, Vec3, Vec4, w, h)
            plt.imshow(np.clip(intermediate_output[:, :, :3], 0.0, 1.0))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"images/render_step_{step:03d}.png", dpi=300)
            plt.close()


    # 保存结果
    np.save("optimized_environment.npy", environment)

    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimization_curve.png", dpi=150)
        
