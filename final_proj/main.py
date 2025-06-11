import os
import sys
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v3 as iio


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler

def render_scene(environment, mainImage, Vec3, Vec4, w, h, env_width, env_height):
    output = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            fragCoord = Vec3(x, y, 0.0)
            fragColor = Vec4(0.0, 0.0, 0.0, 0.0)
            mainImage(fragCoord,
                      float(w), float(h), 0.0,
                      environment.ctypes.data_as(ctypes.POINTER(Vec4)),
                      env_width, env_height,
                      ctypes.byref(fragColor))
            output[y, x, 0] = fragColor.x
            output[y, x, 1] = fragColor.y
            output[y, x, 2] = fragColor.z
            output[y, x, 3] = fragColor.w
    return output


if __name__ == '__main__':

    # image size
    w = 1280
    h = 720

    hdr_env = iio.imread('night_environment.hdr')
    max_val = np.max(hdr_env)
    hdr_env = hdr_env / max_val
    env_h, env_w = hdr_env.shape[:2]
    if hdr_env.shape[2] == 3:
        alpha = np.ones((env_h, env_w, 1), dtype=np.float32)
        environment = np.concatenate([hdr_env.astype(np.float32), alpha], axis=-1)
    else:
        environment = hdr_env.astype(np.float32)

    
    # environment = np.zeros((h, w, 4), dtype=np.float32)
    # for y in range(h):
    #     for x in range(w):
    #         environment[y, x, 0] = x / w           # R: 横向渐变
    #         environment[y, x, 1] = y / h           # G: 纵向渐变
    #         environment[y, x, 2] = 0.2             # B: 常数值
    #         environment[y, x, 3] = 1.0             # A: 不透明

    #### render the initial image ####
    with open('dragon_ball_render.py') as f:
        structs, lib = compiler.compile(f.read(),
                                        target='c',
                                        output_filename='_code/dragon_ball_render')

    mainImage = lib.mainImage
    Vec3 = structs['Vec3']
    Vec4 = structs['Vec4']

    initial_output = render_scene(environment, mainImage, Vec3, Vec4, w, h, env_w, env_h)

    plt.imshow(np.clip(initial_output[:, :, :3], 0.0, 1.0))
    plt.axis('off')
    plt.savefig('dragon_ball_initial.png', dpi=300, bbox_inches='tight', pad_inches=0)


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

    lr = 5e-2
    losses = []

    for step in range(300):
        grad_fn(
            environment.ctypes.data_as(ctypes.POINTER(Vec4)),
            g_env.ctypes.data_as(ctypes.POINTER(Vec4)),
            w,
            h,
            env_w, env_h,
            target_flat.ctypes.data_as(ctypes.POINTER(Vec4)),
            g_target.ctypes.data_as(ctypes.POINTER(Vec4))
        )
        g_env[:, :, 3] = 0.0

        environment[:, :, :3] -= lr * g_env[:, :, :3]
        g_env.fill(0.0)

        loss = loss_fn(
            environment.ctypes.data_as(ctypes.POINTER(Vec4)),
            w, h,
            env_w, env_h,
            target_flat.ctypes.data_as(ctypes.POINTER(Vec4)),
        )
        losses.append(loss)
        print(f"[Step {step}] Loss = {loss:.6f}")

        if step % 10 == 0:
            intermediate_output = render_scene(environment, mainImage, Vec3, Vec4, w, h, env_w, env_h)
            plt.imshow(np.clip(intermediate_output[:, :, :3], 0.0, 1.0))
            plt.axis('off')
            plt.savefig(f"images/render_step_{step:03d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

    
    intermediate_output = render_scene(environment, mainImage, Vec3, Vec4, w, h, env_w, env_h)
    plt.imshow(np.clip(intermediate_output[:, :, :3], 0.0, 1.0))
    plt.axis('off')
    plt.savefig(f"images/render_step_final.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimization_curve.png", dpi=150)
        
