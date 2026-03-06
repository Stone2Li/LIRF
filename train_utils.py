import torch
def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

def pachify(images, patch_size, full_size):
    """
    输入:
        images: (B, C, H, W) - 全尺寸 Latent
        patch_size: int - 目标裁剪尺寸 (比如 16 或 32)
        full_size: int - 原始尺寸 (比如 32)
    输出:
        patches: (B, C, patch_size, patch_size)
        coords: (B, 2, patch_size, patch_size) - 归一化到 [-1, 1]
    """
    device = images.device
    batch_size, channels, h, w = images.shape
    
    # 如果 patch_size 等于原图，直接返回全图和全图坐标
    if patch_size == h and patch_size == w:
        # 生成全图坐标网格
        y_ids = torch.arange(h, device=device).float()
        x_ids = torch.arange(w, device=device).float()
        # align_corners=False 的归一化公式: 2 * (i + 0.5) / size - 1
        y_coords = 2 * (y_ids + 0.5) / h - 1.0
        x_coords = 2 * (x_ids + 0.5) / w - 1.0
        
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij') # (H, W)
        coords = torch.stack([grid_x, grid_y], dim=0) # (2, H, W)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (B, 2, H, W)
        return images, coords

    # 随机裁剪逻辑
    # 随机选择左上角 (i, j)
    # i: height index, j: width index
    i = torch.randint(0, h - patch_size + 1, (batch_size,), device=device)
    j = torch.randint(0, w - patch_size + 1, (batch_size,), device=device)

    # 构建裁剪后的图像 Patch
    # 使用高级索引进行批量裁剪
    rows = torch.arange(patch_size, device=device).view(1, -1) + i.view(-1, 1) # (B, patch_size)
    cols = torch.arange(patch_size, device=device).view(1, -1) + j.view(-1, 1) # (B, patch_size)
    
    # Advanced indexing: images[b, c, rows[b], cols[b]]
    # 需要构造完整的 grid 索引
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1)
    rows_grid = rows.view(batch_size, patch_size, 1).repeat(1, 1, patch_size)
    cols_grid = cols.view(batch_size, 1, patch_size).repeat(1, patch_size, 1)
    
    patches = images[batch_idx, :, rows_grid, cols_grid]
    # Advanced indexing yields (B, P, P, C); convert back to NCHW.
    patches = patches.permute(0, 3, 1, 2).contiguous()

    # 构建坐标 Patch (归一化到 -1, 1 用于 grid_sample)
    # 公式: coord = 2 * (index + 0.5) / full_size - 1
    # 我们直接用 rows_grid 和 cols_grid (它们包含了在大图中的绝对索引)
    
    y_coords = 2 * (rows_grid.float() + 0.5) / full_size - 1.0
    x_coords = 2 * (cols_grid.float() + 0.5) / full_size - 1.0
    
    coords = torch.stack([x_coords, y_coords], dim=1) # (B, 2, P, P)
    
    return patches, coords
