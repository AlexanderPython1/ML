

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


VAE_LATENT_FACTOR = 8  
VAE_CHANNELS = 128
UNET_BASE = 64
VOXEL_GRID = (32, 32, 8)
FEATURE_DIM = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def read_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def load_image_original(path: Path) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img)


def homogeneous_transform(points: np.ndarray, mat: np.ndarray, pad: bool = True) -> np.ndarray:
    pts = points
    if pad:
        ones = np.ones((*pts.shape[:-1], 1), dtype=pts.dtype)
        pts = np.concatenate([pts, ones], axis=-1)
    res = pts @ mat.T
    if res.shape[-1] == 3:
        return res
    out = res[..., :-1] / (res[..., -1:] + 1e-8)
    return out


def project_points(points_xyz: np.ndarray, K: np.ndarray, extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cam_pts = homogeneous_transform(points_xyz, extrinsics)
    uv = homogeneous_transform(cam_pts, K, pad=False)
    depths = cam_pts[:, 2]
    return uv, depths


def rasterize_bboxes(projected_boxes: List[Tuple[np.ndarray, int]], image_size: Tuple[int,int]) -> np.ndarray:
    H, W = image_size
    inst = np.zeros((H, W), dtype=np.uint8)
    obj = np.zeros((H, W), dtype=np.uint8)
    for i, (pts, cls) in enumerate(projected_boxes):
        pts_i = np.round(pts).astype(np.int32)
        pts_i[:,0] = np.clip(pts_i[:,0], 0, W-1)
        pts_i[:,1] = np.clip(pts_i[:,1], 0, H-1)
        bottom = pts_i[[0,2,6,4]]
        try:
            cv2.fillPoly(inst, [bottom], color=1)
            cv2.fillPoly(obj, [bottom], color=1)
        except Exception:
            pass
    return np.stack([inst.astype(np.float32), obj.astype(np.float32)], axis=0)


def rasterize_map(map_data: Dict[str, Any], K: np.ndarray, extrinsics: np.ndarray, image_size: Tuple[int,int]) -> np.ndarray:
    H, W = image_size
    lane = np.zeros((H, W), dtype=np.uint8)
    crosswalk = np.zeros((H, W), dtype=np.uint8)
    lights = np.zeros((H, W), dtype=np.uint8)
    for cl in map_data.get('centerlines', []):
        for seg in cl:
            x1,y1,x2,y2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
            p1 = np.array([[x1,y1,0.0]]); p2 = np.array([[x2,y2,0.0]])
            uv1,_ = project_points(p1, K, extrinsics); uv2,_ = project_points(p2, K, extrinsics)
            u1,v1 = uv1[0]; u2,v2 = uv2[0]
            if np.isfinite(u1) and np.isfinite(u2):
                cv2.line(lane, (int(u1), int(v1)), (int(u2), int(v2)), color=1, thickness=2)
    for cw in map_data.get('crosswalks', []):
        for seg in cw:
            x1,y1,x2,y2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
            p1 = np.array([[x1,y1,0.0]]); p2 = np.array([[x2,y2,0.0]])
            uv1,_ = project_points(p1, K, extrinsics); uv2,_ = project_points(p2, K, extrinsics)
            u1,v1 = uv1[0]; u2,v2 = uv2[0]
            if np.isfinite(u1) and np.isfinite(u2):
                cv2.line(crosswalk, (int(u1), int(v1)), (int(u2), int(v2)), color=1, thickness=3)
    for tl in map_data.get('traffic_lights', []):
        x,y,z = float(tl[0]), float(tl[1]), float(tl[2])
        p = np.array([[x,y,z]])
        uv,_ = project_points(p, K, extrinsics)
        u,v = uv[0]
        if np.isfinite(u) and 0<=int(u)<W and 0<=int(v)<H:
            cv2.circle(lights, (int(u), int(v)), radius=4, color=1, thickness=-1)
    return np.stack([lane.astype(np.float32), crosswalk.astype(np.float32), lights.astype(np.float32)], axis=0)


class SceneDataset(Dataset):
    def __init__(self, data_root: str, input_cam: str='right_rear', target_cameras: List[str]=None):
        self.root = Path(data_root)
        self.scenes = [p for p in self.root.iterdir() if (p / 'ride_id.json').exists()]
        self.input_cam = input_cam
        self.target_cameras = target_cameras or ['front_left','front_center','front_right','rear_center','rear_left']
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_dir = self.scenes[idx]
        bboxes = read_json(scene_dir / 'bboxes_3d_data.json')
        map_data = read_json(scene_dir / 'map_data.json')
        cam_params = read_json(scene_dir / 'camera_params.json')

        src_path = scene_dir / 'images_input' / f'{self.input_cam}.png'
        src_img = load_image_original(src_path)

        tgt_cam = np.random.choice(self.target_cameras)
        tgt_path = scene_dir / 'images_gt' / f'{tgt_cam}.png'
        tgt_img = load_image_original(tgt_path)

        H, W = tgt_img.shape[:2]
        cam_dict = cam_params[tgt_cam]
        K = np.array(cam_dict['intrinsics'], dtype=float).reshape(3,3)
        Rt = np.array(cam_dict['extrinsics'], dtype=float).reshape(4,4)

        projected = []
        bboxes_list = bboxes.get('bboxes', bboxes) if isinstance(bboxes, dict) else bboxes
        for bb in bboxes_list:
            pts = np.array(bb['corners_3d'], dtype=float).reshape(-1,3)
            uv, depths = project_points(pts, K, Rt)
            projected.append((uv, int(bb.get('class_id', 1))))

        bbox_mask = rasterize_bboxes(projected, (H,W))
        map_mask = rasterize_map(map_data, K, Rt, (H,W))

        # conditioning channels: bbox_mask(2) + map_mask(3) = 5 channels
        cond = np.concatenate([bbox_mask, map_mask], axis=0)

        src_t = self.to_tensor(src_img)
        tgt_t = self.to_tensor(tgt_img)
        cond_t = torch.from_numpy(cond).float()

        return {
            'source_img': src_t,
            'conditioning': cond_t,
            'target_img': tgt_t,
            'intrinsics': K,
            'extrinsics': Rt,
            'target_cam': tgt_cam,
            'scene_id': scene_dir.name,
            'orig_size': (H,W)
        }


class ConvEncoder(nn.Module):
    def __init__(self, in_ch=3, base=64, latent_ch=VAE_CHANNELS, factor=VAE_LATENT_FACTOR):
        super().__init__()
        self.factor = factor
        layers = []
        ch = in_ch
        out = base
        steps = int(math.log2(factor))
        for i in range(steps):
            layers += [nn.Conv2d(ch, out, 4, stride=2, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            ch = out; out = min(out*2, latent_ch)
        layers += [nn.Conv2d(ch, latent_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvDecoder(nn.Module):
    def __init__(self, out_ch=3, base=64, latent_ch=VAE_CHANNELS, factor=VAE_LATENT_FACTOR):
        super().__init__()
        self.factor = factor
        steps = int(math.log2(factor))
        layers = []
        ch = latent_ch
        out = max(ch//2, base)
        layers += [nn.Conv2d(ch, out, 3, padding=1), nn.ReLU(inplace=True)]
        for i in range(steps):
            layers += [nn.ConvTranspose2d(out, max(out//2, 16), 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            out = max(out//2, 16)
        layers += [nn.Conv2d(out, out_ch, 3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
                                 nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU())
    def forward(self,x): return self.net(x)

class LatentUNet(nn.Module):
    def __init__(self, in_ch, base=UNET_BASE):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)

class FeatureProjector3D(nn.Module):
    def __init__(self, feat_dim=FEATURE_DIM, voxel_grid=VOXEL_GRID, x_range=(0.0,50.0), y_range=(-25.0,25.0), z_range=(-1.0,3.0)):
        super().__init__()
        self.feat_dim = feat_dim
        self.vx, self.vy, self.vz = voxel_grid
        self.x_range = x_range; self.y_range = y_range; self.z_range = z_range
        self.proc = nn.Sequential(nn.Conv3d(feat_dim, feat_dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv3d(feat_dim, feat_dim, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, img_feats: torch.Tensor, K: np.ndarray, Rt: np.ndarray):
        B, C, Hf, Wf = img_feats.shape
        device = img_feats.device
        xs = np.linspace(self.x_range[0], self.x_range[1], self.vx)
        ys = np.linspace(self.y_range[0], self.y_range[1], self.vy)
        zs = np.linspace(self.z_range[0], self.z_range[1], self.vz)
        grid_pts = np.stack(np.meshgrid(xs, ys, zs, indexing='xy'), axis=-1).reshape(-1, 3)
        Kt = torch.from_numpy(K).to(device).float()
        Rtt = torch.from_numpy(Rt).to(device).float()
        pts_h = torch.from_numpy(np.concatenate([grid_pts, np.ones((len(grid_pts),1))], axis=1)).to(device).float()
        cam_pts = (Rtt @ pts_h.t()).t()[:, :3]
        proj = (Kt @ cam_pts.t()).t()
        uv = proj[:, :2] / (proj[:, 2:3] + 1e-6)
        # map to feature coords
        orig_H = img_feats.shape[2] * VAE_LATENT_FACTOR
        orig_W = img_feats.shape[3] * VAE_LATENT_FACTOR
        scale_x = float(Wf) / float(orig_W); scale_y = float(Hf) / float(orig_H)
        u_feat = uv[:,0] * scale_x; v_feat = uv[:,1] * scale_y
        x_norm = (u_feat / (Wf - 1)) * 2.0 - 1.0
        y_norm = (v_feat / (Hf - 1)) * 2.0 - 1.0
        sample_grid = torch.stack([x_norm, y_norm], dim=-1).view(1, len(grid_pts), 1, 2)
        feats = F.grid_sample(img_feats, sample_grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        feats = feats.view(B, C, self.vx, self.vy, self.vz)
        vol = self.proc(feats)
        return vol


class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.timesteps = timesteps
        self.beta = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

    def q_sample(self, x0: torch.Tensor, t: int, noise: torch.Tensor):
        a = float(self.alpha_bar[t])
        return torch.sqrt(torch.tensor(a, device=x0.device)) * x0 + torch.sqrt(1.0 - a) * noise


class FullModel(nn.Module):
    def __init__(self, cond_ch=5):
        super().__init__()
        self.encoder = ConvEncoder(in_ch=3, base=64, latent_ch=VAE_CHANNELS, factor=VAE_LATENT_FACTOR)
        self.decoder = ConvDecoder(out_ch=3, base=64, latent_ch=VAE_CHANNELS, factor=VAE_LATENT_FACTOR)
        self.projector_encoder = nn.Sequential(nn.Conv2d(3, FEATURE_DIM, 7, stride=2, padding=3), nn.ReLU(inplace=True), nn.Conv2d(FEATURE_DIM, FEATURE_DIM, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.feature_projector = FeatureProjector3D(feat_dim=FEATURE_DIM, voxel_grid=VOXEL_GRID)
        unet_in_ch = VAE_CHANNELS + cond_ch + FEATURE_DIM
        self.unet = LatentUNet(in_ch=unet_in_ch, base=UNET_BASE)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward_denoise(self, z_noisy: torch.Tensor, src_img: torch.Tensor, cond: torch.Tensor, K: np.ndarray, Rt: np.ndarray):
        feats = self.projector_encoder(src_img)
        vol = self.feature_projector(feats, K, Rt)
        vol_collapsed = vol.mean(dim=[3,4])
        vol_vec = vol_collapsed.mean(dim=2, keepdim=True)
        B, F, _ = vol_vec.shape
        H_lat, W_lat = z_noisy.shape[2], z_noisy.shape[3]
        vol_map = vol_vec.view(B,F,1,1).repeat(1,1,H_lat,W_lat)
        inp = torch.cat([z_noisy, cond, vol_map], dim=1)
        eps_pred = self.unet(inp)
        return eps_pred


def train_one_epoch(loader: DataLoader, model: FullModel, optim, scheduler, device: str):
    model.train()
    total = 0.0
    ns = NoiseScheduler(timesteps=1000)
    for batch in tqdm(loader):
        src = batch['source_img'].to(device)
        cond = batch['conditioning'].to(device)
        tgt = batch['target_img'].to(device)
        K = batch['intrinsics'][0] if isinstance(batch['intrinsics'], list) else batch['intrinsics']
        Rt = batch['extrinsics'][0] if isinstance(batch['extrinsics'], list) else batch['extrinsics']
        if isinstance(K, np.ndarray):
            K_np = K; Rt_np = Rt
        else:
            K_np = np.array(K); Rt_np = np.array(Rt)

        z0 = model.encode(tgt)
        t = np.random.randint(0, ns.timesteps)
        noise = torch.randn_like(z0).to(device)
        zt = ns.q_sample(z0, t, noise)

        cond_down = F.interpolate(cond, size=(zt.shape[2], zt.shape[3]), mode='bilinear', align_corners=False)

        eps_pred = model.forward_denoise(zt, src, cond_down, K_np, Rt_np)
        loss_diff = F.mse_loss(eps_pred, noise)

        recon = model.decode(z0)
        l1_img = F.l1_loss(recon, tgt)

        loss = loss_diff + 0.5 * l1_img
        optim.zero_grad(); loss.backward(); optim.step()
        total += loss.item()
    return total / len(loader)



def sample_from_noise(model: FullModel, src_img: torch.Tensor, cond: torch.Tensor, K: np.ndarray, Rt: np.ndarray, steps=50, device: str='cuda') -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        B = src_img.shape[0]
        z_shape = model.encode(src_img).shape
        z = torch.randn(z_shape, device=device)
        cond_down = F.interpolate(cond, size=(z.shape[2], z.shape[3]), mode='bilinear', align_corners=False)
        ns = NoiseScheduler(timesteps=1000)
        timesteps = np.linspace(0, ns.timesteps-1, steps).astype(int)[::-1]
        for t in timesteps:
            eps_pred = model.forward_denoise(z, src_img, cond_down, K, Rt)
            step_factor = 0.1
            z = z - step_factor * eps_pred
        img = model.decode(z)
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','infer'], default='train')
    parser.add_argument('--data_root', type=str, default='data/scenes')
    parser.add_argument('--scene', type=str, default='')
    parser.add_argument('--model', type=str, default='checkpoint_diff.pth')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    if args.mode == 'train':
        ds = SceneDataset(args.data_root)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        model = FullModel(cond_ch=5).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        for e in range(args.epochs):
            avg = train_one_epoch(dl, model, optim, None, DEVICE)
            print(f"Epoch {e} avg loss {avg:.4f}")
            torch.save(model.state_dict(), f'checkpoint_diff_epoch{e}.pth')
    else:
        if args.scene == '':
            raise ValueError('Provide --scene for inference')
        model = FullModel(cond_ch=5).to(DEVICE)
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))
        ds = SceneDataset(args.scene)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        os.makedirs('out', exist_ok=True)
        for i, batch in enumerate(dl):
            src = batch['source_img'].to(DEVICE)
            cond = batch['conditioning'].to(DEVICE)
            K = batch['intrinsics'][0]
            Rt = batch['extrinsics'][0]
            img = sample_from_noise(model, src, cond, K, Rt, steps=50, device=DEVICE)
            out = (img[0].cpu().numpy().transpose(1,2,0)+1.0)/2.0
            out = np.clip(out*255.0,0,255).astype(np.uint8)
            Image.fromarray(out).save(f'out/pred_{i}.png')
            print('Saved out/pred_{i}.png')
