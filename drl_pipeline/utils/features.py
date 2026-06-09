import torch
import numpy as np

from utils.orbit_lite.adapter import single_obs_to_tensor
from utils.orbit_lite.obs import parse_obs
from utils.orbit_lite.movement import MovementConfig
from utils.orbit_lite.movement_step import ensure_planet_movement
from utils.orbit_lite.distance_cache import build_distance_cache
from utils.orbit_lite.planner_core import (
    largest_initial_player_count,
    safe_drain,
    capture_floor,
    is_comet_planet,
    cheap_enemy_pressure,
    friendly_flip_targets,
    reachable_mask,
    reinforcement_timing_factor
)
from utils.orbit_lite.intercept_aim import intercept_angle
from agents.drl.utils.orbit_physics_jit import jit_point_to_segment_dist
from numba import njit
import math

@njit(cache=True)
def jit_all_pairs_intercept_eta(planet_matrix, ang_vels, speed, H):
    N = planet_matrix.shape[0]
    eta_out = np.full((N, N), np.inf, dtype=np.float32)
    viable = np.zeros((N, N), dtype=np.bool_)
    
    for src in range(N):
        sx = planet_matrix[src, 0]
        sy = planet_matrix[src, 1]
        sr = planet_matrix[src, 2]
        for tgt in range(N):
            tx = planet_matrix[tgt, 0]
            ty = planet_matrix[tgt, 1]
            tr = planet_matrix[tgt, 2]
            
            if src == tgt:
                eta_out[src, tgt] = 0.0
                viable[src, tgt] = True
                continue
                
            gap = sr + 0.1 + tr + 0.0
            d0 = math.hypot(tx - sx, ty - sy)
            t_star = max(0.0, min(float(H), (d0 - gap) / speed))
            
            omega = ang_vels[tgt]
            R = math.hypot(tx - 50.0, ty - 50.0)
            a0 = math.atan2(ty - 50.0, tx - 50.0)
            
            for _ in range(6):
                ang = a0 + omega * t_star
                cx = 50.0 + R * math.cos(ang)
                cy = 50.0 + R * math.sin(ang)
                d = math.hypot(cx - sx, cy - sy)
                t_star = max(0.0, min(float(H), (d - gap) / speed))
                
            ang = a0 + omega * t_star
            cx = 50.0 + R * math.cos(ang)
            cy = 50.0 + R * math.sin(ang)
            angle = math.atan2(cy - sy, cx - sx)
            
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            launch_x = sx + cos_a * (sr + 0.1)
            launch_y = sy + sin_a * (sr + 0.1)
            
            # Broad phase cull
            eta_cap = min(float(H), t_star + 2.0)
            seg_len = speed * eta_cap + tr + 2.0
            end_x = launch_x + cos_a * seg_len
            end_y = launch_y + sin_a * seg_len
            seg_xmin = min(launch_x, end_x)
            seg_xmax = max(launch_x, end_x)
            seg_ymin = min(launch_y, end_y)
            seg_ymax = max(launch_y, end_y)
            
            contact_planet = -1
            
            for tick in range(1, int(H) + 1):
                fx0 = launch_x + cos_a * speed * (tick - 1)
                fy0 = launch_y + sin_a * speed * (tick - 1)
                fx1 = launch_x + cos_a * speed * tick
                fy1 = launch_y + sin_a * speed * tick
                
                if fx1 < 0 or fx1 > 100 or fy1 < 0 or fy1 > 100:
                    break
                
                vx = fx1 - fx0
                vy = fy1 - fy0
                wx = 50.0 - fx0
                wy = 50.0 - fy0
                vv = max(1e-12, vx*vx + vy*vy)
                t_sun = max(0.0, min(1.0, (wx*vx + wy*vy)/vv))
                cxp = fx0 + t_sun * vx
                cyp = fy0 + t_sun * vy
                if (cxp - 50.0)**2 + (cyp - 50.0)**2 < 64.0:
                    break
                    
                hit_p = -1
                for p in range(N):
                    px = planet_matrix[p, 0]
                    py = planet_matrix[p, 1]
                    pr = planet_matrix[p, 2]
                    
                    if (seg_xmax < px - pr or seg_xmin > px + pr or
                        seg_ymax < py - pr or seg_ymin > py + pr):
                        continue
                        
                    if ang_vels[p] != 0.0:
                        pr_ang = math.atan2(py - 50.0, px - 50.0) + ang_vels[p] * tick
                        pr_R = math.hypot(px - 50.0, py - 50.0)
                        px1 = 50.0 + pr_R * math.cos(pr_ang)
                        py1 = 50.0 + pr_R * math.sin(pr_ang)
                    else:
                        px1 = px
                        py1 = py
                        
                    dist = jit_point_to_segment_dist(px1, py1, fx0, fy0, fx1, fy1)
                    if dist <= pr:
                        hit_p = p
                        break
                        
                if hit_p != -1:
                    contact_planet = hit_p
                    break
                    
            if contact_planet == tgt:
                viable[src, tgt] = True
                eta_out[src, tgt] = t_star
                
    return eta_out, viable

def build_graph_features(obs, config):
    player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    device = obs_tensors["planets"].device
    
    o = parse_obs(obs_tensors)
    player_count = largest_initial_player_count(obs_tensors)
    
    H = 15
    m_config = MovementConfig(
        movement_horizon=H,
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=player_count,
        max_tracked_fleets=128
    )
    movement = ensure_planet_movement(obs_tensors=obs_tensors, expected_cfg=m_config, cached_movement=None)
    cache = build_distance_cache(movement, max_k=H)
    status = movement.garrison_status(max_horizon=H)
    
    P = int(o.P)
    
    V = torch.zeros((P, 13), dtype=torch.float32, device=device)
    E = torch.zeros((P, P, 4), dtype=torch.float32, device=device)
    
    if P == 0:
        return V.numpy(), E.numpy()
        
    V[:, 0] = movement.radii / 50.0
    V[:, 1] = movement.planet_prod / 5.0
    
    garrison_at_H = status.ships[:, -1]
    V[:, 2] = torch.log1p(garrison_at_H.clamp(min=0.0)) / 10.0
    
    owner_now = o.owner_abs
    V[:, 3] = (owner_now == player_id).float()
    V[:, 7] = ((owner_now == 4) | (owner_now == -1)).float()
    for rel_idx in range(3):
        V[:, 4 + rel_idx] = (owner_now == ((player_id + 1 + rel_idx) % 4)).float()
        
    comet_mask = is_comet_planet(obs_tensors, P, device)
    if comet_mask is not None:
        V[:, 8] = comet_mask.float()
        
    pressure = cheap_enemy_pressure(o, cache, horizon=float(H), player_id=player_id)
    V[:, 9] = pressure / 100.0
    
    all_targets = torch.arange(P, device=device)
    c_floor = capture_floor(status, target_idx=all_targets, k_max=H, capture_overhead=1.0, player_id=player_id)
    if c_floor.shape[-1] > 0:
        V[:, 10] = c_floor[:, -1] / 100.0
    
    H_eff = torch.full((), float(H), device=device)
    s_drain = safe_drain(status, source_idx=all_targets, source_ships=o.ships, H_eff=H_eff, player_id=player_id)
    V[:, 11] = s_drain / 100.0
    
    flip_mask, urgency = friendly_flip_targets(o, status, H=H, prod=movement.planet_prod)
    V[:, 12] = torch.where(flip_mask, urgency, torch.zeros_like(urgency)) / 100.0
    
    src_idx = torch.arange(P, device=device)
    tgt_idx = torch.arange(P, device=device)
    dummy_sizes = torch.full((P, P), 10.0, device=device)
    eta_cap = torch.full((P,), float(H), device=device)
    
    # Calculate planet matrix and angular velocities for Numba
    t0x, t0y = movement.position_at_slots(tgt_idx, 0)
    t1x, t1y = movement.position_at_slots(tgt_idx, 1)
    a0 = torch.atan2(t0y - 50.0, t0x - 50.0)
    a1 = torch.atan2(t1y - 50.0, t1x - 50.0)
    omega = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0)).cpu().numpy().astype(np.float32)
    
    planet_matrix = np.zeros((P, 3), dtype=np.float32)
    planet_matrix[:, 0] = t0x.cpu().numpy()
    planet_matrix[:, 1] = t0y.cpu().numpy()
    planet_matrix[:, 2] = movement.radii[:P].cpu().numpy()
    
    speed = 1.0 + 5.0 * (math.log(10.0) / math.log(1000.0)) ** 1.5
    
    eta_np, viable_np = jit_all_pairs_intercept_eta(planet_matrix, omega, speed, float(H))
    
    eta_out = torch.from_numpy(eta_np).to(device)
    viable = torch.from_numpy(viable_np).to(device)
    
    E[:, :, 0] = torch.where(torch.isinf(eta_out), torch.full_like(eta_out, float(H)), eta_out) / float(H)
    E[:, :, 1] = viable.float()
    
    timing = reinforcement_timing_factor(eta_out, eta_free=0.0, eta_scale=float(H))
    E[:, :, 2] = timing
    E[:, :, 3] = cache.cross_dist[0][:P, :P] / 100.0
    # The GAT model and dataset_gen.py hardcode exactly 50 planets. 
    # orbit_lite pads to 64. We slice the first 50.
    return V.cpu().numpy()[:50], E.cpu().numpy()[:50, :50], movement, obs_tensors
